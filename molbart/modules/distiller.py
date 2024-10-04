# import collections
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# import torch.nn.functional as F
from textbrewer.presets import (  # ,; TEMPERATURE_SCHEDULER,; WEIGHT_SCHEDULER,
    KD_LOSS_MAP,
    MATCH_LOSS_MAP,
    PROJ_MAP,
)
from torch.nn import ModuleList

from molbart.models.bart import BARTModel

FEATURES = ["hidden", "attention"]


@dataclass
class ProjectionGroup:
    projection: str
    dim_in: int
    dim_out: int


@dataclass
class IntermediateMatches:
    layer_T: str = "hidden_state_T"
    layer_S: str = "hidden_state"
    feature: str = "hidden"
    loss: str = "hidden_mse"
    weight: float = 1.0
    projection: Optional[ProjectionGroup] = None

    def __post_init__(self) -> None:
        if type(self.projection) is list:
            self.projection = ProjectionGroup(*self.projection)
        elif type(self.projection) is dict:
            self.projection == ProjectionGroup(**self.projection)


@dataclass
class DistillationConfig:
    temperature: int = 4
    temperature_scheduler: Optional[str] = None
    distill_type: str = "relation"  # relation feature
    hard_label_weight: float = 0.0
    hard_loss_type: Optional[str] = "ce"
    hard_label_weight_scheduler: Optional[str] = None
    kd_loss_weight: float = 1.0
    kd_loss_type: Optional[str] = "ce"
    kd_loss_scheduler: Optional[str] = None
    probability_shift: bool = False  # TODO: find out what this does
    intermediate_matches: Optional[List[IntermediateMatches]] = None

    def __post_init__(self) -> None:
        if self.intermediate_matches is not None:
            _intermediate_matches = []
            for match in self.intermediate_matches:
                if type(match) is [omegaconf.listconfig.ListConfig, list]:
                    _intermediate_matches.append(IntermediateMatches(*match))
                elif type(match) in [omegaconf.dictconfig.DictConfig, dict]:
                    _intermediate_matches.append(IntermediateMatches(**match))
            self.intermediate_matches = _intermediate_matches


class Distiller(pl.LightningModule):
    def __init__(
        self,
        distil_cfg: Optional[DistillationConfig] = None,
        cache_teacher_logits: bool = False,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        # Settings
        self.cache_teacher = cache_teacher_logits
        self.batch_first = batch_first

        if distil_cfg is not None:
            self.distill_config = distil_cfg
        else:
            self.distill_config = DistillationConfig()
        self._configure_projections()

        # Outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.prediction_step_outputs = []
        self.prediction_outputs = []

        self._features = {}

        self.has_custom_matches = False
        self.distill_type = self.distill_config.distill_type
        # if distil_cfg.intermediate_matches is not None:
        #     self.has_custom_matches = True

        #     self.handles_T = []
        #     self.handles_S = []
        #     self.custom_matches_cache = {
        #         "hook_outputs_T": [],
        #         "hook_outputs_S": [],
        #         "match_proj_funcs": [],
        #         "match_weights": [],
        #         "match_losses": [],
        #         "match_proj_groups": [],
        #     }
        #     for match in distil_cfg.intermediate_matches:
        #         self.add_match(match)

        # Important: This property activates manual optimization.
        # self.automatic_optimization = False

    def save_outputs_hook(self, model_name: str, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[model_name][layer_id] = output

        return fn

    def post_adaptor(dict_object):
        if "logits" in dict_object:
            logits = dict_object["logits"]
            if not isinstance(logits, (list, tuple)):
                dict_object["logits"] = [logits]
        if "logits_mask" in dict_object:
            logits_mask = dict_object["logits_mask"]
            if not isinstance(logits_mask, (list, tuple)):
                dict_object["logits_mask"] = [logits_mask]
        if "losses" in dict_object:
            losses = dict_object["losses"]
            if not isinstance(losses, (list, tuple)):
                dict_object["losses"] = [losses]
        if "labels" in dict_object:
            labels = dict_object["labels"]
            if not isinstance(labels, (list, tuple)):
                dict_object["labels"] = [labels]
        return dict_object

    def set_teacher(
        self, model_T: List[BARTModel], adaptor_T: List[callable], names_T: Optional[List[str]] = None
    ) -> None:

        self.model_T = ModuleList(model_T)
        self.adaptor_T = adaptor_T
        self.names_T = names_T if names_T is not None else [f"T_{i}" for i in range(len(model_T))]

        for model, name in zip(self.model_T, names_T):
            model.freeze()
            self._features[name] = {}
            # model._features = {layer: torch.empty(0) for layer in layers}

            for match in self.distill_config.intermediate_matches:
                layer_id = match.layer_T
                if layer_id in dict([*model.named_modules()]).keys():
                    layer = dict([*model.named_modules()])[layer_id]
                    layer.register_forward_hook(self.save_outputs_hook(name, layer_id))

    def set_student(
        self, model_S: List[BARTModel], adaptor_S: List[callable], names_S: Optional[List[str]] = None
    ) -> None:
        self.model_S = ModuleList(model_S)
        self.adaptor_S = adaptor_S
        self.names_S = names_S if names_S is not None else [f"S_{i}" for i in range(len(model_S))]

        for model, name in zip(self.model_S, names_S):
            self._features[name] = {}

            for match in self.distill_config.intermediate_matches:
                layer_id = match.layer_S
                if layer_id in dict([*model.named_modules()]).keys():
                    layer = dict([*model.named_modules()])[layer_id]
                    layer.register_forward_hook(self.save_outputs_hook(name, layer_id))

    def return_student(self, idx: Optional[int] = None) -> pl.LightningModule:
        return self.model_S[idx] if idx is not None else self.model_S[0]

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        for model in self.model_T:
            model.freeze()

        if self.cache_teacher and self.cached_logits is not None:
            logits_T = self.gather_cached_logits(batch)  # TODO
        else:
            logits_T = [adaptor(model(batch)) for adaptor, model in zip(self.adaptor_T, self.model_T)]

        logits_S = [adaptor(model(batch)) for adaptor, model in zip(self.adaptor_S, self.model_S)]
        return logits_T, logits_S

    def _construct_decoder_batch(self, batch, memory) -> Dict[str, torch.Tensor]:
        return {**batch, "memory_input": memory, "memory_pad_mask": batch["encoder_pad_mask"].clone()}

    def on_train_start(self):
        self.tracker = self.model_S[0].tracker
        total_parameters = filter(lambda p: p.requires_grad, self.model_S[0].parameters())
        non_trainable_parameters = filter(lambda p: p.requires_grad is False, self.model_S[0].parameters())
        self.logger.log_hyperparams(
            {
                "model_parameters": sum([np.prod(p.size()) for p in total_parameters]),
                "non_trainable_parameters": sum([np.prod(p.size()) for p in non_trainable_parameters]),
            }
        )

        if self.tracker is not None:
            self.tracker.start()
            self.logger.log_hyperparams(
                {
                    k: v[0]
                    for k, v in self.tracker._construct_attributes_dict().items()
                    if k not in ["duration(s)", "power_consumption(kWh)", "CO2_emissions(kg)"]
                }
            )
            self._log_tracker()

    def training_step(self, batch: Any, batch_idx: Any) -> Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        optimizers = self.optimizers()
        if type(optimizers) is not list:
            optimizers = [optimizers]

        metric_dicts = {}

        # Teachers
        self.eval()
        with torch.no_grad():
            token_out_T = []
            for model, name in zip(self.model_T, self.names_T):
                model.freeze()

                model_output: Dict[str, torch.Tensor] = model.forward(batch)
                token_out_T.append(F.log_softmax(model_output["token_output"], dim=2))

                loss = model._calc_loss(batch, model_output)
                token_acc = model._calc_token_acc(batch, model_output)

                metric_dict = {"model_loss": loss, "token_acc": token_acc}
                metric_dict = {f"{name}/{k}": v for k, v in metric_dict.items()}
                metric_dicts = {**metric_dicts, **metric_dict}

        # Students
        self.train()
        for idx, (student_model, optimizer, name) in enumerate(zip(self.model_S, optimizers, self.names_S)):
            student_model.to(self.device)

            # self.toggle_optimizer(optimizer)
            # optimizer.zero_grad()

            model_output = student_model.forward(batch)
            original_loss = student_model._calc_loss(batch, model_output)
            soft_token_out = model_output["token_output"]
            hard_token_out = F.log_softmax(soft_token_out, dim=2)

            loss = self._calc_loss(hard_token_out, token_out_T, batch["target"], original_loss)

            # Optimize student
            # self.manual_backward(loss)
            # optimizer.step()
            # self.untoggle_optimizer(optimizer)

            token_acc = student_model._calc_token_acc(batch, model_output)
            if len(self.model_S) > 1:
                metric_dict = {f"{name}/model_loss": original_loss, f"{name}/token_acc": token_acc}
            else:
                metric_dict = {"original_loss": original_loss}

            metric_dict["loss"] = loss
            metric_dicts = {**metric_dicts, **metric_dict}

            # self.log("loss", loss, logger=True, batch_size=model_output["batch_size"], prog_bar=True)

        # Set outputs
        self.log("loss", loss, logger=True, batch_size=model_output["batch_size"], prog_bar=True)
        # self.training_step_outputs.append(metric_dicts)
        return loss

    # def on_train_epoch_end(self):
    #     outputs = self.training_step_outputs
    #     self.training_step_outputs = []
    #     avg_outputs = self._avg_dicts(outputs)
    #     self._log_dict(avg_outputs)

    def on_train_epoch_end(self):
        self._log_tracker()

    def on_train_end(self):
        if self.tracker is not None:
            self.tracker.stop()

    def validation_step(self, batch, batch_idx) -> None:
        metric_dicts = {}
        for student_model, name in zip(self.model_S, self.names_S):
            student_model.to(self.device)
            metric_dict = student_model.validation_step(batch, batch_idx)
            # validation_loss = {"validation_loss": metric_dict["validation_loss"]}
            if len(self.model_S) > 1:
                metric_dict = {f"{name}/{k}": v for k, v in metric_dict.items()}
            metric_dicts = {**metric_dicts, **metric_dict}

        self.validation_step_outputs.append(metric_dicts)

        self.log("validation_loss", metric_dict["validation_loss"], logger=True, batch_size=metric_dicts["batch_size"])

        return metric_dicts

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        self.validation_step_outputs = []  # free memory
        for student_model in self.model_S:
            student_model.validation_step_outputs = []  # free memory
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {k[10:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model_S.0.")}
        super().on_save_checkpoint(checkpoint)

    def on_test_start(self):
        if self.tracker is not None:
            self.tracker.start()
            self._log_tracker("test_")

    def test_step(self, batch, batch_idx) -> None:
        metric_dicts = {}
        for student_model, name in zip(self.model_S, self.names_S):
            student_model.to(self.device)
            metric_dict = student_model.test_step(batch, batch_idx)
            validation_loss = {"test_loss": metric_dict["test_loss"]}
            if len(self.model_S) > 1:
                metric_dict = {f"{name}/{k}": v for k, v in metric_dict.items()}
            metric_dicts = {**metric_dicts, **metric_dict, **validation_loss}

        self.test_step_outputs.append(metric_dicts)

        return metric_dicts

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        self.test_step_outputs = []  # free memory
        for student_model in self.model_S:
            student_model.validation_step_outputs = []  # free memory
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)
        self._log_tracker("test_")

    def on_test_end(self):
        if self.tracker is not None:
            self.tracker.stop()

    def configure_optimizers(self):
        optimizers, schedulers = [], []
        for model in self.model_S:
            optim, sch = model.configure_optimizers()
            optimizers.append(*optim)
            schedulers.append(*sch)
        # optimizers = [torch.optim.AdamW(model.parameters(), model.lr) for model in self.model_S]
        return optimizers, schedulers

    def _calc_loss(
        self,
        logits: torch.Tensor,
        logits_T: List[torch.Tensor],
        hard_labels: torch.Tensor,
        hard_label_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: schedule temperature
        temperature = self.distill_config.temperature
        loss = torch.tensor(0.0, device=self.device)

        norm_weights = (
            self.distill_config.hard_label_weight
            + self.distill_config.kd_loss_weight
            + sum([match_dict.weight for match_dict in self.distill_config.intermediate_matches])
        )
        assert norm_weights >= 0.0

        # Hard=label loss
        if self.distill_config.hard_label_weight > 0.0:
            if hard_label_loss is None:
                hard_label_loss = KD_LOSS_MAP[self.distill_config.hard_loss_type](logits, hard_labels, temperature)

            loss = loss + hard_label_loss * self.distill_config.hard_label_weight / norm_weights

        # Response-based loss
        if self.distill_config.kd_loss_weight > 0.0:
            kd_loss = torch.stack(
                [
                    KD_LOSS_MAP[self.distill_config.kd_loss_type](logits, teacher_logits, temperature)
                    for teacher_logits in logits_T
                ]
            ).mean()
            loss = loss + kd_loss * self.distill_config.kd_loss_weight / norm_weights

        # Relation-based loss
        if self.distill_config.intermediate_matches and self.distill_type == "relation":
            for proj, match_dict in zip(self.projs, self.distill_config.intermediate_matches):
                match_S = self._features[self.names_S[0]][match_dict.layer_S]
                if proj is not None:
                    proj.to(self.device)
                    match_S = proj(match_S)
                match_S_dist = self._calculate_distance(match_S)
                match_S_angl = self._calculate_angles(match_S)

                match_T = [self._features[name][match_dict.layer_T] for name in self.names_T]
                match_T_dist = [self._calculate_distance(match) for match in match_T]
                match_T_angl = [self._calculate_angles(match) for match in match_T]

                dist_loss = torch.stack(
                    [
                        KD_LOSS_MAP[self.distill_config.kd_loss_type](match_S_dist, match_dist)
                        for match_dist in match_T_dist
                    ]
                ).mean()
                angl_loss = torch.stack(
                    [
                        KD_LOSS_MAP[self.distill_config.kd_loss_type](match_S_angl, match_dist)
                        for match_dist in match_T_angl
                    ]
                ).mean()

                loss = loss + dist_loss * match_dict.weight / 2 / norm_weights
                loss = loss + angl_loss * match_dict.weight / 2 / norm_weights

        # Feature-based loss
        if self.distill_config.intermediate_matches and self.distill_type == "feature":
            for proj, match_dict in zip(self.projs, self.distill_config.intermediate_matches):
                match_S = self._features[self.names_S[0]][match_dict.layer_S]
                if proj is not None:
                    proj.to(self.device)
                    match_S = proj(match_S)

                match_T = [self._features[name][match_dict.layer_T] for name in self.names_T]
                match_loss = torch.stack(
                    [MATCH_LOSS_MAP[match_dict.loss](match_S, teacher_logits) for teacher_logits in match_T]
                ).mean()
                loss = loss + match_loss * match_dict.weight / norm_weights
        return loss

    def _calculate_distance(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.cdist(tensor, tensor, p=3)

    def _calculate_angles(self, tensor: torch.tensor) -> torch.Tensor:
        nVs = tensor / torch.norm(tensor, p=3, dim=-1, keepdim=True)
        return torch.einsum("bni,bmi->bnm", nVs, nVs)

    def _configure_projections(self) -> None:
        self.projs = []
        for im in self.distill_config.intermediate_matches:
            if im.projection is not None:
                self.projs.append(PROJ_MAP[im.projection.projection](im.projection.dim_in, im.projection.dim_out))
                self.projs[-1].to(self.device)
            else:
                self.projs.append(None)

    def _avg_dicts(self, colls):
        complete_dict = {k: [c[k] for c in colls] for k in colls[0].keys()}  # list[dict] -> dict[list]

        avg_dict = {k: sum(l) / len(l) for k, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val, sync_dist=True)

    def gather_cached_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def _log_tracker(self, pre: str = "") -> None:
        if self.tracker is not None:
            attr_dict = {
                f"{pre}{k}": float(v[0])
                for k, v in self.tracker._func_for_sched().items()
                if k in ["duration(s)", "power_consumption(kWh)", "CO2_emissions(kg)"]
            }
            self.log_dict(attr_dict, logger=True)

    # def _calc_loss()

    #                 if self.d_config.kd_loss_weight_scheduler is not None:
    #                 self.d_config.kd_loss_weight = \
    #                     self.d_config.kd_loss_weight_scheduler(global_step/total_global_steps)
    #             if self.d_config.hard_label_weight_scheduler is not None:
    #                 self.d_config.hard_label_weight = \
    #                     self.d_config.hard_label_weight_scheduler(global_step/total_global_steps)

    #             if (global_step) % print_every == 0:
    #                 logger.info(f"Global step: {global_step}, epoch step:{step+1}")
    #             if (global_step%ckpt_steps==0) or global_step==total_global_steps:
    #                 self.save_and_callback(global_step, step, 0, callback)
