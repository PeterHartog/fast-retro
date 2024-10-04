import math
import time
from typing import Any, Dict, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from molbart.models.util import FuncLR

# from functools import partial


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------- Base Transformer Model -----------------------------------------
# ----------------------------------------------------------------------------------------------------------


class _AbsTransformerModel(pl.LightningModule):
    def __init__(
        self,
        pad_token_idx: int,
        vocabulary_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_feedforward: int,
        lr: float,
        weight_decay: float,
        activation: str,
        num_steps: int,
        max_seq_len: int,
        schedule: str = "cycle",
        warm_up_steps: Optional[int] = None,
        dropout: float = 0.1,
        batch_first: bool = False,
        **kwargs,
    ):
        super().__init__()
        # Model arguments
        self.batch_first = batch_first
        self.pad_token_idx = pad_token_idx
        self.vocabulary_size = vocabulary_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.activation = activation
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # Optimizer arguments
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_steps = num_steps
        self.schedule = schedule
        self.warm_up_steps = warm_up_steps
        if self.schedule == "transformer":
            assert warm_up_steps is not None, "A value for warm_up_steps is required for transformer LR schedule"

        # Additional args passed in to **kwargs in init will also be saved
        self.save_hyperparameters()

        # Outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.prediction_step_outputs = []

        self.tracker = None

        self.emb = nn.Embedding(vocabulary_size, d_model, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_emb", self._positional_embs())

    def set_sampler(self, decode_sampler, num_beams: int = 10, n_unique_beams: int = 10) -> None:
        self.sampler = decode_sampler
        self.val_sampling_alg = None  # "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = num_beams
        self.n_unique_beams = n_unique_beams

    def set_tracker(self, tracker) -> None:
        self.tracker = tracker

    def forward(self, x):
        raise NotImplementedError()

    def _calc_loss(self, batch_input, model_output):
        """Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor)
        """

        raise NotImplementedError()

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        raise NotImplementedError()

    def on_train_start(self):
        total_parameters = filter(lambda p: p.requires_grad, self.parameters())
        non_trainable_parameters = filter(lambda p: p.requires_grad is False, self.parameters())
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
        self.train()

        model_output = self.forward(batch)
        loss = self._calc_loss(batch, model_output)

        self.log("loss", loss, logger=True, batch_size=model_output["batch_size"], sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        self._log_tracker()

    def on_train_end(self):
        if self.tracker is not None:
            self.tracker.stop()

    def validation_step(self, batch, batch_idx):
        self.eval()

        with torch.no_grad():
            model_output = self.forward(batch)
            loss = self._calc_loss(batch, model_output)
            token_acc = self._calc_token_acc(batch, model_output)
            perplexity = self._calc_perplexity(batch, model_output)

            metrics = {
                "batch_idx": batch_idx,
                "batch_size": model_output["batch_size"],
                "validation_loss": loss,
                "val_perplexity": perplexity,
                "val_token_accuracy": token_acc,
            }

            if self.val_sampling_alg is not None:
                target_smiles = batch["target_smiles"]
                t0 = time.time()
                sampled_smiles, log_likelihoods = self.sample_molecules(batch, sampling_alg=self.val_sampling_alg)
                sample_time = time.time() - t0

                # TODO: metrics computation should be outside sampler...
                sampled_metrics = self.sampler.compute_sampling_metrics(sampled_smiles, target_smiles)
                sampled_metrics["sample_time"] = sample_time / 60
                sampled_metrics["rate_correct_score"] = sampled_metrics["accuracy"] / sample_time
                sampled_metrics = {f"val_{k}": v for k, v in sampled_metrics.items()}

                metrics.update(sampled_metrics)

                # sampled_values = {
                #     "sampled_molecules": sampled_smiles,
                #     "log_lhs": log_likelihoods,
                #     "target_smiles": target_smiles,
                # }

        self.log("validation_loss", loss, logger=True, batch_size=model_output["batch_size"], sync_dist=True)
        self.validation_step_outputs.append(metrics)

        return metrics

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        self.validation_step_outputs = []  # free memory
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def on_test_start(self):
        if self.tracker is not None:
            self.tracker.start()
            self._log_tracker("test_")

    def test_step(self, batch, batch_idx):
        self.eval()

        with torch.no_grad():
            model_output = self.forward(batch)
            loss = self._calc_loss(batch, model_output)
            token_acc = self._calc_token_acc(batch, model_output)
            perplexity = self._calc_perplexity(batch, model_output)

            metrics = {
                "batch_idx": batch_idx,
                "batch_size": model_output["batch_size"],
                "test_loss": loss,
                "test_perplexity": perplexity,
                "test_token_accuracy": token_acc,
            }

            if self.test_sampling_alg is not None:
                target_smiles = batch["target_smiles"]
                t0 = time.time()
                sampled_smiles, log_likelihoods = self.sample_molecules(batch, sampling_alg=self.test_sampling_alg)
                sample_time = time.time() - t0

                # TODO: metrics computation should be outside sampler, preferably callback
                sampled_metrics = self.sampler.compute_sampling_metrics(sampled_smiles, target_smiles)
                sampled_metrics["sample_time"] = sample_time
                sampled_metrics["rate_correct_score"] = (
                    sampled_metrics["accuracy"] / sample_time
                    if self.test_sampling_alg == "greedy"
                    else sampled_metrics["accuracy_top_1"] / sample_time
                )
                sampled_metrics = {f"test_{k}": v for k, v in sampled_metrics.items()}

                metrics.update(sampled_metrics)

                metrics.update(
                    {
                        "sampled_molecules": sampled_smiles,
                        "log_lhs": log_likelihoods,
                        "target_smiles": target_smiles,
                    }
                )

        self.test_step_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        self.test_step_outputs = []
        outputs = [
            {k: v for k, v in out.items() if k not in ["sampled_molecules", "log_lhs", "target_smiles"]}
            for out in outputs
        ]

        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)
        self._log_tracker("test_")
        # return

    def on_test_end(self):
        if self.tracker is not None:
            self.tracker.stop()

    def configure_optimizers(self):
        params = self.parameters()
        optim = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))

        if self.schedule == "const":
            print("Using constant LR schedule.")
            const_sch = FuncLR(optim, lr_lambda=self._const_lr)
            sch = {"scheduler": const_sch, "interval": "step"}

        elif self.schedule == "cycle":
            print("Using cyclical LR schedule.")
            cycle_sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)
            sch = {"scheduler": cycle_sch, "interval": "step"}

        elif self.schedule == "transformer":
            print("Using original transformer schedule.")
            trans_sch = FuncLR(optim, lr_lambda=self._transformer_lr)
            sch = {"scheduler": trans_sch, "interval": "step"}

        else:
            raise ValueError(f"Unknown schedule {self.schedule}")

        return [optim], [sch]

    def _transformer_lr(self, step):
        mult = self.d_model**-0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step**-0.5, step * (self.warm_up_steps**-1.5))
        return self.lr * mult * lr

    def _const_lr(self, step):
        if self.warm_up_steps is not None and step < self.warm_up_steps:
            return (self.lr / self.warm_up_steps) * step

        return self.lr

    def _construct_input(self, token_ids: torch.Tensor, sentence_masks=None):
        seq_len = token_ids.shape[1 if self.batch_first else 0]
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0)
        positional_embs = positional_embs.transpose(0, 1) if not self.batch_first else positional_embs
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs

    def _positional_embs(self):
        """Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000**encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[: self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz, device="cpu"):
        """
        Method copied from Pytorch nn.Transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode
        """

        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _calc_perplexity(self, batch_input, model_output):  # TODO: move to callback
        target_ids = batch_input["target"]
        target_mask = batch_input["target_mask"]
        vocab_dist_output = model_output["token_output"]

        inv_target_mask = ~(target_mask > 0)
        log_probs = vocab_dist_output.gather(2, target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask
        log_probs = log_probs.sum(dim=0)

        seq_lengths = inv_target_mask.sum(dim=0)
        exp = -(1 / seq_lengths)
        perp = torch.pow(log_probs.exp(), exp)
        return perp.mean()

    def _calc_token_acc(self, batch_input, model_output):  # TODO: move to callback
        token_ids = batch_input["target"]
        target_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()

        accuracy = num_correct / total
        return accuracy

    def _avg_dicts(self, colls):
        complete_dict = {k: [c[k] for c in colls] for k in colls[0].keys()}  # list[dict] -> dict[list]

        avg_dict = {k: sum(l) / len(l) for k, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val, sync_dist=True)

    def _log_tracker(self, pre: str = "") -> None:
        if self.tracker is not None:
            attr_dict = {
                f"{pre}{k}": float(v[0])
                for k, v in self.tracker._func_for_sched().items()
                if k in ["duration(s)", "power_consumption(kWh)", "CO2_emissions(kg)"]
            }
            self.log_dict(attr_dict, logger=True, sync_dist=True)
