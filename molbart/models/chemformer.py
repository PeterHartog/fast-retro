import os
from argparse import Namespace
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import molbart.modules.util as util
from molbart.models.bart import BARTModel
from molbart.modules.data.base import SimpleReactionListDataModule
from molbart.modules.decoder import BeamSearchSampler
from molbart.modules.distiller import DistillationConfig, Distiller, IntermediateMatches
from molbart.modules.tokenizer import ChemformerTokenizer
from molbart.modules.utils import trainer_utils
from molbart.modules.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class Chemformer:
    """
    Class for building (synthesis) Chemformer model, fine-tuning seq-seq model,
    and predicting/scoring model.
    """

    def __init__(
        self,
        config: DictConfig,
        vocabulary_path: str,
        model_args: Namespace,
        data_args: Namespace,
        model_path: Optional[str] = None,
        n_gpus: int = 1,
        n_beams: int = 1,
        n_unique_beams: Optional[int] = None,
        batch_first: bool = False,
        norm_first: bool = True,
        datamodule_type: str = "seq2seq",
        train_mode: str = "train",
        device: str = "cuda",
        data_device: str = "cuda",
        sample_unique: bool = False,
        resume_training: bool = False,
    ) -> None:
        """
        Args:
            vocabulary_path (str): path to bart_vocabulary.
            model_args (Namespace): Arguments for building the chemformer model.
            data_args (Namespace): Arguments for building torch datamodule.
            model_path (Optional[str]): Path to model weights.
            n_gpus (int): Number of GPUs to use.
            n_beams (int): Number of beams in beam search.
            n_unique_beams (Optional[int]): Restrict number of unique beam search solutions.
                If None => return all unique solutions.
            datamodule_type (str): The type of datamodule to build.
            train_model (str): Whether to train the model ("training") or use
                model for evaluations ("eval").
            sampler (str): Which beam search sampler to use ("optimized" => GPU
                optimized beam search).
            device (str): Which device to run model and beam search on ("cuda" / "cpu").
            data_device (str): device used for handling the data in optimized beam search.
                If memory issues, could help to set data_device="cpu"
            sample_unique (bool): Whether to return unique beam search solutions from the
                optimized beam search.
            resume_training (bool): Whether to continue training from the supplied
                .ckpt file.
        """

        self.config = config
        self.tracker = None
        self.model_args = model_args
        self.batch_first = batch_first
        self.norm_first = norm_first

        self.train_mode = train_mode
        self.resume_training = resume_training
        if resume_training:
            log.info("Resuming training.")

        if n_gpus < 1:
            device = "cpu"
            data_device = "cpu"

        self.device = device

        self.tokenizer = ChemformerTokenizer(filename=vocabulary_path)
        self.train_tokens = data_args.train_tokens
        self.n_buckets = data_args.n_buckets

        self.model_type = model_args.model_type
        self.model_path = model_path

        self.data_args = data_args
        self.n_gpus = n_gpus
        if datamodule_type is not None:
            self.init_datamodule(datamodule_type=datamodule_type)
            self.set_datamodule()

        log.info(f"Vocabulary_size: {len(self.tokenizer)}")
        self.vocabulary_size = len(self.tokenizer)

        self.sampler = BeamSearchSampler(
            tokenizer=self.tokenizer,
            scorers=trainer_utils.instantiate_scorers(self.config.get("scorers")),
            batch_first = self.batch_first,
            max_sequence_length=data_args.max_seq_len,
            device=device,
            data_device=data_device,
            sample_unique=sample_unique,
        )
        self.num_beams = n_beams
        self.n_unique_beams = n_beams if n_unique_beams is None else np.min(np.array([self.num_beams, n_unique_beams]))

        model = self.build_model(model_args)
        self.set_model(model)

        self.trainer = None
        if "trainer" in self.config:
            self.trainer = trainer_utils.build_trainer(config, n_gpus, data_args.dataset_type)

        # self.model = self.model.to(device)
        return

    def encode(
        self,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
    ) -> List[torch.Tensor]:
        """
        Compute memory from transformer inputs.

        Args:
            dataset (str): (Which part of the dataset to use (["train", "val", "test",
                "full"]).)
            dataloader (DataLoader): (If None -> dataloader
                will be retrieved from self.datamodule)
        Returns:
            List[torch.Tensor]: Tranformer memory
        """

        self.model.to(self.device)
        self.model.eval()

        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        X_encoded = []
        for b_idx, batch in enumerate(dataloader):
            batch = self.on_device(batch)
            with torch.no_grad():
                batch_encoded = self.model.encode(batch).permute(
                    1, 0, 2
                )  # Return on shape [n_samples, n_tokens, max_seq_length]

            X_encoded.extend(batch_encoded)
        return X_encoded

    def decode(
        self,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Output token probabilities from a given decoder input

        Args:
            memory_input (torch.Tensor): tensor from encoded input of shape (src_len,
                batch_size, d_model)
            memory_pad_mask (torch.Tensor): bool tensor of memory padding mask of shape
                (src_len, batch_size)
            decoder_input (torch.Tensor): tensor of decoder token_ids of shape (tgt_len,
                batch_size)
        """
        self.model.to(self.device)
        self.model.eval()

        batch_input = {
            "memory_input": memory,
            "memory_pad_mask": memory_pad_mask.permute(1, 0),
            "decoder_input": decoder_input.permute(1, 0),
            "decoder_pad_mask": torch.zeros_like(decoder_input, dtype=bool).permute(1, 0),
        }
        with torch.no_grad():
            return self.model.decode(batch_input)

    def init_datamodule(self, datamodule_type: str) -> None:
        """
        Create a new datamodule by either supplying a datamodule (created elsewhere) or
        a pre-defined datamodule type as input.

        Args:
            datamodule_type (Optional[str]): The type of datamodule to build if no
                datamodule is given as input.
        """
        log.info("Datamodule type: " + str(datamodule_type))
        if datamodule_type == "seq2seq":
            self.datamodule = util.build_seq2seq_datamodule(
                self.data_args, self.tokenizer, self.data_args.forward_prediction
            )
        elif datamodule_type == "simple_reaction_list":
            self.datamodule = SimpleReactionListDataModule(
                dataset_path=self.data_args.reactants_path,
                tokenizer=self.tokenizer,
                batch_size=self.data_args.batch_size,
                max_seq_len=self.data_args.max_seq_len,
                reverse=not self.data_args.forward_prediction,
                num_buckets=self.n_buckets,
            )
        else:
            raise ValueError(f"Unknown datamodule type: {datamodule_type}")

    def fit(self) -> None:
        """
        Fit model to training data in self.datamodule and using parameters specified in
        the trainer object.
        """
        self.trainer.fit(self.model, datamodule=self.datamodule)

    def validate(self) -> None:
        """
        Run validation of model to validation data in self.datamodule and using parameters specified in
        the trainer object.
        """
        self.trainer.validate(self.model, datamodule=self.datamodule)

    def test(self) -> None:
        """
        Test model to test data in self.datamodule and using parameters specified in
        the trainer object.
        """
        # if self.trainer.checkpoint_callback is not None:
        #     ckpt_path = self.trainer.checkpoint_callback.best_model_path
        # else:
        #     ckpt_path = None
        self.trainer.test(self.model, datamodule=self.datamodule)  # , ckpt_path=ckpt_path)

    def _predict(self) -> Dict:  # TODO: rename to predict after predict has been moved to model
        """
        Predict model to test data in self.datamodule and using parameters specified in
        the trainer object.
        """
        return self.trainer.predict(self.model, datamodule=self.datamodule)

    def distill(self) -> None:
        """
        #TODO: Feature request
        Uses knowledge distillation to train a model to training data in self.datamodule and
        using parameters specified in the trainer object.
        (Required) self.teacher is set.
        """
        log.info("Instantiating distiller...")
        if self.config.get("dist_settings") is not None:
            distil_cfg = DistillationConfig(**self.config.get("dist_settings"))
        else:
            distil_cfg = DistillationConfig(
                intermediate_matches=[
                    IntermediateMatches(projection=["linear", self.config.student_d_model, self.model_args.d_model])
                ]
            )
        self.distiller = Distiller(distil_cfg=distil_cfg)

        student_args = self.model_args
        student_args.d_model = self.config.student_d_model
        student_args.n_layers = self.config.student_n_layers
        student_args.n_encoder_layers = self.config.student_n_encoder_layers
        student_args.n_decoder_layers = self.config.student_n_decoder_layers
        student_args.n_heads = self.config.student_n_heads
        student_args.d_feedforward = self.config.student_d_feedforward

        setting_args = {
            k: v
            for k, v in self.config.get("dist_settings").items()
            if k in ["temperature", "hard_label_weight", "kd_loss_weight"]
        }
        setting_args = {
            **setting_args,
            **{
                f"match_{i}_weight": l.weight
                for i, l in enumerate(self.config.get("dist_settings").intermediate_matches)
                if l is not None
            },
        }

        self.distiller.set_teacher([self.model], [], ["big"])
        self.distiller.set_student([self.build_model(student_args, random_init=True, **setting_args)], [], ["small"])

        self.trainer.fit(self.distiller, datamodule=self.datamodule)
        self.model = self.distiller.return_student()

    def explain(self) -> Dict:
        """
        #TODO: Feature request
        Uses XAI methods to explain predictions.
        """
        raise NotImplementedError()

    def parameters(self) -> Iterator:
        return self.model.parameters()

    def set_tracker(self, tracker: Optional[Any] = None) -> None:
        self.tracker = tracker
        self.model.set_tracker(self.tracker)

    def set_trainer(self, trainer: Optional[pl.Trainer]) -> None:
        if trainer is not None:
            self.trainer = trainer
        else:
            pass

    def set_tokenizer(self, tokenizer: Any) -> None:
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            pass

    def set_datamodule(self, datamodule: Optional[pl.LightningDataModule] = None) -> None:
        if datamodule is not None:
            self.datamodule = datamodule

        assert self.datamodule is not None

        self.datamodule.setup()  # TODO: move to datamodule
        n_cpus = len(os.sched_getaffinity(0))
        if self.n_gpus > 0:
            n_workers = n_cpus // self.n_gpus
        else:
            n_workers = n_cpus
        self.datamodule._num_workers = n_workers
        log.info(f"Using {str(n_workers)} workers for data module.")

        if self.train_mode.startswith("train"):
            self.train_steps = trainer_utils.calc_train_steps(self.config, self.datamodule, self.n_gpus)
            log.info(f"Train steps: {self.train_steps}")

    def set_sampler(self, sampler: Optional[Any]) -> None:
        assert self.tokenizer is not None, "Sampler is dependent on tokenizer, which has not been initialized."

        if sampler is not None:
            self.sampler = sampler
        else:
            pass

    def set_model(self, model: Optional[pl.LightningModule]) -> None:
        """
        Build transformer model, either
        1. By loading pre-trained model from checkpoint file, or
        2. Initializing new model with random weight initialization

        Args:
            args (Namespace): Grouped model arguments.
        """
        if model is not None:
            self.model = model
        else:
            pass

    def _random_initialization(self, args: Namespace, extra_args: Dict[str, Any], pad_token_idx: int) -> BARTModel:
        """
        Constructing a model with randomly initialized weights.

        Args:
            args (Namespace): Grouped model arguments.
            extra_args (Dict[str, Any]): Extra arguments passed to the BARTModel.
            Will be saved as hparams by pytorchlightning.
            pad_token_idx: The index denoting padding in the vocabulary.
        """

        if self.train_mode.startswith("train"):
            total_steps = self.train_steps + 1
        else:
            total_steps = 0

        model = BARTModel(
            pad_token_idx=pad_token_idx,
            vocabulary_size=self.vocabulary_size,
            batch_first=args.batch_first,
            norm_first=self.norm_first,
            d_model=args.d_model,
            num_layers=args.n_layers,
            num_heads=args.n_heads,
            d_feedforward=args.d_feedforward,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            activation=args.activation,
            num_steps=total_steps,
            max_seq_len=args.max_seq_len,
            schedule=args.schedule,
            warm_up_steps=args.warm_up_steps,
            dropout=args.dropout,
            num_encoder_layers=args.n_encoder_layers,
            num_decoder_layers=args.n_decoder_layers,
            moe=args.moe,
            perceiver=args.perceiver,
            share_weights=args.share_weights,
            project_dim=args.project_dim,
            **extra_args,
        )
        model.set_sampler(self.sampler, self.num_beams, self.n_unique_beams)
        model.set_tracker(self.tracker)

        return model

    def _initialize_from_ckpt(self, args: Namespace, extra_args: Dict[str, Any], pad_token_idx: int) -> BARTModel:
        """
        Constructing a model with weights from a ckpt-file.

        Args:
            args (Namespace): Grouped model arguments.
            extra_args (Dict[str, Any]): Extra arguments passed to the BARTModel.
            Will be saved as hparams by pytorchlightning.
            pad_token_idx: The index denoting padding in the vocabulary.
        """
        if self.train_mode == "training" or self.train_mode == "train":
            total_steps = self.train_steps + 1
        else:
            total_steps = 0

        if self.train_mode == "training" or self.train_mode == "train":
            if self.resume_training:
                model = BARTModel.load_from_checkpoint(
                    self.model_path,
                    num_steps=total_steps,
                    pad_token_idx=pad_token_idx,
                    vocabulary_size=self.vocabulary_size,
                )
            else:
                model = BARTModel.load_from_checkpoint(
                    self.model_path,
                    d_model=args.d_model,
                    batch_first=self.batch_first,
                    norm_first=self.norm_first,
                    num_layers=args.n_layers,
                    num_heads=args.n_heads,
                    d_feedforward=args.d_feedforward,
                    activation=args.activation,
                    max_seq_len=args.max_seq_len,
                    num_encoder_layers=args.n_encoder_layers,
                    num_decoder_layers=args.n_decoder_layers,
                    pad_token_idx=pad_token_idx,
                    vocabulary_size=self.vocabulary_size,
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    num_steps=total_steps,
                    schedule=args.schedule,
                    warm_up_steps=args.warm_up_steps,
                    **extra_args,
                )
        elif (
            self.train_mode == "validation"
            or self.train_mode == "val"
            or self.train_mode == "test"
            or self.train_mode == "testing"
            or self.train_mode == "eval"
        ):
            try:
                model = BARTModel.load_from_checkpoint(self.model_path)
                model.eval()
            except TypeError:

                model = BARTModel.load_from_checkpoint(
                    self.model_path,
                    d_model=args.d_model,
                    batch_first=self.batch_first,
                    norm_first=self.norm_first,
                    num_layers=args.n_layers,
                    num_heads=args.n_heads,
                    d_feedforward=args.d_feedforward,
                    activation=args.activation,
                    max_seq_len=args.max_seq_len,
                    num_encoder_layers=args.n_encoder_layers,
                    num_decoder_layers=args.n_decoder_layers,
                    pad_token_idx=pad_token_idx,
                    vocabulary_size=self.vocabulary_size,
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    num_steps=total_steps,
                )
                model.eval()
        else:
            raise ValueError(f"Unknown training mode: {self.train_mode}")

        model.set_sampler(self.sampler, self.num_beams, self.n_unique_beams)
        model.set_tracker(self.tracker)

        return model

    def build_model(self, args: Namespace, random_init: bool = False, **kwargs) -> pl.LightningModule:
        """
        Build transformer model, either
        1. By loading pre-trained model from checkpoint file, or
        2. Initializing new model with random weight initialization

        Args:
            args (Namespace): Grouped model arguments.
        """

        pad_token_idx = self.tokenizer["pad"]

        # These args don't affect the model directly but will be saved by lightning as hparams
        # Tensorboard doesn't like None so we need to convert to string
        train_tokens = "None" if self.train_tokens is None else self.train_tokens
        n_buckets = "None" if self.n_buckets is None else self.n_buckets

        if self.train_mode == "training" or self.train_mode == "train":
            extra_args = {
                "batch_size": self.datamodule.batch_size,
                "acc_batches": args.acc_batches,
                "epochs": args.n_epochs,
                "clip_grad": args.clip_grad,
                "augment": args.augmentation_strategy,
                "aug_prob": args.augment_prob,
                "train_tokens": train_tokens,
                "n_buckets": n_buckets,
                "limit_val_batches": args.limit_val_batches,
            }
            extra_args = {**extra_args, **kwargs}
        else:
            extra_args = {}

        # If no model is given, use random init
        if (self.model_path is not None and self.model_path != "") and not random_init:
            model = self._initialize_from_ckpt(args, extra_args, pad_token_idx)
        else:
            model = self._random_initialization(args, extra_args, pad_token_idx)

        model.num_beams = self.num_beams
        model.n_unique_beams = self.n_unique_beams
        return model

    def get_dataloader(self, dataset: str, datamodule: Optional[pl.LightningDataModule] = None) -> DataLoader:
        """
        Get the dataloader for a subset of the data from a specific datamodule.

        Args:
            dataset (str): One in ["full", "train", "val", "test"].
                Specifies which part of the data to return.
            datamodule (Optional[pl.LightningDataModule]): pytorchlightning datamodule.
                If None -> Will use self.datamodule.
        """
        if dataset not in ["full", "train", "val", "test"]:
            raise ValueError(f"Unknown dataset : {dataset}. Should be either 'full', 'train', 'val' or 'test'.")

        if datamodule is None:
            datamodule = self.datamodule

        dataloader = None
        if dataset == "full":
            dataloader = datamodule.full_dataloader()
        elif dataset == "train":
            dataloader = datamodule.train_dataloader()
        elif dataset == "val":
            dataloader = datamodule.val_dataloader()
        elif dataset == "test":
            dataloader = datamodule.test_dataloader()

        return dataloader

    @torch.no_grad()
    def log_likelihood(
        self,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
    ) -> List[float]:
        """
        Computing the likelihood of the encoder_input SMILES and decoder_input SMILES
        pairs.

        Args:
            dataset (str): Which part of the dataset to use (["train", "val", "test",
                "full"]).
            dataloader (Optional[DataLoader]): If None -> dataloader
                will be retrieved from self.datamodule.
        Returns:
            List[float]: List with log-likelihoods of each reactant/product pairs.
        """

        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        self.model.to(self.device)
        self.model.eval()

        log_likelihoods = []
        for batch in dataloader:
            batch = self.on_device(batch)
            output = self.model.forward(batch)
            log_probabilities = self.model.generator(output["model_output"])

            target_ids_lst = batch["decoder_input"].permute(1, 0)

            for target_ids, log_prob in zip(target_ids_lst[:, 1::], log_probabilities.permute(1, 0, 2)):
                llhs = 0.0
                for i_token, token in enumerate(target_ids):
                    llhs += log_prob[i_token, token].item()
                    break_condition = token == self.tokenizer["end"] or token == self.tokenizer["pad"]
                    if break_condition:
                        break

                log_likelihoods.append(llhs)
        return log_likelihoods

    def on_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move data in "batch" to the current model device.

        Args:
            batch (Dict[str, Any]): batch input data to model.
        Returns:
            Dict[str, Any]: batch data on current device.
        """
        device_batch = {
            key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()
        }
        return device_batch

    def predict(  # TODO: move to model and have call here
        self,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
        return_tokenized: bool = False,
        i_chunk: int = 0,
        n_chunks: int = 1,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Predict SMILES output given dataloader, specified by 'dataset'.
        Args:
            dataset: Which part of the dataset to use (["train", "val", "test",
                "full"]).
            dataloader: If None -> dataloader
                will be retrieved from self.datamodule.
            return_tokenized: Whether to return the tokenized beam search
                solutions instead of strings.
            i_chunk: Index of chunk of batches to process.
            n_chunks: Number of chunks to divide batches in.
        Returns:
            (sampled_smiles List[np.ndarray], log_lhs List[np.ndarray], target_smiles List[np.ndarray])
        """

        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        # Divide batches into chunks of batches
        n_batches_in_chunk = int(len(dataloader) / float(n_chunks))
        start_batch_idx = i_chunk * n_batches_in_chunk

        self.model.to(self.device)
        self.model.eval()

        sampled_smiles = []
        log_lhs = []
        target_smiles = []
        for b_idx, batch in enumerate(dataloader):
            if n_chunks > 1:
                if b_idx < start_batch_idx:
                    continue

                if i_chunk != n_chunks - 1:
                    if i_chunk != n_chunks - 1 and b_idx == start_batch_idx + n_batches_in_chunk:
                        break

            batch = self.on_device(batch)
            with torch.no_grad():
                smiles_batch, log_lhs_batch = self.model.sample_molecules(
                    batch, sampling_alg="beam", return_tokenized=return_tokenized
                )
                if self.model.sampler.sample_unique:
                    smiles_batch = self.sampler.smiles_unique
                    log_lhs_batch = self.sampler.log_lhs_unique

            sampled_smiles.extend(smiles_batch)
            log_lhs.extend(log_lhs_batch)
            target_smiles.extend(batch["target_smiles"])

        return sampled_smiles, log_lhs, target_smiles

    def score_model(  # TODO: move to testing and have a call here
        self,
        n_unique_beams: Optional[int] = None,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
        i_chunk: int = 0,
        n_chunks: int = 1,
        output_scores: Optional[str] = None,
        output_sampled_smiles: Optional[str] = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Score model performance on dataset in terms of accuracy (top-1 and top-K) and
        similarity of top-1 molecules. Also collects basic logging scores (loss, etc.).

        Args:
            n_unique_beams: Number of unique beams after canonicalizing sampled
                SMILES strings.
            dataset: Which part of the dataset to use (["train", "val", "test",
                "full"]).
            dataloader (DataLoader): If None -> dataloader will be
                retrieved from self.datamodule.
            output_scores: Path to output .csv file with model performance. If None ->
                Will not write DataFrame to file.
            output_sampled_smiles: Path to output .json file with sampled smiles.
                If None -> Will not write DataFrame to file.
        Returns:
            [pandas.DataFrame with calculated scores/metrics, pandas.DataFrame with
                sampled SMILES]
            or
            pandas.DataFrame with calculated scores/metrics
        """

        if output_scores and output_sampled_smiles:
            for callback in self.trainer.callbacks:
                if hasattr(callback, "set_output_files"):
                    callback.set_output_files(output_scores, output_sampled_smiles)

        if n_unique_beams is None and self.sampler.smiles_unique:
            n_unique_beams = self.model.num_beams
        self.model.n_unique_beams = n_unique_beams

        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        # Divide batches into chunks of batches
        n_batches_in_chunk = int(len(dataloader) / float(n_chunks))
        start_batch_idx = i_chunk * n_batches_in_chunk

        self.model.eval()
        self.model.to(self.device)

        for b_idx, batch in enumerate(dataloader):
            if n_chunks > 1:
                if b_idx < start_batch_idx:
                    continue

                if i_chunk != n_chunks - 1:
                    if i_chunk != n_chunks - 1 and b_idx == start_batch_idx + n_batches_in_chunk:
                        break

            batch = self.on_device(batch)
            metrics = self.model.test_step(batch, b_idx)

            if self.model.sampler.sample_unique:
                sampled_smiles_unique = self.model.sampler.smiles_unique
                log_lhs_unique = self.model.sampler.log_lhs_unique

                # Get data of unique SMILES/solutions (keeping both non-unique
                # and unique metrics)
                metrics_unique = self.model.sampler.compute_sampling_metrics(
                    sampled_smiles_unique, metrics["target_smiles"], is_canonical=False
                )

                metrics_unique.update(
                    {
                        "sampled_molecules": sampled_smiles_unique,
                        "log_lhs": log_lhs_unique,
                    }
                )

                drop_cols = [
                    "fraction_invalid",
                    "fraction_unique",
                    "top1_tanimoto_similarity",
                ]
                metrics_unique = {f"{key}(unique)": val for key, val in metrics_unique.items() if key not in drop_cols}
                metrics.update(metrics_unique)

            for callback in self.trainer.callbacks:
                if not isinstance(callback, pl.callbacks.progress.ProgressBar):
                    callback.on_test_batch_end(self.trainer, self.model, metrics, batch, b_idx)
