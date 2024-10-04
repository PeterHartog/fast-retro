""" Module containing a class for the DataSet used as well as base-classes for DataModules"""

import functools
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pysmilesutils.augment import SMILESAugmenter
from pysmilesutils.datautils import TokenSampler
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset

from molbart.modules.data.util import (
    BatchEncoder,
    build_attention_mask,
    build_target_mask,
)
from molbart.modules.tokenizer import ChemformerTokenizer, TokensMasker
from molbart.modules.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@dataclass
class DataParserDataClass:
    file_type: Optional[str] = None
    in_memory: bool = False
    n_buckets: Optional[int] = None
    n_chunks: Optional[int] = None
    i_chunk: Optional[int] = None


class DataParser:
    def __init__(
        self,
        dynamic: bool = False,
        i_chunk: int = 0,
        n_chunks: int = 1,
    ) -> None:
        pass


class ABCDataModule(pl.LightningDataModule):
    """Base class for all DataModules"""

    train_dataset: Optional[Dataset] = None
    val_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    predict_dataset: Optional[Dataset] = None

    _all_data: Dict[str, Any] = {}

    def __init__(
        self,
        config,
        # dataset_path: str,
        # # tokenizer: ChemformerTokenizer,
        # # batch_size: int,
        # # max_seq_len: int,
        # # train_token_batch_size: int = None,
        # # num_buckets: int = None,
        # # val_idxs: Sequence[int] = None,
        # # test_idxs: Sequence[int] = None,
        # # train_idxs: Sequence[int] = None,
        # # train_set_rest: bool = True,
        # # split_perc: float = 0.2,
        # # pin_memory: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.df = self.load_data(dataset_path=config.dataset_path)

        # Sampling
        if config.train_token_batch_size is not None and config.num_buckets is not None:
            self.use_sampler = True
            log.info(
                f"""Training with approx. {config.train_token_batch_size} tokens per batch"""
                f""" and {config.num_buckets} buckets in the sampler."""
            )
        else:
            self.use_sampler = False
            log.info(f"Using a batch size of {str(config.batch_size)}.")
        self.batch_size = config.batch_size
        self.max_seq_len = config.max_seq_len
        self.train_token_batch_size = config.train_token_batch_size
        self.num_buckets = config.num_buckets

        # Chunking

        # Splitting
        if config.val_idxs is not None and config.test_idxs is not None:
            idxs_intersect = set(config.val_idxs).intersection(set(config.test_idxs))
            if len(idxs_intersect) > 0:
                raise ValueError("Val idxs and test idxs overlap")
        self.val_idxs = config.val_idxs
        self.test_idxs = config.test_idxs
        self.train_idxs = config.train_idxs
        self.train_set_rest = config.train_set_rest
        self.split_perc = config.split_perc

        # Data arguments
        self.num_workers = config.num_workers if torch.cuda.is_available() else 0
        self.persistent_workers = config.persistent_workers if torch.cuda.is_available() else False
        self.pin_memory = config.pin_memory if torch.cuda.is_available() else False
        self.drop_last = config.drop_last if torch.cuda.is_available() else False

        # Collate arguments
        self.batch_first = config.batch_first
        self.pad_to_max = config.pad_to_max
        self.mask_sequences = config.mask_sequences
        self.randomize_smiles = config.randomize_smiles if not self.enumerate else False
        self.random_type = config.random_type

    def load_data(self, dataset_path: str) -> pd.DataFrame:
        if dataset_path.endswith(".csv"):
            return pd.read_csv(dataset_path, sep="\t")
        elif dataset_path.endswith(".pkl") or dataset_path.endswith(".pickle"):
            return pd.read_pickle(dataset_path)
        else:
            raise ValueError(f"File format of file {dataset_path} is not supported.")

    def get_dataloader(self, dataset, shuffle: bool = False, token_batching: bool = False):
        if token_batching:
            sampler = TokenSampler(
                self.num_buckets,
                self.train_dataset.seq_lengths,
                self.train_token_batch_size,
                shuffle=shuffle,
            )
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            batch_sampler=sampler,
            shuffle=shuffle,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate,
        )

    def train_dataloader(self):
        """Returns the DataLoader for the training set"""
        return self.get_dataloader(self.train_dataset, True, self.use_sampler)

    def val_dataloader(self):
        """Returns the DataLoader for the validation set"""
        return self.get_dataloader(self.val_dataset, False, False)

    def test_dataloader(self):
        """Returns the DataLoader for the test set"""
        return self.get_dataloader(self.test_dataset, False, False)

    def predict_dataloader(self):
        return self.get_dataloader(self.predict_dataset, False, False)

    def full_dataloader(self, train: bool = False):
        """Returns a DataLoader for all the data"""
        return self.get_dataloader(self.full_dataset, train)

    def prepare_data(self) -> None:
        self.collate = self.prepare_collate_fn()

        self.load_data(self.config.dataset_path)
        self.split_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.prepare_data()  # for distributed training

        if stage == "fit" or stage is None:
            pass
        if stage == "test" or stage is None:
            pass
        if stage == "predict" or stage is None:
            pass
            log.warn("No prediction set detected, using test set.")
