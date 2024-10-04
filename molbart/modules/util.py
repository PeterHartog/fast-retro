import copy
from argparse import Namespace

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from molbart.models.bart import BARTModel
from molbart.modules.data.mol_data import ChemblDataModule, ZincDataModule
from molbart.modules.data.seq2seq_data import (
    MolOptDataModule,
    SynthesisDataModule,
    Uspto50DataModule,
    UsptoMixedDataModule,
    UsptoSepDataModule,
)

# Default model hyperparams
DEFAULT_D_MODEL = 512
DEFAULT_NUM_LAYERS = 6
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FEEDFORWARD = 2048
DEFAULT_ACTIVATION = "gelu"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_DROPOUT = 0.1

DEFAULT_MODEL = "bart"
DEFAULT_DATASET_TYPE = "synthesis"
DEFAULT_DEEPSPEED_CONFIG_PATH = "ds_config.json"
DEFAULT_LOG_DIR = "tb_logs"
DEFAULT_VOCAB_PATH = "bart_vocab.json"
DEFAULT_CHEM_TOKEN_START = 272
REGEX = r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

DEFAULT_GPUS = 1
DEFAULT_NUM_NODES = 1

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()


def build_molecule_datamodule(args, tokenizer, masker=None):
    dm_cls = {
        "chembl": ChemblDataModule,
        "zinc": ZincDataModule,
    }

    dm = dm_cls[args.dataset_type](
        task=args.task,
        augment_prob=args.augmentation_probability,
        masker=masker,
        dataset_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        train_token_batch_size=args.train_tokens,
        num_buckets=args.n_buckets,
        unified_model=args.model_type == "unified",
    )
    return dm


def build_seq2seq_datamodule(args, tokenizer, forward=True):
    dm_cls = {
        "uspto_50": Uspto50DataModule,
        "uspto_50_with_type": Uspto50DataModule,
        "uspto_mixed": UsptoMixedDataModule,
        "uspto_sep": UsptoSepDataModule,
        "mol_opt": MolOptDataModule,
        "synthesis": SynthesisDataModule,
    }
    kwargs = {
        "uspto_50_with_type": {
            "include_type_token": True,
        }
    }

    dm = dm_cls[args.dataset_type](
        augment_prob=args.augment_prob,
        reverse=not forward,
        dataset_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=getattr(args, "max_seq_len", DEFAULT_MAX_SEQ_LEN),
        train_token_batch_size=args.train_tokens,
        num_buckets=args.n_buckets,
        unified_model=args.model_type == "unified",
        **kwargs.get(args.dataset_type, {}),
    )
    return dm


def seed_everything(seed):
    pl.seed_everything(seed)


def load_bart(args, sampler):
    model = BARTModel.load_from_checkpoint(args.model_path, decode_sampler=sampler)
    model.eval()
    return model


def get_chemformer_args(args):
    if args.task in ["forward_prediction", "mol_opt"]:
        forward_prediction = True
    elif args.task == "backward_prediction":
        forward_prediction = False
    else:
        raise ValueError(f"Unknown task {args.task}")

    data_args = {
        "data_path": getattr(args, "data_path", ""),
        "reactants_path": getattr(args, "reactants_path", ""),
        "dataset_type": getattr(args, "dataset_type", ""),
        "max_seq_len": getattr(args, "max_seq_len", DEFAULT_MAX_SEQ_LEN),
        "augmentation_strategy": getattr(args, "augmentation_strategy", None),
        "augment_prob": getattr(args, "augmentation_probability", 0.0),
        "batch_size": args.batch_size,
        "train_tokens": getattr(args, "train_tokens", None),
        "n_buckets": getattr(args, "n_buckets", None),
        "model_type": getattr(args, "model_type", "bart"),
        "forward_prediction": forward_prediction,
    }

    model_args = {
        "output_directory": getattr(args, "output_directory", ""),
        "task": args.task,
        "model_type": getattr(args, "model_type", "bart"),
        "acc_batches": getattr(args, "acc_batches", None),
        "batch_first": getattr(args, "batch_first", False),
        "d_model": getattr(args, "d_model", None),
        "n_layers": getattr(args, "n_layers", None),
        "n_encoder_layers": getattr(args, "n_encoder_layers", None),
        "n_decoder_layers": getattr(args, "n_decoder_layers", None),
        "moe": getattr(args, "moe", None),
        "perceiver": getattr(args, "perceiver", None),
        "share_weights": getattr(args, "share_weights", None),
        "project_dim": getattr(args, "project_dim", None),
        "n_heads": getattr(args, "n_heads", None),
        "d_feedforward": getattr(args, "d_feedforward", None),
        "activation": getattr(args, "activation", DEFAULT_ACTIVATION),
        "max_seq_len": getattr(args, "max_seq_len", DEFAULT_MAX_SEQ_LEN),
        "dropout": getattr(args, "dropout", DEFAULT_DROPOUT),
        "n_epochs": getattr(args, "n_epochs", None),
        "augmentation_strategy": getattr(args, "augmentation_strategy", None),
        "augment_prob": getattr(args, "augmenation_probability", 0.0),
        "warm_up_steps": getattr(args, "warm_up_steps", None),
        "deepspeed_config_path": getattr(args, "deepspeed_config_path", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "clip_grad": getattr(args, "clip_grad", None),
        "schedule": getattr(args, "schedule", None),
        "limit_val_batches": getattr(args, "limit_val_batches", None),
        "check_val_every_n_epoch": getattr(args, "check_val_every_n_epoch", None),
        # "checkpoint_every_n_step": getattr(args, "checkpoint_every_n_step", None),
        "n_nodes": getattr(args, "n_nodes", DEFAULT_NUM_NODES),
    }

    return Namespace(**model_args), Namespace(**data_args)


def _clean_string(x, expr_list):
    y = copy.copy(x)
    y = y.replace("''", "&")  # Mark empty SMILES string with dummy character
    for expr in expr_list:
        y = y.replace(expr, "")
    return y


def _convert_to_array(data_list):
    data_new = np.zeros(len(data_list), dtype="object")
    for ix, x in enumerate(data_list):
        data_new[ix] = x
    return data_new


def read_score_tsv(
    filename,
    str_to_list_columns,
    is_numeric,
    expr_list1=["'", "[array([", "array([", "[array(", "array(", " ", "\n"],
):
    """
    Read TSV-file generated by the Chemformer.score_model() function.
    Args:
    - filename: str (path to .csv file)
    - str_to_list_columns: list(str) (list of columns to convert from string to nested list)
    - is_numeric: list(bool) (list denoting which columns contain strings that should be converted to lists of floats)
    """

    sep = ","
    numeric_expr_list = ["(", ")", "[", "]", "\n"]
    data = pd.read_csv(filename, sep="\t")

    for col, to_float in zip(str_to_list_columns, is_numeric):
        print("Converting string to data of column: " + col)
        data_str = data[col].values

        data_list = []
        for X in data_str:
            X = [x for x in X.split(sep) if "dtype=" not in x]
            inner_list = []
            X_new = []
            is_last_molecule = False
            for x in X:
                x = _clean_string(x, expr_list1)

                if x == "":
                    continue

                if x[-1] == ")" and sum([token == "(" for token in x]) < sum([token == ")" for token in x]):
                    x = x[:-1]
                    is_last_molecule = True

                if x[-1] == "]" and sum([token == "[" for token in x]) < sum([token == "]" for token in x]):
                    x = x[:-1]
                    is_last_molecule = True

                inner_list.append(x)

                if is_last_molecule:
                    if to_float:
                        inner_list = [_clean_string(element, numeric_expr_list) for element in inner_list]
                        inner_list = [float(element) if element != "" else np.nan for element in inner_list]
                    X_new.append(inner_list)
                    inner_list = []
                    is_last_molecule = False

            print("Batch size after cleaning (for validating cleaning): " + str(len(X_new)))
            data_list.append(X_new)
        data.drop(columns=[col], inplace=True)
        data_list = _convert_to_array(data_list)
        data[col] = data_list
    return data
