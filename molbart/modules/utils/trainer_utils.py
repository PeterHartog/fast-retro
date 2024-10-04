import math
from typing import Iterable, List, Optional, Tuple, Union

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from molbart.modules.callbacks import CallbackCollection
from molbart.modules.scores import ScoreCollection
from molbart.modules.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def instantiate_callbacks(callbacks_config: Optional[DictConfig]) -> CallbackCollection:
    """Instantiates callbacks from config."""
    callbacks = CallbackCollection()

    if not callbacks_config:
        log.info("No callbacks configs found! Skipping...")
        return callbacks

    callbacks.load_from_config(callbacks_config)
    return callbacks


def instantiate_scorers(scorer_config: Optional[DictConfig]) -> CallbackCollection:
    """Instantiates scorer from config."""

    scorer = ScoreCollection()
    if not scorer_config:
        log.info("No scorer configs found! Skipping...")
        return scorer

    scorer.load_from_config(scorer_config)
    return scorer


def instantiate_logger(logger_config: Optional[DictConfig]) -> TensorBoardLogger:
    """Instantiates logger from config."""
    logger: TensorBoardLogger = []

    if not logger_config:
        log.warn("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_config, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    if isinstance(logger_config, DictConfig) and "_target_" in logger_config:
        log.info(f"Instantiating logger <{logger_config._target_}>")
        logger = hydra.utils.instantiate(logger_config)

    return logger


def calc_train_steps(args, dm, n_gpus=None):
    n_gpus = getattr(args, "n_gpus", n_gpus)

    if n_gpus is not None and n_gpus > 0:
        batches_per_gpu = math.ceil(len(dm.train_dataloader()) / float(n_gpus))
    else:
        log.warn("Number of GPUs should be > 0 in training.")
        batches_per_gpu = math.ceil(len(dm.train_dataloader()))
    train_steps = math.ceil(batches_per_gpu / args.acc_batches) * args.n_epochs
    return train_steps


def build_trainer(config, n_gpus=None, dataset=None):
    dataset = getattr(config, "dataset_type", dataset)

    log.info("Instantiating loggers...")
    logger = instantiate_logger(config.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: CallbackCollection = instantiate_callbacks(config.get("callbacks"))

    if n_gpus > 1:
        config.trainer.strategy = "ddp"
    # else:
    #     plugins = None

    log.info("Building trainer...")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks.objects(),
        logger=logger,  # plugins=plugins
    )
    log.info("Finished trainer.")

    log.info(f"Default logging and checkpointing directory: {trainer.default_root_dir} or {trainer.ckpt_path}")
    # weights_save_path}")
    return trainer


def get_metric_value(metric_dict: dict, metric_name: Union[str, Iterable[str]]) -> Optional[Tuple[float]]:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    metric_names = [metric_name] if type(metric_name) is str else metric_name
    metric_values = []
    for metric_name in metric_names:
        if metric_name not in metric_dict:
            log.warn(
                f"Metric value not found! <metric_name={metric_name}>\n"
                "Make sure metric name logged in LightningModule is correct!\n"
                "Make sure `optimized_metric` name in `hparams_search` config is correct!"
            )
            metric_values.append(None)
        else:
            metric_value = metric_dict[metric_name].item()
            log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")
            metric_values.append(metric_value)

    return metric_values


@rank_zero_only
def log_hyperparameters(trainer: pl.Trainer, config: dict, selection: Optional[List[str]] = None) -> None:
    """Controls which config parts are saved by lightning loggers."""

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    config = config if selection is None else {k: v for k, v in config.items() if k in selection}

    for logger in trainer.loggers:
        logger.log_hyperparams(config)
