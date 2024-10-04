import signal
from typing import Dict, Optional, Tuple

import hydra
import omegaconf

import molbart.modules.util as util
from molbart.models import Chemformer
from molbart.modules.utils.pylogger import get_pylogger
from molbart.modules.utils.trainer_utils import get_metric_value

log = get_pylogger(__name__)


class TimeoutException(Exception):
    pass


def handler(signum, frame):
    raise TimeoutException("Timed out!")


def finish_wandb():
    try:
        import wandb

        # finish wandb run
        wandb.finish()
    except Exception:
        pass


def invalid_value(num) -> bool:
    try:
        if float("-inf") < float(num) < float("inf"):
            return False
        else:
            return True
    except TypeError:
        return True


def pipeline(config: omegaconf.DictConfig) -> Dict:
    util.seed_everything(config.seed)

    log.info("Running pipeline.")
    model_args, data_args = util.get_chemformer_args(config)

    kwargs = {
        "config": config,
        "vocabulary_path": config.vocabulary_path,
        "n_gpus": config.n_gpus,
        "model_path": config.model_path,
        "model_args": model_args,
        "data_args": data_args,
        "resume_training": config.resume,
        "n_beams": config.n_beams,
    }

    log.info("Setting up Chemformer...")
    chemformer = Chemformer(**kwargs)
    log.info("Chemformer set up.")

    if config.track:
        import eco2ai

        tracker = eco2ai.Tracker(
            project_name=config.project_name,
            experiment_description=f"{config.logger.name}",
            measure_period=600,
            alpha_2_code="SE",
            file_name=config.track_path,
            ignore_warnings=True,
        )
        chemformer.set_tracker(tracker)

    callback_metrics = {}

    if config.train:
        log.info("Training model...")
        chemformer.fit()
        callback_metrics = {**callback_metrics, **chemformer.trainer.callback_metrics}
        log.info("Training finished.")

    if config.fine_tune:
        log.info("Fine-tuning model...")
        chemformer.fit()
        callback_metrics = {**callback_metrics, **chemformer.trainer.callback_metrics}
        log.info("Fine-tuning finished.")

    if config.distill:
        log.info("Distilling model...")
        chemformer.distill()
        callback_metrics = {**callback_metrics, **chemformer.trainer.callback_metrics}
        log.info("Distilation finished.")

    if config.validate:
        log.info("Validating model...")
        chemformer.model.val_sampling_alg = "greedy"
        chemformer.validate()
        callback_metrics = {**callback_metrics, **chemformer.trainer.callback_metrics}
        log.info("Validation finished.")

    if config.test:
        log.info("Evaluating model...")
        try:
            if config.max_test_time is not None:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(config.max_test_time)

            chemformer.test()

        except TimeoutException:
            pass
        callback_metrics = {**callback_metrics, **chemformer.trainer.callback_metrics}
        log.info("Evaluation finished.")

    if config.score:
        log.info("Scoring model...")
        chemformer.score_model(
            n_unique_beams=config.n_unique_beams,
            dataset=config.dataset_part,
            i_chunk=config.i_chunk,
            n_chunks=config.n_chunks,
            output_scores=config.output_score_data,
            output_sampled_smiles=config.output_sampled_smiles,
        )
        log.info("Scoring finished.")

    if config.predict:
        log.info("Predicting model...")
        chemformer.predict()
        callback_metrics = {**callback_metrics, **chemformer.trainer.callback_metrics}
        log.info("Predictions finished.")

    return callback_metrics


@hydra.main(version_base=None, config_path="config", config_name="pipeline")
def main(cfg: omegaconf.DictConfig) -> Optional[Tuple[float]]:
    # run the model
    metric_dict = pipeline(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_values = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # make optuna impervious to None
    if metric_values is not None:
        metric_values = tuple(999 if invalid_value(metr) else metr for metr in metric_values)
        if len(metric_values) == 1:
            metric_values = metric_values[0]
    else:
        metric_values = 999
    log.info(f"Metric values: {metric_values}")

    # close wandb if logger exists
    finish_wandb()

    # return optimized metric
    return metric_values


if __name__ == "__main__":
    main()
