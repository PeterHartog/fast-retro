import os
from typing import List, Optional, Tuple

import eco2ai
import fastapi
import hydra
import omegaconf as oc

import molbart.modules.util as util
from molbart.models import Chemformer
from service.service_utils import (
    calculate_llhs,
    estimate_compound_llhs,
    get_predictions,
)


@hydra.main(version_base=None, config_path="../molbart/config", config_name="inference_score")
def main(cfg: oc.DictConfig) -> Optional[Tuple]:
    config = cfg

    config.batch_size = 64
    config.n_gpus = 1
    config.model_type = "bart"
    config.n_beams = 10
    config.task = os.environ["CHEMFORMER_TASK"]
    config.vocabulary_path = os.environ["CHEMFORMER_VOCAB"]
    model_args, data_args = util.get_chemformer_args(config)

    config.track_file = os.environ["TRACK_FILE"]
    config.track_project_name = os.environ["JOB_NAME"]
    config.port = os.environ["PORT"]
    if config.port is None:
        config.port = 8003

    tracker = eco2ai.Tracker(
        project_name=config.track_project_name,
        experiment_description="Multistep chemrformer API tracker",
        measure_period=600,
        alpha_2_code="SE",
        file_name=config.track_file,
        ignore_warnings=True,
    )

    def startup():
        tracker.start()
        return fastapi.Response(status_code=200, content="Server starting up...")

    def shutdown():
        tracker.stop()
        return fastapi.Response(status_code=200, content="Server shutting down...")

    app = fastapi.FastAPI(on_startup=[startup], on_shutdown=[shutdown])

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(config.port),
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
