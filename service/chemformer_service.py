# import os
# from typing import List

# import eco2ai
# import fastapi
# import omegaconf as oc
# from fastapi import FastAPI
# from service_utils import calculate_llhs, estimate_compound_llhs, get_predictions

# import molbart.modules.util as util
# from molbart.models import Chemformer

# # Container for data, classes that can be loaded upon startup of the REST API
# config = oc.OmegaConf.load("../molbart/config/inference_score.yaml")

# config.batch_size = 64
# config.n_gpus = 1
# config.model_path = os.environ["CHEMFORMER_MODEL"]
# config.model_type = "bart"
# config.n_beams = 10
# config.task = os.environ["CHEMFORMER_TASK"]
# config.vocabulary_path = os.environ["CHEMFORMER_VOCAB"]
# model_args, data_args = util.get_chemformer_args(config)

# config.track_file = os.environ["TRACK_FILE"]
# config.track_project_name = os.environ["JOB_NAME"]
# config.port = os.environ["PORT"]
# if config.port is None:
#     config.port = 8003

# I, J, K = 0, 0, 0
# kwargs = {
#     "config": config,
#     "vocabulary_path": config.vocabulary_path,
#     "n_gpus": config.n_gpus,
#     "model_path": config.model_path,
#     "model_args": model_args,
#     "data_args": data_args,
#     "n_beams": config.n_beams,
#     "train_mode": "eval",
#     "datamodule_type": None,
#     "sample_unique": False,
# }

# global_items = {"chemformer": Chemformer(**kwargs)}
# tracker = eco2ai.Tracker(
#     project_name=config.track_project_name,
#     experiment_description="Multistep chemrformer API tracker",
#     measure_period=600,
#     alpha_2_code="SE",
#     file_name=config.track_file,
#     ignore_warnings=True,
# )


# def startup():
#     tracker.start()
#     return fastapi.Response(status_code=200, content="Server starting up...")


# def shutdown():
#     tracker.stop()
#     return fastapi.Response(status_code=200, content="Server shutting down...")


# app = FastAPI(on_startup=[startup], on_shutdown=[shutdown])


# @app.post("/chemformer-api/predict")
# def predict(smiles_list: List[str], n_beams: int = 10):
#     smiles, log_lhs, original_smiles = get_predictions(global_items["chemformer"], smiles_list, n_beams)

#     output = []
#     for item_pred, item_lhs, item_smiles in zip(smiles, log_lhs, original_smiles):
#         output.append(
#             {
#                 "input": item_smiles,
#                 "output": list(item_pred),
#                 "lhs": [float(val) for val in item_lhs],
#             }
#         )
#     return output


# @app.post("/chemformer-api/log_likelihood")
# def log_likelihood(reactants: List[str], products: List[str]):
#     log_lhs = calculate_llhs(global_items["chemformer"], reactants, products)

#     output = []
#     for prod_smi, react_smi, llhs in zip(products, reactants, log_lhs):
#         output.append(
#             {
#                 "product_smiles": str(prod_smi),
#                 "reactant_smiles": str(react_smi),
#                 "log_likelihood": float(llhs),
#             }
#         )
#     return output


# @app.post("/chemformer-api/compound_log_likelihood")
# def compound_log_likelihood(reactants: List[str], products: List[str], n_augments: int = 10):
#     log_lhs = estimate_compound_llhs(global_items["chemformer"], reactants, products, n_augments=n_augments)

#     output = []
#     for prod_smi, react_smi, llhs in zip(products, reactants, log_lhs):
#         output.append(
#             {
#                 "product_smiles": str(prod_smi),
#                 "reactant_smiles": str(react_smi),
#                 "log_likelihood": float(llhs),
#             }
#         )
#     return output


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(
#         "chemformer_service:app",
#         host="0.0.0.0",
#         port=int(config.port),
#         log_level="info",
#         reload=False,
#     )
