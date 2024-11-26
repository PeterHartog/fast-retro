import os
import pickle
import sys

import pandas as pd
import tqdm
from rxnutils.routes.readers import read_aizynthfinder_dict

# To use rxnmapper, this assumes your conda and the
# environments are installed in your home directory - update if necessary
conda_path = os.path.expanduser("~") + "/miniconda3"
os.environ["CONDA_PATH"] = f"{conda_path}/bin/"
os.environ["RXNMAPPER_ENV_PATH"] = f"{conda_path}/envs/rxnmapper/"

idx = int(sys.argv[1])

root = f"multistep/last_epoch/output"
files = {
    k: [
        f"{root}/{k}/{f}"
        for f in os.listdir(f"{root}/{k}")
        if os.path.isfile(os.path.join(f"{root}/{k}", f)) and f.endswith(".json.gz")
    ]
    for k in os.listdir(root)
}
dfs: dict[str, pd.DataFrame] = {k: pd.concat(pd.read_json(f, orient="table") for f in fs) for k, fs in files.items()}

for setkey in dfs.keys():
    routes = []
    for i in tqdm.tqdm(range(len(dfs[setkey]))):
        route = read_aizynthfinder_dict(dfs[setkey].trees.iloc[i][0])  # Top route for each molecule
        if route.max_depth > 0:
            route.assign_atom_mapping(only_rxnmapper=True, overwrite=True)
        routes.append(route.reaction_tree)

    with open(f"{root}/{setkey}_trees.pickle", "wb") as fileobj:
        pickle.dump(routes, fileobj)
