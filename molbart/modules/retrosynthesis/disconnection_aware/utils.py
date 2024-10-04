"""Module containing auxiliary functions needed to run the disconnection-Chemformer"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem

# fmt: off
_TAG_CONVERT_SCRIPT = Path(__file__).parent / "tag_converter.py"
# fmt: on


def get_model_input(
    products_atom_map_tagged: List[str],
    rxnutils_env_path: Optional[str] = None,
    conda_path: Optional[str] = None,
) -> str:
    """
    Transform mapped product SMILES and current bond to tagged input SMILES for the
    disconnection-aware Chemformer.

    Args:
        product_atom_map_tagged: product SMILES with atom-map tagged disconnection site
        rxnutils_env_path: path to conda evironment with rxnutils
        conda_path: path to conda executable
    Returns:
        A SMILES with atoms in current bond tagged by "<atom>!"
    """
    output_handle, output_path = tempfile.mkstemp(suffix=".csv")

    args = f"--input_smiles {' '.join(products_atom_map_tagged)} --output {output_path}"
    cmd = f"python {_TAG_CONVERT_SCRIPT} {args}"

    if rxnutils_env_path:
        cmd = f"conda run -p {rxnutils_env_path} " + cmd

        if conda_path:
            cmd = conda_path + cmd

    subprocess.check_output(cmd.split())

    with open(output_path, "r") as fid:
        smiles = fid.read().split()

    os.remove(output_path)
    os.close(output_handle)
    fid.close()
    return smiles


def verify_disconnection(
    reactants_mapped: str,
    bond_atom_inds: List[int],
    mapping_to_index: Optional[Dict[int, int]] = None,
) -> bool:
    """
    Check that current bond was broken in a specific prediction (reactants_smiles).

    Args:
        reactants_mapped: Atom-mapped reactant SMILES
        bond_atom_inds: The bond which should be broken
    Returns:
        whether the bonds was broken or not.
    """
    mol = Chem.MolFromSmiles(reactants_mapped)

    if not mapping_to_index:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in bond_atom_inds:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomMapNum() in bond_atom_inds:
                        return False
    else:
        counter = 0
        for ind in bond_atom_inds:
            try:
                atom = mol.GetAtomWithIdx(mapping_to_index[ind])
            except KeyError:
                counter += 1
                continue
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomMapNum() in bond_atom_inds:
                    return False
        if counter == len(bond_atom_inds):
            return False
    return True
