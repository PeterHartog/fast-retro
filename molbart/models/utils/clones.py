"""Modules used for multiplying the same module N times."""
import copy

import torch.nn as nn


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_duplicates(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([module for _ in range(N)])
