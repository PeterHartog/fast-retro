"""Decoder model for transformer architecture."""

# from typing import Optional

# import torch
import torch.nn as nn

from molbart.models.utils.clones import _get_clones, _get_duplicates


class TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None, share_weights: bool = False):
        super().__init__(decoder_layer, num_layers, norm)
        self.layers = (
            _get_clones(decoder_layer, num_layers) if not share_weights else _get_duplicates(decoder_layer, num_layers)
        )
