"""Encoder model for transformer architecture."""

from typing import Optional

import torch
import torch.nn as nn

from molbart.models.utils.clones import _get_clones, _get_duplicates


class TransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None, share_weights: bool = False):
        super().__init__(encoder_layer, num_layers, norm)
        self.layers = (
            _get_clones(encoder_layer, num_layers) if not share_weights else _get_duplicates(encoder_layer, num_layers)
        )

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)  # , output, output,

        if self.norm is not None:
            output = self.norm(output)

        return output


class PerceiverEncoder(TransformerEncoder):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        share_weights: bool = None,
        project_dim: int = 256,
        dropout: float = 0.1,
        nhead: int = 8,
        batch_first: bool = False,
    ):
        super().__init__(encoder_layer, num_layers, norm, share_weights)
        self.layers = (
            _get_clones(encoder_layer, num_layers) if not share_weights else _get_duplicates(encoder_layer, num_layers)
        )
        proj_dim, embed_dim = project_dim, encoder_layer.linear1.in_features
        self.projection = nn.Parameter(torch.randn(proj_dim, embed_dim), requires_grad=True)
        self.batch_first = False

        # projection
        self.norm_first = encoder_layer.norm_first
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        context = src

        if context.dim() == 3:
            batch_size = context.shape[0 if self.batch_first else 1]
            output = (
                self.projection.unsqueeze(0).repeat(batch_size, 1, 1)
                if self.batch_first
                else self.projection.unsqueeze(1).repeat(1, batch_size, 1)
            )
        else:
            output = self.projection

        output = self._mha_block(output, context, attn_mask=mask, key_padding_mask=src_key_padding_mask)

        for mod in self.layers:
            output = mod(output, src_mask=None, src_key_padding_mask=None)

        if self.norm is not None:
            output = self.norm(output)

        return output

    # multihead attention block
    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.norm_first:
            x = self.norm1(x)
            x = self.multihead_attn(
                x,
                mem,
                mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                need_weights=False,
            )[0]
        else:
            x = self.multihead_attn(
                x,
                mem,
                mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                need_weights=False,
            )[0]
            x = self.norm1(x)
        return self.dropout1(x)
