"""Decoder layers for transformer architecture."""

import torch
import torch.nn as nn

from molbart.models.layers.switch_layer import SwitchFeedForward


class SwitchDecoderLayer(nn.TransformerDecoderLayer):

    def _create_MoE(
        self,
        d_model: int,
        n_experts: int = 4,
        capacity_factor: float = 1.0,
        drop_tokens: bool = False,
        is_scale_prob: bool = True,
        batch_first: bool = False,
    ) -> None:
        assert d_model == self.linear1.in_features

        ff = nn.Sequential(
            self.linear1,
            nn.GELU(),
            self.dropout,
            self.linear2,
        )
        self.ff = SwitchFeedForward(
            ff,
            d_model,
            capacity_factor=capacity_factor,
            drop_tokens=drop_tokens,
            is_scale_prob=is_scale_prob,
            n_experts=n_experts,
            batch_first=batch_first,
        )

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff(x)[0]
        return self.dropout3(x)


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Self attention block
        query = self.norm1(tgt)
        query = self.self_attn(
            query,
            query,
            query,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        query = tgt + self.dropout1(query)

        # Context attention block
        att = self.norm2(query)
        att = self.multihead_attn(
            att,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        att = query + self.dropout2(att)

        # Feedforward block
        out = self.norm3(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout3(out)
        return out
