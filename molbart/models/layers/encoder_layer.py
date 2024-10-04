"""Encoder layers for transformer architecture."""

from typing import Optional

import torch
import torch.nn as nn

from molbart.models.layers.switch_layer import SwitchFeedForward
from molbart.models.utils.clones import _get_clones


class SwitchEncoderLayer(nn.TransformerEncoderLayer):

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
        return self.dropout2(x)


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormEncoderLayer(nn.TransformerEncoderLayer):
    def forward(
        self,
        query: torch.Tensor,
        # key: torch.Tensor,
        # value: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # if query is key is value:
        src = query

        # Self attention block
        att = self.norm1(src)
        att = self.self_attn(att, att, att, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        att = src + self.dropout1(att)
        # else:
        #     src_mask = None
        #     # Self attention block
        #     att, key, value = self.norm1(query), self.norm1(key), self.norm1(value)
        #     att = self.self_attn(att, key, value, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        #     att = query + self.dropout1(att)

        # Feedforward block
        out = self.norm2(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout2(out)
        return out


class SparseGate(nn.Module):
    def __init__(self, d_model: int, n_experts: int, k: int = 1) -> None:
        super().__init__()

        self.k = k
        self.gate = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated_values = self.gate(x)
        gated_values = self.softmax(gated_values)
        _, indices = torch.topk(gated_values, self.k, dim=-1)
        return indices  # [batch_size, max_sequence_length]


class MoEEncoderLayer(nn.TransformerEncoderLayer):
    # def forward(
    #     self,
    #     query: torch.Tensor,
    #     # key: torch.Tensor,
    #     # value: torch.Tensor,
    #     src_mask: Optional[torch.Tensor] = None,
    #     src_key_padding_mask: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    # # Self attention block
    # att, key, value = self.norm1(query), self.norm1(key), self.norm1(value)

    # src = query

    # # Self attention block
    # att = self.norm1(src)
    # att = self.self_attn(att, att, att, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
    # att = src + self.dropout1(att)
    # att = self.self_attn(att, key, value, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
    # att = query + self.dropout1(att)

    # # Feedforward block
    # gate_values = self.gate(att)
    # masks = [gate_values.reshape(-1) == i for i in range(self.n_experts + 1)]

    # if self.k > 1:
    #     masks = [mask.reshape(-1, 2).any(-1) for mask in masks]

    # expert_out = []
    # for i in range(self.n_experts):
    #     mask = masks[i]
    #     if not mask.any():
    #         print(i)
    #         out = torch.zeros_like(att, requires_grad=True)
    #     else:
    #         masked_tensor = att
    #         masked_tensor.reshape(-1, self.d_model)[~mask.reshape(-1, self.k).any(-1)] = 0
    #         if torch.nonzero(masked_tensor).numel() == 0:
    #             print("hmm")
    #             out = torch.zeros_like(att, requires_grad=True)
    #         else:
    #             out = self.experts[i](masked_tensor)

    #             print(torch.nonzero(masked_tensor).numel())
    #         # masked_tensor = masked_tensor.to_sparse()
    #         # print(masked_tensor)
    #         # print(self.norm1(masked_tensor))
    #     expert_out.append(out)  # .to_dense())
    # print([exp.shape for exp in expert_out])
    # out = att + torch.stack(expert_out, dim=-1).mean(dim=-1)
    # # out = self.norm2(att)
    # # out = self.linear2(self.dropout(self.activation(self.linear1(out))))
    # # out = att + self.dropout2(out)
    # return out

    def _create_MoE(self, d_model: int, n_experts: int, k: int = 1) -> None:
        self.k = k
        assert d_model == self.linear1.in_features

        ff = nn.Sequential(
            self.linear1,
            nn.GELU(),
            self.dropout,
            self.linear2,
        )

        # layers = _get_clones(ff, n_experts)

        self.gate = SparseGate(d_model, n_experts=n_experts + 1)
        self.experts = _get_clones(ff, n_experts)
        self.d_model = d_model
        self.k = k
        self.n_experts = n_experts

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:

        # Sparsity gate
        gate_values = self.gate(x)
        masks = [gate_values.reshape(-1) == i for i in range(self.n_experts + 1)]
        if self.k > 1:
            masks = [mask.reshape(-1, 2).any(-1) for mask in masks]

        # Feedforward block
        expert_out = []
        for i in range(self.n_experts):
            mask = masks[i]

            if not mask.any():
                out = torch.zeros_like(x, requires_grad=True)
            else:
                masked_tensor = x.clone()
                masked_tensor.reshape(-1, self.d_model)[~mask.reshape(-1, self.k).any(-1)] = 0
                # import pdb

                # pdb.set_trace()

                if torch.nonzero(masked_tensor).numel() == 0:
                    out = torch.zeros_like(x, requires_grad=True)
                    print("hmm")
                else:
                    masked_tensor = masked_tensor.reshape(-1, self.d_model)
                    masked_tensor = masked_tensor.to_sparse()
                    out = self.experts[i](masked_tensor)

                    out = out.to_dense().reshape_as(x)
            expert_out.append(out.to_dense())
        out = torch.stack(expert_out, dim=-1).mean(dim=-1)
        return self.dropout2(out)
