from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from molbart.models.abs_transformer import _AbsTransformerModel
from molbart.models.layers.decoder_layer import SwitchDecoderLayer
from molbart.models.layers.encoder_layer import SwitchEncoderLayer
from molbart.models.modules.decoder import TransformerDecoder
from molbart.models.modules.encoder import PerceiverEncoder, TransformerEncoder


class BARTModel(_AbsTransformerModel):
    def __init__(
        self,
        # decode_sampler,
        pad_token_idx: int,
        vocabulary_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_feedforward: int,
        lr: float,
        weight_decay: float,
        activation: str,
        num_steps: int,
        max_seq_len: int,
        norm_first: bool = True,
        schedule: str = "cycle",
        warm_up_steps: Optional[int] = None,
        dropout: float = 0.1,
        moe: bool = False,  # TODO: fix
        perceiver: bool = False,
        share_weights: bool = False,
        project_dim: Optional[int] = None,
        batch_first: bool = False,
        num_encoder_layers: Optional[int] = None,
        num_decoder_layers: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_idx=pad_token_idx,
            vocabulary_size=vocabulary_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_feedforward=d_feedforward,
            lr=lr,
            weight_decay=weight_decay,
            activation=activation,
            num_steps=num_steps,
            max_seq_len=max_seq_len,
            schedule=schedule,
            warm_up_steps=warm_up_steps,
            dropout=dropout,
            batch_first=batch_first,
            **kwargs,
        )

        if moe:
            encoder_layer = SwitchEncoderLayer(
                d_model,
                num_heads,
                d_feedforward // 8,
                dropout,
                activation,
                norm_first=norm_first,
                batch_first=batch_first,
            )
            encoder_layer._create_MoE(d_model=d_model, n_experts=8, batch_first=batch_first)
            decoder_layer = SwitchDecoderLayer(
                d_model,
                num_heads,
                d_feedforward // 8,
                dropout,
                activation,
                norm_first=norm_first,
                batch_first=batch_first,
            )
            decoder_layer._create_MoE(d_model=d_model, n_experts=8, batch_first=batch_first)

        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, num_heads, d_feedforward, dropout, activation, norm_first=norm_first, batch_first=batch_first
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model, num_heads, d_feedforward, dropout, activation, norm_first=norm_first, batch_first=batch_first
            )

        if not perceiver:
            self.encoder = TransformerEncoder(
                encoder_layer,
                num_layers if num_encoder_layers is None else num_encoder_layers,
                norm=nn.LayerNorm(d_model),
                share_weights=share_weights,
            )
            self.perceiver: bool = False
        else:
            self.encoder = PerceiverEncoder(
                encoder_layer,
                num_layers if num_encoder_layers is None else num_encoder_layers,
                norm=nn.LayerNorm(d_model),
                share_weights=share_weights,
                project_dim=project_dim,
                dropout=dropout,
                nhead=num_heads,
                batch_first=batch_first,
            )
            self.perceiver: bool = True

        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers if num_decoder_layers is None else num_decoder_layers,
            norm=nn.LayerNorm(d_model),
            share_weights=share_weights,
        )

        self.loss_function = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)

        self.token_fc = nn.Linear(d_model, vocabulary_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """
        batch["memory_input"] = self.encode(batch)
        batch["memory_pad_mask"] = batch["encoder_pad_mask"].clone() if not self.perceiver else None

        model_output = self.decode(batch)

        token_output = self.token_fc(model_output)

        output = {
            "model_output": model_output,
            "token_output": token_output,
            "batch_size": batch["encoder_input"].shape[0 if self.batch_first else 1],
        }

        return output

    def encode(self, batch):
        """Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """

        encoder_input = batch["encoder_input"].transpose(0, 1) if self.batch_first else batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        encoder_embs = self._construct_input(encoder_input)

        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)

        return model_output

    def decode(self, batch):
        """Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """
        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1) if not self.perceiver else None

        decoder_input = batch["decoder_input"] if not self.batch_first else batch["decoder_input"].transpose(0, 1)
        decoder_pad_mask = batch.get("decoder_pad_mask")
        decoder_pad_mask = decoder_pad_mask.transpose(0, 1) if decoder_pad_mask is not None else None

        decoder_embeddings = self._construct_input(decoder_input)

        seq_len = decoder_embeddings.shape[0 if not self.batch_first else 1]
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=decoder_embeddings.device)

        decoder_output = self.decoder(
            tgt=decoder_embeddings,
            memory=memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask,
        )
        return decoder_output if not self.batch_first else decoder_output.transpose(0, 1)

    def decode_probs(self, batch):
        """Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """
        decoder_output = self.decode(batch)
        token_log_probabilities = self.generator(decoder_output)
        return token_log_probabilities

    def generator(self, decoder_output):
        token_log_probabilities = self.log_softmax(self.token_fc(decoder_output))
        return token_log_probabilities

    def _calc_loss(self, batch_input, model_output):
        """Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input["target"]
        pad_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)

        return token_mask_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocabulary_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_function(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss

    def sample_molecules(self, batch_input, sampling_alg="greedy", return_tokenized=False):
        """Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        # Freezing the weights reduces the amount of memory leakage in the transformer
        self.freeze()

        if hasattr(self.sampler, "sample_molecules"):
            mol_strs, log_lhs = self.sampler.sample_molecules(
                self,
                batch_input,
                self.num_beams,
                sampling_alg,
                return_tokenized=return_tokenized,
            )
        else:
            enc_input = batch_input["encoder_input"]
            enc_mask = batch_input["encoder_pad_mask"]
            encode_input = {"encoder_input": enc_input, "encoder_pad_mask": enc_mask}
            memory = self.encode(encode_input)
            mem_mask = enc_mask.clone()

            _, batch_size, _ = tuple(memory.size())

            decode_fn = partial(self._decode_fn, memory=memory, mem_pad_mask=mem_mask)

            if sampling_alg == "greedy":
                mol_strs, log_lhs = self.sampler.greedy_decode(decode_fn, batch_size, memory.device)

            elif sampling_alg == "beam":
                mol_strs, log_lhs = self.sampler.beam_decode(decode_fn, batch_size, memory.device, k=self.num_beams)

            else:
                raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        # Must remember to unfreeze!
        self.unfreeze()

        return mol_strs, log_lhs

    def _decode_fn(self, token_ids, pad_mask, memory, mem_pad_mask):
        decode_input = {
            "decoder_input": token_ids,
            "decoder_pad_mask": pad_mask,
            "memory_input": memory,
            "memory_pad_mask": mem_pad_mask,
        }
        model_output = self.decode_probs(decode_input)
        return model_output

    def decode_batch(self, batch, return_last=True):
        """Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """
        batch["memory_input"] = batch["memory_input"].permute(1, 0, 2)
        batch["decoder_input"] = batch["decoder_input"].transpose(0, 1)
        batch["memory_pad_mask"] = batch["memory_pad_mask"].transpose(0, 1) if not self.perceiver else None
        token_probabilities = self.decode_probs(batch)

        if return_last:
            return token_probabilities[-1, :, :]
        else:
            return token_probabilities
