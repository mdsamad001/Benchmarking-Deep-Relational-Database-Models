import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph
from ml_dtypes import bfloat16
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from .griffin_float_embedder import FloatEncoder, GriffinFloatEmbedder

allow_ops_in_compiled_graph()
flex_attention = torch.compile(flex_attention)


class MaskedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, block_mask):
        q = rearrange(self.wq(x), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.wk(x), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.wv(x), "b s (h d) -> b h s d", h=self.num_heads)
        if block_mask is None:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = F.scaled_dot_product_attention(q, k, v)
        else:
            x = flex_attention(q, k, v, block_mask=block_mask)
        return self.wo(rearrange(x, "b h s d -> b s (h d)"))


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff,    bias=False)
        self.w2 = nn.Linear(d_ff,    d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff,    bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RelationalBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norms = nn.ModuleDict({l: nn.RMSNorm(d_model) for l in ["feat","nbr","col","full","ffn"]})
        self.attns = nn.ModuleDict({l: MaskedAttention(d_model, num_heads) for l in ["feat","nbr","col","full"]})
        self.ffn   = FFN(d_model, d_ff)

    def forward(self, x, block_masks):
        for l in ["col", "feat", "nbr", "full"]:
            x = x + self.attns[l](self.norms[l](x), block_mask=block_masks[l])
        return x + self.ffn(self.norms["ffn"](x))


def _make_block_mask(mask, batch_size, seq_len, device):
    def _mod(b, h, q_idx, kv_idx):
        return mask[b, q_idx, kv_idx]
    return create_block_mask(
        mask_mod=_mod, B=batch_size, H=None,
        Q_LEN=seq_len, KV_LEN=seq_len, device=device, _compile=True,
    )


class RelationalTransformer(nn.Module):
    def __init__(self, num_blocks, d_model, d_text, num_heads, d_ff, float_ckpt_path=None):
        super().__init__()

        if float_ckpt_path is not None:
            numeric_encoder = nn.Sequential(
                GriffinFloatEmbedder(dim=64, hidden_dim=256, pretrained_path=float_ckpt_path),
                nn.Linear(64, d_model),
            )
        else:
            numeric_encoder = nn.Linear(1, d_model, bias=True)

        self.enc_dict = nn.ModuleDict({
            "number":   numeric_encoder,
            "text":     nn.Linear(d_text, d_model, bias=True),
            "datetime": nn.Linear(1,      d_model, bias=True),
            "col_name": nn.Linear(d_text, d_model, bias=True),
            "boolean":  nn.Linear(1,      d_model, bias=True),
        })
        self.dec_dict = nn.ModuleDict({
            "number":   nn.Linear(d_model, 1,      bias=True),
            "text":     nn.Linear(d_model, d_text, bias=True),
            "datetime": nn.Linear(d_model, 1,      bias=True),
            "boolean":  nn.Linear(d_model, 1,      bias=True),
        })
        self.norm_dict = nn.ModuleDict({
            t: nn.RMSNorm(d_model) for t in ["number","text","datetime","col_name","boolean"]
        })
        self.mask_embs = nn.ParameterDict({
            t: nn.Parameter(torch.randn(d_model)) for t in ["number","text","datetime","boolean"]
        })
        self.blocks   = nn.ModuleList([RelationalBlock(d_model, num_heads, d_ff) for _ in range(num_blocks)])
        self.norm_out = nn.RMSNorm(d_model)
        self.d_model  = d_model

    def forward(self, batch):
        node_idxs     = batch["node_idxs"]
        f2p_nbr_idxs  = batch["f2p_nbr_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        is_padding    = batch["is_padding"]
        batch_size, seq_len = node_idxs.shape
        device        = node_idxs.device

        pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])

        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]

        kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(-1)

        q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)

        same_col_table = (
            (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) &
            (table_name_idxs[:, :, None] == table_name_idxs[:, None, :])
        )

        attn_masks = {
            "feat": (same_node | kv_in_f2p) & pad,
            "nbr":  q_in_f2p & pad,
            "col":  same_col_table & pad,
            "full": pad,
        }
        for l in attn_masks:
            attn_masks[l] = attn_masks[l].contiguous()

        make_bm   = partial(_make_block_mask, batch_size=batch_size, seq_len=seq_len, device=device)
        block_masks = {l: make_bm(m) for l, m in attn_masks.items()}

        x = self.norm_dict["col_name"](self.enc_dict["col_name"](batch["col_name_values"])) * (~is_padding)[..., None]

        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            visible = (batch["sem_types"] == i) & ~batch["masks"] & ~is_padding
            masked  = (batch["sem_types"] == i) &  batch["masks"] & ~is_padding
            x = x + self.norm_dict[t](self.enc_dict[t](batch[f"{t}_values"])) * visible[..., None]
            x = x + self.mask_embs[t] * masked[..., None]

        for block in self.blocks:
            x = block(x, block_masks)
        x = self.norm_out(x)

        loss_out = x.new_zeros(())
        yhat_out = {t: None for t in ["number", "text", "datetime", "boolean"]}

        sem_types = batch["sem_types"]
        masks     = batch["masks"].bool()

        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            yhat          = self.dec_dict[t](x)
            y             = batch[f"{t}_values"]
            sem_type_mask = (sem_types == i) & masks

            if not sem_type_mask.any():
                loss_out    = loss_out + yhat.sum() * 0.0
                yhat_out[t] = yhat
                continue

            if t in ("number", "datetime"):
                loss_t = F.huber_loss(yhat, y, reduction="none").mean(-1)
            elif t == "boolean":
                loss_t = F.binary_cross_entropy_with_logits(
                    yhat, (y > 0).float(), reduction="none"
                ).mean(-1)
            elif t == "text":
                raise ValueError("masking text not supported")

            loss_out    = loss_out + (loss_t * sem_type_mask).sum()
            yhat_out[t] = yhat

        loss_out = loss_out / masks.sum()
        return loss_out, yhat_out
