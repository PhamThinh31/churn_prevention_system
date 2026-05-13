"""SASRec — Self-Attentive Sequential Recommendation.

Production-ready version of the teacher's notebook model. Adds:
  - causal attention mask (true autoregressive)
  - position embeddings
  - dropout
  - method to extract user vector (last non-pad hidden state) and item embeddings
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_len: int = 50,
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.item_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(emb_dim)

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) -> hidden (B, L, D). Auto-moves x to the model's device."""
        x = x.to(self.item_emb.weight.device)
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.item_emb(x) + self.pos_emb(pos)
        h = self.dropout(h)

        h = self.encoder(
            h,
            mask=self._causal_mask(L, x.device),
            src_key_padding_mask=(x == self.pad_id),
        )
        return self.norm(h)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Tie weights with item embedding for next-item prediction."""
        return hidden @ self.item_emb.weight.T

    def user_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Last non-pad hidden state per row -> (B, D). Auto-moves x to model device.

        Assumes RIGHT-padded sequences (real items first, PAD last). With left-pad,
        position 0 would be PAD and the encoder would emit NaN — see pad_right.
        Rows that are entirely PAD (rare) are returned as zeros: depending on the
        PyTorch version the transformer may emit either NaN (older builds) or some
        arbitrary finite value (newer builds with safer attention), so we detect
        both conditions defensively.
        """
        x = x.to(self.item_emb.weight.device)
        h = self.forward(x)
        real_len = (x != self.pad_id).sum(dim=1)
        idx = (real_len.clamp(min=1) - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        out = h.gather(1, idx).squeeze(1)

        # Zero out rows that had no real tokens — covers the all-PAD edge case
        # regardless of how the underlying attention handles a fully-masked row.
        empty = (real_len == 0).unsqueeze(1)
        nan = torch.isnan(out).any(dim=1, keepdim=True)
        bad = empty | nan
        if bad.any():
            out = torch.where(bad, torch.zeros_like(out), out)
        return out

    def item_matrix(self) -> torch.Tensor:
        """Item embedding matrix (V, D). Skip PAD row for downstream FAISS."""
        return self.item_emb.weight.detach()
