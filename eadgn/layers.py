# -*- coding: utf-8 -*-
"""Model layers (Python 3.6 compatible).

We implement a lightweight EADGN-style block:
- Dynamic adjacency (computed outside, passed in)
- Spatial modeling via diffusion graph convolution
- Temporal modeling via multi-head self-attention (custom, no torch>=1.9 deps)
- Bridge attention (encoder->decoder mapping)
"""

from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def graph_matmul(A, X):
    """Multiply adjacency A (N,N) with node features X (B,T,N,C)."""
    return torch.einsum('nm,btnc->btmc', A, X)


class DiffusionConv(nn.Module):
    """Diffusion graph convolution with K steps.

    Y = Linear([X, A X, A^2 X, ...])
    """

    def __init__(self, in_dim, out_dim, K=2, dropout=0.0):
        super(DiffusionConv, self).__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.K = int(K)
        self.proj = nn.Linear((self.K + 1) * self.in_dim, self.out_dim)
        self.dropout = float(dropout)

    def forward(self, X, A):
        feats = [X]
        Xk = X
        for _ in range(self.K):
            Xk = graph_matmul(A, Xk)
            feats.append(Xk)
        H = torch.cat(feats, dim=-1)
        H = self.proj(H)
        if self.dropout > 0.0:
            H = F.dropout(H, p=self.dropout, training=self.training)
        return H


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention along the time axis (per node).

    Input: (B,T,N,C)
    Output: (B,T,N,C)

    We implement it manually for compatibility across torch versions (and to avoid
    relying on MultiheadAttention's `batch_first` flag).
    """

    def __init__(self, dim, num_heads=4, dropout=0.0):
        super(TemporalSelfAttention, self).__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError('dim must be divisible by num_heads. Got dim={}, heads={}'.format(self.dim, self.num_heads))
        self.head_dim = self.dim // self.num_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.out = nn.Linear(self.dim, self.dim)
        self.dropout = float(dropout)

    def forward(self, X):
        # X: (B,T,N,C)
        B, T, N, C = X.size()
        # (B,N,T,C)
        Xn = X.permute(0, 2, 1, 3).contiguous()
        qkv = self.qkv(Xn)  # (B,N,T,3C)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # split heads: (B,N,heads,T,head_dim)
        def split_heads(t):
            t = t.view(B, N, T, self.num_heads, self.head_dim)
            return t.permute(0, 1, 3, 2, 4).contiguous()

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # attention: (B,N,heads,T,T)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        if self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)  # (B,N,heads,T,head_dim)
        # merge heads
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, N, T, C)
        out = self.out(out)

        # back to (B,T,N,C)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out


class STBlock(nn.Module):
    """Spatial-Temporal block.

    - Spatial diffusion conv with dynamic adjacency
    - Temporal attention (with optional graph-injected features)
    """

    def __init__(self, hidden_dim, diffusion_steps=2, num_heads=4, dropout=0.0, ste_dim=0, event_dim=0):
        super(STBlock, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.dropout = float(dropout)

        self.spatial = DiffusionConv(hidden_dim, hidden_dim, K=self.diffusion_steps, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # temporal attention input dimension: X (+ STE) (+ graph-injected Xn) (+ event)
        self.use_ste = int(ste_dim) > 0
        self.use_event = int(event_dim) > 0

        attn_in_dim = hidden_dim
        if self.use_ste:
            attn_in_dim += int(ste_dim)
        # graph injected features
        attn_in_dim += hidden_dim
        if self.use_event:
            attn_in_dim += int(event_dim)

        self.attn_in = nn.Linear(attn_in_dim, hidden_dim)
        self.temporal = TemporalSelfAttention(hidden_dim, num_heads=num_heads, dropout=self.dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, X, A, ste=None, eve=None):
        # X: (B,T,N,C)
        # A: (N,N)
        # ste: (B,T,N,ste_dim) or None
        # eve: (B,T,N,event_dim) or None

        # spatial
        H = self.spatial(X, A)
        X = self.norm1(X + H)

        # temporal (inject A @ X)
        Xn = torch.einsum('nm,btnc->btmc', A, X)
        parts = [X, Xn]
        if ste is not None:
            parts.append(ste)
        if eve is not None:
            parts.append(eve)
        H_in = torch.cat(parts, dim=-1)
        H_in = self.attn_in(H_in)

        H = self.temporal(H_in)
        X = self.norm2(X + H)

        H = self.ffn(X)
        X = self.norm3(X + H)
        return X


class BridgeAttention(nn.Module):
    """Bridge attention mapping encoder length P to decoder length Q.

    Inputs:
        enc: (B,P,N,C)
        ste_enc: (B,P,N,D)
        ste_dec: (B,Q,N,D)

    Output:
        dec_init: (B,Q,N,C)
    """

    def __init__(self, hidden_dim, ste_dim, dropout=0.0):
        super(BridgeAttention, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.ste_dim = int(ste_dim)
        self.dropout = float(dropout)

        # project STE to queries/keys, encoder states to values
        self.q_proj = nn.Linear(self.ste_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.ste_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, enc, ste_enc, ste_dec):
        B, P, N, C = enc.size()
        Q = ste_dec.size(1)

        # compute per-node attention over time
        # reshape to (B,N,P,*) and (B,N,Q,*)
        enc_bn = enc.permute(0, 2, 1, 3).contiguous()  # (B,N,P,C)
        ste_enc_bn = ste_enc.permute(0, 2, 1, 3).contiguous()  # (B,N,P,D)
        ste_dec_bn = ste_dec.permute(0, 2, 1, 3).contiguous()  # (B,N,Q,D)

        q = self.q_proj(ste_dec_bn)  # (B,N,Q,C)
        k = self.k_proj(ste_enc_bn)  # (B,N,P,C)
        v = self.v_proj(enc_bn)      # (B,N,P,C)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(C))  # (B,N,Q,P)
        attn = torch.softmax(attn, dim=-1)
        if self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)  # (B,N,Q,C)
        out = self.out_proj(out)

        # back to (B,Q,N,C)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out
