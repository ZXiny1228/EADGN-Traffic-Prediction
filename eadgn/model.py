# -*- coding: utf-8 -*-
"""EADGN main model (Python 3.6 compatible).

This is a pragmatic re-implementation inspired by the paper idea:
- Event-adaptive dynamic adjacency (history-driven + event-enhanced)
- Encoder -> Bridge -> Decoder with ST blocks

Shapes:
    X:  (B, P, N, 1)
    TE: (B, P+Q, 2)
    E:  (B, P+Q, N, 1) optional
    Y:  (B, Q, N, 1)
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .embedding import STEmbedding, EventEmbedding
from .graph import DynamicGraphLearner
from .layers import STBlock, BridgeAttention


class EADGN(nn.Module):
    def __init__(self, adj, se, cfg):
        super(EADGN, self).__init__()

        self.cfg = cfg
        self.P = int(cfg.task.P)
        self.Q = int(cfg.task.Q)

        # Derive num_nodes from actual SE matrix (more robust than trusting config)
        try:
            self.num_nodes = int(se.shape[0])
        except Exception:
            self.num_nodes = int(cfg.model.num_nodes)
        self.in_dim = int(cfg.model.in_dim)
        self.out_dim = int(cfg.model.out_dim)
        self.hidden_dim = int(cfg.model.hidden_dim)

        # Derive se_dim from actual SE matrix to avoid shape mismatch
        try:
            self.se_dim = int(se.shape[1])
        except Exception:
            self.se_dim = int(cfg.model.se_dim)

        # keep cfg in sync (best-effort)
        try:
            cfg.model.num_nodes = self.num_nodes
            cfg.model.se_dim = self.se_dim
        except Exception:
            pass
        self.te_dim = int(cfg.model.te_dim)
        self.ste_dim = self.se_dim + self.te_dim
        self.event_dim = int(cfg.model.event_dim)
        self.event_enabled = bool(cfg.data.event_enabled)

        self.dropout = float(cfg.model.dropout)
        self.num_blocks = int(cfg.model.num_blocks)
        self.num_heads = int(cfg.model.num_heads)
        self.diffusion_steps = int(cfg.model.diffusion_steps)

        # Embeddings
        self.st_embedding = STEmbedding(
            se=se,
            steps_per_day=int(cfg.data.steps_per_day),
            te_dim=self.te_dim,
            trainable_se=False,
        )
        self.event_embedding = EventEmbedding(self.event_dim) if self.event_enabled else None

        # Input projection
        self.input_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.ste_proj = nn.Linear(self.ste_dim, self.hidden_dim)

        # Dynamic graph learner
        self.graph_learner = DynamicGraphLearner(
            base_adj=adj,
            topk=int(cfg.model.topk),
            alpha_init=float(cfg.model.alpha_init),
            use_base_adaptive_adj=bool(cfg.model.use_base_adaptive_adj),
            adaptive_adj_dim=int(cfg.model.adaptive_adj_dim),
        )

        # Encoder / Decoder blocks
        self.encoder = nn.ModuleList([
            STBlock(
                hidden_dim=self.hidden_dim,
                diffusion_steps=self.diffusion_steps,
                num_heads=self.num_heads,
                dropout=self.dropout,
                ste_dim=self.ste_dim,
                event_dim=self.event_dim if self.event_enabled else 0,
            ) for _ in range(self.num_blocks)
        ])

        self.bridge = BridgeAttention(self.hidden_dim, ste_dim=self.ste_dim, dropout=self.dropout)

        self.decoder = nn.ModuleList([
            STBlock(
                hidden_dim=self.hidden_dim,
                diffusion_steps=self.diffusion_steps,
                num_heads=self.num_heads,
                dropout=self.dropout,
                ste_dim=self.ste_dim,
                event_dim=self.event_dim if self.event_enabled else 0,
            ) for _ in range(self.num_blocks)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

        # 获取消融配置
        ablation_flags = {
            'use_eq7': cfg.model.get('use_eq7', True),
            'use_eq9': cfg.model.get('use_eq9', True)
        }

        # Dynamic graph learner
        self.graph_learner = DynamicGraphLearner(
            base_adj=adj,
            topk=int(cfg.model.topk),
            alpha_init=float(cfg.model.alpha_init),
            use_base_adaptive_adj=bool(cfg.model.use_base_adaptive_adj),
            adaptive_adj_dim=int(cfg.model.adaptive_adj_dim),
            ablation_flags=ablation_flags  
        )

    def forward(self, x, te, event=None, return_adj=False):
        """Forward.

        Args:
            x: (B,P,N,1)
            te: (B,P+Q,2)
            event: (B,P+Q,N,1) or None
        """
        B, P, N, C = x.size()
        if P != self.P:
            self.P = P

        # STE for full horizon
        ste_all = self.st_embedding(te)  # (B,P+Q,N,ste_dim)
        ste_enc = ste_all[:, :P]
        ste_dec = ste_all[:, P:]

        # Event embedding
        eve_all = None
        if self.event_enabled and (event is not None) and (self.event_embedding is not None):
            eve_all = self.event_embedding(event)
        eve_enc = eve_all[:, :P] if eve_all is not None else None
        eve_dec = eve_all[:, P:] if eve_all is not None else None

        # Project input
        x_h = self.input_proj(x)  # (B,P,N,H)
        x_h = x_h + self.ste_proj(ste_enc)

        # Dynamic adjacency (one per batch)
        A_dyn = self.graph_learner(x_h, event=event)

        # Encoder
        h = x_h
        for blk in self.encoder:
            h = blk(h, A_dyn, ste=ste_enc, eve=eve_enc)

        # Bridge to Q length
        dec = self.bridge(h, ste_enc=ste_enc, ste_dec=ste_dec)

        # Decoder
        for blk in self.decoder:
            dec = blk(dec, A_dyn, ste=ste_dec, eve=eve_dec)

        y = self.output_proj(dec)  # (B,Q,N,out_dim)

        if return_adj:
            return y, A_dyn
        return y
