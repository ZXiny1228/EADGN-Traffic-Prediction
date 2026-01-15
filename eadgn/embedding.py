# -*- coding: utf-8 -*-
"""Embedding modules (Python 3.6 compatible)."""

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn


class TemporalEmbedding(nn.Module):
    """Time embedding from (time_of_day, day_of_week).

    TE expected shape: (B, T, 2) long.
        TE[..., 0] = time-of-day id in [0, steps_per_day)
        TE[..., 1] = day-of-week id in [0,7)
    """

    def __init__(self, steps_per_day, te_dim):
        super(TemporalEmbedding, self).__init__()
        self.steps_per_day = int(steps_per_day)
        self.te_dim = int(te_dim)
        self.tod = nn.Embedding(self.steps_per_day, self.te_dim)
        self.dow = nn.Embedding(7, self.te_dim)

    def forward(self, te):
        # te: (B,T,2)
        tod_id = te[..., 0].clamp(min=0, max=self.steps_per_day - 1)
        dow_id = te[..., 1].clamp(min=0, max=6)
        emb = self.tod(tod_id) + self.dow(dow_id)
        return emb  # (B,T,te_dim)


class STEmbedding(nn.Module):
    """Spatial + Temporal embedding.

    Spatial embedding (SE) is provided as numpy array (N, se_dim).
    Temporal embedding is learned.

    Returns:
        ste: (B, T, N, se_dim + te_dim)
    """

    def __init__(self, se, steps_per_day, te_dim, trainable_se=False):
        super(STEmbedding, self).__init__()
        se = np.asarray(se, dtype=np.float32)
        self.num_nodes = int(se.shape[0])
        self.se_dim = int(se.shape[1])
        self.te_dim = int(te_dim)
        self.temporal = TemporalEmbedding(steps_per_day=steps_per_day, te_dim=self.te_dim)

        se_tensor = torch.from_numpy(se)
        if trainable_se:
            self.se = nn.Parameter(se_tensor)
        else:
            self.register_buffer('se', se_tensor)

    def forward(self, te):
        # te: (B,T,2)
        B = te.size(0)
        T = te.size(1)

        se = self.se.view(1, 1, self.num_nodes, self.se_dim).expand(B, T, self.num_nodes, self.se_dim)
        te_emb = self.temporal(te).view(B, T, 1, self.te_dim).expand(B, T, self.num_nodes, self.te_dim)
        ste = torch.cat([se, te_emb], dim=-1)
        return ste


class EventEmbedding(nn.Module):
    """Optional event embedding.

    E expected shape: (B, T, N, 1) float.
    Returns: (B, T, N, event_dim)
    """

    def __init__(self, event_dim):
        super(EventEmbedding, self).__init__()
        self.event_dim = int(event_dim)
        self.proj = nn.Sequential(
            nn.Linear(1, self.event_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.event_dim, self.event_dim),
        )

    def forward(self, e):
        if e is None:
            return None
        # (B,T,N,1) -> (B,T,N,event_dim)
        return self.proj(e)
