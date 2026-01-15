# -*- coding: utf-8 -*-
"""Graph utilities and dynamic graph learner (Python 3.6 compatible)."""

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn


def normalize_adj(adj, add_self_loops=True, eps=1e-6):
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    # torch.isinf is not available in some older torch builds; keep a safe fallback.
    try:
        inf_mask = torch.isinf(deg_inv_sqrt)
        deg_inv_sqrt = torch.where(inf_mask, torch.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
    except Exception:
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    D = torch.diag(deg_inv_sqrt)
    return torch.matmul(torch.matmul(D, adj), D)


def topk_sparse(adj, k):
    """Keep top-k entries per row (including self if present), zero others."""
    if k is None or k <= 0:
        return adj
    N = adj.size(0)
    k = int(min(k, N))
    # indices: (N,k)
    vals, idx = torch.topk(adj, k=k, dim=1)
    mask = torch.zeros_like(adj)
    mask.scatter_(1, idx, 1.0)
    return adj * mask


class BaseAdaptiveAdj(nn.Module):
    """Learnable adaptive adjacency similar to many traffic baselines.

    A_adp = softmax(relu(nodevec1 @ nodevec2))
    """

    def __init__(self, num_nodes, adp_dim):
        super(BaseAdaptiveAdj, self).__init__()
        self.num_nodes = int(num_nodes)
        self.adp_dim = int(adp_dim)
        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, self.adp_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(self.adp_dim, self.num_nodes), requires_grad=True)

    def forward(self):
        a = torch.matmul(self.nodevec1, self.nodevec2)
        a = torch.relu(a)
        a = torch.softmax(a, dim=1)
        return a


class DynamicGraphLearner(nn.Module):
    def __init__(self, base_adj, topk=20, alpha_init=0.5, use_base_adaptive_adj=True, adaptive_adj_dim=10,
                 ablation_flags=None):
        """
        ablation_flags (dict):
            'use_eq7': bool, 是否使用距离衰减 (Eq. 7)
            'use_eq9': bool, 是否使用动态融合 (Eq. 9)
        """
        super(DynamicGraphLearner, self).__init__()
        base_adj = np.asarray(base_adj, dtype=np.float32)
        self.N = int(base_adj.shape[0])
        self.register_buffer('base_adj', torch.from_numpy(base_adj))
        self.register_buffer('dist_mx', torch.from_numpy(base_adj > 0).float())

        self.topk = int(topk)

        # Eq. 9 的融合参数 alpha 
        a0 = float(alpha_init)
        logit = np.log(a0 / (1.0 - a0))
        self._alpha_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

        self.use_base_adaptive_adj = bool(use_base_adaptive_adj)
        self.base_adp = BaseAdaptiveAdj(self.N, adaptive_adj_dim) if self.use_base_adaptive_adj else None

        # 默认开启所有功能
        self.flags = ablation_flags if ablation_flags is not None else {'use_eq7': True, 'use_eq9': True}

    def alpha(self):
        return torch.sigmoid(self._alpha_logit)

    def forward(self, x_hist, event=None):
        # x_hist: (B,P,N,C)
        A_h = self._history_adj(x_hist)

        if event is not None:
            A_e = self._event_adj(event)
        else:
            A_e = torch.zeros_like(A_h)

        # base static
        A_base = self.base_adj
        if self.use_base_adaptive_adj and self.base_adp is not None:
            A_base = A_base + self.base_adp()

        # --- 消融核心: Eq. 9 (Fusion) ---
        # A_dyn = gamma * A_h + (1-gamma) * A_e
        if self.flags.get('use_eq9', True):
            gamma = self.alpha()
            A_dyn = gamma * A_h + (1.0 - gamma) * A_e
        else:
            A_dyn = 0.5 * A_h + 0.5 * A_e

        A_dyn = 0.5 * A_dyn + 0.5 * A_base
        A_dyn = topk_sparse(A_dyn, self.topk)
        A_dyn = normalize_adj(A_dyn, add_self_loops=True)
        return A_dyn

    def _history_adj(self, x_hist):
        B, P, N, C = x_hist.size()
        feat = x_hist.mean(dim=1).mean(dim=0)
        feat_norm = feat / (torch.norm(feat, dim=1, keepdim=True) + 1e-6)
        sim = torch.relu(torch.matmul(feat_norm, feat_norm.t()))
        return torch.softmax(sim, dim=1)

    def _event_adj(self, event):
        B, T, N, _ = event.size()
        sev = event.mean(dim=1).mean(dim=0).squeeze(-1) 
        impacted = (sev > 0.0).float() 
        A = torch.zeros((N, N), device=event.device)

        if torch.sum(impacted) == 0:
            return A

        if self.flags.get('use_eq7', True):
            lambda_val = 1.0
            dist_weight = self.dist_mx * 0.5
            eta = torch.exp(-dist_weight)  
        else:
            eta = torch.ones_like(self.base_adj)

        sev_matrix = sev.view(N, 1).expand(N, N)
        A = sev_matrix * eta * self.base_adj

        return torch.softmax(torch.relu(A), dim=1)