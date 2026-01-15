# -*- coding: utf-8 -*-
"""Utility functions (Python 3.6 compatible)."""

from __future__ import absolute_import, division, print_function

import os
import random
import torch

import numpy as np

try:
    import torch
except Exception:
    torch = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class StandardScaler(object):
    def __init__(self, mean, std, eps=1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def transform(self, data):
        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data):
        return data * (self.std + self.eps) + self.mean


class MinMaxScaler(object):
    def __init__(self, data_min, data_max, eps=1e-6):
        self.data_min = data_min
        self.data_max = data_max
        self.eps = eps

    def transform(self, data):
        return (data - self.data_min) / (self.data_max - self.data_min + self.eps)

    def inverse_transform(self, data):
        return data * (self.data_max - self.data_min + self.eps) + self.data_min


def _get_mask(y_true, null_val=0.0):
    if torch is not None and torch.is_tensor(y_true):
        if np.isnan(null_val):
            mask = ~torch.isnan(y_true)
        else:
            mask = (y_true != null_val)
        mask = mask.float()
        mask = mask / (torch.mean(mask) + 1e-6)
        return mask
    else:
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = (y_true != null_val)
        mask = mask.astype(np.float32)
        mask = mask / (np.mean(mask) + 1e-6)
        return mask


def masked_mae(y_pred, y_true, null_val=0.0):
    mask = _get_mask(y_true, null_val)
    if torch is not None and torch.is_tensor(y_true):
        loss = torch.abs(y_pred - y_true)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)
    else:
        loss = np.abs(y_pred - y_true)
        loss = loss * mask
        loss = np.where(np.isnan(loss), 0.0, loss)
        return np.mean(loss)


def masked_mse(y_pred, y_true, null_val=0.0):
    mask = _get_mask(y_true, null_val)
    if torch is not None and torch.is_tensor(y_true):
        loss = (y_pred - y_true) ** 2
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)
    else:
        loss = (y_pred - y_true) ** 2
        loss = loss * mask
        loss = np.where(np.isnan(loss), 0.0, loss)
        return np.mean(loss)


def masked_rmse(y_pred, y_true, null_val=0.0):
    if torch is not None and torch.is_tensor(y_true):
        return torch.sqrt(masked_mse(y_pred, y_true, null_val) + 1e-6)
    return np.sqrt(masked_mse(y_pred, y_true, null_val) + 1e-6)


def masked_mape(preds, labels, null_val=0.0, eps=1e-5):
    if torch.is_tensor(null_val) and torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask_mean = torch.mean(mask)
    if mask_mean.item() == 0:
        return torch.tensor(0.0, device=labels.device)

    mask = mask / mask_mean
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    denom = torch.abs(labels)
    denom = torch.clamp(denom, min=eps)

    loss = torch.abs((preds - labels) / denom)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = torch.where(torch.isinf(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mbe(preds, labels, null_val=0.0):
    if torch.is_tensor(null_val) and torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask_mean = torch.mean(mask)
    if mask_mean.item() == 0:
        return torch.tensor(0.0, device=labels.device)

    mask = mask / mask_mean
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = (preds - labels) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = torch.where(torch.isinf(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class EarlyStopping(object):
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad_count = 0

    def step(self, value):
        stop = False
        if self.best is None or (value < self.best - self.min_delta):
            self.best = value
            self.bad_count = 0
        else:
            self.bad_count += 1
            if self.bad_count >= self.patience:
                stop = True
        return stop


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def to_device(x, device):
    if torch is None:
        return x
    if torch.is_tensor(x):
        return x.to(device)
    return x
