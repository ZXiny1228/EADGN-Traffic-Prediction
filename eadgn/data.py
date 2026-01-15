# -*- coding: utf-8 -*-
"""Data loading and windowing.

This module is written to be *robust* to common traffic datasets:
- CSV can be (T, N) numeric with no timestamp column
- or first column might be a timestamp column

It produces:
- X: (num_samples, P, N, 1)
- Y: (num_samples, Q, N, 1)
- TE: (num_samples, P+Q, 2)  -> [time_of_day_id, day_of_week_id]
- optional E: (num_samples, P+Q, N, 1) event severity (if enabled)

Python 3.6 compatible (no dataclasses).
"""

from __future__ import absolute_import, division, print_function

import os
import pickle

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None
    Dataset = object
    DataLoader = None

from .utils import StandardScaler, MinMaxScaler


def load_adj_pkl(adj_path):
    """Load adjacency matrix from a pkl file.

    This is intentionally *robust* because different repos store adjacency pickles
    with different wrappers.

    Common formats:
    - (sensor_ids, sensor_id_to_ind, adj_mx)  (len==3)
    - (sensor_ids, sensor_id_to_ind, adj_mx, ...)  (len>=3)
    - dict with keys like 'adj_mx' / 'adj'
    - scipy.sparse matrix
    - plain numpy array

    Returns:
        adj: np.ndarray float32, shape (N, N)
    """

    def _to_dense(a):
        # scipy.sparse -> dense
        if hasattr(a, 'toarray'):
            try:
                a = a.toarray()
            except Exception:
                pass
        # numpy.matrix -> ndarray
        if hasattr(a, 'A'):
            try:
                a = a.A
            except Exception:
                pass
        return a

    def _is_square_numeric(a):
        try:
            a = _to_dense(a)
            arr = np.asarray(a)
            if arr.ndim != 2:
                return False
            if arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
                return False
            if arr.dtype.kind in 'biufc':
                return True
            arr.astype(np.float32)
            return True
        except Exception:
            return False

    def _as_float32(a):
        a = _to_dense(a)
        arr = np.asarray(a)
        if arr.dtype.kind not in 'biufc':
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float32, copy=False)
        return arr

    with open(adj_path, 'rb') as f:
        try:
            obj = pickle.load(f)
        except (TypeError, UnicodeDecodeError):
            f.seek(0)
            obj = pickle.load(f, encoding='latin1')

    # 1) dict-style
    if isinstance(obj, dict):
        for k in ['adj_mx', 'adj', 'A', 'matrix']:
            if k in obj:
                return _as_float32(obj[k])
        for v in obj.values():
            if _is_square_numeric(v):
                return _as_float32(v)

    # 2) tuple/list-style
    if isinstance(obj, (tuple, list)):
        if len(obj) >= 3 and _is_square_numeric(obj[2]):
            return _as_float32(obj[2])
        for v in reversed(list(obj)):
            if _is_square_numeric(v):
                return _as_float32(v)

    # 3) direct array/sparse
    if _is_square_numeric(obj):
        return _as_float32(obj)

    raise ValueError(
        'Cannot parse adjacency from pkl: {} (loaded type: {})'.format(adj_path, type(obj))
    )


def load_se_txt(se_path, num_nodes=None):
    """Load spatial embedding (SE) from a txt file.

    This loader is robust to common node2vec/word2vec text formats.

    Supported formats:
    1) word2vec header + vectors:
        <num_nodes> <dim>
        <node_id> <v1> <v2> ... <vD>
        ...

    2) vectors without header:
        <node_id> <v1> ... <vD>
        ...

    3) vectors without node_id:
        <v1> ... <vD>

    It returns:
        se: np.ndarray float32 of shape (num_nodes, dim)

    Notes:
    - If node ids are 1..N (instead of 0..N-1), we auto-shift by -1.
    - If some nodes are missing, they remain zeros.
    """
    lines = []
    with open(se_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    if len(lines) == 0:
        raise ValueError('Empty SE file: {}'.format(se_path))

    header_n = None
    header_d = None
    start_idx = 0
    first = lines[0].replace(',', ' ').split()
    if len(first) == 2:
        try:
            header_n = int(first[0])
            header_d = int(first[1])
            if header_n > 0 and header_d > 0:
                start_idx = 1
        except Exception:
            header_n = None
            header_d = None
            start_idx = 0

    dim_expected = header_d
    vec_by_id = {}
    seq_vecs = []

    def _is_float_token(s):
        try:
            float(s)
            return True
        except Exception:
            return False

    for line in lines[start_idx:]:
        # normalize separators (commas / brackets)
        ln = line.replace(',', ' ').replace('[', ' ').replace(']', ' ')
        parts = ln.split()
        if not parts:
            continue

        node_token = None
        vec_tokens = None

        if dim_expected is not None:
            if len(parts) == dim_expected + 1:
                node_token = parts[0]
                vec_tokens = parts[1:]
            elif len(parts) == dim_expected:
                node_token = None
                vec_tokens = parts

        if vec_tokens is None:
            # Heuristic: treat first as node_id if it's int-like and the rest are floats
            if len(parts) >= 2:
                is_int_id = False
                try:
                    int(parts[0])
                    is_int_id = True
                except Exception:
                    is_int_id = False
                if is_int_id and all([_is_float_token(x) for x in parts[1:]]):
                    node_token = parts[0]
                    vec_tokens = parts[1:]

        if vec_tokens is None:
            # No node id: all floats
            if all([_is_float_token(x) for x in parts]):
                node_token = None
                vec_tokens = parts

        if vec_tokens is None:
            # Fallback: regex-extract numbers
            import re as _re
            nums = _re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', ln)
            if len(nums) == 0:
                continue
            if dim_expected is not None and len(nums) == dim_expected + 1:
                node_token = nums[0]
                vec_tokens = nums[1:]
            else:
                node_token = None
                vec_tokens = nums

        # Convert tokens to float vector
        vec = []
        for t in vec_tokens:
            try:
                vec.append(float(t))
            except Exception:
                vec.append(0.0)

        if dim_expected is None:
            dim_expected = len(vec)

        # pad/truncate to dim_expected
        if len(vec) < dim_expected:
            vec = vec + [0.0] * (dim_expected - len(vec))
        elif len(vec) > dim_expected:
            vec = vec[:dim_expected]

        if node_token is None:
            seq_vecs.append(vec)
        else:
            try:
                nid = int(node_token)
                vec_by_id[nid] = vec
            except Exception:
                # can't map, fall back to sequential order
                seq_vecs.append(vec)

    if dim_expected is None:
        raise ValueError('Failed to infer SE dimension from file: {}'.format(se_path))

    # Decide output num_nodes
    if num_nodes is not None:
        N = int(num_nodes)
    elif header_n is not None:
        N = int(header_n)
    elif len(vec_by_id) > 0:
        N = int(max(vec_by_id.keys())) + 1
    else:
        N = len(seq_vecs)

    se = np.zeros((N, dim_expected), dtype=np.float32)

    if len(vec_by_id) > 0:
        # Detect 1-based ids and shift to 0-based
        keys = sorted(vec_by_id.keys())
        shift = 0
        if (len(keys) == N) and (keys[0] == 1) and (keys[-1] == N) and (0 not in vec_by_id):
            shift = -1

        for nid, vec in vec_by_id.items():
            idx = nid + shift
            if 0 <= idx < N:
                se[idx, :] = np.asarray(vec, dtype=np.float32)
    else:
        m = min(len(seq_vecs), N)
        for i in range(m):
            se[i, :] = np.asarray(seq_vecs[i], dtype=np.float32)

    return se


def _maybe_parse_time_column(df):
    """Try to detect and parse a timestamp column."""
    if df.shape[1] <= 1:
        return None, df

    # Common patterns: first column named 'date', 'time', 'timestamp'
    first_col = df.columns[0]
    if str(first_col).lower() in ['date', 'time', 'timestamp', 'datetime']:
        time = df.iloc[:, 0]
        values = df.iloc[:, 1:]
        return time, values

    # If first column is non-numeric and looks like date strings, parse
    if not np.issubdtype(df.dtypes[0], np.number):
        time = df.iloc[:, 0]
        values = df.iloc[:, 1:]
        return time, values

    return None, df


def load_traffic_csv(csv_path):
    """Load traffic series from CSV.

    Returns:
        data: np.ndarray, shape (T, N)
        timestamps: pd.Series or None
    """
    if pd is None:
        raise ImportError('pandas is required for CSV loading. Please `pip install pandas`.')

    df = pd.read_csv(csv_path)
    time_col, values_df = _maybe_parse_time_column(df)
    values = values_df.values.astype(np.float32)

    if time_col is not None:
        # keep raw; parse later if possible
        timestamps = time_col
    else:
        timestamps = None

    # If values are shaped (N, T), transpose to (T, N)
    if values.shape[0] < values.shape[1]:
        # heuristic: traffic datasets usually T >> N
        pass
    else:
        # could still be (T,N); don't transpose blindly
        pass

    return values, timestamps


def build_time_features(num_steps, timestamps=None, steps_per_day=288):
    """Build TE features: time-of-day id and day-of-week id.

    If timestamps are available and parseable, use them; otherwise fall back
    to index-based features.

    Returns:
        te: np.ndarray of shape (T, 2) int64
    """
    if timestamps is not None and pd is not None:
        try:
            ts = pd.to_datetime(timestamps)
            # time of day bucket
            # if timestamps are evenly spaced, infer step within day
            minutes = ts.dt.hour * 60 + ts.dt.minute
            # map minutes to [0, steps_per_day)
            # assume 24h -> steps_per_day
            tod = ((minutes / (24.0 * 60.0)) * steps_per_day).astype(np.int64) % steps_per_day
            dow = ts.dt.dayofweek.astype(np.int64)
            te = np.stack([tod.values, dow.values], axis=1).astype(np.int64)
            if te.shape[0] == num_steps:
                return te
        except Exception:
            pass

    # fallback: index-based
    idx = np.arange(num_steps)
    tod = (idx % steps_per_day).astype(np.int64)
    dow = ((idx // steps_per_day) % 7).astype(np.int64)
    te = np.stack([tod, dow], axis=1).astype(np.int64)
    return te


def windowed_dataset(data, te, P, Q, event=None):
    """Generate sliding windows.

    Args:
        data: (T, N)
        te: (T, 2)
        P: history length
        Q: prediction length
        event: optional (T, N) or (T, N, 1)

    Returns:
        X: (S, P, N, 1)
        Y: (S, Q, N, 1)
        TE: (S, P+Q, 2)
        E: (S, P+Q, N, 1) or None
    """
    T, N = data.shape
    S = T - (P + Q) + 1
    X = np.zeros((S, P, N, 1), dtype=np.float32)
    Y = np.zeros((S, Q, N, 1), dtype=np.float32)
    TE = np.zeros((S, P + Q, 2), dtype=np.int64)
    E = None

    if event is not None:
        if event.ndim == 2:
            event = event[:, :, None]
        E = np.zeros((S, P + Q, N, 1), dtype=np.float32)

    for i in range(S):
        X[i, :, :, 0] = data[i:i + P]
        Y[i, :, :, 0] = data[i + P:i + P + Q]
        TE[i] = te[i:i + P + Q]
        if E is not None:
            E[i] = event[i:i + P + Q]

    return X, Y, TE, E


def split_dataset(X, Y, TE, E, train_ratio=0.7, val_ratio=0.1):
    S = X.shape[0]
    n_train = int(S * train_ratio)
    n_val = int(S * val_ratio)
    n_test = S - n_train - n_val

    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, S)

    def _slice(arr, slc):
        if arr is None:
            return None
        return arr[slc]

    out = {
        'train': (X[train_slice], Y[train_slice], TE[train_slice], _slice(E, train_slice)),
        'val': (X[val_slice], Y[val_slice], TE[val_slice], _slice(E, val_slice)),
        'test': (X[test_slice], Y[test_slice], TE[test_slice], _slice(E, test_slice)),
    }
    return out


class NumpyDataset(Dataset):
    def __init__(self, X, Y, TE, E=None):
        self.X = X
        self.Y = Y
        self.TE = TE
        self.E = E

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        te = self.TE[idx]
        if self.E is None:
            return x, y, te, None
        return x, y, te, self.E[idx]


def _collate_with_optional_event(batch):
    """Custom collate_fn to support optional event tensors.

    PyTorch's default collate behavior for None can differ across versions.
    To be safe (especially for older torch + Python 3.6 environments), we
    explicitly handle the case where the 4th field is None.
    """
    xs, ys, tes, es = [], [], [], []
    for item in batch:
        xs.append(item[0])
        ys.append(item[1])
        tes.append(item[2])
        es.append(item[3])

    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    te = torch.stack(tes, dim=0)

    if es[0] is None:
        e = None
    else:
        e = torch.stack(es, dim=0)
    return x, y, te, e


def build_dataloaders(cfg):
    """Build train/val/test dataloaders and scalers.

    Returns:
        loaders: dict with keys train/val/test
        scaler: fitted scaler
        adj: (N,N) numpy
        se: (N,D) numpy
    """
    data_dir = cfg.paths.data_dir
    traffic_path = os.path.join(data_dir, cfg.data.traffic_file)
    adj_path = os.path.join(data_dir, cfg.data.adj_file)
    se_path = os.path.join(data_dir, cfg.data.se_file)

    data, timestamps = load_traffic_csv(traffic_path)
    # ensure (T,N)
    if data.ndim != 2:
        raise ValueError('Traffic data should be 2D (T,N). Got shape {}'.format(data.shape))

    T, N = data.shape
    if cfg.model.num_nodes is not None:
        # best-effort: align N
        if N > cfg.model.num_nodes:
            data = data[:, :cfg.model.num_nodes]
        elif N < cfg.model.num_nodes:
            pad = np.zeros((T, cfg.model.num_nodes - N), dtype=np.float32)
            data = np.concatenate([data, pad], axis=1)

    T, N = data.shape

    te = build_time_features(T, timestamps=timestamps, steps_per_day=int(cfg.data.steps_per_day))

    # optional event
    event = None
    if bool(cfg.data.event_enabled):
        event_file = str(cfg.data.event_file)
        if event_file:
            event_path = os.path.join(data_dir, event_file)
            # expected a (T,N) csv with same layout as traffic
            ev, _ = load_traffic_csv(event_path)
            if ev.shape[0] != T:
                raise ValueError('Event T mismatch: traffic T={}, event T={}'.format(T, ev.shape[0]))
            if ev.shape[1] != N:
                # best-effort align
                if ev.shape[1] > N:
                    ev = ev[:, :N]
                else:
                    pad = np.zeros((T, N - ev.shape[1]), dtype=np.float32)
                    ev = np.concatenate([ev, pad], axis=1)
            event = ev.astype(np.float32)

    P = int(cfg.task.P)
    Q = int(cfg.task.Q)

    X, Y, TE, E = windowed_dataset(data, te, P, Q, event=event)

    splits = split_dataset(X, Y, TE, E, train_ratio=float(cfg.data.train_ratio), val_ratio=float(cfg.data.val_ratio))

    # fit scaler on train X (all timesteps & nodes)
    train_X = splits['train'][0]  # (S,P,N,1)
    train_flat = train_X.reshape((-1, 1))

    scaler_name = str(cfg.data.scaler).lower()
    if scaler_name == 'minmax':
        data_min = float(np.min(train_flat))
        data_max = float(np.max(train_flat))
        scaler = MinMaxScaler(data_min, data_max)
    else:
        mean = float(np.mean(train_flat))
        std = float(np.std(train_flat))
        scaler = StandardScaler(mean, std)

    # transform X and Y
    def _transform(arr):
        return scaler.transform(arr)

    def _apply_transform(x, y):
        return _transform(x), _transform(y)

    train_X, train_Y, train_TE, train_E = splits['train']
    val_X, val_Y, val_TE, val_E = splits['val']
    test_X, test_Y, test_TE, test_E = splits['test']

    train_X, train_Y = _apply_transform(train_X, train_Y)
    val_X, val_Y = _apply_transform(val_X, val_Y)
    test_X, test_Y = _apply_transform(test_X, test_Y)

    # load adj and se
    adj = load_adj_pkl(adj_path)
    se = load_se_txt(se_path, num_nodes=N)

    # torch dataloaders
    if torch is None:
        raise ImportError('PyTorch is required for training. Please install torch.')

    def _to_tensor(arr, dtype=torch.float32):
        if arr is None:
            return None
        if dtype is None:
            return torch.from_numpy(arr)
        return torch.from_numpy(arr).type(dtype)

    train_ds = NumpyDataset(_to_tensor(train_X), _to_tensor(train_Y), _to_tensor(train_TE, dtype=None).long(),
                            _to_tensor(train_E) if train_E is not None else None)
    val_ds = NumpyDataset(_to_tensor(val_X), _to_tensor(val_Y), _to_tensor(val_TE, dtype=None).long(),
                          _to_tensor(val_E) if val_E is not None else None)
    test_ds = NumpyDataset(_to_tensor(test_X), _to_tensor(test_Y), _to_tensor(test_TE, dtype=None).long(),
                           _to_tensor(test_E) if test_E is not None else None)

    loaders = {
        'train': DataLoader(train_ds, batch_size=int(cfg.training.batch_size), shuffle=True,
                            num_workers=int(cfg.training.num_workers), collate_fn=_collate_with_optional_event),
        'val': DataLoader(val_ds, batch_size=int(cfg.training.batch_size), shuffle=False,
                          num_workers=int(cfg.training.num_workers), collate_fn=_collate_with_optional_event),
        'test': DataLoader(test_ds, batch_size=int(cfg.training.batch_size), shuffle=False,
                           num_workers=int(cfg.training.num_workers), collate_fn=_collate_with_optional_event),
    }

    return loaders, scaler, adj, se
