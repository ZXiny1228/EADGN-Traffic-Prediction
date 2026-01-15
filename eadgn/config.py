from __future__ import absolute_import, division, print_function

import copy
import os

try:
    import yaml
except Exception:
    yaml = None


class AttrDict(dict):
    def __getattr__(self, item):
        if item in self:
            v = self[item]
            if isinstance(v, dict) and not isinstance(v, AttrDict):
                v = AttrDict(v)
                self[item] = v
            return v
        raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value


def deep_update(base, override):
    if override is None:
        return base
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def get_default_config():
    return {
        'seed': 1234,
        'device': 'cuda',
        'paths': {
            'data_dir': 'data',
            'run_dir': 'runs',
        },
        'data': {
            'traffic_file': 'BJ500.csv',
            'adj_file': 'adj_mx_BJ.pkl',
            'se_file': 'SE(BJ500).txt',
            'event_enabled': False,
            'event_file': '',
            'steps_per_day': 288,  
            'train_ratio': 0.7,
            'val_ratio': 0.1,
            'scaler': 'standard',  
        },
        'model': {
            'num_nodes': 500,
            'in_dim': 1,
            'out_dim': 1,
            'hidden_dim': 64,
            'se_dim': 64,
            'te_dim': 16,
            'event_dim': 8,
            'num_heads': 4,
            'dropout': 0.1,
            'num_blocks': 2,
            'diffusion_steps': 2,
            'topk': 20,
            'use_base_adaptive_adj': True,
            'adaptive_adj_dim': 10,
            'alpha_init': 0.5,
        },
        'task': {
            'P': 12,
            'Q': 12,
        },
        'training': {
            'batch_size': 16,
            'epochs': 50,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'clip_grad': 5.0,
            'patience': 10,
            'log_every': 50,
            'num_workers': 0,
        },
        'logging': {
            'experiment_name': 'EADGN_BJ500',
        },
    }


def load_yaml_config(path, default_cfg=None):
    """Load YAML config and merge into defaults."""
    if default_cfg is None:
        default_cfg = get_default_config()
    cfg = copy.deepcopy(default_cfg)

    if path is None:
        return AttrDict(cfg)

    if yaml is None:
        raise ImportError('PyYAML is required. Please `pip install pyyaml`.')

    with open(path, 'r') as f:
        user_cfg = yaml.safe_load(f)
    if user_cfg is None:
        user_cfg = {}

    deep_update(cfg, user_cfg)

    cfg.setdefault('paths', {})
    cfg['paths']['config_path'] = path
    cfg['paths']['config_dir'] = os.path.dirname(os.path.abspath(path))

    data_dir = cfg['paths'].get('data_dir', 'data')
    if not os.path.isabs(data_dir):
        cfg['paths']['data_dir'] = os.path.abspath(os.path.join(cfg['paths']['config_dir'], data_dir))

    run_dir = cfg['paths'].get('run_dir', 'runs')
    if not os.path.isabs(run_dir):
        cfg['paths']['run_dir'] = os.path.abspath(os.path.join(cfg['paths']['config_dir'], run_dir))

    return AttrDict(cfg)
