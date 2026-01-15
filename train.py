# -*- coding: utf-8 -*-
"""Training entrypoint (Python 3.6 compatible).

Run:
    python train.py --config configs/bj500.yaml

This script is intentionally self-contained and avoids modern Python syntax.
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import time

import numpy as np
import torch

from eadgn.config import load_yaml_config
from eadgn.data import build_dataloaders
from eadgn.model import EADGN
from eadgn.utils import masked_mae, masked_mape, masked_rmse, masked_mbe
from eadgn.utils import set_seed, ensure_dir, EarlyStopping

try:
    from thop import profile as thop_profile
    from thop import file_count
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

def _cfg_get(cfg, key, default=None):
    try:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)
    except Exception:
        return default


def _to_cpu_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    if torch.is_tensor(x):
        return x.detach().cpu().float()
    # fallback
    return torch.tensor(x).float()


def _ensure_BQN(x):
    if torch.is_tensor(x) is False:
        x = _to_cpu_tensor(x)

    # squeeze last dim if 1
    if x.dim() == 4 and x.size(-1) == 1:
        x = x.squeeze(-1)

    if x.dim() != 3:
        raise ValueError("Expect 3D tensor after squeeze, got shape={}".format(list(x.size())))

    B, D1, D2 = x.size(0), x.size(1), x.size(2)

    # Heuristic: Q 通常很小(<=24)，N 通常很大(>>Q)
    if D1 <= 24 and D2 > D1:
        # [B,Q,N] OK
        return x
    if D2 <= 24 and D1 > D2:
        # [B,N,Q] -> [B,Q,N]
        return x.permute(0, 2, 1).contiguous()

    # ambiguous: default treat as [B,Q,N]
    return x


def _compute_metrics(pred, true, null_val=0.0):
    pred = _to_cpu_tensor(pred)
    true = _to_cpu_tensor(true)

    mae = masked_mae(pred, true, null_val).item()
    mape = masked_mape(pred, true, null_val).item() * 100.0  # % 输出
    rmse = masked_rmse(pred, true, null_val).item()
    mbe = masked_mbe(pred, true, null_val).item()
    return mae, mape, rmse, mbe


def report_horizon_metrics(pred, true, cfg, split_name="test", save_dir=None, null_val=0.0):
    data_cfg = _cfg_get(cfg, "data", None)
    interval_min = float(_cfg_get(data_cfg, "time_interval_min", _cfg_get(cfg, "time_interval_min", 5.0)))

    pred = _ensure_BQN(_to_cpu_tensor(pred))
    true = _ensure_BQN(_to_cpu_tensor(true))

    Q = pred.size(1)

    # overall all steps
    all_mae, all_mape, all_rmse, all_mbe = _compute_metrics(pred, true, null_val)

    # 60min mapping
    step60 = int(round(60.0 / interval_min))
    if step60 < 1:
        step60 = 1

    # within 60 minutes = first step60 steps
    within60_mae = within60_mape = within60_rmse = within60_mbe = None
    if step60 <= Q:
        within60_mae, within60_mape, within60_rmse, within60_mbe = _compute_metrics(
            pred[:, :step60, :], true[:, :step60, :], null_val
        )

    # 60-min ahead = step60-th step
    h60_mae = h60_mape = h60_rmse = h60_mbe = None
    if step60 <= Q:
        h60_mae, h60_mape, h60_rmse, h60_mbe = _compute_metrics(
            pred[:, step60 - 1, :], true[:, step60 - 1, :], null_val
        )

    print("\n========== {} Horizon Report ==========".format(split_name.upper()))
    print("interval_min = {} min, pred_len(Q) = {}".format(interval_min, Q))
    print("[All steps 1..{}]   MAE {:.4f} | MAPE {:.2f}% | RMSE {:.4f} | MBE {:.4f}".format(
        Q, all_mae, all_mape, all_rmse, all_mbe
    ))

    if within60_mae is not None:
        print("[Within 60min <= step {}] MAE {:.4f} | MAPE {:.2f}% | RMSE {:.4f} | MBE {:.4f}".format(
            step60, within60_mae, within60_mape, within60_rmse, within60_mbe
        ))
    else:
        print("[Within 60min] skipped (need Q >= {}, but Q={})".format(step60, Q))

    if h60_mae is not None:
        print("[60min ahead = step {}]  MAE {:.4f} | MAPE {:.2f}% | RMSE {:.4f} | MBE {:.4f}".format(
            step60, h60_mae, h60_mape, h60_rmse, h60_mbe
        ))
    else:
        print("[60min ahead] skipped (need Q >= {}, but Q={})".format(step60, Q))

    print("\nPer-horizon metrics:")
    print(" step | minutes |   MAE   |  MAPE%  |  RMSE  |   MBE")
    rows = []
    for h in range(1, Q + 1):
        mins = h * interval_min
        mae, mape, rmse, mbe = _compute_metrics(pred[:, h - 1, :], true[:, h - 1, :], null_val)
        print(" {:>4d} | {:>7.1f} | {:>7.4f} | {:>7.2f} | {:>7.4f} | {:>7.4f}".format(
            h, mins, mae, mape, rmse, mbe
        ))
        rows.append([h, mins, mae, mape, rmse, mbe])

    # optional: save csv
    if save_dir is not None:
        try:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            out_csv = os.path.join(save_dir, "{}_horizon_metrics.csv".format(split_name))
            with open(out_csv, "w") as f:
                w = csv.writer(f)
                w.writerow(["step", "minutes", "MAE", "MAPE_percent", "RMSE", "MBE"])
                for r in rows:
                    w.writerow(r)
            print("\nSaved horizon metrics to: {}".format(out_csv))
        except Exception as e:
            print("\n[Warn] failed to save horizon csv: {}".format(e))

    print("======================================\n")


def _evaluate(model, loader, device, scaler, null_val=0.0):
    """Evaluate metrics on *original scale* (inverse-transformed).

    We use masked metrics to ignore missing values (commonly stored as 0).
    """
    model.eval()
    maes, mapes, rmses, mbes = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y, te, e = batch
            x = x.to(device)
            y = y.to(device)
            te = te.to(device)
            if e is not None:
                e = e.to(device)

            pred = model(x, te, event=e)

            # inverse transform to original scale for metrics
            pred = scaler.inverse_transform(pred)
            y_true = scaler.inverse_transform(y)

            # masked metrics
            mae = masked_mae(pred, y_true, null_val=null_val)
            mape = masked_mape(pred, y_true, null_val=null_val) * 100.0  # percent
            rmse = masked_rmse(pred, y_true, null_val=null_val)
            mbe = masked_mbe(pred, y_true, null_val=null_val)

            maes.append(float(mae.detach().cpu().item()))
            mapes.append(float(mape.detach().cpu().item()))
            rmses.append(float(rmse.detach().cpu().item()))
            mbes.append(float(mbe.detach().cpu().item()))

    if len(maes) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.mean(maes)), float(np.mean(mapes)), float(np.mean(rmses)), float(np.mean(mbes))


def _predict_outputs(model, loader, device, scaler):
    """Collect predictions and labels on original scale.

    Returns:
        preds: torch.Tensor on CPU, shape [B, Q, N, 1] or [B, Q, N]
        trues: torch.Tensor on CPU, same shape as preds
    """
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            x, y, te, e = batch
            x = x.to(device)
            y = y.to(device)
            te = te.to(device)
            if e is not None:
                e = e.to(device)

            pred = model(x, te, event=e)

            # original scale
            pred = scaler.inverse_transform(pred)
            y_true = scaler.inverse_transform(y)

            preds.append(pred.detach().cpu())
            trues.append(y_true.detach().cpu())

    if len(preds) == 0:
        return None, None
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0)


def run_profiling(model, loader, device, cfg):
    print("\n========== Efficiency Profiling & Analysis ==========")
    model.eval()

    # 1. 获取一个 Batch 的数据作为样本
    iterator = iter(loader)
    batch = next(iterator)
    x, y, te, e = batch
    x = x.to(device)
    te = te.to(device)
    if e is not None:
        e = e.to(device)

    # ---------------------------------------------------------
    # A. Model Parameters & FLOPs
    # ---------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops_str = "N/A (Install 'thop' library)"
    if HAS_THOP:
        try:
            input_args = (x, te)
            macs, _ = thop_profile(model, inputs=input_args, verbose=False)
            flops_str = "{:.2f} GFLOPs".format(macs / 1e9)
        except Exception as e:
            flops_str = "Failed to calc FLOPs: {}".format(e)

    print(f"Total Params: {total_params / 1e6:.2f} M")
    print(f"Trainable Params: {trainable_params / 1e6:.2f} M")
    print(f"Computational Complexity (FLOPs): {flops_str}")

    # ---------------------------------------------------------
    # B. Memory Usage (GPU only)
    # ---------------------------------------------------------
    memory_str = "N/A (CPU)"
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model(x, te, event=e)

        peak_mem = torch.cuda.max_memory_allocated()
        memory_str = "{:.2f} MB".format(peak_mem / 1024 / 1024)

    print(f"Peak GPU Memory (Inference): {memory_str}")

    # ---------------------------------------------------------
    # C. Inference Latency (Time)
    # ---------------------------------------------------------
    print("Running warmup (10 steps)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, te, event=e)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    runs = 100
    print(f"Profiling inference latency over {runs} runs...")

    t_start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x, te, event=e)
            if device.type == 'cuda':
                torch.cuda.synchronize()  
    t_end = time.time()

    avg_time = (t_end - t_start) / runs
    batch_size = x.size(0)

    print(f"Average Batch Inference Time: {avg_time * 1000:.2f} ms")
    print(f"Average Sample Inference Time: {(avg_time * 1000 / batch_size):.4f} ms")
    print("=====================================================\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='YAML config path')
    parser.add_argument('--device', type=str, default=None, help='override device, e.g. cuda or cpu')
    parser.add_argument('--eval_only', action='store_true', help='Skip training; only run val/test using a checkpoint.')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path (default: <run_dir>/best.pt)')
    parser.add_argument('--profile', action='store_true',
                        help='Run efficiency profiling (FLOPs, Memory, Latency) and exit.')
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if args.device is not None:
        cfg.device = args.device

    # device
    device = torch.device(cfg.device if torch.cuda.is_available() and str(cfg.device).startswith('cuda') else 'cpu')

    # seed
    set_seed(int(cfg.seed))

    # dirs
    run_dir = os.path.join(cfg.paths.run_dir, cfg.logging.experiment_name)
    ensure_dir(run_dir)

    # data
    loaders, scaler, adj, se = build_dataloaders(cfg)

    print('Device: {}'.format(device))
    print('Run dir: {}'.format(run_dir))
    print('Note: train_loss is masked MAE on *normalized* scale (after scaler.transform).')
    print('      val/test metrics are on *original* scale (after scaler.inverse_transform).')
    print('      MAPE is masked (y_true != 0) and reported in %; MBE>0 means over-prediction.')

    # model
    model = EADGN(adj=adj, se=se, cfg=cfg)
    model = model.to(device)

    if args.profile:
        run_profiling(model, loaders['test'], device, cfg)
        return  
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.lr),
                                 weight_decay=float(cfg.training.weight_decay))

    # training
    best_val = None
    best_path = os.path.join(run_dir, 'best.pt')
    stopper = EarlyStopping(patience=int(cfg.training.patience), min_delta=0.0)

    # eval-only mode
    if bool(args.eval_only):
        ckpt_path = args.ckpt if args.ckpt is not None else best_path
        if not os.path.exists(ckpt_path):
            raise ValueError('Checkpoint not found: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        val_mae, val_mape, val_rmse, val_mbe = _evaluate(model, loaders['val'], device, scaler, null_val=0.0)
        test_mae, test_mape, test_rmse, test_mbe = _evaluate(model, loaders['test'], device, scaler, null_val=0.0)
        print('Val   | MAE {:.4f}  MAPE {:.2f}%  RMSE {:.4f}  MBE {:.4f}'.format(val_mae, val_mape, val_rmse, val_mbe))
        print('Test  | MAE {:.4f}  MAPE {:.2f}%  RMSE {:.4f}  MBE {:.4f}'.format(test_mae, test_mape, test_rmse,
                                                                                 test_mbe))
        # Horizon report: per-step (1..Q) metrics and 60-min window metrics
        # (Only prints once; does NOT affect training.)
        try:
            test_pred, test_true = _predict_outputs(model, loaders['test'], device, scaler)
            if test_pred is not None:
                report_horizon_metrics(test_pred, test_true, cfg, split_name='test', save_dir=run_dir, null_val=0.0)
        except Exception as e:
            print('[Warn] horizon report failed: {}'.format(e))
        return

    global_step = 0
    for epoch in range(1, int(cfg.training.epochs) + 1):
        model.train()
        t0 = time.time()
        losses = []

        for batch in loaders['train']:
            global_step += 1
            x, y, te, e = batch
            x = x.to(device)
            y = y.to(device)
            te = te.to(device)
            if e is not None:
                e = e.to(device)

            pred = model(x, te, event=e)

            loss = masked_mae(pred, y, null_val=0.0)

            optimizer.zero_grad()
            loss.backward()

            # grad clip
            if float(cfg.training.clip_grad) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.clip_grad))

            optimizer.step()

            losses.append(float(loss.detach().cpu().item()))

        # validation (original scale)
        val_mae, val_mape, val_rmse, val_mbe = _evaluate(model, loaders['val'], device, scaler, null_val=0.0)
        train_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        # save best
        if best_val is None or val_mae < best_val:
            best_val = val_mae
            torch.save({
                'model': model.state_dict(),
                'cfg': dict(cfg),
                'val_mae': best_val,
                'epoch': epoch,
            }, best_path)

        # log
        dt = time.time() - t0
        msg = 'Epoch {:03d} | train_MAE(z) {:.6f} | val MAE {:.4f} MAPE {:.2f}% RMSE {:.4f} MBE {:.4f} | best {:.4f} | {:.1f}s'.format(
            epoch, train_loss, val_mae, val_mape, val_rmse, val_mbe, best_val, dt
        )
        print(msg)

        # early stop
        if stopper.step(val_mae):
            print('Early stopping triggered. Best val MAE: {:.4f}'.format(best_val))
            break

    # test with best model
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])

    test_mae, test_mape, test_rmse, test_mbe = _evaluate(model, loaders['test'], device, scaler, null_val=0.0)
    print('Test  | MAE {:.4f}  MAPE {:.2f}%  RMSE {:.4f}  MBE {:.4f}'.format(test_mae, test_mape, test_rmse, test_mbe))

    # Horizon report on test set (per-step and 60-min window)
    try:
        test_pred, test_true = _predict_outputs(model, loaders['test'], device, scaler)
        if test_pred is not None:
            report_horizon_metrics(test_pred, test_true, cfg, split_name='test', save_dir=run_dir, null_val=0.0)
    except Exception as e:
        print('[Warn] horizon report failed: {}'.format(e))


if __name__ == '__main__':
    main()
