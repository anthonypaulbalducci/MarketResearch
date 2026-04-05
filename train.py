"""
Training pipeline for iTransformer V2.

Usage:
    python train.py
    python train.py --epochs 100 --lr 5e-5 --batch-size 128
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import Config
from dataset import load_and_split_data
from model import build_model


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def get_loss_fn(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "huber":
        return nn.HuberLoss(delta=1.0)
    else:
        raise ValueError(f"Unknown loss: {name}")


def get_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    if cfg.train.optimizer == "adamw":
        return AdamW(model.parameters(), lr=cfg.train.learning_rate,
                     weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == "adam":
        return Adam(model.parameters(), lr=cfg.train.learning_rate,
                    weight_decay=cfg.train.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.train.optimizer}")


def get_scheduler(optimizer, cfg, steps_per_epoch: int):
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = cfg.train.warmup_steps

    if cfg.train.scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
        return LambdaLR(optimizer, lr_lambda)
    elif cfg.train.scheduler == "none":
        return LambdaLR(optimizer, lambda step: 1.0)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.train.scheduler}")


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn,
                    device, max_grad_norm, log_interval) -> dict:
    model.train()
    total_loss = 0.0
    n_batches = 0
    epoch_start = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)  # (B, lookback, N, F)
        y = y.to(device)  # (B, forecast_len)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()

        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            lr = scheduler.get_last_lr()[0]
            print(f"    Step {batch_idx+1}/{len(loader)}: "
                  f"loss={avg_loss:.6f}, lr={lr:.2e}")

    return {"loss": total_loss / max(n_batches, 1),
            "time": time.time() - epoch_start,
            "n_batches": n_batches}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)
        mae = F.l1_loss(pred, y)

        total_loss += loss.item()
        total_mae += mae.item()
        n_batches += 1

        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Flatten for correlation / directional accuracy
    pred_flat = preds.mean(axis=-1) if preds.ndim > 1 else preds
    target_flat = targets.mean(axis=-1) if targets.ndim > 1 else targets

    corr = np.corrcoef(pred_flat, target_flat)[0, 1] if len(pred_flat) > 1 else 0.0
    dir_acc = np.mean(np.sign(pred_flat) == np.sign(target_flat))

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": total_mae / max(n_batches, 1),
        "correlation": float(corr) if not np.isnan(corr) else 0.0,
        "directional_accuracy": float(dir_acc),
        "predictions": preds,
        "targets": targets,
        "pred_mean": float(pred_flat.mean()),
        "pred_std": float(pred_flat.std()),
    }


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
    }, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0)


def train(cfg: Config):
    """Full training pipeline."""
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    device = get_device(cfg.train.device)
    print(f"Device: {device}")

    data = load_and_split_data(cfg)

    model = build_model(cfg, spy_index=data["spy_index"])
    model = model.to(device)

    loss_fn = get_loss_fn(cfg.train.loss_fn)
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg, steps_per_epoch=len(data["train_loader"]))
    early_stopping = EarlyStopping(patience=cfg.train.patience)

    history = {"train_loss": [], "val_loss": [], "val_mae": [],
               "val_corr": [], "val_dir_acc": [], "lr": [],
               "pred_mean": [], "pred_std": []}

    best_val_loss = float("inf")
    best_epoch = 0
    print(f"\n{'='*70}")
    print(f"Training for {cfg.train.epochs} epochs  |  Loss: {cfg.train.loss_fn}  |  "
          f"LR: {cfg.train.learning_rate}  |  Batch: {cfg.train.batch_size}")
    print(f"Lookback: {cfg.model.lookback_len} days  |  "
          f"Forecast: {cfg.model.forecast_len} day(s)  |  "
          f"Variates: {cfg.model.num_variates}  |  "
          f"Features: {cfg.model.n_features}")
    print(f"{'='*70}\n")

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = train_one_epoch(
            model, data["train_loader"], optimizer, scheduler,
            loss_fn, device, cfg.train.max_grad_norm, cfg.train.log_interval,
        )
        val_metrics = evaluate(model, data["val_loader"], loss_fn, device)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_corr"].append(val_metrics["correlation"])
        history["val_dir_acc"].append(val_metrics["directional_accuracy"])
        history["lr"].append(scheduler.get_last_lr()[0])
        history["pred_mean"].append(val_metrics["pred_mean"])
        history["pred_std"].append(val_metrics["pred_std"])

        print(
            f"Epoch {epoch:3d}/{cfg.train.epochs} | "
            f"Train: {train_metrics['loss']:.6f} | "
            f"Val: {val_metrics['loss']:.6f} | "
            f"MAE: {val_metrics['mae']:.6f} | "
            f"Corr: {val_metrics['correlation']:.4f} | "
            f"DirAcc: {val_metrics['directional_accuracy']:.4f} | "
            f"PredMean: {val_metrics['pred_mean']:.6f} | "
            f"PredStd: {val_metrics['pred_std']:.6f} | "
            f"{train_metrics['time']:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics["loss"],
                cfg.train.checkpoint_dir / "best_model.pt",
            )
            print(f"  ✓ New best (val_loss={best_val_loss:.6f})")

        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics["loss"],
                cfg.train.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
            )

        if early_stopping(val_metrics["loss"]):
            print(f"\nEarly stopping at epoch {epoch} (patience={cfg.train.patience})")
            break

    print(f"\n{'='*70}")
    print(f"Best val_loss={best_val_loss:.6f} at epoch {best_epoch}")
    print(f"{'='*70}\n")

    # Load best and evaluate on test set
    load_checkpoint(cfg.train.checkpoint_dir / "best_model.pt", model)
    model = model.to(device)

    print("Evaluating on test set...")
    test_metrics = evaluate(model, data["test_loader"], loss_fn, device)
    print(f"Test Loss:    {test_metrics['loss']:.6f}")
    print(f"Test MAE:     {test_metrics['mae']:.6f}")
    print(f"Test Corr:    {test_metrics['correlation']:.4f}")
    print(f"Test DirAcc:  {test_metrics['directional_accuracy']:.4f}")
    print(f"Test PredMean:{test_metrics['pred_mean']:.6f}")
    print(f"Test PredStd: {test_metrics['pred_std']:.6f}")

    # Save results
    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_mae": test_metrics["mae"],
        "test_correlation": test_metrics["correlation"],
        "test_directional_accuracy": test_metrics["directional_accuracy"],
        "test_pred_mean": test_metrics["pred_mean"],
        "test_pred_std": test_metrics["pred_std"],
        "history": history,
        "config": {
            "lookback_len": cfg.model.lookback_len,
            "forecast_len": cfg.model.forecast_len,
            "num_variates": cfg.model.num_variates,
            "n_features": cfg.model.n_features,
            "d_model": cfg.model.d_model,
            "n_heads": cfg.model.n_heads,
            "n_layers": cfg.model.n_layers,
            "learning_rate": cfg.train.learning_rate,
            "batch_size": cfg.train.batch_size,
            "loss_fn": cfg.train.loss_fn,
        },
        "tickers": data["tickers"],
        "feature_names": data["feature_names"],
    }

    results_path = cfg.train.results_dir / "training_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))

    np.savez(
        cfg.train.results_dir / "test_predictions.npz",
        predictions=test_metrics["predictions"],
        targets=test_metrics["targets"],
    )

    print(f"\nResults saved to {cfg.train.results_dir}/")
    return model, history, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train iTransformer V2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--forecast", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.lr:
        cfg.train.learning_rate = args.lr
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lookback:
        cfg.model.lookback_len = args.lookback
    if args.forecast:
        cfg.model.forecast_len = args.forecast
    if args.d_model:
        cfg.model.d_model = args.d_model
    if args.n_layers:
        cfg.model.n_layers = args.n_layers
    if args.device:
        cfg.train.device = args.device

    train(cfg)


if __name__ == "__main__":
    main()
