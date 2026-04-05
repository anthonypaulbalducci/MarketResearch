"""
Online (incremental) training module for the iTransformer.

Downloads new data since the last update, preprocesses it using the original
training statistics, and fine-tunes the model with a conservative learning rate.

The original model checkpoint and training data are never modified.

Usage:
    # Update with latest data and fine-tune
    python online_train.py

    # Update with specific date range
    python online_train.py --since 2026-01-15

    # Download new data only (no training)
    python online_train.py --download-only

    # Fine-tune only (new data already downloaded)
    python online_train.py --skip-download

    # Rollback to original model
    python online_train.py --rollback
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from sp500_tickers import get_ticker_set, get_all_tickers, TARGET_TICKER
from download_data import create_s3_client, download_single_file, build_s3_key
from preprocess import (
    read_daily_flat_file, filter_market_hours, aggregate_to_daily,
    compute_features, _compute_rsi,
)
from dataset import SPYForecastDataset
from model import build_model
from train import get_device, get_loss_fn, evaluate, save_checkpoint, load_checkpoint


class OnlineTrainer:
    """
    Manages incremental model updates with new market data.

    Directory structure:
        checkpoints/
            best_model.pt           ← original, never modified
            online_model.pt         ← latest fine-tuned version
            online_history.json     ← log of all online updates
        data/
            processed/
                features.npy        ← original training features
                metadata.json       ← original metadata
                online_features.npy ← accumulated new data features
                online_meta.json    ← online data metadata
                scaler_params.npz   ← normalization stats from training
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device(cfg.train.device)
        self.ticker_set = get_ticker_set()
        self.all_tickers = get_all_tickers()

        # Paths
        self.original_ckpt = cfg.train.checkpoint_dir / "best_model.pt"
        self.online_ckpt = cfg.train.checkpoint_dir / "online_model.pt"
        self.history_path = cfg.train.checkpoint_dir / "online_history.json"
        self.online_features_path = cfg.data.processed_data_dir / "online_features.npy"
        self.online_meta_path = cfg.data.processed_data_dir / "online_meta.json"
        self.scaler_path = cfg.data.processed_data_dir / "scaler_params.npz"

        # Load original metadata
        meta_path = cfg.data.processed_data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError("Original metadata not found. Run the full pipeline first.")
        self.orig_metadata = json.loads(meta_path.read_text())

        self.tickers = self.orig_metadata["tickers"]
        self.feature_names = self.orig_metadata["feature_names"]
        self.spy_index = self.orig_metadata["spy_index"]

    def get_last_date(self) -> str:
        """Get the last date in the dataset (original or online)."""
        if self.online_meta_path.exists():
            online_meta = json.loads(self.online_meta_path.read_text())
            return online_meta["dates"][-1]
        return self.orig_metadata["dates"][-1]

    def get_update_history(self) -> list:
        """Load the history of online updates."""
        if self.history_path.exists():
            return json.loads(self.history_path.read_text())
        return []

    def _save_history(self, history: list):
        self.history_path.write_text(json.dumps(history, indent=2, default=str))

    # ------------------------------------------------------------------
    # Step 1: Download new data
    # ------------------------------------------------------------------
    def download_new_data(self, since: str | None = None) -> list[Path]:
        """
        Download flat files for dates after the last known date.
        Returns list of newly downloaded file paths.
        """
        access_key = self.cfg.data.s3_access_key or os.environ.get("MASSIVE_S3_ACCESS_KEY", "")
        secret_key = self.cfg.data.s3_secret_key or os.environ.get("MASSIVE_S3_SECRET_KEY", "")

        if not access_key or not secret_key:
            print("ERROR: S3 credentials required.")
            print("Set MASSIVE_S3_ACCESS_KEY and MASSIVE_S3_SECRET_KEY env vars.")
            sys.exit(1)

        last_date = since or self.get_last_date()
        today = datetime.now().strftime("%Y-%m-%d")

        # Start from the day after the last known date
        start = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
        end = datetime.strptime(today, "%Y-%m-%d")

        if start > end:
            print(f"Already up to date (last date: {last_date})")
            return []

        # Generate weekdays in range
        dates = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        if not dates:
            print("No new trading days to download.")
            return []

        print(f"Downloading data from {dates[0]} to {dates[-1]} ({len(dates)} days)")

        s3_client = create_s3_client(
            access_key, secret_key, self.cfg.data.s3_endpoint, self.cfg.data.s3_region
        )

        new_files = []
        for i, date_str in enumerate(dates, 1):
            s3_key = build_s3_key(date_str, self.cfg.data.minute_aggs_prefix)
            local_path = self.cfg.data.raw_data_dir / f"{date_str}.csv.gz"
            result = download_single_file(
                s3_client, self.cfg.data.s3_bucket, s3_key, local_path,
                skip_existing=self.cfg.data.skip_existing,
            )
            if result["status"] == "downloaded":
                new_files.append(local_path)
                size_mb = result["size"] / (1024 * 1024)
                print(f"  [{i}/{len(dates)}] {date_str}: Downloaded ({size_mb:.1f} MB)")
            elif result["status"] == "skipped":
                # File exists, still include it for processing
                new_files.append(local_path)
            elif result["status"] == "not_found":
                pass  # Holiday
            else:
                print(f"  [{i}/{len(dates)}] {date_str}: {result['status']}")

        print(f"Downloaded/found {len(new_files)} new files")
        return new_files

    # ------------------------------------------------------------------
    # Step 2: Preprocess new data
    # ------------------------------------------------------------------
    def preprocess_new_data(self, file_paths: list[Path]) -> np.ndarray | None:
        """
        Process new flat files into feature tensor using original training stats.
        Returns features array of shape (new_days, num_variates, n_features) or None.
        """
        if not file_paths:
            print("No new files to preprocess.")
            return None

        print(f"\nPreprocessing {len(file_paths)} new files...")

        # Aggregate minute bars to daily OHLCV
        daily_rows = []
        for filepath in sorted(file_paths):
            try:
                df = read_daily_flat_file(filepath, self.ticker_set)
                if df.empty:
                    continue
                df = filter_market_hours(df, self.cfg.data.market_open, self.cfg.data.market_close)
                if df.empty:
                    continue
                daily = aggregate_to_daily(df)
                if daily is not None and not daily.empty:
                    daily_rows.append(daily)
            except Exception as e:
                print(f"  Warning: Failed to process {filepath.name}: {e}")

        if not daily_rows:
            print("No valid data in new files.")
            return None

        new_daily = pd.concat(daily_rows, axis=0, ignore_index=True)
        new_daily["date"] = pd.to_datetime(new_daily["date"])
        new_daily = new_daily.sort_values(["date", "ticker"]).reset_index(drop=True)

        print(f"New daily records: {len(new_daily):,}")
        print(f"New date range: {new_daily['date'].min()} to {new_daily['date'].max()}")

        # We need historical context for rolling features (RSI, SMA, etc.)
        # Load the existing OHLCV data to prepend as context
        context_features = self._build_features_with_context(new_daily)
        return context_features

    def _build_features_with_context(self, new_daily: pd.DataFrame) -> np.ndarray | None:
        """
        Build features for new data, using existing data as context for rolling calcs.
        """
        # Load existing full features to get the trailing OHLCV context
        orig_features_path = self.cfg.data.processed_data_dir / "features.npy"
        if not orig_features_path.exists():
            print("ERROR: Original features not found.")
            return None

        # We need raw OHLCV context, not just features. Reload from cache.
        # Get the last N days of cached daily data for rolling window context.
        context_days = 30  # Enough for RSI(14), SMA(20), etc.

        # Find cached daily parquet files
        cache_dir = self.cfg.data.cache_dir
        cached_files = sorted(cache_dir.glob("*_daily.parquet"))
        context_files = cached_files[-context_days:] if len(cached_files) >= context_days else cached_files

        context_rows = []
        for cf in context_files:
            try:
                context_rows.append(pd.read_parquet(cf))
            except Exception:
                continue

        if not context_rows:
            print("Warning: No cached context data. Rolling features may have NaN.")
            combined = new_daily
        else:
            context_df = pd.concat(context_rows, axis=0, ignore_index=True)
            context_df["date"] = pd.to_datetime(context_df["date"])
            combined = pd.concat([context_df, new_daily], axis=0, ignore_index=True)
            combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
            combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

        # Pivot to wide OHLCV
        valid_tickers = self.tickers  # Use same tickers as original
        combined = combined[combined["ticker"].isin(valid_tickers)]

        ohlcv_wide = {}
        for field in ["open", "high", "low", "close", "volume"]:
            pivoted = combined.pivot_table(
                index="date", columns="ticker", values=field, aggfunc="last"
            )
            # Ensure same column order
            for t in valid_tickers:
                if t not in pivoted.columns:
                    pivoted[t] = np.nan
            pivoted = pivoted[valid_tickers]
            pivoted = pivoted.sort_index().ffill().bfill()
            ohlcv_wide[field] = pivoted

        # Compute features on full context + new data
        all_features = compute_features(ohlcv_wide, valid_tickers)
        all_dates = ohlcv_wide["close"].index

        # Find where new data starts
        new_start_date = pd.Timestamp(new_daily["date"].min())
        new_mask = all_dates >= new_start_date
        new_start_idx = np.argmax(new_mask)

        # Extract only the new portion
        new_features = all_features[new_start_idx:]
        new_dates = all_dates[new_start_idx:]

        # Replace NaN/inf
        np.nan_to_num(new_features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply normalization using original training statistics
        new_features = self._normalize_with_original_stats(new_features)

        # Cache the new daily data
        for _, row_group in new_daily.groupby("date"):
            date_str = str(row_group["date"].iloc[0].date())
            cache_path = cache_dir / f"{date_str}_daily.parquet"
            if not cache_path.exists():
                row_group.to_parquet(cache_path)

        print(f"New features shape: {new_features.shape}")
        print(f"New dates: {new_dates[0]} to {new_dates[-1]}")

        # Save / append online features
        self._save_online_features(new_features, new_dates)

        return new_features

    def _normalize_with_original_stats(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using the original training set statistics."""
        if self.cfg.features.normalization != "zscore":
            return features

        if self.scaler_path.exists():
            scaler = np.load(self.scaler_path)
            mean = scaler["mean"]
            std = scaler["std"]
        else:
            # Fall back: load original features and compute stats
            orig_features = np.load(self.cfg.data.processed_data_dir / "features.npy")
            train_end = int(len(orig_features) * self.cfg.train.train_ratio)
            train_data = orig_features[:train_end]
            mean = train_data.mean(axis=0)
            std = train_data.std(axis=0)
            std[std < 1e-8] = 1.0

            # Cache for future use
            np.savez(self.scaler_path, mean=mean, std=std)

        features = (features - mean) / std
        np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def _save_online_features(self, new_features: np.ndarray, new_dates):
        """Save or append new features to the online dataset."""
        date_strs = [str(d.date()) if hasattr(d, 'date') else str(d) for d in new_dates]

        if self.online_features_path.exists():
            existing = np.load(self.online_features_path)
            existing_meta = json.loads(self.online_meta_path.read_text())
            existing_dates = existing_meta["dates"]

            # Avoid duplicates
            new_mask = [d not in existing_dates for d in date_strs]
            if not any(new_mask):
                print("All new dates already in online dataset.")
                return

            new_features_filtered = new_features[new_mask]
            new_dates_filtered = [d for d, m in zip(date_strs, new_mask) if m]

            combined = np.concatenate([existing, new_features_filtered], axis=0)
            combined_dates = existing_dates + new_dates_filtered
        else:
            combined = new_features
            combined_dates = date_strs

        np.save(self.online_features_path, combined.astype(np.float32))

        online_meta = {
            "shape": list(combined.shape),
            "dates": combined_dates,
            "tickers": self.tickers,
            "feature_names": self.feature_names,
            "spy_index": self.spy_index,
            "n_days": len(combined_dates),
        }
        self.online_meta_path.write_text(json.dumps(online_meta, indent=2, default=str))

        print(f"Online dataset: {combined.shape[0]} total days")

    # ------------------------------------------------------------------
    # Step 3: Fine-tune the model
    # ------------------------------------------------------------------
    def fine_tune(
        self,
        epochs: int = 10,
        lr: float = 1e-5,
        min_new_samples: int = 5,
    ) -> dict:
        """
        Fine-tune the model on new data.

        Uses the last `lookback_len + forecast_len` days of original data as
        context, plus all online data, to create a fine-tuning dataset.

        Args:
            epochs: Number of fine-tuning epochs
            lr: Learning rate (lower than original training)
            min_new_samples: Minimum new samples required to fine-tune
        """
        if not self.online_features_path.exists():
            print("No online data available. Run download + preprocess first.")
            return {}

        # Load data
        orig_features = np.load(self.cfg.data.processed_data_dir / "features.npy")
        online_features = np.load(self.online_features_path)
        online_meta = json.loads(self.online_meta_path.read_text())

        # Get SPY returns for online data (feature index 0 = returns)
        online_spy_returns = online_features[:, self.spy_index, 0]

        # Need context from original data for the lookback window
        lookback = self.cfg.model.lookback_len
        context_len = lookback + 5  # Extra buffer
        context_features = orig_features[-context_len:]
        context_spy_returns = context_features[:, self.spy_index, 0]

        # Combine context + online data
        combined_features = np.concatenate([context_features, online_features], axis=0)
        combined_spy_returns = np.concatenate([context_spy_returns, online_spy_returns])

        # Apply normalization to context (online data already normalized)
        combined_features[:context_len] = self._normalize_with_original_stats(
            orig_features[-context_len:]
        )

        n_samples = len(combined_features) - lookback - self.cfg.model.forecast_len + 1
        if n_samples < min_new_samples:
            print(f"Only {n_samples} samples available (need {min_new_samples}). "
                  f"Collect more data before fine-tuning.")
            return {}

        print(f"\nFine-tuning dataset: {n_samples} samples "
              f"({context_len} context + {len(online_features)} new days)")

        # Create dataset and loader
        dataset = SPYForecastDataset(
            combined_features, combined_spy_returns,
            lookback, self.cfg.model.forecast_len,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(self.cfg.train.batch_size, n_samples),
            shuffle=True, num_workers=0, drop_last=False,
        )

        # Load model (online if exists, otherwise original)
        self.cfg.model.num_variates = combined_features.shape[1]
        self.cfg.model.n_features = combined_features.shape[2]

        model = build_model(self.cfg, spy_index=self.spy_index)

        if self.online_ckpt.exists():
            print("Loading online model checkpoint...")
            load_checkpoint(self.online_ckpt, model)
        elif self.original_ckpt.exists():
            print("Loading original model checkpoint...")
            load_checkpoint(self.original_ckpt, model)
        else:
            print("ERROR: No model checkpoint found.")
            return {}

        model = model.to(self.device)

        # Fine-tune with lower learning rate
        loss_fn = get_loss_fn(self.cfg.train.loss_fn)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        print(f"\nFine-tuning for {epochs} epochs (lr={lr:.1e})")
        print("-" * 60)

        best_loss = float("inf")
        metrics_history = []

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            n_batches = 0

            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.train.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Quick eval on same data (we don't have a separate val set for online)
            eval_metrics = evaluate(model, loader, loss_fn, self.device)

            print(f"  Epoch {epoch:2d}/{epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Corr: {eval_metrics['correlation']:.4f} | "
                  f"DirAcc: {eval_metrics['directional_accuracy']:.4f}")

            metrics_history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "correlation": eval_metrics["correlation"],
                "directional_accuracy": eval_metrics["directional_accuracy"],
            })

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, None, epoch, avg_loss, self.online_ckpt)

        # Log the update
        update_record = {
            "timestamp": datetime.now().isoformat(),
            "new_days": len(online_features),
            "n_samples": n_samples,
            "epochs": epochs,
            "lr": lr,
            "best_loss": best_loss,
            "final_correlation": metrics_history[-1]["correlation"],
            "final_dir_acc": metrics_history[-1]["directional_accuracy"],
            "online_dates": online_meta["dates"],
        }

        history = self.get_update_history()
        history.append(update_record)
        self._save_history(history)

        print(f"\nFine-tuning complete. Best loss: {best_loss:.6f}")
        print(f"Online model saved to {self.online_ckpt}")
        print(f"Update #{len(history)} logged.")

        return update_record

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def rollback(self):
        """Rollback to the original model, discarding online updates."""
        if self.online_ckpt.exists():
            # Keep a backup
            backup_path = self.cfg.train.checkpoint_dir / "online_model_backup.pt"
            shutil.copy2(self.online_ckpt, backup_path)
            self.online_ckpt.unlink()
            print(f"Removed online model (backed up to {backup_path})")

        if self.online_features_path.exists():
            self.online_features_path.unlink()
            print("Removed online features")

        if self.online_meta_path.exists():
            self.online_meta_path.unlink()
            print("Removed online metadata")

        print("Rolled back to original model.")

    def status(self):
        """Print current status of online training."""
        print("\n=== Online Training Status ===")
        print(f"Original model:  {'✓' if self.original_ckpt.exists() else '✗'}")
        print(f"Online model:    {'✓' if self.online_ckpt.exists() else '✗ (using original)'}")

        last_date = self.get_last_date()
        print(f"Last data date:  {last_date}")

        if self.online_features_path.exists():
            online_meta = json.loads(self.online_meta_path.read_text())
            print(f"Online data:     {online_meta['n_days']} days "
                  f"({online_meta['dates'][0]} to {online_meta['dates'][-1]})")
        else:
            print("Online data:     None")

        history = self.get_update_history()
        print(f"Updates:         {len(history)}")
        if history:
            last = history[-1]
            print(f"Last update:     {last['timestamp']}")
            print(f"  Correlation:   {last['final_correlation']:.4f}")
            print(f"  Dir. Accuracy: {last['final_dir_acc']:.4f}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Online training for iTransformer")
    parser.add_argument("--since", type=str, default=None,
                        help="Download data since this date (YYYY-MM-DD)")
    parser.add_argument("--download-only", action="store_true",
                        help="Download new data without training")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, fine-tune on existing online data")
    parser.add_argument("--rollback", action="store_true",
                        help="Rollback to original model")
    parser.add_argument("--status", action="store_true",
                        help="Show current online training status")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Fine-tuning epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Fine-tuning learning rate (default: 1e-5)")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="Minimum new samples before fine-tuning (default: 5)")
    args = parser.parse_args()

    cfg = Config()
    trainer = OnlineTrainer(cfg)

    if args.status:
        trainer.status()
        return

    if args.rollback:
        trainer.rollback()
        return

    # Step 1: Download
    if not args.skip_download:
        new_files = trainer.download_new_data(since=args.since)
        if not new_files and not args.skip_download:
            print("No new data. Model is up to date.")
            trainer.status()
            return

        # Step 2: Preprocess
        trainer.preprocess_new_data(new_files)

    if args.download_only:
        print("Download complete (--download-only). Skipping training.")
        trainer.status()
        return

    # Step 3: Fine-tune
    trainer.fine_tune(
        epochs=args.epochs,
        lr=args.lr,
        min_new_samples=args.min_samples,
    )

    trainer.status()


if __name__ == "__main__":
    main()
