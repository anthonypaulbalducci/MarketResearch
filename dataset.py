"""
PyTorch Dataset for the iTransformer (V2).

Loads the 3D feature tensor (days, variates, features) and creates
rolling windows of (lookback, forecast) pairs.

Input shape:  (lookback_len, num_variates, n_features)
Target shape: (forecast_len,) — future SPY daily returns
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


class SPYForecastDataset(Dataset):
    """
    Dataset producing (input, target) pairs for SPY forecasting.

    Input:  (lookback_len, num_variates, n_features)
    Target: (forecast_len,) — future SPY returns
    """

    def __init__(
        self,
        features: np.ndarray,
        spy_returns: np.ndarray,
        lookback_len: int,
        forecast_len: int,
    ):
        """
        Args:
            features: (T, N, F) array — T days, N variates, F features
            spy_returns: (T,) array — daily SPY returns
            lookback_len: days of history as input
            forecast_len: days ahead to predict
        """
        self.features = features.astype(np.float32)
        self.spy_returns = spy_returns.astype(np.float32)
        self.lookback_len = lookback_len
        self.forecast_len = forecast_len
        self.total_len = lookback_len + forecast_len

        self.n_samples = len(features) - self.total_len + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"Data length ({len(features)}) too short for "
                f"lookback={lookback_len} + forecast={forecast_len}"
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Input: (lookback_len, num_variates, n_features)
        x = self.features[idx : idx + self.lookback_len]

        # Target: (forecast_len,) — SPY returns in the forecast window
        target_start = idx + self.lookback_len
        target_end = target_start + self.forecast_len
        y = self.spy_returns[target_start:target_end]

        return torch.from_numpy(x), torch.from_numpy(y)


def load_and_split_data(cfg: Config) -> dict:
    """
    Load processed 3D features and create train/val/test datasets.
    """
    data_dir = cfg.data.processed_data_dir
    features_path = data_dir / "features.npy"
    returns_path = data_dir / "spy_returns.npy"
    meta_path = data_dir / "metadata.json"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {features_path}. Run preprocess.py first."
        )

    print("Loading processed dataset...")
    features = np.load(features_path)       # (T, N, F)
    spy_returns = np.load(returns_path)      # (T,)
    metadata = json.loads(meta_path.read_text())

    tickers = metadata["tickers"]
    feature_names = metadata["feature_names"]
    spy_index = metadata["spy_index"]
    dates = metadata["dates"]

    T, N, F = features.shape
    print(f"Features shape: ({T} days, {N} variates, {F} features)")
    print(f"Feature names: {feature_names}")
    print(f"SPY index: {spy_index} ('{tickers[spy_index]}')")

    # Chronological split
    n = T
    train_end = int(n * cfg.train.train_ratio)
    val_end = train_end + int(n * cfg.train.val_ratio)

    train_features = features[:train_end]
    val_features = features[train_end:val_end]
    test_features = features[val_end:]

    train_returns = spy_returns[:train_end]
    val_returns = spy_returns[train_end:val_end]
    test_returns = spy_returns[val_end:]

    print(f"\nTrain: {len(train_features):,} days ({dates[0]} to {dates[train_end-1]})")
    print(f"Val:   {len(val_features):,} days ({dates[train_end]} to {dates[val_end-1]})")
    print(f"Test:  {len(test_features):,} days ({dates[val_end]} to {dates[-1]})")

    # Z-score normalization using training statistics
    # Compute per-feature, per-variate mean and std from training data
    scaler_params = None
    if cfg.features.normalization == "zscore":
        # (N, F) statistics
        mean = train_features.mean(axis=0)  # (N, F)
        std = train_features.std(axis=0)    # (N, F)
        std[std < 1e-8] = 1.0

        train_features = (train_features - mean) / std
        val_features = (val_features - mean) / std
        test_features = (test_features - mean) / std

        scaler_params = {"mean": mean, "std": std}

        # Save scaler params for online training
        scaler_path = cfg.data.processed_data_dir / "scaler_params.npz"
        np.savez(scaler_path, mean=mean, std=std)
        print("Applied z-score normalization (train stats, per-variate per-feature)")
        print(f"Saved scaler params to {scaler_path}")

    # Clean up any NaN/inf
    for arr in [train_features, val_features, test_features]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Create datasets
    lookback = cfg.model.lookback_len
    forecast = cfg.model.forecast_len

    train_dataset = SPYForecastDataset(train_features, train_returns, lookback, forecast)
    val_dataset = SPYForecastDataset(val_features, val_returns, lookback, forecast)
    test_dataset = SPYForecastDataset(test_features, test_returns, lookback, forecast)

    print(f"\nSamples — Train: {len(train_dataset):,}  "
          f"Val: {len(val_dataset):,}  Test: {len(test_dataset):,}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # Update model config with actual dimensions
    cfg.model.num_variates = N
    cfg.model.n_features = F
    print(f"Variates: {N}, Features per variate: {F}")

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "tickers": tickers,
        "feature_names": feature_names,
        "spy_index": spy_index,
        "scaler_params": scaler_params,
        "num_variates": N,
        "n_features": F,
        "dates": dates,
    }
