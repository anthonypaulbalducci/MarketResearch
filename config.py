"""
Configuration for SPY iTransformer forecasting project.
V2: Daily bars, top 50 stocks by market cap, richer features.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data download and storage configuration."""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_endpoint: str = "https://files.massive.com"
    s3_bucket: str = "flatfiles"
    s3_region: str = "us-east-1"
    minute_aggs_prefix: str = "us_stocks_sip/minute_aggs_v1"

    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")

    start_date: str = "2016-04-01"
    end_date: str = "2026-03-31"

    market_open: str = "09:30"
    market_close: str = "16:00"
    include_extended_hours: bool = False

    max_download_workers: int = 4
    retry_attempts: int = 3
    skip_existing: bool = True


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    features: tuple = (
        "returns",           # Close-to-close return
        "log_returns",       # Log close-to-close return
        "intraday_range",    # (high - low) / close
        "open_close_return", # (close - open) / open
        "log_volume",        # log(volume + 1) z-scored
        "relative_volume",   # volume / 20-day avg volume
        "rsi_14",            # 14-day RSI (rescaled 0-1)
        "price_vs_sma20",    # close / SMA(20) - 1
        "realized_vol_10",   # 10-day rolling std of returns
        "return_rank",       # Cross-sectional percentile rank of daily return
    )

    normalization: str = "zscore"
    fill_method: str = "ffill"
    max_missing_ratio: float = 0.1


@dataclass
class ModelConfig:
    """iTransformer architecture configuration."""
    lookback_len: int = 20        # Trading days of history
    forecast_len: int = 1         # Trading days ahead to predict
    num_variates: int = 51        # Top 50 + SPY (set dynamically)
    n_features: int = 10          # Features per variate per day (set dynamically)

    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"

    predict_spy_only: bool = True
    norm_type: str = "pre"


@dataclass
class TrainConfig:
    """Training configuration."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15

    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 200

    loss_fn: str = "huber"

    max_grad_norm: float = 1.0
    device: str = "auto"

    log_interval: int = 50
    checkpoint_dir: Path = Path("checkpoints")
    results_dir: Path = Path("results")

    seed: int = 42


@dataclass
class Config:
    """Top-level configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        for d in [
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.data.cache_dir,
            self.train.checkpoint_dir,
            self.train.results_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
