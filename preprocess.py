"""
Preprocess minute flat files into daily feature-rich multivariate time series.

V2 Pipeline:
1. Read daily .csv.gz files (minute bars)
2. Filter for top 50 tickers + SPY, regular market hours only
3. Aggregate minute bars → daily OHLCV per ticker
4. Compute rich features per ticker per day
5. Save as 3D array: (trading_days, num_variates, n_features)

Usage:
    python preprocess.py
    python preprocess.py --start-date 2020-01-01
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from sp500_tickers import get_ticker_set, get_all_tickers, TARGET_TICKER


def read_daily_flat_file(filepath: Path, ticker_set: set) -> pd.DataFrame:
    """Read a daily minute aggregates .csv.gz and filter for target tickers."""
    try:
        with gzip.open(filepath, "rt") as f:
            df = pd.read_csv(f)
    except Exception as e:
        print(f"  Warning: Could not read {filepath.name}: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    df = df[df["ticker"].isin(ticker_set)].copy()
    if df.empty:
        return df

    # Convert nanosecond timestamps to Eastern Time
    df["timestamp"] = pd.to_datetime(df["window_start"], unit="ns", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")

    cols = ["ticker", "timestamp", "open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]]
    return df


def filter_market_hours(df: pd.DataFrame, market_open: str, market_close: str) -> pd.DataFrame:
    """Filter to regular trading hours only."""
    if df.empty:
        return df
    time_idx = df["timestamp"].dt.time
    open_time = pd.Timestamp(market_open).time()
    close_time = pd.Timestamp(market_close).time()
    mask = (time_idx >= open_time) & (time_idx < close_time)
    return df[mask].copy()


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate minute bars to daily OHLCV per ticker.
    Returns DataFrame with columns: ticker, date, open, high, low, close, volume
    """
    if df.empty:
        return pd.DataFrame()

    df["date"] = df["timestamp"].dt.date

    daily = df.groupby(["ticker", "date"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    return daily


def process_single_day(filepath: Path, ticker_set: set,
                       market_open: str, market_close: str) -> pd.DataFrame | None:
    """Read one flat file and return daily OHLCV for target tickers."""
    df = read_daily_flat_file(filepath, ticker_set)
    if df.empty:
        return None

    df = filter_market_hours(df, market_open, market_close)
    if df.empty:
        return None

    daily = aggregate_to_daily(df)
    if daily.empty:
        return None

    return daily


def compute_features(ohlcv_wide: dict, tickers: list) -> np.ndarray:
    """
    Compute rich features from daily OHLCV data.

    Args:
        ohlcv_wide: dict with keys 'open', 'high', 'low', 'close', 'volume',
                     each a DataFrame of shape (trading_days, num_tickers)
        tickers: list of ticker names (column order)

    Returns:
        features: np.ndarray of shape (trading_days, num_tickers, n_features)
    """
    close = ohlcv_wide["close"]
    opn = ohlcv_wide["open"]
    high = ohlcv_wide["high"]
    low = ohlcv_wide["low"]
    volume = ohlcv_wide["volume"]

    n_days, n_tickers = close.shape
    feature_list = []

    # 1. Returns (close-to-close)
    returns = close.pct_change()
    feature_list.append(returns)

    # 2. Log returns
    log_returns = np.log(close / close.shift(1))
    feature_list.append(log_returns)

    # 3. Intraday range: (high - low) / close
    intraday_range = (high - low) / close
    feature_list.append(intraday_range)

    # 4. Open-to-close return: (close - open) / open
    oc_return = (close - opn) / opn
    feature_list.append(oc_return)

    # 5. Log volume (z-scored per ticker later)
    log_vol = np.log1p(volume)
    feature_list.append(log_vol)

    # 6. Relative volume: volume / 20-day rolling average
    vol_ma20 = volume.rolling(window=20, min_periods=5).mean()
    relative_vol = volume / vol_ma20
    feature_list.append(relative_vol)

    # 7. RSI (14-day), rescaled to 0-1
    rsi = _compute_rsi(close, period=14)
    rsi = rsi / 100.0  # Rescale to 0-1
    feature_list.append(rsi)

    # 8. Price vs SMA(20): close / SMA(20) - 1
    sma20 = close.rolling(window=20, min_periods=5).mean()
    price_vs_sma = close / sma20 - 1.0
    feature_list.append(price_vs_sma)

    # 9. Realized volatility: 10-day rolling std of returns
    realized_vol = returns.rolling(window=10, min_periods=3).std()
    feature_list.append(realized_vol)

    # 10. Cross-sectional return rank (percentile among tickers each day)
    return_rank = returns.rank(axis=1, pct=True)
    feature_list.append(return_rank)

    # Stack: (n_days, n_tickers, n_features)
    features = np.stack([f.values for f in feature_list], axis=-1)

    return features


def _compute_rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute RSI for each column in a DataFrame."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Use EMA-style smoothing after initial SMA
    for i in range(period, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def preprocess_all(cfg: Config):
    """Process all flat files into a feature-rich daily dataset."""
    ticker_set = get_ticker_set()
    all_tickers = get_all_tickers()
    raw_dir = cfg.data.raw_data_dir
    cache_dir = cfg.data.cache_dir

    gz_files = sorted(raw_dir.glob("*.csv.gz"))
    if not gz_files:
        print(f"No .csv.gz files found in {raw_dir}")
        print("Run download_data.py first.")
        sys.exit(1)

    print(f"Found {len(gz_files)} daily files to process")
    print(f"Target tickers: {len(ticker_set)}")

    # Step 1: Aggregate all files to daily OHLCV
    print("\n--- Step 1: Aggregating minute bars to daily OHLCV ---")
    daily_rows = []
    start_time = time.time()

    for i, filepath in enumerate(gz_files, 1):
        date_str = filepath.stem.replace(".csv", "")
        cache_path = cache_dir / f"{date_str}_daily.parquet"

        if cache_path.exists():
            try:
                day_df = pd.read_parquet(cache_path)
            except Exception:
                cache_path.unlink(missing_ok=True)
                day_df = process_single_day(
                    filepath, ticker_set, cfg.data.market_open, cfg.data.market_close
                )
                if day_df is not None and not day_df.empty:
                    day_df.to_parquet(cache_path)
        else:
            day_df = process_single_day(
                filepath, ticker_set, cfg.data.market_open, cfg.data.market_close
            )
            if day_df is not None and not day_df.empty:
                day_df.to_parquet(cache_path)

        if day_df is not None and not day_df.empty:
            daily_rows.append(day_df)

        if i % 100 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(gz_files) - i) / rate if rate > 0 else 0
            print(f"  Processed {i}/{len(gz_files)} days "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    if not daily_rows:
        print("No data was processed successfully.")
        sys.exit(1)

    # Concatenate all daily data
    all_daily = pd.concat(daily_rows, axis=0, ignore_index=True)
    all_daily["date"] = pd.to_datetime(all_daily["date"])
    all_daily = all_daily.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"\nTotal daily records: {len(all_daily):,}")
    print(f"Date range: {all_daily['date'].min()} to {all_daily['date'].max()}")
    print(f"Unique tickers: {all_daily['ticker'].nunique()}")
    print(f"Unique trading days: {all_daily['date'].nunique()}")

    # Step 2: Pivot to wide format for each OHLCV field
    print("\n--- Step 2: Pivoting to wide format ---")

    # Determine tickers with sufficient coverage
    coverage = all_daily.groupby("ticker")["date"].count()
    total_days = all_daily["date"].nunique()
    min_days = int(total_days * (1 - cfg.features.max_missing_ratio))
    valid_tickers = coverage[coverage >= min_days].index.tolist()

    # Ensure SPY is present
    if TARGET_TICKER not in valid_tickers:
        print(f"ERROR: {TARGET_TICKER} has insufficient data!")
        sys.exit(1)

    # Keep only valid tickers
    valid_tickers = sorted(valid_tickers)
    print(f"Tickers with ≥{100*(1-cfg.features.max_missing_ratio):.0f}% coverage: "
          f"{len(valid_tickers)} / {len(all_tickers)}")

    all_daily = all_daily[all_daily["ticker"].isin(valid_tickers)]

    ohlcv_wide = {}
    for field in ["open", "high", "low", "close", "volume"]:
        pivoted = all_daily.pivot_table(
            index="date", columns="ticker", values=field, aggfunc="last"
        )
        pivoted = pivoted[valid_tickers]  # Consistent column order
        pivoted = pivoted.sort_index()
        # Forward-fill then back-fill missing days
        pivoted = pivoted.ffill().bfill()
        ohlcv_wide[field] = pivoted

    dates = ohlcv_wide["close"].index
    tickers = valid_tickers
    print(f"OHLCV matrix shape: ({len(dates)} days, {len(tickers)} tickers)")

    # Step 3: Compute features
    print("\n--- Step 3: Computing features ---")
    features_3d = compute_features(ohlcv_wide, tickers)
    print(f"Feature tensor shape: {features_3d.shape}  "
          f"(days, tickers, features)")

    # Trim leading NaN rows (from rolling windows)
    # Find first row with no NaN
    valid_mask = ~np.isnan(features_3d).any(axis=(1, 2))
    first_valid = np.argmax(valid_mask)
    features_3d = features_3d[first_valid:]
    dates = dates[first_valid:]
    print(f"After trimming leading NaN rows: {features_3d.shape}")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    # Replace remaining NaN/inf with 0
    np.nan_to_num(features_3d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 4: Save
    print("\n--- Step 4: Saving ---")
    output_dir = cfg.data.processed_data_dir

    # Save feature tensor
    np.save(output_dir / "features.npy", features_3d.astype(np.float32))

    # Save target (SPY daily return = feature index 0 of SPY's variate)
    spy_idx = tickers.index(TARGET_TICKER)
    spy_returns = features_3d[:, spy_idx, 0]  # Feature 0 = returns
    np.save(output_dir / "spy_returns.npy", spy_returns.astype(np.float32))

    # Save metadata
    feature_names = list(cfg.features.features)
    metadata = {
        "shape": list(features_3d.shape),
        "tickers": tickers,
        "spy_index": spy_idx,
        "dates": [str(d) for d in dates],
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_variates": len(tickers),
        "n_days": len(dates),
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    print(f"Saved features.npy: {features_3d.shape} "
          f"({features_3d.nbytes / 1024**2:.1f} MB)")
    print(f"Saved spy_returns.npy: {spy_returns.shape}")
    print(f"Saved metadata.json")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Features: {feature_names}")
    print(f"  SPY index: {spy_idx}")
    print(f"\nPreprocessing complete!")


def main():
    parser = argparse.ArgumentParser(description="Preprocess flat files (V2)")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.start_date:
        cfg.data.start_date = args.start_date
    if args.end_date:
        cfg.data.end_date = args.end_date

    preprocess_all(cfg)


if __name__ == "__main__":
    main()
