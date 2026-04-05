"""
Run the full SPY iTransformer V2 pipeline.

Usage:
    python run.py --access-key YOUR_KEY --secret-key YOUR_SECRET
    python run.py --skip-download
    python run.py --only-train
    python run.py --only-evaluate
"""
import argparse
import os
import sys

from config import Config


def main():
    parser = argparse.ArgumentParser(description="SPY iTransformer V2 Pipeline")
    parser.add_argument("--access-key", type=str, default=None)
    parser.add_argument("--secret-key", type=str, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--only-train", action="store_true")
    parser.add_argument("--only-evaluate", action="store_true")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--forecast", type=int, default=None)
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

    # Step 1: Download
    if not args.skip_download and not args.only_train and not args.only_evaluate:
        print("=" * 60)
        print("STEP 1: Downloading flat files from Massive.com")
        print("=" * 60)
        access_key = args.access_key or os.environ.get("MASSIVE_S3_ACCESS_KEY", "")
        secret_key = args.secret_key or os.environ.get("MASSIVE_S3_SECRET_KEY", "")
        if not access_key or not secret_key:
            print("ERROR: S3 credentials required for download.")
            print("Use --access-key/--secret-key or set env vars.")
            print("Or use --skip-download if data is already downloaded.")
            sys.exit(1)
        from download_data import download_all
        download_all(cfg, access_key, secret_key)
        print()

    # Step 2: Preprocess
    if not args.only_train and not args.only_evaluate:
        print("=" * 60)
        print("STEP 2: Preprocessing → daily OHLCV + rich features")
        print("=" * 60)
        from preprocess import preprocess_all
        preprocess_all(cfg)
        print()

    # Step 3: Train
    if not args.only_evaluate:
        print("=" * 60)
        print("STEP 3: Training iTransformer V2")
        print("=" * 60)
        from train import train
        train(cfg)
        print()

    # Step 4: Evaluate
    print("=" * 60)
    print("STEP 4: Generating evaluation plots")
    print("=" * 60)
    from evaluate import generate_all_plots
    generate_all_plots(cfg)
    print()

    print("Pipeline complete!")


if __name__ == "__main__":
    main()
