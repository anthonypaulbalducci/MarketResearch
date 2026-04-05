"""
Download minute aggregate flat files from Massive.com via S3.

Usage:
    python download_data.py --access-key YOUR_KEY --secret-key YOUR_SECRET

    Or set environment variables:
        export MASSIVE_S3_ACCESS_KEY=your_key
        export MASSIVE_S3_SECRET_KEY=your_secret
        python download_data.py
"""
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from config import Config


def get_trading_days(start_date: str, end_date: str) -> list[str]:
    """Generate weekdays between start and end."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


def build_s3_key(date_str: str, prefix: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{prefix}/{dt.year}/{dt.month:02d}/{date_str}.csv.gz"


def create_s3_client(access_key: str, secret_key: str, endpoint: str, region: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=BotoConfig(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=60,
        ),
    )


def download_single_file(s3_client, bucket, s3_key, local_path, skip_existing=True):
    result = {"key": s3_key, "status": "unknown", "size": 0}

    if skip_existing and local_path.exists() and local_path.stat().st_size > 0:
        result["status"] = "skipped"
        return result

    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        data = response["Body"].read()
        local_path.write_bytes(data)
        result["status"] = "downloaded"
        result["size"] = len(data)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("NoSuchKey", "404"):
            result["status"] = "not_found"
        else:
            result["status"] = f"error: {error_code}"
    except Exception as e:
        result["status"] = f"error: {str(e)}"

    return result


def download_all(cfg: Config, access_key: str, secret_key: str):
    s3_client = create_s3_client(
        access_key, secret_key, cfg.data.s3_endpoint, cfg.data.s3_region
    )

    trading_days = get_trading_days(cfg.data.start_date, cfg.data.end_date)
    print(f"Date range: {cfg.data.start_date} to {cfg.data.end_date}")
    print(f"Potential trading days: {len(trading_days)}")

    stats = {"downloaded": 0, "skipped": 0, "not_found": 0, "error": 0}

    def _download_day(date_str):
        s3_key = build_s3_key(date_str, cfg.data.minute_aggs_prefix)
        local_path = cfg.data.raw_data_dir / f"{date_str}.csv.gz"
        return download_single_file(
            s3_client, cfg.data.s3_bucket, s3_key, local_path, cfg.data.skip_existing
        )

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=cfg.data.max_download_workers) as executor:
        futures = {executor.submit(_download_day, day): day for day in trading_days}

        for i, future in enumerate(as_completed(futures), 1):
            day = futures[future]
            result = future.result()
            status = result["status"]

            if status == "downloaded":
                stats["downloaded"] += 1
                size_mb = result["size"] / (1024 * 1024)
                print(f"  [{i}/{len(trading_days)}] {day}: Downloaded ({size_mb:.1f} MB)")
            elif status == "skipped":
                stats["skipped"] += 1
            elif status == "not_found":
                stats["not_found"] += 1
            else:
                stats["error"] += 1
                print(f"  [{i}/{len(trading_days)}] {day}: {status}")

            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(trading_days) - i) / rate
                print(f"  Progress: {i}/{len(trading_days)} "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"\nDownload complete in {elapsed:.1f}s")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Skipped (existing): {stats['skipped']}")
    print(f"  Not found (holidays): {stats['not_found']}")
    print(f"  Errors: {stats['error']}")


def main():
    parser = argparse.ArgumentParser(description="Download Massive.com flat files")
    parser.add_argument("--access-key", type=str, default=None)
    parser.add_argument("--secret-key", type=str, default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    access_key = args.access_key or os.environ.get("MASSIVE_S3_ACCESS_KEY", "")
    secret_key = args.secret_key or os.environ.get("MASSIVE_S3_SECRET_KEY", "")

    if not access_key or not secret_key:
        print("ERROR: S3 credentials required.")
        print("Either pass --access-key/--secret-key or set environment variables:")
        print("  export MASSIVE_S3_ACCESS_KEY=your_key")
        print("  export MASSIVE_S3_SECRET_KEY=your_secret")
        sys.exit(1)

    cfg = Config()
    if args.start_date:
        cfg.data.start_date = args.start_date
    if args.end_date:
        cfg.data.end_date = args.end_date
    if args.workers:
        cfg.data.max_download_workers = args.workers

    download_all(cfg, access_key, secret_key)


if __name__ == "__main__":
    main()
