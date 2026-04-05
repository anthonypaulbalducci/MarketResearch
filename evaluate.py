"""
Evaluation and visualization for iTransformer V2.

Generates:
- Training curves
- Prediction vs actual plots
- Prediction distribution (bias diagnostic)
- Cumulative return simulation (with de-biased variant)
- Attention heatmaps

Usage:
    python evaluate.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config


def plot_training_curves(results: dict, save_dir: Path):
    """Training and validation loss, MAE, correlation, directional accuracy."""
    history = results["history"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training History (V2: Daily Bars, Top 50 + SPY, Rich Features)",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train", alpha=0.8)
    ax.plot(history["val_loss"], label="Validation", alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history["val_mae"], color="orange", alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE"); ax.set_title("Validation MAE")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(history["val_corr"], color="green", alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Correlation"); ax.set_title("Pred-Target Correlation")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history["val_dir_acc"], color="purple", alpha=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Directional Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(history.get("pred_mean", []), color="red", alpha=0.8, label="Mean")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Pred Mean"); ax.set_title("Prediction Bias")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(history.get("pred_std", []), color="teal", alpha=0.8, label="Std")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Pred Std"); ax.set_title("Prediction Spread")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved training_curves.png")


def plot_predictions(predictions: np.ndarray, targets: np.ndarray, save_dir: Path):
    """Predicted vs actual, scatter, error distribution, and prediction histogram."""
    pred = predictions.flatten() if predictions.ndim > 1 else predictions
    tgt = targets.flatten() if targets.ndim > 1 else targets

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Test Set Analysis", fontsize=14, fontweight="bold")

    # Time series overlay
    ax = axes[0, 0]
    n_show = min(300, len(tgt))
    ax.plot(tgt[:n_show], label="Actual", alpha=0.7, linewidth=0.8)
    ax.plot(pred[:n_show], label="Predicted", alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Day"); ax.set_ylabel("Return")
    ax.set_title(f"SPY Returns (first {n_show} days)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Scatter
    ax = axes[0, 1]
    ax.scatter(tgt, pred, alpha=0.4, s=15, color="steelblue")
    lim = max(abs(tgt).max(), abs(pred).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    corr = np.corrcoef(pred, tgt)[0, 1]
    ax.set_title(f"Scatter (corr={corr:.4f})")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Prediction distribution (bias diagnostic)
    ax = axes[1, 0]
    ax.hist(pred, bins=60, alpha=0.6, color="steelblue", label="Predictions", density=True)
    ax.hist(tgt, bins=60, alpha=0.6, color="orange", label="Actual", density=True)
    ax.axvline(x=pred.mean(), color="blue", linestyle="--", alpha=0.8,
               label=f"Pred mean={pred.mean():.5f}")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Return"); ax.set_ylabel("Density")
    ax.set_title("Prediction vs Actual Distribution")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Error distribution
    ax = axes[1, 1]
    errors = pred - tgt
    ax.hist(errors, bins=60, alpha=0.7, color="coral", edgecolor="white")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Error"); ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution (mean={errors.mean():.5f}, std={errors.std():.5f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved predictions.png")


def plot_cumulative_returns(predictions: np.ndarray, targets: np.ndarray, save_dir: Path):
    """
    Cumulative return simulation with both raw and de-biased strategy signals.
    """
    pred = predictions.flatten() if predictions.ndim > 1 else predictions
    tgt = targets.flatten() if targets.ndim > 1 else targets

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- Raw signal strategy ---
    ax = axes[0]
    positions_raw = np.sign(pred)
    strategy_raw = positions_raw * tgt
    cum_strategy_raw = np.cumsum(strategy_raw)
    cum_buyhold = np.cumsum(tgt)

    ax.plot(cum_strategy_raw, label="Strategy (raw signal)", alpha=0.8, linewidth=1.2)
    ax.plot(cum_buyhold, label="Buy & Hold SPY", alpha=0.8, linewidth=1.2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Day"); ax.set_ylabel("Cumulative Return")
    ax.set_title("Raw Signal Strategy vs Buy & Hold")

    pct_short = (positions_raw < 0).mean() * 100
    ax.annotate(f"Short {pct_short:.0f}% of the time", xy=(0.02, 0.92),
                xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- De-biased signal strategy ---
    ax = axes[1]
    # Use median-centered predictions so bias doesn't determine all positions
    pred_centered = pred - np.median(pred)
    positions_debiased = np.sign(pred_centered)
    strategy_debiased = positions_debiased * tgt
    cum_strategy_debiased = np.cumsum(strategy_debiased)

    ax.plot(cum_strategy_debiased, label="Strategy (de-biased signal)", alpha=0.8,
            linewidth=1.2, color="green")
    ax.plot(cum_buyhold, label="Buy & Hold SPY", alpha=0.8, linewidth=1.2, color="orange")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Day"); ax.set_ylabel("Cumulative Return")
    ax.set_title("De-biased Signal Strategy vs Buy & Hold")

    if strategy_debiased.std() > 0:
        sharpe = strategy_debiased.mean() / strategy_debiased.std() * np.sqrt(252)
        ax.annotate(f"Annualized Sharpe: {sharpe:.2f}", xy=(0.02, 0.92),
                    xycoords="axes fraction", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    pct_short_db = (positions_debiased < 0).mean() * 100
    ax.annotate(f"Short {pct_short_db:.0f}% of the time", xy=(0.02, 0.82),
                xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved cumulative_returns.png")


def generate_all_plots(cfg: Config):
    """Generate all evaluation plots from saved results."""
    results_dir = cfg.train.results_dir

    results_path = results_dir / "training_results.json"
    if not results_path.exists():
        print(f"No results found at {results_path}. Run train.py first.")
        return

    results = json.loads(results_path.read_text())
    plot_training_curves(results, results_dir)

    pred_path = results_dir / "test_predictions.npz"
    if pred_path.exists():
        data = np.load(pred_path)
        predictions = data["predictions"]
        targets = data["targets"]

        plot_predictions(predictions, targets, results_dir)
        plot_cumulative_returns(predictions, targets, results_dir)
    else:
        print("No test predictions found. Skipping prediction plots.")

    print(f"\nAll plots saved to {results_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate iTransformer V2")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.results_dir:
        cfg.train.results_dir = Path(args.results_dir)

    generate_all_plots(cfg)


if __name__ == "__main__":
    main()
