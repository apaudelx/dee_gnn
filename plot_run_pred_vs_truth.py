#!/usr/bin/env python3
"""
Build side-by-side Truth vs Prediction scatter plots from a training run folder
(e.g. results/20260422_213856), matching the style of train/test summary figures.

Example:
  python plot_run_pred_vs_truth.py results/20260422_213856
  python plot_run_pred_vs_truth.py results/20260422_213856 --splits train val test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLIT_SPECS = {
    "train": ("train_predictions.csv", "Training"),
    "val": ("val_predictions.csv", "Validation"),
    "test": ("test_predictions.csv", "Testing"),
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae, rmse, r2


def _style_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("Truth", fontweight="bold")
    ax.set_ylabel("Prediction", fontweight="bold")
    ax.grid(True, linestyle=":", color="0.75", linewidth=0.8)
    ax.tick_params(axis="both", direction="in", colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def _plot_split(ax: plt.Axes, csv_path: Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    if "target" not in df.columns or "predicted" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'target' and 'predicted' columns")
    y_true = df["target"].to_numpy(dtype=float)
    y_pred = df["predicted"].to_numpy(dtype=float)
    mae, rmse, r2 = _metrics(y_true, y_pred)

    ax.scatter(
        y_true,
        y_pred,
        s=28,
        facecolors="#74F5F8",
        edgecolors="black",
        linewidths=0.55,
        alpha=0.95,
        zorder=2,
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1.2, zorder=1)

    _style_axis(ax)
    ax.set_title(title, fontweight="bold", pad=10)

    text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\n$R^2$: {r2:.4f}"
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        family="sans-serif",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Truth vs Prediction panel plot from run prediction CSVs."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Run directory containing train_predictions.csv (and optionally val/test).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_SPECS.keys()),
        default=("train", "test"),
        help="Which splits to plot (default: train test — two panels like a train/test figure).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <results_dir>/pred_vs_truth_panels.png).",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI (default: 200)")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Not a directory: {results_dir}")

    panels: list[tuple[Path, str]] = []
    for key in args.splits:
        fname, title = SPLIT_SPECS[key]
        path = results_dir / fname
        if not path.is_file():
            raise SystemExit(f"Missing {fname} in {results_dir} (needed for --splits {key})")
        panels.append((path, title))

    n = len(panels)
    fig_w = 4.2 * n
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.2), squeeze=False)
    row = axes[0]
    for ax, (csv_path, title) in zip(row, panels):
        _plot_split(ax, csv_path, title)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.12, wspace=0.25)
    out = args.output or (results_dir / "pred_vs_truth_panels.png")
    out = out.resolve()
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
