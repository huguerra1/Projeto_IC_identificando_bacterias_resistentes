#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def try_load(path: Path) -> np.ndarray:
    """Load CSV or NPY file with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".npy":
        return np.load(path)
    else:
        return np.loadtxt(path, delimiter=",", skiprows=1, usecols=1)


def plot_band_importance(importance: np.ndarray,
                         selected_indices: np.ndarray | None,
                         output_path: Path) -> None:
    """Plot spectral band importance with optional selected band markers."""
    plt.figure(figsize=(10, 5))
    plt.plot(importance, label="RF Importance", color="blue", lw=2)
    if selected_indices is not None:
        plt.scatter(selected_indices,
                    importance[selected_indices],
                    color="red",
                    label="Selected Bands",
                    zorder=5)
    plt.xlabel("Band Index")
    plt.ylabel("Importance")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Random Forest spectral band importance."
    )
    parser.add_argument(
        "--importance",
        type=Path,
        default=Path("output_analysis/rf_importance_selected.csv"),  # corrigido aqui
        help="Path to CSV/NPY with RF importance scores."
    )
    parser.add_argument(
        "--selected",
        type=Path,
        default=None,
        help="Optional path to NPY file with selected band indices."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/band_scores_rf_en.png"),
        help="Output path for the plot."
    )

    parser.add_argument(
    "--importance",
    type=Path,
    default=Path("output_analysis/rf_importances_selected.csv"),  # corrigido o nome
    help="Path to CSV/NPY with RF importance scores."
)


    band_importance = try_load(args.importance)
    selected_indices = None
    if args.selected and args.selected.exists():
        selected_indices = np.load(args.selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_band_importance(band_importance, selected_indices, args.output)


if __name__ == "__main__":
    main()
