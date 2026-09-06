#!/usr/bin/env python3
"""Bar chart: PR #6 gdr_param vs adaptive select on headline cells."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Error_mitigation" / "out_research" / "figures"

# Single-draw TVDs from PR #6 out/ and the adaptive hybrid (NOTEBOOK / adaptive_recipe.md).
# Bootstrap σ on select where leftover_bootstrap ran (8×8192).
CELLS = [
    ("ECD rand loss 0.1", 0.298, 0.373, 0.203, 0.012),
    ("ECD rand comp. 0.1", 0.403, 0.539, 0.342, 0.013),
    ("ECD opt comp. 0.1", 0.909, 0.343, 0.343, 0.008),
    ("SNAP rand comp. 0.003", 0.037, 0.045, 0.0369, 0.004),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = [c[0] for c in CELLS]
    raw = np.array([c[1] for c in CELLS])
    base = np.array([c[2] for c in CELLS])
    adapt = np.array([c[3] for c in CELLS])
    yerr = np.array([c[4] for c in CELLS])
    x = np.arange(len(CELLS))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - w, raw, w, label="raw", color="0.65")
    ax.bar(x, base, w, label="PR #6 gdr_param", color="#2980b9")
    ax.bar(x + w, adapt, w, label="adaptive select", color="#2c3e50", yerr=yerr, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("TVD (lower is better)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Hard cells: PR #6 gdr_param vs adaptive recipe")
    ax.text(
        0.0,
        -0.22,
        "Select error bars: bootstrap σ from 8×8192 resamples (Phase 9).",
        transform=ax.transAxes,
        fontsize=7,
        color="0.35",
    )
    fig.tight_layout()
    path = OUT / "hard_cells_adaptive.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
