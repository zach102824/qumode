"""Paper-style figures (Figs. 4, 5, 8, 9, 14)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .hamiltonian import TARGET_QNM


def _energy_xy(
    history: list[dict],
    start_iteration: int,
    stride: int,
) -> tuple[list[int], list[float]]:
    """Sample the paper's logged grid: iterations 10, 20, …, 200."""
    by_iter = {int(h["iteration"]): float(h["energy_physical"]) for h in history}
    if not by_iter:
        return [], []
    max_it = max(by_iter)
    start = max(int(start_iteration), min(by_iter))
    step = max(int(stride), 1)
    xs: list[int] = []
    ys: list[float] = []
    it = start
    while it <= max_it:
        nearest = min(by_iter, key=lambda k, target=it: abs(k - target))
        xs.append(int(it))
        ys.append(by_iter[nearest])
        it += step
    if max_it not in xs:
        xs.append(max_it)
        ys.append(by_iter[max_it])
    return xs, ys


def plot_energy_overlay(
    series: dict[str, list[dict]],
    path: Path,
    exact: float = -12.0,
    ylabel: str = "Trial energy (a.u.)",
    title: str = "",
    *,
    start_iteration: int = 10,
    stride: int = 10,
    styles: dict[str, dict] | None = None,
) -> None:
    """Paper-style overlay of several BFGS energy histories."""
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    default_cycle = (
        {"marker": "o", "color": "green"},
        {"marker": "s", "color": "darkorange"},
        {"marker": "x", "color": "royalblue"},
        {"marker": "^", "color": "crimson"},
    )
    for i, (label, history) in enumerate(series.items()):
        style = dict(default_cycle[i % len(default_cycle)])
        if styles and label in styles:
            style.update(styles[label])
        iters, energies = _energy_xy(history, start_iteration, stride)
        ax.plot(
            iters,
            energies,
            linestyle="-",
            markersize=7,
            label=label,
            **style,
        )
    ax.axhline(exact, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    xmax = 200
    for history in series.values():
        if history:
            xmax = max(xmax, max(int(h["iteration"]) for h in history))
    ax.set_xlim(0, xmax + 5)
    ax.legend(frameon=False, fontsize=12)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_energy_history(
    history: list[dict],
    path: Path,
    exact: float = -12.0,
    ylabel: str = "Trial energy (a.u.)",
    title: str = "",
    *,
    start_iteration: int = 10,
    stride: int = 10,
    marker: str = "o",
    color: str = "green",
) -> None:
    """Match ``qumode_data_reduced.ipynb``: markers every 10 iterations from 10."""
    iters, energies = _energy_xy(history, start_iteration, stride)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.plot(iters, energies, marker=marker, linestyle="-", color=color, markersize=7)
    ax.axhline(exact, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(0, max(iters, default=200) + 5)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _qnm_label(qnm: tuple[int, int, int]) -> str:
    q, n, m = qnm
    return rf"$|{q},{n},{m}\rangle$"


def top_k_basis_states(prob_arrays: list[np.ndarray], k: int = 5) -> list[tuple[int, int, int]]:
    """Rank computational states by the largest population in any of the arrays."""
    stacked = np.stack([np.asarray(p, dtype=float) for p in prob_arrays], axis=0)
    score = stacked.max(axis=0)
    order = np.argsort(score, axis=None)[::-1][: int(k)]
    return [tuple(int(v) for v in np.unravel_index(int(i), score.shape)) for i in order]


def plot_population_comparison(
    snapshots: dict[int, dict[str, np.ndarray]],
    path: Path,
    k: int = 5,
    colors: dict[str, str] | None = None,
    highlight: tuple[int, int, int] = TARGET_QNM,
    ylabel: str = r"$S_{q,n,m}$",
    title: str = "",
) -> None:
    """Fig. 5-style grouped bars: top-``k`` states at selected iterations."""
    iterations = sorted(snapshots)
    n = len(iterations)
    ncols = n if n <= 3 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.9 * nrows), sharey=True, squeeze=False
    )
    default_colors = {
        "noiseless": "green",
        r"paper photon loss ($\kappa\tau\simeq 0.003$)": "darkorange",
        "typical device": "royalblue",
    }
    palette = colors or default_colors
    handles = None
    legend_labels: list[str] = []
    for ax, it in zip(axes.ravel(), iterations):
        series = snapshots[it]
        labels = list(series)
        arrays = [np.asarray(series[lab], dtype=float) for lab in labels]
        states = top_k_basis_states(arrays, k=k)
        x = np.arange(len(states))
        width = 0.72 / max(len(labels), 1)
        for i, lab in enumerate(labels):
            heights = [float(series[lab][q, n_ph, m]) for q, n_ph, m in states]
            offset = (i - 0.5 * (len(labels) - 1)) * width
            bars = ax.bar(
                x + offset,
                heights,
                width=width * 0.95,
                color=palette.get(lab, f"C{i}"),
                label=lab,
                zorder=2,
            )
            if highlight in states:
                hi = states.index(highlight)
                bars[hi].set_edgecolor("black")
                bars[hi].set_linewidth(0.8)
        ax.set_xticks(x, [_qnm_label(s) for s in states], rotation=40, ha="right", fontsize=10)
        ax.set_title(f"Iteration {it}", fontsize=13)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        if handles is None:
            handles, legend_labels = ax.get_legend_handles_labels()
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].set_ylabel(ylabel, fontsize=14)
    if nrows > 1:
        axes[1, 0].set_ylabel(ylabel, fontsize=14)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        fontsize=11,
    )
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92 if nrows == 1 else 0.93))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _states_above_threshold(probs: np.ndarray, threshold: float):
    labels = []
    heights = []
    qnms = []
    for q in range(probs.shape[0]):
        for n in range(probs.shape[1]):
            for m in range(probs.shape[2]):
                p = float(probs[q, n, m])
                if p >= threshold:
                    qnms.append((q, n, m))
                    heights.append(p)
                    labels.append(rf"$|{q},{n},{m}\rangle$")
    return qnms, heights, labels


def plot_histogram(
    probs: np.ndarray,
    path: Path,
    title: str = "",
    threshold: float = 0.01,
    highlight: tuple[int, int, int] = TARGET_QNM,
) -> None:
    qnms, heights, labels = _states_above_threshold(probs, threshold)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.38 * max(len(labels), 1)), 3.8))
    colors = ["C3" if t == highlight else "C0" for t in qnms]
    ax.bar(range(len(heights)), heights, color=colors)
    ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel(r"$S_{q,n,m}$")
    ax.set_title(title or r"$S_{q,n,m}$")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_iteration_histograms(
    snapshots: dict[int, np.ndarray],
    path: Path,
    highlight: tuple[int, int, int],
    threshold: float = 0.02,
    title: str = "",
) -> None:
    """Fig. 5 / Fig. 9 style: probability histograms at selected iterations."""
    items = sorted(snapshots.items())
    n = len(items)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.4 * nrows), squeeze=False)
    for ax, (it, probs) in zip(axes.ravel(), items):
        qnms, heights, labels = _states_above_threshold(probs, threshold)
        colors = ["C3" if t == highlight else "steelblue" for t in qnms]
        ax.bar(range(len(heights)), heights, color=colors, width=0.8)
        ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel(r"$S_{q,n,m}$")
        ax.set_title(f"Iteration {it}")
        ax.set_ylim(0.0, max(1.0, max(heights, default=1.0) * 1.05))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_kappa_overlay(
    series: dict[str, np.ndarray],
    path: Path,
    threshold: float = 0.02,
    highlight: tuple[int, int, int] = TARGET_QNM,
    title: str = "",
) -> None:
    """Fig. 14 style overlay of photon-loss histograms."""
    keys = list(series.keys())
    ref = next(iter(series.values()))
    labels = []
    indices = []
    for q in range(ref.shape[0]):
        for n in range(ref.shape[1]):
            for m in range(ref.shape[2]):
                if any(float(series[k][q, n, m]) >= threshold for k in keys):
                    indices.append((q, n, m))
                    labels.append(rf"$|{q},{n},{m}\rangle$")
    x = np.arange(len(indices))
    n_series = max(len(keys), 1)
    width = 0.78 / n_series
    fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * max(len(labels), 1)), 4.2))
    for i, key in enumerate(keys):
        heights = [float(series[key][q, n, m]) for q, n, m in indices]
        offset = (i - 0.5 * (n_series - 1)) * width
        ax.bar(x + offset, heights, width=width * (0.7 + 0.3 * i / n_series), label=key)
    ax.set_xticks(x, labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel(r"$S_{q,n,m}$")
    ax.set_title(title or "Photon-loss sweep after 80 iterations")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
