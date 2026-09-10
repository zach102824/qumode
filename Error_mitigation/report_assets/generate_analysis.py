#!/usr/bin/env python3
"""Recompute the official gdr_param scoreboard and report figures.

This reads only Error_mitigation/out/results.json and uses the stored
gdr_param metrics for every cell.  It does not stitch later research
variants (gdr_damped, gdr_floor, gdr_select, span-twin replays).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "figures"
METHOD = "gdr_param"
# Matched mid-noise, strong-readout cells: joint TVD + energy-error
# improvement on optimized circuits. SNAP has three resolved peaks;
# ECD is a nearly pure |070> Fock state.
HISTOGRAM_CASES = (
    {
        "ansatz": "snap",
        "params": "optimized",
        "family": "loss_thermal_dephasing",
        "kappa_tau": 0.03,
        "readout": "readout_strong",
        "stem": "case_histogram_snap",
    },
    {
        "ansatz": "ecd",
        "params": "optimized",
        "family": "loss_thermal_dephasing",
        "kappa_tau": 0.03,
        "readout": "readout_strong",
        "stem": "case_histogram_ecd",
    },
)


def load_payload(relative: str = "out/results.json") -> dict:
    with (ROOT / relative).open() as handle:
        return json.load(handle)


def as_prob(values) -> np.ndarray:
    hist = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(hist.sum())
    if total <= 0.0:
        return np.full(hist.shape, 1.0 / hist.size, dtype=float)
    return hist / total


def match_case(case: dict, target: dict) -> bool:
    return (
        case["ansatz"] == target["ansatz"]
        and case["params"] == target["params"]
        and case["family"] == target["family"]
        and float(case["kappa_tau"]) == float(target["kappa_tau"])
        and case["readout"] == target["readout"]
    )


def select_histogram_case(cases: list[dict], target: dict) -> dict:
    for case in cases:
        if match_case(case, target):
            raw_tvd = float(case["metrics"]["raw"]["tvd"])
            mit_tvd = float(case["metrics"][METHOD]["tvd"])
            raw_de = float(case["metrics"]["raw"]["dE"])
            mit_de = float(case["metrics"][METHOD]["dE"])
            if not (mit_tvd < raw_tvd and mit_de < raw_de):
                raise RuntimeError(
                    f"histogram case {target['ansatz']} no longer shows joint improvement"
                )
            return case
    raise RuntimeError(f"histogram case {target['ansatz']} missing from official results")


def histogram_case_summary(case: dict) -> dict:
    p_true = as_prob(case["p_ideal"])
    return {
        "ansatz": case["ansatz"],
        "params": case["params"],
        "family": case["family"],
        "kappa_tau": float(case["kappa_tau"]),
        "readout": case["readout"],
        "raw_tvd": float(case["metrics"]["raw"]["tvd"]),
        "mit_tvd": float(case["metrics"][METHOD]["tvd"]),
        "raw_dE": float(case["metrics"]["raw"]["dE"]),
        "mit_dE": float(case["metrics"][METHOD]["dE"]),
        "n_significant_true_bins": int(np.sum(p_true > 0.02)),
        "true_peak_qnm": [int(v) for v in np.unravel_index(int(p_true.argmax()), p_true.shape)],
    }


def key(record: dict) -> tuple:
    return (
        record["ansatz"],
        record["params"],
        record["family"],
        float(record["kappa_tau"]),
        record["readout"],
    )


def metric(record: dict, method: str, name: str) -> float:
    value = record["metrics"][method][name]
    if value is None:
        raise ValueError(f"missing {name} for {method} in {key(record)}")
    return float(value)


def optional_metric(record: dict, method: str, name: str) -> float | None:
    block = record["metrics"].get(method)
    if not block:
        return None
    value = block.get(name)
    return None if value is None else float(value)


def build_rows(records: list[dict]) -> list[dict]:
    if len(records) != 108 or len({key(r) for r in records}) != 108:
        raise RuntimeError("official matrix must contain 108 unique cells")

    rows = []
    for record in records:
        readout_tvd = optional_metric(record, "readout_only", "tvd")
        oracle_tvd = optional_metric(record, "oracle_binomial", "tvd")
        zne_tvd = optional_metric(record, "zne_idle", "tvd")
        rows.append(
            {
                "ansatz": record["ansatz"],
                "params": record["params"],
                "family": record["family"],
                "kappa_tau": float(record["kappa_tau"]),
                "readout": record["readout"],
                "recipe": "official default twins + parametric GDR",
                "method": METHOD,
                "raw_tvd": metric(record, "raw", "tvd"),
                "mit_tvd": metric(record, METHOD, "tvd"),
                "raw_dE": metric(record, "raw", "dE"),
                "mit_dE": metric(record, METHOD, "dE"),
                "readout_only_tvd": readout_tvd,
                "oracle_tvd": oracle_tvd,
                "zne_tvd": zne_tvd,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    raw_tvd = np.array([r["raw_tvd"] for r in rows])
    mit_tvd = np.array([r["mit_tvd"] for r in rows])
    raw_de = np.array([r["raw_dE"] for r in rows])
    mit_de = np.array([r["mit_dE"] for r in rows])

    readout_pairs = [
        (r["mit_tvd"], r["readout_only_tvd"])
        for r in rows
        if r["readout_only_tvd"] is not None
    ]
    oracle_pairs = [
        (r["mit_tvd"], r["oracle_tvd"]) for r in rows if r["oracle_tvd"] is not None
    ]
    zne_pairs = [(r["mit_tvd"], r["zne_tvd"]) for r in rows if r["zne_tvd"] is not None]

    grouped = {}
    for field in ("ansatz", "params", "family", "kappa_tau", "readout"):
        grouped[field] = {}
        for value in sorted({r[field] for r in rows}, key=str):
            subset = [r for r in rows if r[field] == value]
            raw = np.array([r["raw_tvd"] for r in subset])
            mit = np.array([r["mit_tvd"] for r in subset])
            grouped[field][str(value)] = {
                "n": len(subset),
                "raw_mean_tvd": float(raw.mean()),
                "mitigated_mean_tvd": float(mit.mean()),
                "pooled_reduction_percent": float(100 * (1 - mit.sum() / raw.sum())),
                "tvd_improved": int(np.sum(mit < raw)),
                "tvd_worsened": int(np.sum(mit > raw)),
            }

    return {
        "sources": {
            "official": "out/results.json",
            "method": METHOD,
        },
        "n_cells": len(rows),
        "raw_mean_tvd": float(raw_tvd.mean()),
        "mitigated_mean_tvd": float(mit_tvd.mean()),
        "raw_median_tvd": float(np.median(raw_tvd)),
        "mitigated_median_tvd": float(np.median(mit_tvd)),
        "pooled_tvd_reduction_percent": float(100 * (1 - mit_tvd.sum() / raw_tvd.sum())),
        "median_cellwise_tvd_reduction_percent": float(
            100 * np.median((raw_tvd - mit_tvd) / raw_tvd)
        ),
        "tvd_improved": int(np.sum(mit_tvd < raw_tvd)),
        "tvd_tied": int(np.sum(mit_tvd == raw_tvd)),
        "tvd_worsened": int(np.sum(mit_tvd > raw_tvd)),
        "tvd_not_worse": int(np.sum(mit_tvd <= raw_tvd)),
        "beats_readout_only": int(sum(m < other for m, other in readout_pairs)),
        "n_readout_only": len(readout_pairs),
        "beats_oracle": int(sum(m < other for m, other in oracle_pairs)),
        "n_oracle": len(oracle_pairs),
        "beats_zne_idle": int(sum(m < other for m, other in zne_pairs)),
        "n_zne_idle": len(zne_pairs),
        "energy_error_improved": int(np.sum(mit_de < raw_de)),
        "energy_error_worsened": int(np.sum(mit_de > raw_de)),
        "raw_mean_abs_energy_error": float(raw_de.mean()),
        "mitigated_mean_abs_energy_error": float(mit_de.mean()),
        "raw_median_abs_energy_error": float(np.median(raw_de)),
        "mitigated_median_abs_energy_error": float(np.median(mit_de)),
        "grouped": grouped,
        "rows": rows,
    }


def selected_bins(p_true: np.ndarray, p_raw: np.ndarray, p_mit: np.ndarray, k: int = 8) -> list[tuple[int, int, int]]:
    """Keep the largest true peaks and the main raw-leakage bins."""
    score = np.maximum(np.maximum(p_true, p_raw), p_mit).ravel()
    order = np.argsort(score)[::-1][:k]
    return [tuple(int(v) for v in np.unravel_index(int(i), p_true.shape)) for i in order]


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def paired_tvd(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    colors = {"loss": "#2878B5", "loss_thermal_dephasing": "#DDAA33", "comprehensive": "#C44E52"}
    labels = {
        "loss": "Loss",
        "loss_thermal_dephasing": "Loss + thermal + dephasing",
        "comprehensive": "Comprehensive",
    }
    for family in colors:
        group = [r for r in rows if r["family"] == family]
        ax.scatter(
            [r["raw_tvd"] for r in group],
            [r["mit_tvd"] for r in group],
            s=22,
            alpha=0.78,
            color=colors[family],
            edgecolor="white",
            linewidth=0.35,
            label=f"{labels[family]} (n={len(group)})",
        )
    limit = max(max(r["raw_tvd"], r["mit_tvd"]) for r in rows) * 1.03
    ax.plot([0, limit], [0, limit], "--", color="0.35", linewidth=1, label="No change")
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Raw TVD", ylabel="Mitigated TVD")
    ax.set_title("All 108 trials: points below diagonal improve")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "paired_tvd.pdf", bbox_inches="tight")
    fig.savefig(OUT / "paired_tvd.png", bbox_inches="tight")
    plt.close(fig)


def by_noise_strength(rows: list[dict]) -> None:
    families = ["loss", "loss_thermal_dephasing", "comprehensive"]
    titles = ["Loss", "Loss + thermal + dephasing", "Comprehensive"]
    kappas = [0.003, 0.03, 0.1]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75), sharey=True)
    for ax, family, title in zip(axes, families, titles):
        raw, mit = [], []
        for kt in kappas:
            group = [r for r in rows if r["family"] == family and r["kappa_tau"] == kt]
            raw.append(np.mean([r["raw_tvd"] for r in group]))
            mit.append(np.mean([r["mit_tvd"] for r in group]))
        ax.plot(kappas, raw, "o-", color="0.45", label="Raw")
        ax.plot(kappas, mit, "o-", color="#2878B5", label="Parametric GDR")
        ax.set_xscale("log")
        ax.set_xticks(kappas, ["0.003", "0.03", "0.1"])
        ax.set_title(title)
        ax.set_xlabel(r"Per-application noise $\kappa\tau$")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Mean TVD across ansatz, parameter, readout")
    axes[0].legend(frameon=False)
    fig.suptitle("Mitigation remains beneficial as simulated noise increases", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "tvd_by_noise.pdf", bbox_inches="tight")
    fig.savefig(OUT / "tvd_by_noise.png", bbox_inches="tight")
    plt.close(fig)


def improvement_heatmap(rows: list[dict]) -> None:
    row_keys = [("ecd", "random"), ("ecd", "optimized"), ("snap", "random"), ("snap", "optimized")]
    families = ["loss", "loss_thermal_dephasing", "comprehensive"]
    kappas = [0.003, 0.03, 0.1]
    values = np.zeros((len(row_keys), len(families) * len(kappas)))
    for i, (ansatz, params) in enumerate(row_keys):
        for j, (family, kt) in enumerate((f, k) for f in families for k in kappas):
            group = [
                r
                for r in rows
                if r["ansatz"] == ansatz
                and r["params"] == params
                and r["family"] == family
                and r["kappa_tau"] == kt
            ]
            values[i, j] = 100 * (
                1 - sum(r["mit_tvd"] for r in group) / sum(r["raw_tvd"] for r in group)
            )
    fig, ax = plt.subplots(figsize=(7.4, 2.55))
    vmin = min(0.0, float(values.min()))
    vmax = max(70.0, float(values.max()))
    image = ax.imshow(values, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_yticks(range(4), ["ECD random", "ECD optimized", "SNAP random", "SNAP optimized"])
    ax.set_xticks(
        range(9),
        [f"{name}\n{kt:g}" for name in ("L", "T", "C") for kt in kappas],
    )
    ax.set_xlabel(
        r"Family (L=loss, T=thermal/dephasing, C=comprehensive) and $\kappa\tau$;"
        "\npooled over three readout levels"
    )
    ax.set_title("Pooled TVD reduction (%) — larger is better")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="TVD reduction (%)", fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "improvement_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "improvement_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def case_histogram(case: dict, stem: str) -> None:
    p_true = as_prob(case["p_ideal"])
    p_raw = as_prob(case["hists"]["raw"])
    p_mit = as_prob(case["hists"][METHOD])
    bins = selected_bins(p_true, p_raw, p_mit)
    labels = [rf"$|{q}{n}{m}\rangle$" for q, n, m in bins]
    series = (
        ("True", p_true, "#1A1A1A"),
        ("Raw noisy", p_raw, "#8A8A8A"),
        ("Mitigated after GDR", p_mit, "#2878B5"),
    )
    raw_tvd = float(case["metrics"]["raw"]["tvd"])
    mit_tvd = float(case["metrics"][METHOD]["tvd"])
    raw_de = float(case["metrics"]["raw"]["dE"])
    mit_de = float(case["metrics"][METHOD]["dE"])

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.85), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})
    x = np.arange(len(bins))
    width = 0.26
    ax = axes[0]
    for j, (name, hist, color) in enumerate(series):
        vals = [float(hist[q, n, m]) for q, n, m in bins]
        ax.bar(
            x + (j - 1) * width,
            vals,
            width=width,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title(r"Top $|q,n_1,n_2\rangle$ bins")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.98,
        0.62,
        f"TVD {raw_tvd:.3f}→{mit_tvd:.3f}\n"
        rf"$|\Delta E|$ {raw_de:.2f}→{mit_de:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "0.8", "pad": 3},
    )

    n1 = np.arange(p_true.shape[1])
    n2 = np.arange(p_true.shape[2])
    for axm, mode, ticks in (
        (axes[1], 1, n1),
        (axes[2], 2, n2),
    ):
        for name, hist, color in series:
            marg = hist.sum(axis=(0, 2)) if mode == 1 else hist.sum(axis=(0, 1))
            axm.plot(ticks, marg, "-o", color=color, lw=1.4, ms=3.5, label=name)
        axm.set_xlabel(rf"$n_{mode}$")
        axm.set_ylabel(rf"$P(n_{mode})$")
        axm.set_title(rf"Marginal $n_{mode}$")
        axm.set_xticks(ticks)
        axm.grid(alpha=0.18)
    axes[1].legend(frameon=False, loc="upper left")

    family = {
        "loss": "pure loss",
        "loss_thermal_dephasing": "loss + thermal + dephasing",
        "comprehensive": "comprehensive noise",
    }[case["family"]]
    readout = {
        "ideal": "ideal readout",
        "readout_realistic": "realistic readout error",
        "readout_strong": "strong readout error",
    }[case["readout"]]
    fig.suptitle(
        case.get(
            "figure_title",
            f"{case['ansatz'].upper()} {case['params']}, {family}, "
            rf"$\kappa\tau={float(case['kappa_tau']):g}$, {readout}",
        ),
        y=1.03,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def energy_tradeoff(rows: list[dict]) -> None:
    x = np.array([r["raw_tvd"] - r["mit_tvd"] for r in rows])
    y = np.array([r["raw_dE"] - r["mit_dE"] for r in rows])
    n = len(rows)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    colors = np.where(y >= 0, "#2878B5", "#C44E52")
    ax.scatter(x, y, s=22, c=colors, alpha=0.76, edgecolor="white", linewidth=0.35)
    ax.axhline(0, color="0.35", linewidth=0.9)
    ax.axvline(0, color="0.35", linewidth=0.9)
    ax.set_xlabel(r"TVD improvement = TVD$_{raw}$ − TVD$_{mit}$")
    ax.set_ylabel(r"Energy-error improvement = $|\Delta E|_{raw}-|\Delta E|_{mit}$")
    ax.set_title("Distribution recovery does not guarantee energy recovery")
    ax.grid(alpha=0.18)
    ax.text(
        0.98,
        0.04,
        f"Energy error improves: {int(np.sum(y > 0))}/{n}\n"
        f"Energy error worsens: {int(np.sum(y < 0))}/{n}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "pad": 3},
    )
    fig.tight_layout()
    fig.savefig(OUT / "energy_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(OUT / "energy_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    payload = load_payload()
    rows = build_rows(payload["records"])
    summary = summarize(rows)
    chosen_cases = []
    for target in HISTOGRAM_CASES:
        chosen = select_histogram_case(payload["cases"], target)
        chosen_cases.append((chosen, target["stem"]))
    summary["histogram_cases"] = [histogram_case_summary(case) for case, _ in chosen_cases]
    (HERE / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    paired_tvd(summary["rows"])
    by_noise_strength(summary["rows"])
    improvement_heatmap(summary["rows"])
    for chosen, stem in chosen_cases:
        case_histogram(chosen, stem)
    energy_tradeoff(summary["rows"])
    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "grouped"}}, indent=2))


if __name__ == "__main__":
    main()
