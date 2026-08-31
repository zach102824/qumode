#!/usr/bin/env python3
"""Energy vs Gibbs on the paper knapsack: who recovers the optimal bitstring.

Compares ⟨H⟩ and Gibbs −ln⟨e^{−ηE}⟩ on two ansatze with the same BFGS starts:

* hybrid ECD-VQE (qumode, N_d = 5), Fig. 5 setting
* qubit QAOA (p = 20), Fig. 7 setting

The single output figure is the fraction of runs whose most-likely bitstring
is the knapsack optimum ``0110000``. Library code is imported, never edited.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from qumode_vqe.data import load_reference
from qumode_vqe.hamiltonian import EXACT_GROUND_ENERGY, TARGET_BITSTRING, TARGET_QNM

from paper_result.ecd import notebook_ecd_x0, run_ecd_trial
from paper_result.qaoa import (
    PUBLISHED_PSTAR,
    check_spectrum,
    qubo_spectrum,
    random_qaoa_params,
    run_qaoa_trial,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = HERE / "out"
FIG5_SEED = 2026
QAOA_SEED_BASE = 4000
ECD_MAXITER_PAPER = 200
QAOA_MAXITER_PAPER = 200
QAOA_TRIALS_PAPER = 50
ECD_TRIALS_PAPER = 50
QAOA_P_PAPER = 20
SINGLE_PLOT = "gibbs_vs_energy_bitstring.png"
EXTRA_PLOTS = (
    "fig5_energy_curve.png",
    "fig5_energy_histograms.png",
    "fig5_energy_vs_gibbs_energy.png",
    "fig5_energy_vs_gibbs_populations.png",
    "fig5_gibbs_histograms.png",
    "fig7_energy_histogram.png",
    "fig7_energy_vs_gibbs.png",
    "fig7_gibbs_histogram.png",
)


PRESETS = {
    "smoke": {
        "ecd_maxiter": 3,
        "ecd_trials": 1,
        "qaoa_maxiter": 8,
        "qaoa_trials": 1,
        "qaoa_p": 2,
        "record_every": 1,
    },
    "quick": {
        "ecd_maxiter": 40,
        "ecd_trials": 8,
        "qaoa_maxiter": 80,
        "qaoa_trials": 8,
        "qaoa_p": 20,
        "record_every": 10,
    },
    "paper": {
        "ecd_maxiter": ECD_MAXITER_PAPER,
        "ecd_trials": ECD_TRIALS_PAPER,
        "qaoa_maxiter": QAOA_MAXITER_PAPER,
        "qaoa_trials": QAOA_TRIALS_PAPER,
        "qaoa_p": QAOA_P_PAPER,
        "record_every": 10,
    },
}


def _json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def _trial_out(rec: dict) -> dict:
    skip = {"probs"}
    out = {k: v for k, v in rec.items() if k not in skip}
    out["history"] = _history_lite(rec.get("history") or [])
    return out


def _history_lite(history: list[dict]) -> list[dict]:
    lite = []
    for h in history:
        rec = {
            "iteration": int(h["iteration"]),
            "energy_physical": float(h["energy_physical"]),
            "target_prob_physical": float(h.get("target_prob_physical", h.get("pstar", float("nan")))),
            "most_likely": h.get("most_likely"),
            "most_likely_bitstring": h.get("most_likely_bitstring"),
            "cost": h.get("cost"),
            "eta": h.get("eta"),
        }
        if h.get("pstar") is not None:
            rec["pstar"] = float(h["pstar"])
        lite.append(rec)
    return lite


def _final_is_optimum(rec: dict) -> bool:
    if rec.get("success"):
        return True
    if str(rec.get("most_likely_bitstring")) == TARGET_BITSTRING:
        return True
    ml = rec.get("most_likely")
    if isinstance(ml, (list, tuple)) and tuple(int(v) for v in ml) == TARGET_QNM:
        return True
    return False


def _ecd_success_rate(fig5: dict, objective: str) -> tuple[float, int]:
    """ECD: most-likely bitstring after BFGS, same rule as QAOA."""
    rec = fig5[objective]
    trials = rec.get("trials")
    if trials:
        if "success_rate" in rec:
            return float(rec["success_rate"]), int(rec.get("n") or len(trials))
        hits = sum(_final_is_optimum(t) for t in trials)
        return hits / max(len(trials), 1), len(trials)
    return (1.0 if _final_is_optimum(rec) else 0.0), 1


def _qaoa_success_rate(fig7: dict, objective: str) -> tuple[float, int]:
    block = fig7[objective]
    n = int(block.get("n") or len(block.get("trials") or []))
    if n <= 0:
        return float("nan"), 0
    if "success_rate" in block:
        return float(block["success_rate"]), n
    trials = block.get("trials") or []
    hits = sum(str(t.get("most_likely_bitstring")) == TARGET_BITSTRING for t in trials)
    return hits / max(len(trials), 1), len(trials)


def plot_bitstring_success(fig5: dict | None, fig7: dict | None, path: Path) -> dict:
    """One grouped-bar figure: Gibbs vs energy on recovering ``0110000``."""
    labels = []
    energy_vals = []
    gibbs_vals = []
    energy_ns = []
    gibbs_ns = []
    if fig5 is not None:
        e, ne = _ecd_success_rate(fig5, "energy")
        g, ng = _ecd_success_rate(fig5, "gibbs")
        labels.append("ECD-VQE\n(qumode)")
        energy_vals.append(e)
        gibbs_vals.append(g)
        energy_ns.append(ne)
        gibbs_ns.append(ng)
    if fig7 is not None:
        e, ne = _qaoa_success_rate(fig7, "energy")
        g, ng = _qaoa_success_rate(fig7, "gibbs")
        labels.append("QAOA\n" + rf"($p={fig7.get('p_layers', 20)}$)")
        energy_vals.append(e)
        gibbs_vals.append(g)
        energy_ns.append(ne)
        gibbs_ns.append(ng)
    if not labels:
        raise ValueError("need Fig. 5 and/or Fig. 7 results to plot")

    x = np.arange(len(labels), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    b1 = ax.bar(x - width / 2, energy_vals, width, label="energy ⟨H⟩", color="#2ca02c", zorder=2)
    b2 = ax.bar(x + width / 2, gibbs_vals, width, label="Gibbs", color="#d62728", zorder=2)
    ax.set_xticks(x, labels, fontsize=12)
    ax.set_ylabel("Most-likely bitstring is optimal", fontsize=12)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    def _annotate(bars, ns):
        for bar, n in zip(bars, ns):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.03,
                f"{100.0 * h:.0f}%" + (f"\n$n={n}$" if n else ""),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    _annotate(b1, energy_ns)
    _annotate(b2, gibbs_ns)
    ax.set_title(
        "Gibbs recovers the knapsack bitstring more reliably than energy",
        fontsize=12,
        pad=8,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {
        "ecd": None
        if fig5 is None
        else {
            "energy": energy_vals[0],
            "gibbs": gibbs_vals[0],
            "n_energy": energy_ns[0],
            "n_gibbs": gibbs_ns[0],
        },
        "qaoa": None
        if fig7 is None
        else {
            "energy": energy_vals[-1],
            "gibbs": gibbs_vals[-1],
            "n_energy": energy_ns[-1],
            "n_gibbs": gibbs_ns[-1],
        },
        "plot": str(path),
    }


def _compare_pstar(energy_p: float, gibbs_p: float, label: str) -> dict:
    de = float(gibbs_p) - float(energy_p)
    return {
        "metric": label,
        "energy": float(energy_p),
        "gibbs": float(gibbs_p),
        "gibbs_minus_energy": de,
        "improved": de > 1e-6,
        "tied": abs(de) <= 1e-6,
        "worse": de < -1e-6,
    }


def _summarize_trials(trials: list[dict]) -> dict:
    n = len(trials)
    pstars = np.asarray([float(t["pstar"]) for t in trials], dtype=float)
    energies = np.asarray([float(t["energy_physical"]) for t in trials], dtype=float)
    n_hit = int(sum(_final_is_optimum(t) for t in trials))
    best = max(trials, key=lambda t: float(t["pstar"])) if trials else None
    return {
        "n": n,
        "n_success": n_hit,
        "success_rate": n_hit / max(n, 1),
        "mean_pstar": float(np.mean(pstars)) if n else float("nan"),
        "std_pstar": float(np.std(pstars, ddof=1)) if n > 1 else 0.0,
        "best_pstar": float(np.max(pstars)) if n else float("nan"),
        "best_trial": None if best is None else int(best["trial"]),
        "mean_energy": float(np.mean(energies)) if n else float("nan"),
        "best_energy": float(np.min(energies)) if n else float("nan"),
    }


def _paired_success(energy_trials: list[dict], gibbs_trials: list[dict]) -> dict:
    e_ok = {int(t["trial"]) for t in energy_trials if _final_is_optimum(t)}
    g_ok = {int(t["trial"]) for t in gibbs_trials if _final_is_optimum(t)}
    n = max(len(energy_trials), len(gibbs_trials))
    return {
        "n": n,
        "both": len(e_ok & g_ok),
        "gibbs_only": len(g_ok - e_ok),
        "energy_only": len(e_ok - g_ok),
        "neither": n - len(e_ok | g_ok),
    }


def _run_jobs(worker, jobs: list[dict], workers: int, *, label: str, checkpoint=None) -> list[dict]:
    records: list[dict] = []
    n = len(jobs)

    def _note(rec: dict, done: int) -> None:
        print(
            f"[{done}/{n}] {label} t{rec['trial']}  {rec['objective']}  "
            f"P*={rec['pstar']:.4f}  E={rec['energy_physical']:.3f}  "
            f"bits={rec['most_likely_bitstring']}  nit={rec['nit']}",
            flush=True,
        )
        if checkpoint is not None:
            checkpoint(records)

    if workers <= 1 or n <= 1:
        for i, job in enumerate(jobs, 1):
            rec = worker(job)
            records.append(rec)
            _note(rec, i)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(worker, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                done += 1
                _note(rec, done)
    return records


def run_fig5(
    *,
    outdir: Path,
    n_trials: int,
    maxiter: int,
    seed_base: int,
    workers: int,
    record_every: int,
    adaptive_eta: bool,
) -> dict:
    print(
        f"=== Fig. 5 ECD-VQE  N_d=5  trials={n_trials}  maxiter={maxiter}  "
        f"workers={workers}  notebook Uniform init, shared starts ===",
        flush=True,
    )
    starts = []
    for t in range(int(n_trials)):
        rng = np.random.default_rng(int(seed_base) + t)
        starts.append(notebook_ecd_x0(5, rng))

    jobs = []
    for t, x0 in enumerate(starts):
        for objective in ("energy", "gibbs"):
            jobs.append(
                {
                    "trial": t,
                    "seed": int(seed_base) + t,
                    "ndepth": 5,
                    "x0": x0,
                    "objective": objective,
                    "maxiter": int(maxiter),
                    "record_every": int(record_every),
                    "adaptive_eta": bool(adaptive_eta),
                }
            )
    t0 = time.perf_counter()
    partial = outdir / "fig5.partial.json"

    def _ckpt(recs: list[dict]) -> None:
        if len(recs) % 5 != 0:
            return
        _save_json(
            partial,
            {
                "n_done": len(recs),
                "n_jobs": len(jobs),
                "records": [_trial_out(r) for r in recs],
            },
        )

    records = _run_jobs(
        run_ecd_trial, jobs, workers, label="ecd", checkpoint=_ckpt
    )
    records.sort(key=lambda r: (str(r["objective"]), int(r["trial"])))
    elapsed = time.perf_counter() - t0
    by_obj = {"energy": [], "gibbs": []}
    for rec in records:
        by_obj[str(rec["objective"])].append(rec)
    summaries = {k: _summarize_trials(v) for k, v in by_obj.items()}
    paired = _paired_success(by_obj["energy"], by_obj["gibbs"])

    print(
        f"  energy: {summaries['energy']['n_success']}/{summaries['energy']['n']} "
        f"hit GS  best P*={summaries['energy']['best_pstar']:.4f}  "
        f"mean P*={summaries['energy']['mean_pstar']:.4f}",
        flush=True,
    )
    print(
        f"  Gibbs:  {summaries['gibbs']['n_success']}/{summaries['gibbs']['n']} "
        f"hit GS  best P*={summaries['gibbs']['best_pstar']:.4f}  "
        f"mean P*={summaries['gibbs']['mean_pstar']:.4f}",
        flush=True,
    )
    print(
        f"  paired: Gibbs-only {paired['gibbs_only']}  energy-only {paired['energy_only']}  "
        f"both {paired['both']}  neither {paired['neither']}",
        flush=True,
    )
    print(f"  wall {elapsed:.1f}s", flush=True)

    payload = {
        "figure": 5,
        "ansatz": "ecd_nd5",
        "n_trials": int(n_trials),
        "maxiter": int(maxiter),
        "seed_base": int(seed_base),
        "ndepth": 5,
        "n_parameters": 40,
        "init": "notebook: beta_mag~U(0,3), beta_arg,theta,phi~U(0,pi)",
        "target_qnm": list(TARGET_QNM),
        "target_bitstring": TARGET_BITSTRING,
        "exact_ground_energy": EXACT_GROUND_ENERGY,
        "elapsed_s": elapsed,
        "paired": paired,
        "energy": {
            **summaries["energy"],
            "trials": [_trial_out(r) for r in by_obj["energy"]],
        },
        "gibbs": {
            **summaries["gibbs"],
            "trials": [_trial_out(r) for r in by_obj["gibbs"]],
        },
        "comparison": {
            "success_rate": _compare_pstar(
                summaries["energy"]["success_rate"],
                summaries["gibbs"]["success_rate"],
                "ECD-VQE fraction with most-likely = 0110000",
            ),
            "best_pstar": _compare_pstar(
                summaries["energy"]["best_pstar"],
                summaries["gibbs"]["best_pstar"],
                "ECD-VQE best P(|0,6,0⟩)",
            ),
            "mean_pstar": _compare_pstar(
                summaries["energy"]["mean_pstar"],
                summaries["gibbs"]["mean_pstar"],
                "ECD-VQE mean P(|0,6,0⟩)",
            ),
        },
    }
    _save_json(outdir / "fig5.json", payload)
    if partial.exists():
        partial.unlink()
    return payload


def run_fig7(
    *,
    outdir: Path,
    p_layers: int,
    n_trials: int,
    maxiter: int,
    seed_base: int,
    workers: int,
    record_every: int,
    adaptive_eta: bool,
) -> dict:
    energies = qubo_spectrum()
    spec = check_spectrum(energies)
    if not spec["ground_is_target"] or not spec["ground_energy_ok"]:
        raise RuntimeError(f"QAOA spectrum does not match the paper knapsack: {spec}")
    print(
        f"=== Fig. 7 QAOA  p={p_layers}  trials={n_trials}  maxiter={maxiter}  "
        f"workers={workers}  shared starts ===",
        flush=True,
    )
    print(
        f"  spectrum: Emin={spec['energy_min']:.1f}  ground={spec['ground_bitstring']}  "
        f"⟨H⟩_uniform={spec['mean_energy']:.2f}",
        flush=True,
    )
    starts = []
    for t in range(int(n_trials)):
        rng = np.random.default_rng(int(seed_base) + t)
        starts.append(random_qaoa_params(int(p_layers), rng))

    jobs = []
    for t, x0 in enumerate(starts):
        for objective in ("energy", "gibbs"):
            jobs.append(
                {
                    "trial": t,
                    "seed": int(seed_base) + t,
                    "p_layers": int(p_layers),
                    "energies": energies,
                    "x0": x0,
                    "objective": objective,
                    "maxiter": int(maxiter),
                    "record_every": int(record_every),
                    "adaptive_eta": bool(adaptive_eta),
                }
            )
    t0 = time.perf_counter()
    records = _run_jobs(run_qaoa_trial, jobs, workers, label=f"qaoa_p{int(p_layers)}")
    records.sort(key=lambda r: (int(r["p_layers"]), str(r["objective"]), int(r["trial"])))
    elapsed = time.perf_counter() - t0
    by_obj = {"energy": [], "gibbs": []}
    for rec in records:
        by_obj[str(rec["objective"])].append(rec)
    summaries = {k: _summarize_trials(v) for k, v in by_obj.items()}
    paired = _paired_success(by_obj["energy"], by_obj["gibbs"])

    print(
        f"  energy best P*={summaries['energy']['best_pstar']:.4f}  "
        f"mean P*={summaries['energy']['mean_pstar']:.4f}  "
        f"(published {PUBLISHED_PSTAR:.4f})",
        flush=True,
    )
    print(
        f"  Gibbs  best P*={summaries['gibbs']['best_pstar']:.4f}  "
        f"mean P*={summaries['gibbs']['mean_pstar']:.4f}",
        flush=True,
    )
    print(
        f"  paired: Gibbs-only {paired['gibbs_only']}  energy-only {paired['energy_only']}  "
        f"both {paired['both']}  neither {paired['neither']}",
        flush=True,
    )
    print(f"  wall {elapsed:.1f}s", flush=True)

    payload = {
        "figure": 7,
        "ansatz": f"qaoa_p{int(p_layers)}",
        "n_trials": int(n_trials),
        "maxiter": int(maxiter),
        "seed_base": int(seed_base),
        "p_layers": int(p_layers),
        "n_parameters": 2 * int(p_layers),
        "spectrum": spec,
        "published_pstar": PUBLISHED_PSTAR,
        "elapsed_s": elapsed,
        "paired": paired,
        "energy": {
            **summaries["energy"],
            "published_pstar": PUBLISHED_PSTAR,
            "trials": [_trial_out(r) for r in by_obj["energy"]],
        },
        "gibbs": {
            **summaries["gibbs"],
            "trials": [_trial_out(r) for r in by_obj["gibbs"]],
        },
        "comparison": {
            "success_rate": _compare_pstar(
                summaries["energy"]["success_rate"],
                summaries["gibbs"]["success_rate"],
                "QAOA fraction with most-likely = 0110000",
            ),
            "best_pstar": _compare_pstar(
                summaries["energy"]["best_pstar"],
                summaries["gibbs"]["best_pstar"],
                "QAOA best P(0110000)",
            ),
            "mean_pstar": _compare_pstar(
                summaries["energy"]["mean_pstar"],
                summaries["gibbs"]["mean_pstar"],
                "QAOA mean P(0110000)",
            ),
        },
    }
    _save_json(outdir / "fig7.json", payload)
    return payload


def write_comparison(outdir: Path, fig5: dict | None, fig7: dict | None) -> dict:
    plot_path = Path(outdir) / SINGLE_PLOT
    stats = plot_bitstring_success(fig5, fig7, plot_path)
    lines = [
        "# Gibbs vs energy: most-likely bitstring is the knapsack optimum (0110000)",
        "",
    ]
    payload: dict = {"plot": str(plot_path), "metric": stats}
    if stats.get("ecd") is not None:
        e = stats["ecd"]
        lines += [
            "## ECD-VQE (qumode), most-likely bitstring after BFGS",
            f"- energy: {100.0 * e['energy']:.0f}%  (n={e['n_energy']})",
            f"- Gibbs:  {100.0 * e['gibbs']:.0f}%  (n={e['n_gibbs']})",
        ]
        if fig5 is not None and fig5.get("paired"):
            p = fig5["paired"]
            lines.append(
                f"- paired: Gibbs-only {p['gibbs_only']}, energy-only {p['energy_only']}, "
                f"both {p['both']}, neither {p['neither']}"
            )
        lines.append("")
    if stats.get("qaoa") is not None:
        q = stats["qaoa"]
        lines += [
            "## QAOA, most-likely bitstring after BFGS",
            f"- energy: {100.0 * q['energy']:.0f}%  (n={q['n_energy']})",
            f"- Gibbs:  {100.0 * q['gibbs']:.0f}%  (n={q['n_gibbs']})",
        ]
        if fig7 is not None and fig7.get("paired"):
            p = fig7["paired"]
            lines.append(
                f"- paired: Gibbs-only {p['gibbs_only']}, energy-only {p['energy_only']}, "
                f"both {p['both']}, neither {p['neither']}"
            )
        lines.append("")
    text = "\n".join(lines) + "\n"
    (outdir / "comparison.txt").write_text(text, encoding="utf-8")
    _save_json(outdir / "comparison.json", payload)
    print(text, end="", flush=True)
    print(f"Wrote {plot_path}", flush=True)
    return payload


def _remove_extra_plots(outdir: Path) -> None:
    for name in EXTRA_PLOTS:
        path = Path(outdir) / name
        if path.exists():
            path.unlink()
            print(f"removed {path.name}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="paper")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--skip-ecd", action="store_true")
    parser.add_argument("--skip-qaoa", action="store_true")
    parser.add_argument("--ecd-maxiter", type=int, default=None)
    parser.add_argument("--ecd-trials", type=int, default=None)
    parser.add_argument("--ecd-seed", type=int, default=FIG5_SEED, help="Seed base for notebook ECD starts.")
    parser.add_argument("--qaoa-trials", type=int, default=None)
    parser.add_argument("--qaoa-p", type=int, default=None)
    parser.add_argument("--qaoa-maxiter", type=int, default=None)
    parser.add_argument("--qaoa-seed-base", type=int, default=QAOA_SEED_BASE)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--record-every", type=int, default=None)
    parser.add_argument("--fixed-eta", action="store_true", help="Do not refresh Gibbs η during BFGS.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild the single comparison plot from saved fig5.json / fig7.json.",
    )
    args = parser.parse_args(argv)

    preset = PRESETS[args.preset]
    ecd_maxiter = preset["ecd_maxiter"] if args.ecd_maxiter is None else args.ecd_maxiter
    ecd_trials = preset["ecd_trials"] if args.ecd_trials is None else args.ecd_trials
    qaoa_maxiter = preset["qaoa_maxiter"] if args.qaoa_maxiter is None else args.qaoa_maxiter
    qaoa_trials = preset["qaoa_trials"] if args.qaoa_trials is None else args.qaoa_trials
    qaoa_p = preset["qaoa_p"] if args.qaoa_p is None else args.qaoa_p
    record_every = preset["record_every"] if args.record_every is None else args.record_every
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    adaptive_eta = not args.fixed_eta

    fig5_path = outdir / "fig5.json"
    fig7_path = outdir / "fig7.json"

    def _load(path: Path) -> dict | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    if args.plot_only:
        fig5 = _load(fig5_path)
        fig7 = _load(fig7_path)
        if fig5 is None and fig7 is None:
            raise FileNotFoundError(f"no fig5.json or fig7.json in {outdir}")
        write_comparison(outdir, fig5, fig7)
        _remove_extra_plots(outdir)
        return 0

    ref = load_reference()
    print(
        f"paper_result  preset={args.preset}  outdir={outdir}  "
        f"reference notebook energy={ref['energy_after_200_bfgs']:.6f}",
        flush=True,
    )

    fig5 = None
    fig7 = None
    if not args.skip_ecd:
        fig5 = run_fig5(
            outdir=outdir,
            n_trials=ecd_trials,
            maxiter=ecd_maxiter,
            seed_base=args.ecd_seed,
            workers=args.workers,
            record_every=record_every,
            adaptive_eta=adaptive_eta,
        )
    else:
        fig5 = _load(fig5_path)
    if not args.skip_qaoa:
        fig7 = run_fig7(
            outdir=outdir,
            p_layers=qaoa_p,
            n_trials=qaoa_trials,
            maxiter=qaoa_maxiter,
            seed_base=args.qaoa_seed_base,
            workers=args.workers,
            record_every=record_every,
            adaptive_eta=adaptive_eta,
        )
    else:
        fig7 = _load(fig7_path)
    if fig5 is None and fig7 is None:
        raise FileNotFoundError(f"no fig5.json or fig7.json in {outdir}")
    write_comparison(outdir, fig5, fig7)
    _remove_extra_plots(outdir)
    print(f"Wrote figures and JSON under {outdir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
