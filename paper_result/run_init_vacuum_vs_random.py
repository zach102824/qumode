#!/usr/bin/env python3
"""ECD-VQE: vacuum |0,0,0⟩ vs a frozen random product state.

Same notebook ECD starts (40 parameters, N_d = 5) and BFGS as Fig. 5.
The only change is the ket the circuit acts on:

* vacuum: |0⟩_q ⊗ |0⟩ ⊗ |0⟩  (paper / notebook)
* random: Ry(θ)|0⟩ ⊗ |α₁⟩ ⊗ |α₂⟩, 5 parameters drawn once and frozen

Prep is not optimized. Output is the fraction of paired trials whose
most-likely bitstring is the knapsack optimum ``0110000``.
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

from qumode_vqe.hamiltonian import (
    DEFAULT_NFOCKS,
    EXACT_GROUND_ENERGY,
    TARGET_BITSTRING,
    TARGET_QNM,
)

from paper_result.ecd import notebook_ecd_x0, random_product_prep, run_ecd_trial

HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = HERE / "out"
SEED_BASE = 2026
PREP_SEED_OFFSET = 10_000
PLOT_NAME = "init_vacuum_vs_random.png"
JSON_NAME = "init_vacuum_vs_random.json"
TXT_NAME = "init_vacuum_vs_random.txt"

PRESETS = {
    "smoke": {"maxiter": 2, "trials": 2, "record_every": 1},
    "quick": {"maxiter": 40, "trials": 8, "record_every": 10},
    "paper": {"maxiter": 200, "trials": 50, "record_every": 10},
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


def _history_lite(history: list[dict]) -> list[dict]:
    lite = []
    for h in history or []:
        rec = {
            "iteration": int(h["iteration"]),
            "energy_physical": float(h["energy_physical"]),
            "target_prob_physical": float(
                h.get("target_prob_physical", h.get("pstar", float("nan")))
            ),
            "most_likely": h.get("most_likely"),
            "most_likely_bitstring": h.get("most_likely_bitstring"),
            "cost": h.get("cost"),
            "eta": h.get("eta"),
        }
        lite.append(rec)
    return lite


def _trial_out(rec: dict) -> dict:
    skip = {"probs"}
    out = {k: v for k, v in rec.items() if k not in skip}
    out["history"] = _history_lite(rec.get("history") or [])
    return out


def _is_optimum(rec: dict) -> bool:
    if rec.get("success"):
        return True
    if str(rec.get("most_likely_bitstring")) == TARGET_BITSTRING:
        return True
    ml = rec.get("most_likely")
    if isinstance(ml, (list, tuple)) and tuple(int(v) for v in ml) == TARGET_QNM:
        return True
    return False


def _summarize(trials: list[dict]) -> dict:
    n = len(trials)
    pstars = np.asarray([float(t["pstar"]) for t in trials], dtype=float) if n else np.array([])
    energies = (
        np.asarray([float(t["energy_physical"]) for t in trials], dtype=float) if n else np.array([])
    )
    n_hit = int(sum(_is_optimum(t) for t in trials))
    p0 = [float(t["pstar0"]) for t in trials if t.get("pstar0") is not None]
    return {
        "n": n,
        "n_success": n_hit,
        "success_rate": n_hit / max(n, 1),
        "mean_pstar": float(np.mean(pstars)) if n else float("nan"),
        "std_pstar": float(np.std(pstars, ddof=1)) if n > 1 else 0.0,
        "best_pstar": float(np.max(pstars)) if n else float("nan"),
        "mean_energy": float(np.mean(energies)) if n else float("nan"),
        "best_energy": float(np.min(energies)) if n else float("nan"),
        "mean_pstar0": float(np.mean(p0)) if p0 else None,
    }


def _paired(vacuum: list[dict], random: list[dict]) -> dict:
    v_ok = {int(t["trial"]) for t in vacuum if _is_optimum(t)}
    r_ok = {int(t["trial"]) for t in random if _is_optimum(t)}
    n = max(len(vacuum), len(random))
    return {
        "n": n,
        "both": len(v_ok & r_ok),
        "random_only": len(r_ok - v_ok),
        "vacuum_only": len(v_ok - r_ok),
        "neither": n - len(v_ok | r_ok),
    }


def _run_jobs(jobs: list[dict], workers: int, checkpoint=None) -> list[dict]:
    records: list[dict] = []
    n = len(jobs)

    def _note(rec: dict, done: int) -> None:
        print(
            f"[{done}/{n}] t{rec['trial']}  {rec.get('init', '?')}  {rec['objective']}  "
            f"P*={rec['pstar']:.4f}  E={rec['energy_physical']:.3f}  "
            f"bits={rec['most_likely_bitstring']}  nit={rec['nit']}",
            flush=True,
        )
        if checkpoint is not None:
            checkpoint(records)

    if workers <= 1 or n <= 1:
        for i, job in enumerate(jobs, 1):
            rec = run_ecd_trial(job)
            records.append(rec)
            _note(rec, i)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(run_ecd_trial, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                done += 1
                _note(rec, done)
    records.sort(key=lambda r: (str(r.get("init")), str(r["objective"]), int(r["trial"])))
    return records


def _reuse_vacuum_from_fig5(
    path: Path,
    *,
    n_trials: int,
    maxiter: int,
    seed_base: int,
    objectives: tuple[str, ...],
) -> dict[str, list[dict]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("n_trials") or 0) < int(n_trials):
        return None
    if int(payload.get("maxiter") or 0) != int(maxiter):
        return None
    if int(payload.get("seed_base") or -1) != int(seed_base):
        return None
    reused: dict[str, list[dict]] = {}
    for obj in objectives:
        block = payload.get(obj) or {}
        trials = list(block.get("trials") or [])
        if len(trials) < int(n_trials):
            return None
        rows = []
        for t in trials[: int(n_trials)]:
            rec = dict(t)
            rec["init"] = "vacuum"
            rec["objective"] = obj
            rec["prep"] = None
            rows.append(rec)
        reused[obj] = rows
    print(f"reusing vacuum {objectives} from {path}  (n={n_trials})", flush=True)
    return reused


def plot_comparison(payload: dict, path: Path) -> None:
    labels = []
    vacuum_s = []
    random_s = []
    vacuum_n = []
    random_n = []
    vacuum_p = []
    random_p = []
    for obj in payload["objectives"]:
        v = payload[obj]["vacuum"]
        r = payload[obj]["random_product"]
        labels.append("energy ⟨H⟩" if obj == "energy" else "Gibbs")
        vacuum_s.append(float(v["success_rate"]))
        random_s.append(float(r["success_rate"]))
        vacuum_n.append(int(v["n"]))
        random_n.append(int(r["n"]))
        vacuum_p.append(float(v["mean_pstar"]))
        random_p.append(float(r["mean_pstar"]))

    x = np.arange(len(labels), dtype=float)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))

    ax = axes[0]
    b1 = ax.bar(x - width / 2, vacuum_s, width, label="vacuum |0,0,0⟩", color="#4c72b0", zorder=2)
    b2 = ax.bar(x + width / 2, random_s, width, label="random product", color="#dd8452", zorder=2)
    ax.set_xticks(x, labels, fontsize=12)
    ax.set_ylabel("Most-likely bitstring is optimal", fontsize=11)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bars, ns in ((b1, vacuum_n), (b2, random_n)):
        for bar, n in zip(bars, ns):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.03,
                f"{100.0 * h:.0f}%" + (f"\n$n={n}$" if n else ""),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax = axes[1]
    ax.bar(x - width / 2, vacuum_p, width, label="vacuum |0,0,0⟩", color="#4c72b0", zorder=2)
    ax.bar(x + width / 2, random_p, width, label="random product", color="#dd8452", zorder=2)
    ax.set_xticks(x, labels, fontsize=12)
    ax.set_ylabel(r"Mean $P(|0,6,0\rangle)$", fontsize=11)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (vv, rr) in enumerate(zip(vacuum_p, random_p)):
        ax.text(x[i] - width / 2, vv + 0.03, f"{vv:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width / 2, rr + 0.03, f"{rr:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "ECD-VQE initial state: vacuum vs frozen random product",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_text(outdir: Path, payload: dict) -> str:
    lines = [
        "# ECD-VQE initial state: vacuum vs random product (frozen prep)",
        "",
        "Same notebook ECD x0 and BFGS. Prep is not optimized.",
        "",
    ]
    for obj in payload["objectives"]:
        v = payload[obj]["vacuum"]
        r = payload[obj]["random_product"]
        p = payload[obj]["paired"]
        title = "energy ⟨H⟩" if obj == "energy" else "Gibbs"
        lines += [
            f"## {title}",
            f"- vacuum: {100.0 * v['success_rate']:.0f}%  (n={v['n']})  "
            f"mean P*={v['mean_pstar']:.3f}",
            f"- random: {100.0 * r['success_rate']:.0f}%  (n={r['n']})  "
            f"mean P*={r['mean_pstar']:.3f}",
            f"- paired: random-only {p['random_only']}, vacuum-only {p['vacuum_only']}, "
            f"both {p['both']}, neither {p['neither']}",
            "",
        ]
    text = "\n".join(lines)
    (outdir / TXT_NAME).write_text(text + "\n", encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="paper")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--record-every", type=int, default=None)
    parser.add_argument(
        "--also-gibbs",
        action="store_true",
        help="Also run Gibbs BFGS (default is energy only, the paper ECD-VQE cost).",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Do not reuse vacuum trials from fig5.json; rerun vacuum ECD.",
    )
    parser.add_argument(
        "--fig5",
        type=Path,
        default=None,
        help="fig5.json to reuse vacuum trials from (default: <outdir>/fig5.json).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild the figure from saved JSON.",
    )
    args = parser.parse_args(argv)

    preset = PRESETS[args.preset]
    n_trials = preset["trials"] if args.trials is None else args.trials
    maxiter = preset["maxiter"] if args.maxiter is None else args.maxiter
    record_every = preset["record_every"] if args.record_every is None else args.record_every
    objectives: tuple[str, ...] = ("energy", "gibbs") if args.also_gibbs else ("energy",)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / JSON_NAME
    plot_path = outdir / PLOT_NAME
    nfocks = (int(DEFAULT_NFOCKS[0]), int(DEFAULT_NFOCKS[1]))

    if args.plot_only:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        plot_comparison(payload, plot_path)
        print(write_text(outdir, payload), end="")
        print(f"Wrote {plot_path}", flush=True)
        return 0

    print(
        f"ECD init ablation  preset={args.preset}  trials={n_trials}  "
        f"maxiter={maxiter}  objectives={objectives}  workers={args.workers}",
        flush=True,
    )

    starts = []
    preps = []
    for t in range(int(n_trials)):
        starts.append(notebook_ecd_x0(5, np.random.default_rng(int(args.seed_base) + t)))
        preps.append(
            random_product_prep(
                nfocks,
                np.random.default_rng(int(args.seed_base) + PREP_SEED_OFFSET + t),
            )
        )

    reused = None
    if not args.no_reuse:
        fig5_path = args.fig5 if args.fig5 is not None else outdir / "fig5.json"
        reused = _reuse_vacuum_from_fig5(
            fig5_path,
            n_trials=n_trials,
            maxiter=maxiter,
            seed_base=args.seed_base,
            objectives=objectives,
        )

    jobs = []
    inits = ("random_product",) if reused is not None else ("vacuum", "random_product")
    for t, x0 in enumerate(starts):
        for init in inits:
            for objective in objectives:
                jobs.append(
                    {
                        "trial": t,
                        "seed": int(args.seed_base) + t,
                        "ndepth": 5,
                        "x0": x0,
                        "prep": None if init == "vacuum" else preps[t],
                        "objective": objective,
                        "maxiter": int(maxiter),
                        "record_every": int(record_every),
                        "adaptive_eta": True,
                    }
                )

    partial = outdir / "init_vacuum_vs_random.partial.json"

    def _ckpt(recs: list[dict]) -> None:
        if len(recs) % 5 != 0:
            return
        _save_json(partial, {"n_done": len(recs), "n_jobs": len(jobs), "records": [_trial_out(r) for r in recs]})

    t0 = time.perf_counter()
    records = _run_jobs(jobs, args.workers, checkpoint=_ckpt)
    elapsed = time.perf_counter() - t0

    by: dict[str, dict[str, list[dict]]] = {obj: {"vacuum": [], "random_product": []} for obj in objectives}
    if reused is not None:
        for obj, rows in reused.items():
            by[obj]["vacuum"] = rows
    for rec in records:
        by[str(rec["objective"])][str(rec["init"])].append(rec)

    payload: dict = {
        "ansatz": "ecd_nd5",
        "n_trials": int(n_trials),
        "maxiter": int(maxiter),
        "seed_base": int(args.seed_base),
        "prep_seed_offset": PREP_SEED_OFFSET,
        "ndepth": 5,
        "nfocks": list(nfocks),
        "objectives": list(objectives),
        "prep_frozen": True,
        "prep": "Ry(θ)|0⟩ ⊗ |α1⟩ ⊗ |α2⟩, θ~U[0,π], |α|² uniform on [0, N−1]",
        "vacuum": "|0⟩⊗|0⟩⊗|0⟩",
        "target_qnm": list(TARGET_QNM),
        "target_bitstring": TARGET_BITSTRING,
        "exact_ground_energy": EXACT_GROUND_ENERGY,
        "reused_vacuum": reused is not None,
        "elapsed_s": elapsed,
    }
    for obj in objectives:
        vac = by[obj]["vacuum"]
        rnd = by[obj]["random_product"]
        vac.sort(key=lambda r: int(r["trial"]))
        rnd.sort(key=lambda r: int(r["trial"]))
        payload[obj] = {
            "vacuum": {**_summarize(vac), "trials": [_trial_out(r) for r in vac]},
            "random_product": {**_summarize(rnd), "trials": [_trial_out(r) for r in rnd]},
            "paired": _paired(vac, rnd),
        }
        sv = payload[obj]["vacuum"]
        sr = payload[obj]["random_product"]
        print(
            f"  {obj}: vacuum {sv['n_success']}/{sv['n']}  "
            f"random {sr['n_success']}/{sr['n']}  "
            f"mean P* {sv['mean_pstar']:.3f} vs {sr['mean_pstar']:.3f}",
            flush=True,
        )
    print(f"  wall {elapsed:.1f}s", flush=True)

    _save_json(json_path, payload)
    if partial.exists():
        partial.unlink()
    plot_comparison(payload, plot_path)
    print(write_text(outdir, payload), end="")
    print(f"Wrote {plot_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
