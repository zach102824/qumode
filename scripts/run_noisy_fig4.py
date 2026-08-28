#!/usr/bin/env python3
"""200-step noisy BFGS for Fig. 4 under two error models.

Same random start as the noiseless Fig. 4 run (seed 2026). Cases:

* ``paper_loss`` — paper Kraus photon loss after each UER layer, hardware
  κτ = 2 τ_ECD / T1_cav ≈ 0.003.
* ``typical_device`` — Lindblad after each ECD pair: cavity T1/nth, transmon
  T1/T2, self-Kerr 500 Hz, 1% ECD/rotation amplitude errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qumode_vqe.hamiltonian import EXACT_GROUND_ENERGY
from qumode_vqe.noise import (
    DEFAULT_T1_CAV,
    DEFAULT_TAU_ECD,
    TimingMode,
    comprehensive_config,
    paper_loss_config,
)
from qumode_vqe.params import random_parameters
from qumode_vqe.plotting import plot_energy_overlay, plot_population_comparison
from qumode_vqe.vqe import HybridSimulator, optimize_vqe

OUTDIR = Path("results")
SEED = 2026
MAXITER = 200
PAPER_KAPPA_TAU_LAYER = 2.0 * DEFAULT_TAU_ECD / DEFAULT_T1_CAV

STYLES = {
    "noiseless": {"marker": "o", "color": "green"},
    r"paper photon loss ($\kappa\tau\simeq 0.003$)": {"marker": "s", "color": "darkorange"},
    "typical device": {"marker": "x", "color": "royalblue"},
}


def _json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def _slim_history(history: list[dict]) -> list[dict]:
    out = []
    for h in history:
        rec = {
            "iteration": int(h["iteration"]),
            "energy_physical": float(h["energy_physical"]),
            "target_prob_physical": float(h["target_prob_physical"]),
            "most_likely": list(h["most_likely"]),
        }
        if int(h["iteration"]) % 10 == 0 or int(h["iteration"]) == history[-1]["iteration"]:
            rec["probs"] = np.asarray(h["probs"]).tolist()
        out.append(rec)
    return out


def noise_for(case: str):
    if case == "paper_loss":
        return paper_loss_config(PAPER_KAPPA_TAU_LAYER, TimingMode.PER_UER_LAYER)
    if case == "typical_device":
        return comprehensive_config()
    raise ValueError(f"Unknown case {case}")


def case_label(case: str) -> str:
    if case == "paper_loss":
        return r"paper photon loss ($\kappa\tau\simeq 0.003$)"
    return "typical device"


def run_case(case: str, outdir: Path, maxiter: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x0 = random_parameters(5, rng)
    noise = noise_for(case)
    sim = HybridSimulator(ndepth=5, noise=noise)
    print(
        f"=== {case}: 200-step noisy BFGS, seed={seed}, "
        f"κτ_used={noise.kappa_tau_used():.6f}, timing={noise.timing.value} ===",
        flush=True,
    )
    opt = optimize_vqe(sim, x0, method="BFGS", maxiter=maxiter, record_every=1, verbose=True)
    final = opt.history[-1]
    payload = {
        "case": case,
        "seed": seed,
        "maxiter": maxiter,
        "nit": opt.nit,
        "nfev": opt.nfev,
        "success": opt.success,
        "message": opt.message,
        "kappa_tau_used": noise.kappa_tau_used(),
        "timing": noise.timing.value,
        "loss_model": noise.loss_model.value,
        "enable_transmon": noise.enable_transmon,
        "nth_cav": noise.nth_cav,
        "t1_q": noise.t1_q,
        "t2_q": noise.t2_q,
        "kerr": noise.kerr,
        "rotation_rel_error": noise.rotation_rel_error,
        "ecd_amp_rel_error": noise.ecd_amp_rel_error,
        "final_energy": final["energy_physical"],
        "final_target_prob": final["target_prob_physical"],
        "most_likely": final["most_likely"],
        "x": np.asarray(opt.x, dtype=float).tolist(),
        "history": _slim_history(opt.history),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"fig4_noisy_{case}.json").write_text(
        json.dumps(_json_ready(payload), indent=2), encoding="utf-8"
    )
    print(
        f"Finished {case}: E={payload['final_energy']:.6f}  "
        f"P*={payload['final_target_prob']:.4f}  most_likely={payload['most_likely']}",
        flush=True,
    )
    return payload


def history_from_json_rows(rows: list) -> list[dict]:
    return [
        {
            "iteration": int(r[0] if not isinstance(r, dict) else r["iteration"]),
            "energy_physical": float(r[1] if not isinstance(r, dict) else r["energy_physical"]),
        }
        for r in rows
    ]


POP_ITERS = (20, 60, 80, 200)


def _probs_at(history: list[dict], iteration: int) -> np.ndarray:
    rec = min(history, key=lambda h: abs(int(h["iteration"]) - iteration))
    if "probs" not in rec:
        raise KeyError(f"No probability snapshot near iteration {iteration}")
    return np.asarray(rec["probs"], dtype=float)


def plot_populations(outdir: Path, seed: int) -> None:
    """Fig. 5-style top-5 populations at iterations 20, 60, 80, 200."""
    replay_until = max(it for it in POP_ITERS if it < 200)
    print(f"=== noiseless replay to iteration {replay_until} (seed={seed}) ===", flush=True)
    rng = np.random.default_rng(seed)
    x0 = random_parameters(5, rng)
    opt = optimize_vqe(
        HybridSimulator(ndepth=5),
        x0,
        method="BFGS",
        maxiter=replay_until,
        record_every=1,
        verbose=True,
    )
    snapshots: dict[int, dict[str, np.ndarray]] = {it: {} for it in POP_ITERS}
    for it in POP_ITERS:
        if it < 200:
            snapshots[it]["noiseless"] = _probs_at(opt.history, it)
    fig4 = json.loads((outdir / "paper_figures.json").read_text())["fig4_fig5"]
    snapshots[200]["noiseless"] = np.asarray(
        HybridSimulator(ndepth=5)
        .evaluate(np.asarray(fig4["x"], dtype=float))
        .measurement.physical_probs
    )

    for case in ("paper_loss", "typical_device"):
        path = outdir / f"fig4_noisy_{case}.json"
        rec = json.loads(path.read_text())
        for it in POP_ITERS:
            snapshots[it][case_label(case)] = _probs_at(rec["history"], it)
            used = min(rec["history"], key=lambda h: abs(int(h["iteration"]) - it))
            if int(used["iteration"]) != it:
                print(
                    f"  {case} has no snapshot at {it}; using iteration {used['iteration']}",
                    flush=True,
                )

    out = outdir / "fig5_noisy_populations.png"
    plot_population_comparison(snapshots, out, k=5)
    print(f"Wrote {out}", flush=True)


def plot_overlay(outdir: Path) -> None:
    series: dict[str, list[dict]] = {}
    noiseless_path = outdir / "paper_figures.json"
    if noiseless_path.exists():
        data = json.loads(noiseless_path.read_text())
        series["noiseless"] = history_from_json_rows(data["fig4_fig5"]["history_energies"])
    for case in ("paper_loss", "typical_device"):
        path = outdir / f"fig4_noisy_{case}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        series[case_label(case)] = rec["history"]
    if len(series) < 2:
        raise SystemExit("Need at least two energy histories to overlay.")
    plot_energy_overlay(
        series,
        outdir / "fig4_noisy_overlay.png",
        exact=EXACT_GROUND_ENERGY,
        styles=STYLES,
    )
    print(f"Wrote {outdir / 'fig4_noisy_overlay.png'}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("paper_loss", "typical_device", "overlay", "populations"),
        required=True,
    )
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.case == "overlay":
        plot_overlay(args.outdir)
        return 0
    if args.case == "populations":
        plot_populations(args.outdir, args.seed)
        return 0
    run_case(args.case, args.outdir, args.maxiter, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
