#!/usr/bin/env python3
"""How Gibbs η is chosen: same starts, several η policies.

Isolates the inverse-temperature schedule. Two cheap ansatze, shared
random starts across arms:

* product R_y on the 20 mixed p-spin instances (7 angles, SPSA)
* qubit QAOA p=20 on the paper knapsack (40 parameters, BFGS)

Arms:

* energy ⟨H⟩
* fixed η=0.1          (too soft; close to energy)
* fixed η=1/(0.05·12)  (paper knapsack default; needs that |E_min|)
* fixed η=1/(0.05|E_min|)  (same formula, oracle E_min per instance)
* fixed η=26           (typical adaptive final on p-spin; too sharp at t=0)
* sampled_tail frozen  (histogram 5%/25% quantiles at the start only)
* sampled_tail adaptive  (production: refresh + EMA, no known E_min)
"""

from __future__ import annotations

import argparse
import importlib.util
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

from qumode_vqe.eta import SampledTailEta
from qumode_vqe.hamiltonian import EXACT_GROUND_ENERGY
from qumode_vqe.vqe import gibbs_objective

from paper_result.qaoa import qubo_spectrum, random_qaoa_params, run_qaoa_trial

HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = HERE / "out"
PAPER_KNAPSACK_ETA = 1.0 / (0.05 * abs(EXACT_GROUND_ENERGY))
SHARP_ETA = 26.0
SEED_RY = 3000
SEED_QAOA = 4000
PLOT_NAME = "eta_ablation.png"
JSON_NAME = "eta_ablation.json"
TXT_NAME = "eta_ablation.txt"


def _load_ry():
    path = ROOT / "scripts" / "mixed_p_spin_ry_spsa.py"
    spec = importlib.util.spec_from_file_location("mixed_p_spin_ry_spsa", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ry = _load_ry()


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


def oracle_emin_eta(energies: np.ndarray) -> float:
    emin = float(np.min(np.asarray(energies, dtype=float)))
    return 1.0 / (0.05 * max(abs(emin), 1e-12))


POLICIES = (
    {"name": "energy", "kind": "energy", "label": "energy ⟨H⟩"},
    {"name": "fixed_soft", "kind": "fixed", "eta": 0.1, "label": "fixed η=0.1"},
    {
        "name": "fixed_paper",
        "kind": "fixed",
        "eta": PAPER_KNAPSACK_ETA,
        "label": f"fixed η={PAPER_KNAPSACK_ETA:.3f} (paper BKP)",
    },
    {"name": "fixed_oracle", "kind": "oracle", "label": "fixed η=1/(0.05|Emin|) oracle"},
    {
        "name": "fixed_sharp",
        "kind": "fixed",
        "eta": SHARP_ETA,
        "label": f"fixed η={SHARP_ETA:.0f}",
    },
    {"name": "tail_frozen", "kind": "tail", "adaptive": False, "label": "sampled_tail frozen"},
    {"name": "tail_adaptive", "kind": "tail", "adaptive": True, "label": "sampled_tail adaptive"},
)

POLICY_ORDER = [p["name"] for p in POLICIES]
POLICY_BY_NAME = {p["name"]: p for p in POLICIES}

PRESETS = {
    "smoke": {"ry_trials": 1, "ry_maxiter": 4, "qaoa_trials": 1, "qaoa_maxiter": 4, "qaoa_p": 2},
    "quick": {"ry_trials": 2, "ry_maxiter": 80, "qaoa_trials": 8, "qaoa_maxiter": 80, "qaoa_p": 20},
    "paper": {"ry_trials": 5, "ry_maxiter": 200, "qaoa_trials": 50, "qaoa_maxiter": 200, "qaoa_p": 20},
}


def _held_eta(policy: dict, energies: np.ndarray) -> float | None:
    kind = str(policy["kind"])
    if kind == "fixed":
        return float(policy["eta"])
    if kind == "oracle":
        return oracle_emin_eta(energies)
    return None


def run_ry_eta_trial(job: dict) -> dict:
    """Picklable worker: one R_y start on one Hamiltonian under one η policy."""
    t0 = time.perf_counter()
    n_spins = int(job["num_spins"])
    energies = np.asarray(job["energies"], dtype=float)
    bits = ry.bit_table(n_spins)
    maxiter = int(job["maxiter"])
    policy = POLICY_BY_NAME[str(job["eta_policy"])]
    rng = np.random.default_rng(int(job["seed"]))
    theta0 = ry.wrap_angles(rng.uniform(0.0, 2.0 * np.pi, size=n_spins))
    tail = None
    eta0 = None
    held = _held_eta(policy, energies)

    if policy["kind"] == "energy":

        def fun(theta: np.ndarray) -> float:
            probs = ry.product_probs(theta, bits)
            total = float(probs.sum())
            if total <= 0.0:
                return 0.0
            return float(np.dot(probs / total, energies))

        def before_step(k: int, theta: np.ndarray) -> None:
            del k, theta

    elif held is not None:
        eta0 = float(held)

        def fun(theta: np.ndarray) -> float:
            return gibbs_objective(ry.product_probs(theta, bits), energies, float(held))

        def before_step(k: int, theta: np.ndarray) -> None:
            del k, theta

    else:
        tail = SampledTailEta()
        eta0 = float(tail.initialize(energies, ry.product_probs(theta0, bits)).eta)

        def fun(theta: np.ndarray) -> float:
            return gibbs_objective(ry.product_probs(theta, bits), energies, float(tail.eta))

        def before_step(k: int, theta: np.ndarray) -> None:
            if policy.get("adaptive"):
                tail.maybe_update(k, maxiter, energies, ry.product_probs(theta, bits))

    opt = ry.run_spsa(
        fun,
        theta0,
        maxiter=maxiter,
        rng=rng,
        project=ry.wrap_angles,
        a=float(job["spsa_a"]),
        c=float(job["spsa_c"]),
        A=float(job["spsa_A"]),
        alpha=float(job["spsa_alpha"]),
        gamma=float(job["spsa_gamma"]),
        on_before_step=before_step,
    )
    theta = ry.wrap_angles(opt.x)
    if tail is not None:
        eta_final = float(tail.eta)
        eta_history = list(tail.snapshot()["history"])
    else:
        eta_final = None if held is None else float(held)
        eta_history = (
            []
            if held is None
            else [
                {"step": 0, "eta": float(held)},
                {"step": int(maxiter), "eta": float(held)},
            ]
        )
    init = ry.histogram_stats(ry.product_probs(theta0, bits), energies, n_spins, eta=eta0)
    final = ry.histogram_stats(ry.product_probs(theta, bits), energies, n_spins, eta=eta_final)
    if policy["kind"] == "energy":
        init["cost"] = float(init["energy_physical"])
        final["cost"] = float(final["energy_physical"])
    else:
        init["cost"] = float(init["gibbs_cost"])
        final["cost"] = float(final["gibbs_cost"])
    return {
        "backend": "ry_product",
        "trial": int(job["trial"]),
        "hamiltonian_id": int(job["hamiltonian_id"]),
        "file": str(job["file"]),
        "eta_policy": str(policy["name"]),
        "eta_label": str(policy["label"]),
        "objective": "energy" if policy["kind"] == "energy" else "gibbs",
        "num_spins": n_spins,
        "seed": int(job["seed"]),
        "nit": int(opt.nit),
        "nfev": int(opt.nfev),
        "eta0": eta0,
        "eta": eta_final,
        "eta_history": eta_history,
        "elapsed_s": float(time.perf_counter() - t0),
        "init": init,
        **final,
    }


def run_qaoa_eta_trial(job: dict) -> dict:
    rec = run_qaoa_trial(job)
    rec["backend"] = "qaoa"
    rec["eta_policy"] = str(job["eta_policy"])
    rec["eta_label"] = str(job["eta_label"])
    rec["hamiltonian_id"] = 0
    rec.pop("probs", None)
    rec.pop("x0", None)
    rec.pop("x", None)
    return rec


def _run_jobs(worker, jobs: list[dict], workers: int, *, label: str) -> list[dict]:
    records: list[dict] = []
    n = len(jobs)

    def _note(rec: dict, done: int) -> None:
        hid = rec.get("hamiltonian_id")
        hid_s = "" if hid is None else f" H{int(hid):03d}"
        eta = rec.get("eta")
        eta_s = "" if eta is None else f"  η={float(eta):.3f}"
        print(
            f"[{done}/{n}] {label}{hid_s} t{rec['trial']}  {rec['eta_policy']}"
            f"{eta_s}  E={rec.get('energy_diag', rec.get('energy_physical')):.3f}  "
            f"gs={rec.get('success')}  bits={rec.get('most_likely_bitstring')}",
            flush=True,
        )

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
    records.sort(
        key=lambda r: (
            str(r.get("eta_policy", "")),
            int(r.get("hamiltonian_id", -1)),
            int(r["trial"]),
        )
    )
    return records


def _summarize(records: list[dict]) -> dict:
    n = len(records)
    n_success = int(sum(bool(r.get("success")) for r in records))
    etas = [float(r["eta"]) for r in records if r.get("eta") is not None]
    eta0s = [float(r["eta0"]) for r in records if r.get("eta0") is not None]
    rel = [float(r["rel_gap"]) for r in records if r.get("rel_gap") is not None]
    gaps = [float(r["gap_to_min"]) for r in records if r.get("gap_to_min") is not None]
    pstars = [float(r["pstar"]) for r in records if r.get("pstar") is not None]
    return {
        "n": n,
        "n_success": n_success,
        "success_rate": n_success / max(n, 1),
        "mean_rel_gap": float(np.mean(rel)) if rel else float("nan"),
        "mean_gap_to_min": float(np.mean(gaps)) if gaps else float("nan"),
        "mean_pstar": float(np.mean(pstars)) if pstars else float("nan"),
        "mean_eta0": float(np.mean(eta0s)) if eta0s else float("nan"),
        "mean_eta": float(np.mean(etas)) if etas else float("nan"),
        "median_eta": float(np.median(etas)) if etas else float("nan"),
    }


def _by_policy(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {name: [] for name in POLICY_ORDER}
    for rec in records:
        out.setdefault(str(rec["eta_policy"]), []).append(rec)
    return out


def _pair_key(rec: dict) -> tuple:
    return (int(rec.get("hamiltonian_id", -1)), int(rec["trial"]))


def _paired(by_policy: dict[str, list[dict]], a: str, b: str) -> dict:
    ma = {_pair_key(r): r for r in by_policy.get(a, [])}
    mb = {_pair_key(r): r for r in by_policy.get(b, [])}
    keys = sorted(set(ma) & set(mb))
    a_ok = {k for k in keys if ma[k].get("success")}
    b_ok = {k for k in keys if mb[k].get("success")}
    return {
        "n": len(keys),
        "both": len(a_ok & b_ok),
        f"{a}_only": len(a_ok - b_ok),
        f"{b}_only": len(b_ok - a_ok),
        "neither": len(keys) - len(a_ok | b_ok),
        f"{a}_hits": len(a_ok),
        f"{b}_hits": len(b_ok),
    }


def _lite(rec: dict) -> dict:
    skip = {"init", "eta_history", "history", "theta0", "theta"}
    out = {k: v for k, v in rec.items() if k not in skip}
    hist = rec.get("eta_history") or []
    if hist:
        out["eta_history"] = [
            {"step": int(h["step"]), "eta": float(h["eta"])}
            for h in hist
            if h.get("eta") is not None
        ]
    return out


def run_ry(
    *,
    n_trials: int,
    maxiter: int,
    workers: int,
    seed_base: int,
) -> dict:
    instances = ry.load_instances(ROOT / "Hamiltonians" / "mixed_p_spin")
    jobs: list[dict] = []
    ham_meta = []
    for inst in instances:
        ham_meta.append({k: v for k, v in inst.items() if k != "energies"})
        for trial in range(int(n_trials)):
            seed = int(seed_base) + 100 * int(inst["hamiltonian_id"]) + trial
            for policy in POLICIES:
                jobs.append(
                    {
                        "trial": trial,
                        "hamiltonian_id": int(inst["hamiltonian_id"]),
                        "file": inst["file"],
                        "num_spins": int(inst["num_spins"]),
                        "energies": inst["energies"],
                        "seed": seed,
                        "eta_policy": policy["name"],
                        "maxiter": int(maxiter),
                        "spsa_a": ry.SPSA_A,
                        "spsa_c": ry.SPSA_C,
                        "spsa_A": ry.SPSA_A_STAB,
                        "spsa_alpha": ry.SPSA_ALPHA,
                        "spsa_gamma": ry.SPSA_GAMMA,
                    }
                )
    print(
        f"=== R_y η ablation  {len(instances)} H × {n_trials} starts × {len(POLICIES)} policies  "
        f"maxiter={maxiter}  workers={workers} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    records = _run_jobs(run_ry_eta_trial, jobs, workers, label="ry")
    elapsed = time.perf_counter() - t0
    grouped = _by_policy(records)
    summaries = {name: _summarize(grouped[name]) for name in POLICY_ORDER}
    paired = {
        "energy_vs_tail_adaptive": _paired(grouped, "energy", "tail_adaptive"),
        "fixed_soft_vs_tail_adaptive": _paired(grouped, "fixed_soft", "tail_adaptive"),
        "fixed_paper_vs_tail_adaptive": _paired(grouped, "fixed_paper", "tail_adaptive"),
        "fixed_oracle_vs_tail_adaptive": _paired(grouped, "fixed_oracle", "tail_adaptive"),
        "fixed_sharp_vs_tail_adaptive": _paired(grouped, "fixed_sharp", "tail_adaptive"),
        "tail_frozen_vs_tail_adaptive": _paired(grouped, "tail_frozen", "tail_adaptive"),
    }
    print(f"  wall {elapsed:.1f}s", flush=True)
    for name in POLICY_ORDER:
        s = summaries[name]
        print(
            f"  {name:<16}  hits {s['n_success']}/{s['n']}  "
            f"rate={s['success_rate']:.3f}  rel-gap={s['mean_rel_gap']:.3f}  "
            f"η0={s['mean_eta0']:.3f}  η*={s['mean_eta']:.3f}",
            flush=True,
        )
    return {
        "backend": "ry_product",
        "ansatz": "one_layer_ry",
        "n_hamiltonians": len(instances),
        "n_trials_per_hamiltonian": int(n_trials),
        "maxiter": int(maxiter),
        "seed_base": int(seed_base),
        "elapsed_s": elapsed,
        "summaries": summaries,
        "paired": paired,
        "hamiltonians": ham_meta,
        "trials": [_lite(r) for r in records],
    }


def run_qaoa(
    *,
    n_trials: int,
    maxiter: int,
    p_layers: int,
    workers: int,
    seed_base: int,
) -> dict:
    energies = qubo_spectrum()
    starts = [random_qaoa_params(int(p_layers), np.random.default_rng(int(seed_base) + t)) for t in range(int(n_trials))]
    jobs: list[dict] = []
    for t, x0 in enumerate(starts):
        for policy in POLICIES:
            held = _held_eta(policy, energies)
            jobs.append(
                {
                    "trial": t,
                    "seed": int(seed_base) + t,
                    "p_layers": int(p_layers),
                    "energies": energies,
                    "x0": x0,
                    "objective": "energy" if policy["kind"] == "energy" else "gibbs",
                    "maxiter": int(maxiter),
                    "record_every": max(int(maxiter) // 10, 1),
                    "adaptive_eta": bool(policy.get("adaptive", False)),
                    "fixed_eta": held,
                    "eta_policy": policy["name"],
                    "eta_label": policy["label"],
                }
            )
    print(
        f"=== QAOA η ablation  p={p_layers}  {n_trials} starts × {len(POLICIES)} policies  "
        f"maxiter={maxiter}  workers={workers} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    records = _run_jobs(run_qaoa_eta_trial, jobs, workers, label="qaoa")
    elapsed = time.perf_counter() - t0
    grouped = _by_policy(records)
    summaries = {name: _summarize(grouped[name]) for name in POLICY_ORDER}
    paired = {
        "energy_vs_tail_adaptive": _paired(grouped, "energy", "tail_adaptive"),
        "fixed_soft_vs_tail_adaptive": _paired(grouped, "fixed_soft", "tail_adaptive"),
        "fixed_paper_vs_tail_adaptive": _paired(grouped, "fixed_paper", "tail_adaptive"),
        "fixed_oracle_vs_tail_adaptive": _paired(grouped, "fixed_oracle", "tail_adaptive"),
        "fixed_sharp_vs_tail_adaptive": _paired(grouped, "fixed_sharp", "tail_adaptive"),
        "tail_frozen_vs_tail_adaptive": _paired(grouped, "tail_frozen", "tail_adaptive"),
    }
    print(f"  wall {elapsed:.1f}s", flush=True)
    for name in POLICY_ORDER:
        s = summaries[name]
        print(
            f"  {name:<16}  hits {s['n_success']}/{s['n']}  "
            f"rate={s['success_rate']:.3f}  P*={s['mean_pstar']:.4f}  "
            f"η0={s['mean_eta0']:.3f}  η*={s['mean_eta']:.3f}",
            flush=True,
        )
    return {
        "backend": "qaoa",
        "ansatz": f"qaoa_p{int(p_layers)}",
        "n_trials": int(n_trials),
        "maxiter": int(maxiter),
        "p_layers": int(p_layers),
        "seed_base": int(seed_base),
        "elapsed_s": elapsed,
        "paper_knapsack_eta": PAPER_KNAPSACK_ETA,
        "summaries": summaries,
        "paired": paired,
        "trials": [_lite(r) for r in records],
    }


def _bar_axis(ax, names: list[str], values: list[float], title: str, ylabel: str) -> None:
    colors = []
    for name in names:
        if name == "tail_adaptive":
            colors.append("#d62728")
        elif name == "energy":
            colors.append("#7f7f7f")
        else:
            colors.append("#4c78a8")
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colors, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, max(1.05, max(values) * 1.15 if values else 1.05))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{100.0 * val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_ablation(ry_payload: dict | None, qaoa_payload: dict | None, path: Path) -> None:
    panels = [p for p in (ry_payload, qaoa_payload) if p is not None]
    if not panels:
        raise ValueError("nothing to plot")
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.6), squeeze=False)
    for ax, payload in zip(axes[0], panels):
        names = list(POLICY_ORDER)
        summaries = payload["summaries"]
        values = [float(summaries[n]["success_rate"]) for n in names]
        title = (
            "R_y product, 20 mixed p-spin × 5 starts"
            if payload["backend"] == "ry_product"
            else "QAOA p=20, paper knapsack"
        )
        _bar_axis(ax, names, values, title, "exact ground-state hit rate")
        n = int(summaries[names[0]]["n"])
        ax.text(
            0.0,
            -0.28,
            f"n={n} paired starts per bar  ·  red = production sampled_tail",
            transform=ax.transAxes,
            fontsize=8,
            color="#444444",
        )
    fig.suptitle("Gibbs η policy ablation (same starts)", y=1.02, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_text(outdir: Path, ry_payload: dict | None, qaoa_payload: dict | None) -> str:
    lines = [
        "# Gibbs η ablation: same starts, only the inverse temperature changes",
        "",
        "Cost is −ln⟨e^{−ηE}⟩ except the energy arm, which minimizes ⟨H⟩.",
        "sampled_tail sets η = ln(20)/(q25−q05) from the current Born histogram.",
        f"Paper BKP formula: η = 1/(0.05|E_min|) = {PAPER_KNAPSACK_ETA:.6f} using E_min=−12.",
        "",
    ]

    def _block(title: str, payload: dict) -> list[str]:
        out = [f"## {title}", ""]
        out.append(
            f"{'policy':<16} {'hits':>8} {'rate':>6} {'rel-gap':>8} {'⟨P*⟩':>8} {'η0':>7} {'η*':>7}"
        )
        for name in POLICY_ORDER:
            s = payload["summaries"][name]
            pstar = s["mean_pstar"]
            pstar_s = "     —" if not math.isfinite(pstar) else f"{pstar:8.4f}"
            out.append(
                f"{name:<16} {s['n_success']:3d}/{s['n']:<4d} "
                f"{s['success_rate']:6.3f} {s['mean_rel_gap']:8.3f} "
                f"{pstar_s} {s['mean_eta0']:7.3f} {s['mean_eta']:7.3f}"
            )
        out.append("")
        paired = payload.get("paired") or {}
        key = "tail_frozen_vs_tail_adaptive"
        if key in paired:
            p = paired[key]
            out.append(
                f"paired frozen vs adaptive: both {p['both']}, "
                f"frozen-only {p['tail_frozen_only']}, "
                f"adaptive-only {p['tail_adaptive_only']}, neither {p['neither']}"
            )
        if "energy_vs_tail_adaptive" in paired:
            p = paired["energy_vs_tail_adaptive"]
            out.append(
                f"paired energy vs adaptive: both {p['both']}, "
                f"energy-only {p['energy_only']}, "
                f"adaptive-only {p['tail_adaptive_only']}, neither {p['neither']}"
            )
        out.append("")
        return out

    if ry_payload is not None:
        lines += _block(
            f"R_y product  ({ry_payload['n_hamiltonians']} mixed p-spin × "
            f"{ry_payload['n_trials_per_hamiltonian']} starts, {ry_payload['maxiter']} SPSA)",
            ry_payload,
        )
    if qaoa_payload is not None:
        lines += _block(
            f"QAOA p={qaoa_payload['p_layers']}  "
            f"({qaoa_payload['n_trials']} starts, {qaoa_payload['maxiter']} BFGS)",
            qaoa_payload,
        )
    text = "\n".join(lines) + "\n"
    (outdir / TXT_NAME).write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="paper")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--skip-ry", action="store_true")
    parser.add_argument("--skip-qaoa", action="store_true")
    parser.add_argument("--ry-trials", type=int, default=None)
    parser.add_argument("--ry-maxiter", type=int, default=None)
    parser.add_argument("--qaoa-trials", type=int, default=None)
    parser.add_argument("--qaoa-maxiter", type=int, default=None)
    parser.add_argument("--qaoa-p", type=int, default=None)
    args = parser.parse_args(argv)

    preset = PRESETS[args.preset]
    ry_trials = preset["ry_trials"] if args.ry_trials is None else args.ry_trials
    ry_maxiter = preset["ry_maxiter"] if args.ry_maxiter is None else args.ry_maxiter
    qaoa_trials = preset["qaoa_trials"] if args.qaoa_trials is None else args.qaoa_trials
    qaoa_maxiter = preset["qaoa_maxiter"] if args.qaoa_maxiter is None else args.qaoa_maxiter
    qaoa_p = preset["qaoa_p"] if args.qaoa_p is None else args.qaoa_p
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"eta_ablation  preset={args.preset}  outdir={outdir}  "
        f"paper_BKP_η={PAPER_KNAPSACK_ETA:.6f}",
        flush=True,
    )
    ry_payload = None
    qaoa_payload = None
    if not args.skip_ry:
        ry_payload = run_ry(
            n_trials=int(ry_trials),
            maxiter=int(ry_maxiter),
            workers=int(args.workers),
            seed_base=SEED_RY,
        )
    if not args.skip_qaoa:
        qaoa_payload = run_qaoa(
            n_trials=int(qaoa_trials),
            maxiter=int(qaoa_maxiter),
            p_layers=int(qaoa_p),
            workers=int(args.workers),
            seed_base=SEED_QAOA,
        )
    payload = {
        "policies": list(POLICIES),
        "paper_knapsack_eta": PAPER_KNAPSACK_ETA,
        "sharp_eta": SHARP_ETA,
        "preset": args.preset,
        "ry": ry_payload,
        "qaoa": qaoa_payload,
    }
    _save_json(outdir / JSON_NAME, payload)
    if ry_payload is not None or qaoa_payload is not None:
        plot_ablation(ry_payload, qaoa_payload, outdir / PLOT_NAME)
    text = write_text(outdir, ry_payload, qaoa_payload)
    print(text, end="", flush=True)
    print(f"Wrote {outdir / JSON_NAME}, {outdir / TXT_NAME}, {outdir / PLOT_NAME}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
