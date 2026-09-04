#!/usr/bin/env python3
"""Noiseless Gibbs VQE: joint SPSA on preparation and the ECD ansatz.

Does **not** use a known ground-state label during optimization. Each trial:

1. Starts from the hybrid vacuum ``|0⟩⊗|0⟩⊗|0⟩`` (prep ``(θ, α₁, α₂) = 0``).
2. Jointly SPSA-optimizes those five preparation variables **and** a random
   ECD ansatz. Prep is not frozen unless ``--spsa-iter N`` with ``N>0``.

``--mixed-p-spin`` sweeps ECD depth 1–5 (8–40 ansatz parameters, 4–20
primitive gates) on the saved mixed p-spin NPZ instances, using 200 joint
SPSA steps by default.

Gibbs η is always ``sampled_tail`` (histogram quantiles, no known E_min).
Selection and reporting use only the Gibbs cost and the decoded histogram.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from qumode_vqe.circuit import N_PREP_PARAMS, coherent_radius_max, vacuum_prep_params
from qumode_vqe.hamiltonian import (
    DEFAULT_NFOCKS,
    BKPInstance,
    bits_from_qnm,
    bitstring_from_bits,
    bkp_instance_as_dict,
    energy_spectrum_stats,
    energy_tensor_from_z_terms,
    ground_qnm_from_tensor,
    hybrid_energy_tensor,
    ising_benchmark_terms,
    knapsack_benchmark_instance,
    knapsack_packing_stats,
    knapsack_sweep_instance,
    load_mixed_p_spin_instances,
    max_pauli_weight,
    qubo_energy,
    z_terms_as_list,
)
from qumode_vqe.params import (
    ansatz_inventory,
    paper_bounds,
    random_parameters,
    random_snap_parameters,
)
from qumode_vqe.vqe import (
    DEFAULT_ANSATZ_STEPS,
    DEFAULT_JOINT_STEPS,
    HybridSimulator,
    optimize_gibbs_adaptive,
    optimize_vqe,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
OUTDIR = Path("results")
SEED_BASE = 3000
N_TRIALS = 50
NDEPTH = 5
NFOCKS = DEFAULT_NFOCKS
OUTER_ITERATIONS = DEFAULT_JOINT_STEPS
SPSA_ITERATIONS = DEFAULT_ANSATZ_STEPS
WORKERS = 7

SPSA_A = 0.2
SPSA_C = 0.15
SPSA_A_STAB = 10.0
SPSA_ALPHA = 0.602
SPSA_GAMMA = 0.101

OUTPUT_JSON = "gibbs_adaptive_noiseless.json"
KNAPSACK_JSON = "gibbs_adaptive_knapsack.json"
MIXED_JSON = "gibbs_mixed_40.json"
MIXED_P_SPIN_JSON = "gibbs_mixed_p_spin_ecd.json"
MIXED_P_SPIN_DIR = Path("Hamiltonians") / "mixed_p_spin"
MIXED_P_SPIN_NPZ_GLOB = "mixed_p_spin_p2-4_[0-9][0-9][0-9].npz"
MIXED_P_SPIN_STEPS = 200
MIXED_P_SPIN_ECD_DEPTHS = (1, 2, 3, 4, 5)
MIXED_P_SPIN_SNAP_DEPTHS = (1, 2)
HAMILTONIANS_JSON = "hamiltonians.json"
HAMILTONIANS_NPZ = "hamiltonians.npz"
ENERGY_JSON = "energy_spsa_baseline.json"
SCHEDULE_JSON = "gibbs_schedule_abc.json"
JOINT_STEPS_JSON = "gibbs_joint_step_sweep.json"
JOINT_STEP_BUDGETS = (40, 50, 80, 100, 150, 200)
PREP_STEP_SCALE = 0.25
N_HAMILTONIANS = 0
N_KNAPSACK = 0
N_ISING = 0
HAM_SEED = 8000
PREP_SEED_OFFSET = 50_000


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


def _budget_label(outer_iter: int, spsa_iter: int) -> str:
    if int(spsa_iter) <= 0:
        return f"joint {int(outer_iter)} (prep never frozen)"
    return f"joint {int(outer_iter)} then freeze + ansatz {int(spsa_iter)}"


def _budget_fields(outer_iter: int, spsa_iter: int) -> dict:
    return {
        "outer_iterations": int(outer_iter),
        "spsa_iterations": int(spsa_iter),
        "prep_frozen": int(spsa_iter) > 0,
        "schedule": "freeze" if int(spsa_iter) > 0 else "joint",
    }


def bitstring_energy(qnm) -> float:
    q, n, m = (int(v) for v in qnm)
    return qubo_energy(bits_from_qnm(q, n, m))


def unit_to_prep(u: np.ndarray, nfocks: tuple[int, int]) -> np.ndarray:
    """Map a unit-cube point to (θ, Re α₁, Im α₁, Re α₂, Im α₂).

    θ covers [0, π]. Each cavity uses polar coordinates so |α|² is uniform on
    [0, N−1], i.e. the available Fock range.
    """
    u = np.asarray(u, dtype=float).reshape(N_PREP_PARAMS)
    theta = float(u[0] * np.pi)
    r1 = coherent_radius_max(nfocks[0])
    r2 = coherent_radius_max(nfocks[1])
    mag1 = r1 * np.sqrt(u[1])
    mag2 = r2 * np.sqrt(u[3])
    a1 = mag1 * np.exp(1j * (2.0 * np.pi * u[2]))
    a2 = mag2 * np.exp(1j * (2.0 * np.pi * u[4]))
    return np.array([theta, a1.real, a1.imag, a2.real, a2.imag], dtype=float)


def random_prep(nfocks: tuple[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One uniform random preparation. Returns (unit cube, prep)."""
    rng = np.random.default_rng(int(seed))
    unit = rng.random(N_PREP_PARAMS)
    return unit, unit_to_prep(unit, nfocks)


def vacuum_start() -> tuple[np.ndarray, np.ndarray]:
    """Hybrid vacuum |0,0,0⟩ as (unit cube, prep). Used for every Gibbs trial."""
    zeros = vacuum_prep_params()
    return zeros.copy(), zeros.copy()


def _prep_unit_from(obj: dict) -> np.ndarray:
    raw = obj.get("prep_unit", obj.get("sobol_unit", np.zeros(N_PREP_PARAMS)))
    return np.asarray(raw, dtype=float).reshape(N_PREP_PARAMS)


def _eval_fields(ev: dict, cost: float, energy_tensor: np.ndarray | None = None) -> dict:
    ml = [int(v) for v in ev["most_likely"]]
    if energy_tensor is None:
        e_diag = bitstring_energy(ml)
    else:
        e_diag = float(np.asarray(energy_tensor, dtype=float)[tuple(ml)])
    return {
        "cost": float(cost),
        "energy_physical": float(ev["energy_physical"]),
        "energy_diag": e_diag,
        "energy_qubo": e_diag,
        "most_likely": ml,
        "most_likely_bitstring": str(ev["most_likely_bitstring"]),
    }


def _score_tensor(ml: list[int], energy_tensor: np.ndarray) -> dict:
    e = np.asarray(energy_tensor, dtype=float)
    found = float(e[tuple(int(v) for v in ml)])
    e_min = float(np.min(e))
    e_max = float(np.max(e))
    spread = max(e_max - e_min, 1e-12)
    gs = list(ground_qnm_from_tensor(e))
    hit = bool(np.isclose(found, e_min, atol=1e-8, rtol=0.0))
    return {
        "energy_diag": found,
        "energy_min": e_min,
        "energy_max": e_max,
        "gap_to_min": found - e_min,
        "rel_gap": (found - e_min) / spread,
        "ground_qnm": gs,
        "success": hit,
    }


def run_trial(job: dict) -> dict:
    t0 = time.perf_counter()
    trial = int(job["trial"])
    ndepth = int(job["ndepth"])
    nfocks = (int(job["nfocks"][0]), int(job["nfocks"][1]))
    ansatz = str(job.get("ansatz", "ecd")).lower()
    rng = np.random.default_rng(int(job["seed_base"]) + trial)
    prep0 = np.asarray(job["prep0"], dtype=float)
    if ansatz == "snap":
        drawn = random_snap_parameters(ndepth, nfocks, rng)
    else:
        drawn = random_parameters(ndepth, rng)
    x0 = np.asarray(job["x0"], dtype=float) if job.get("x0") is not None else drawn
    prep_step_scale = float(job.get("prep_step_scale", 1.0))
    energy_tensor = job.get("energy_tensor")
    if energy_tensor is not None:
        energy_tensor = np.asarray(energy_tensor, dtype=float)
    inventory = ansatz_inventory(ansatz, ndepth, nfocks, n_prep_params=N_PREP_PARAMS)
    result = optimize_gibbs_adaptive(
        prep0,
        x0,
        ndepth=ndepth,
        nfocks=nfocks,
        outer_iter=int(job["outer_iter"]),
        spsa_iter=int(job["spsa_iter"]),
        rng=rng,
        a=float(job["spsa_a"]),
        c=float(job["spsa_c"]),
        A=float(job["spsa_A"]),
        alpha=float(job["spsa_alpha"]),
        gamma=float(job["spsa_gamma"]),
        prep_step_scale=prep_step_scale,
        energy_tensor=energy_tensor,
        ansatz=ansatz,
    )
    warm = _eval_fields(result.eval_warmup, result.fun_warmup, energy_tensor)
    final = _eval_fields(result.eval_final, result.fun, energy_tensor)
    rec = {
        "trial": trial,
        "ansatz": ansatz,
        "ndepth": ndepth,
        **inventory,
        "prep_unit": _prep_unit_from(job),
        "prep0": np.asarray(result.prep0, dtype=float),
        "prep": np.asarray(result.prep, dtype=float),
        "x0": np.asarray(result.x0, dtype=float),
        "x_warmup": np.asarray(result.x_warmup, dtype=float),
        "x": np.asarray(result.x, dtype=float),
        "warmup": warm,
        "final": final,
        "cost": final["cost"],
        "energy_physical": final["energy_physical"],
        "energy_qubo": final["energy_qubo"],
        "energy_diag": final["energy_diag"],
        "most_likely": final["most_likely"],
        "most_likely_bitstring": final["most_likely_bitstring"],
        "nit_warmup": int(result.nit_warmup),
        "nit": int(result.nit),
        "nit_total": int(result.nit_warmup) + int(result.nit),
        "nfev_warmup": int(result.nfev_warmup),
        "nfev_ansatz": int(result.nfev_ansatz),
        "nfev": int(result.nfev),
        "eta_policy": "sampled_tail",
        "eta0": float(result.eta0),
        "eta": float(result.eta),
        "n_eta_clamps": int(result.n_eta_clamps),
        "n_eta_fallbacks": int(result.n_eta_fallbacks),
        "outer_iter": int(job["outer_iter"]),
        "spsa_iter": int(job["spsa_iter"]),
        "prep_step_scale": prep_step_scale,
        "schedule": str(job.get("schedule") or ("freeze" if int(job["spsa_iter"]) > 0 else "joint")),
        "prep_frozen": int(job["spsa_iter"]) > 0,
        "elapsed_s": float(time.perf_counter() - t0),
        "file": str(job.get("file", "")),
    }
    if energy_tensor is not None:
        rec["hamiltonian_id"] = int(job.get("hamiltonian_id", trial))
        rec["kind"] = str(job.get("kind", "custom"))
        rec["family"] = str(job.get("family", rec["kind"]))
        rec.update(_score_tensor(final["most_likely"], energy_tensor))
        rec["ground_bitstring"] = bitstring_from_bits(bits_from_qnm(*rec["ground_qnm"]))
        inst_d = job.get("instance")
        if inst_d is not None:
            inst = BKPInstance(
                inst_d["values"], inst_d["weights"], inst_d["capacity"], inst_d["penalty"]
            )
            rec["packing"] = knapsack_packing_stats(
                bits_from_qnm(*final["most_likely"]), inst
            )
    return rec


def summarize_trials(trials: list[dict]) -> dict:
    n = len(trials)
    costs = np.asarray([t["cost"] for t in trials], dtype=float)
    energies = np.asarray([t["energy_qubo"] for t in trials], dtype=float)
    return {
        "n_trials": n,
        "mean_cost": float(np.mean(costs)) if n else float("nan"),
        "std_cost": float(np.std(costs, ddof=1)) if n > 1 else 0.0,
        "mean_energy_qubo": float(np.mean(energies)) if n else float("nan"),
        "std_energy_qubo": float(np.std(energies, ddof=1)) if n > 1 else 0.0,
        "best_cost": float(np.min(costs)) if n else float("nan"),
        "best_trial": int(trials[int(np.argmin(costs))]["trial"]) if n else None,
    }


def _print_job_line(i: int, n: int, rec: dict) -> None:
    extra = ""
    if "success" in rec:
        feas = rec.get("packing", {}).get("feasible")
        feas_s = "" if feas is None else f"  feas={feas}"
        extra = (
            f"  {rec.get('family', rec.get('kind', '?'))}"
            f"/H{rec.get('hamiltonian_id', '?')}"
            f"  η={rec.get('eta', float('nan')):.3f}  "
            f"E={rec['energy_diag']:.3f} (min {rec['energy_min']:.3f}, "
            f"rel={rec['rel_gap']:.3f})  gs={rec['success']}{feas_s}"
        )
    else:
        extra = f"  E_qubo={rec['energy_qubo']:.4f}"
    label = "⟨H⟩" if rec.get("method") == "energy_spsa" else "Gibbs"
    sched = rec.get("schedule") or ""
    sched_s = f"{sched} " if sched else ""
    depth_s = ""
    if rec.get("ndepth") is not None:
        depth_s = f" {rec.get('ansatz', 'ecd')} L{int(rec['ndepth'])}"
    print(
        f"[{i}/{n}] {sched_s}trial {rec['trial']}{depth_s}: {label}={rec['cost']:.4f}{extra}  "
        f"bits={rec['most_likely_bitstring']}",
        flush=True,
    )


def _run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    records: list[dict] = []
    if workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs, 1):
            rec = run_trial(job)
            records.append(rec)
            _print_job_line(i, len(jobs), rec)
        records.sort(
            key=lambda r: (
                str(r.get("schedule", "")),
                str(r.get("family", "")),
                int(r.get("ndepth", 0)),
                int(r.get("hamiltonian_id", 0)),
                int(r["trial"]),
            )
        )
        return records
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futs = {pool.submit(run_trial, job): job for job in jobs}
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            done += 1
            _print_job_line(done, len(jobs), rec)
    records.sort(
        key=lambda r: (
            str(r.get("schedule", "")),
            str(r.get("family", "")),
            int(r.get("ndepth", 0)),
            int(r.get("hamiltonian_id", 0)),
            int(r["trial"]),
        )
    )
    return records


def _summarize_group(recs: list[dict]) -> dict:
    n = len(recs)
    n_success = int(sum(bool(r.get("success")) for r in recs))
    n_feas = int(sum(bool(r.get("packing", {}).get("feasible")) for r in recs))
    etas = [float(r["eta"]) for r in recs if r.get("eta") is not None]
    elapsed = [float(r["elapsed_s"]) for r in recs if r.get("elapsed_s") is not None]
    nfev = [float(r["nfev"]) for r in recs if r.get("nfev") is not None]
    out = {
        "n": n,
        "n_success": n_success,
        "success_rate": n_success / max(n, 1),
        "n_feasible": n_feas,
        "mean_rel_gap": float(np.mean([r["rel_gap"] for r in recs])) if n else float("nan"),
        "mean_gap_to_min": float(np.mean([r["gap_to_min"] for r in recs])) if n else float("nan"),
        "mean_energy_diag": float(np.mean([r["energy_diag"] for r in recs])) if n else float("nan"),
        "mean_energy_physical": float(np.mean([r["energy_physical"] for r in recs])) if n else float("nan"),
        "mean_eta": float(np.mean(etas)) if etas else float("nan"),
        "mean_nfev": float(np.mean(nfev)) if nfev else float("nan"),
        "mean_elapsed_s": float(np.mean(elapsed)) if elapsed else float("nan"),
    }
    if n and recs[0].get("n_ansatz_params") is not None:
        out["n_ansatz_params"] = int(recs[0]["n_ansatz_params"])
        out["n_params"] = int(recs[0]["n_params"])
        out["n_primitive_gates"] = int(recs[0]["n_primitive_gates"])
        out["n_scalar_controls"] = int(recs[0]["n_scalar_controls"])
        out["ndepth"] = int(recs[0]["ndepth"])
    return out


def run_knapsack_hamiltonian_sweep(
    *,
    n_hamiltonians: int,
    n_trials: int,
    outer_iter: int,
    spsa_iter: int,
    workers: int,
    seed_base: int,
    ham_seed: int,
    outdir: Path,
    ndepth: int,
    nfocks: tuple[int, int],
    spsa_a: float,
    spsa_c: float,
    spsa_A: float,
    spsa_alpha: float,
    spsa_gamma: float,
) -> dict:
    """Adaptive Gibbs on knapsack-like QUBOs (paper encoding: 4 items + 3 slack)."""
    n_h = max(int(n_hamiltonians), 1)
    jobs: list[dict] = []
    ham_meta: list[dict] = []
    print("=== Knapsack instances (landscape, before VQE) ===", flush=True)
    print(
        f"{'H':>2}  {'kind':<16}  {'Emin':>8}  {'spread':>8}  {'gap':>6}",
        flush=True,
    )
    for h in range(n_h):
        kind, inst = knapsack_sweep_instance(h, ham_seed)
        tensor = hybrid_energy_tensor(nfocks, inst)
        spec = energy_spectrum_stats(tensor)
        gs_bits = bitstring_from_bits(bits_from_qnm(*spec["ground_qnm"]))
        gs_pack = knapsack_packing_stats(bits_from_qnm(*spec["ground_qnm"]), inst)
        meta = {
            "hamiltonian_id": h,
            "kind": kind,
            "instance": bkp_instance_as_dict(inst),
            "ground_bitstring": gs_bits,
            "ground_packing": gs_pack,
            **spec,
        }
        ham_meta.append(meta)
        print(
            f"{h:2d}  {kind:<16}  {spec['energy_min']:8.2f}  {spec['spread']:8.1f}  "
            f"{spec['gap']:6.2f}",
            flush=True,
        )
        for t in range(n_trials):
            unit, prep0 = vacuum_start()
            jobs.append(
                {
                    "trial": t,
                    "hamiltonian_id": h,
                    "kind": kind,
                    "instance": meta["instance"],
                    "seed_base": seed_base + 100 * h,
                    "ndepth": ndepth,
                    "nfocks": nfocks,
                    "outer_iter": outer_iter,
                    "spsa_iter": spsa_iter,
                    "prep0": prep0,
                    "prep_unit": unit,
                    "energy_tensor": tensor,
                    "spsa_a": spsa_a,
                    "spsa_c": spsa_c,
                    "spsa_A": spsa_A,
                    "spsa_alpha": spsa_alpha,
                    "spsa_gamma": spsa_gamma,
                }
            )
    print(
        f"=== Gibbs adaptive SPSA (sampled_tail η), {n_h} knapsack-like H, "
        f"{n_trials} trial(s) each, {_budget_label(outer_iter, spsa_iter)}, "
        f"workers={workers} ===",
        flush=True,
    )
    records = _run_jobs(jobs, workers)
    by_kind: dict[str, list[dict]] = {}
    for rec in records:
        by_kind.setdefault(str(rec.get("kind", "custom")), []).append(rec)
    kind_summary = {k: _summarize_group(v) for k, v in by_kind.items()}
    n_success = int(sum(bool(r.get("success")) for r in records))
    payload = {
        "n_hamiltonians": n_h,
        "n_trials_per_hamiltonian": n_trials,
        "eta_policy": "sampled_tail",
        **_budget_fields(outer_iter, spsa_iter),
        "ndepth": ndepth,
        "nfocks": list(nfocks),
        "seed_base": seed_base,
        "ham_seed": ham_seed,
        "workers": workers,
        "n_success": n_success,
        "success_rate": n_success / max(len(records), 1),
        "mean_rel_gap": float(np.mean([r["rel_gap"] for r in records])) if records else float("nan"),
        "mean_eta": float(np.mean([r["eta"] for r in records])) if records else float("nan"),
        "by_kind": kind_summary,
        "hamiltonians": ham_meta,
        "trials": records,
        "spsa": {
            "a": spsa_a,
            "c": spsa_c,
            "A": spsa_A,
            "alpha": spsa_alpha,
            "gamma": spsa_gamma,
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / KNAPSACK_JSON
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    print(
        f"  exact-GS hits {n_success}/{len(records)}  "
        f"mean rel-gap {payload['mean_rel_gap']:.3f}  "
        f"mean η {payload['mean_eta']:.3f}",
        flush=True,
    )
    for k, s in kind_summary.items():
        print(
            f"  {k}: {s['n_success']}/{s['n']}  mean rel-gap {s['mean_rel_gap']:.3f}",
            flush=True,
        )
    return payload


def _spsa_job_fields(
    *,
    ndepth: int,
    nfocks: tuple[int, int],
    outer_iter: int,
    spsa_iter: int,
    spsa_a: float,
    spsa_c: float,
    spsa_A: float,
    spsa_alpha: float,
    spsa_gamma: float,
) -> dict:
    return {
        "ndepth": ndepth,
        "nfocks": nfocks,
        "outer_iter": outer_iter,
        "spsa_iter": spsa_iter,
        "spsa_a": spsa_a,
        "spsa_c": spsa_c,
        "spsa_A": spsa_A,
        "spsa_alpha": spsa_alpha,
        "spsa_gamma": spsa_gamma,
    }


def _save_hamiltonians(outdir: Path, ham_meta: list[dict]) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / HAMILTONIANS_JSON
    json_path.write_text(json.dumps(_json_ready(ham_meta), indent=2), encoding="utf-8")
    npz_path = outdir / HAMILTONIANS_NPZ
    families = np.array([str(h["family"]) for h in ham_meta])
    ids = np.array([int(h["hamiltonian_id"]) for h in ham_meta], dtype=int)
    tensors = np.stack([np.asarray(h["energy_tensor"], dtype=float) for h in ham_meta])
    np.savez_compressed(npz_path, family=families, hamiltonian_id=ids, energy_tensor=tensors)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {npz_path}", flush=True)
    return json_path, npz_path


def run_mixed_suite(
    *,
    n_knapsack: int,
    n_ising: int,
    n_trials: int,
    outer_iter: int,
    spsa_iter: int,
    workers: int,
    seed_base: int,
    ham_seed: int,
    outdir: Path,
    ndepth: int,
    nfocks: tuple[int, int],
    spsa_a: float,
    spsa_c: float,
    spsa_A: float,
    spsa_alpha: float,
    spsa_gamma: float,
) -> dict:
    """sampled_tail Gibbs on random knapsacks + random 7-spin Ising, one start each by default."""
    n_k = max(int(n_knapsack), 0)
    n_i = max(int(n_ising), 0)
    n_h = n_k + n_i
    if n_h <= 0:
        raise ValueError("need at least one knapsack or Ising Hamiltonian")
    n_t = max(int(n_trials), 1)
    spsa_fields = _spsa_job_fields(
        ndepth=ndepth,
        nfocks=nfocks,
        outer_iter=outer_iter,
        spsa_iter=spsa_iter,
        spsa_a=spsa_a,
        spsa_c=spsa_c,
        spsa_A=spsa_A,
        spsa_alpha=spsa_alpha,
        spsa_gamma=spsa_gamma,
    )
    jobs: list[dict] = []
    ham_meta: list[dict] = []

    print("=== Hamiltonians (landscape, before VQE) ===", flush=True)
    print(
        f"{'fam':<9} {'H':>2}  {'Emin':>8}  {'spread':>8}  {'gap':>6}  extra",
        flush=True,
    )
    for h in range(n_k):
        inst = knapsack_benchmark_instance(h, ham_seed)
        tensor = hybrid_energy_tensor(nfocks, inst)
        spec = energy_spectrum_stats(tensor)
        gs_bits = bitstring_from_bits(bits_from_qnm(*spec["ground_qnm"]))
        gs_pack = knapsack_packing_stats(bits_from_qnm(*spec["ground_qnm"]), inst)
        meta = {
            "family": "knapsack",
            "kind": "knapsack",
            "hamiltonian_id": h,
            "instance": bkp_instance_as_dict(inst),
            "ground_bitstring": gs_bits,
            "ground_packing": gs_pack,
            "energy_tensor": np.asarray(tensor, dtype=float),
            **spec,
        }
        ham_meta.append(meta)
        print(
            f"{'knapsack':<9} {h:2d}  {spec['energy_min']:8.2f}  {spec['spread']:8.1f}  "
            f"{spec['gap']:6.2f}  C={inst.capacity:.0f} λ={inst.penalty:.2f} "
            f"v={np.asarray(inst.values).tolist()} w={np.asarray(inst.weights).tolist()}",
            flush=True,
        )
        for t in range(n_t):
            unit, prep0 = vacuum_start()
            jobs.append(
                {
                    "trial": t,
                    "hamiltonian_id": h,
                    "family": "knapsack",
                    "kind": "knapsack",
                    "instance": meta["instance"],
                    "seed_base": seed_base + 100 * h,
                    "prep0": prep0,
                    "prep_unit": unit,
                    "energy_tensor": tensor,
                    **spsa_fields,
                }
            )

    for h in range(n_i):
        terms = ising_benchmark_terms(h, ham_seed)
        tensor = energy_tensor_from_z_terms(terms, nfocks)
        spec = energy_spectrum_stats(tensor)
        gs_bits = bitstring_from_bits(bits_from_qnm(*spec["ground_qnm"]))
        meta = {
            "family": "ising",
            "kind": "ising",
            "hamiltonian_id": h,
            "terms": z_terms_as_list(terms),
            "max_pauli_weight": int(max_pauli_weight(terms)),
            "n_terms": len(terms),
            "ground_bitstring": gs_bits,
            "energy_tensor": np.asarray(tensor, dtype=float),
            **spec,
        }
        ham_meta.append(meta)
        print(
            f"{'ising':<9} {h:2d}  {spec['energy_min']:8.2f}  {spec['spread']:8.1f}  "
            f"{spec['gap']:6.2f}  n_terms={len(terms)} weight≤{meta['max_pauli_weight']}",
            flush=True,
        )
        for t in range(n_t):
            unit, prep0 = vacuum_start()
            jobs.append(
                {
                    "trial": t,
                    "hamiltonian_id": h,
                    "family": "ising",
                    "kind": "ising",
                    "seed_base": seed_base + 10_000 + 100 * h,
                    "prep0": prep0,
                    "prep_unit": unit,
                    "energy_tensor": tensor,
                    **spsa_fields,
                }
            )

    _save_hamiltonians(outdir, ham_meta)
    print(
        f"=== Gibbs adaptive SPSA (sampled_tail η), {n_k} knapsack + {n_i} Ising, "
        f"{n_t} trial(s) each, {_budget_label(outer_iter, spsa_iter)}, "
        f"workers={workers} ===",
        flush=True,
    )
    records = _run_jobs(jobs, workers)
    by_family: dict[str, list[dict]] = {}
    for rec in records:
        by_family.setdefault(str(rec.get("family", "custom")), []).append(rec)
    family_summary = {k: _summarize_group(v) for k, v in by_family.items()}
    n_success = int(sum(bool(r.get("success")) for r in records))
    payload = {
        "n_knapsack": n_k,
        "n_ising": n_i,
        "n_hamiltonians": n_h,
        "n_trials_per_hamiltonian": n_t,
        "eta_policy": "sampled_tail",
        **_budget_fields(outer_iter, spsa_iter),
        "ndepth": ndepth,
        "nfocks": list(nfocks),
        "seed_base": seed_base,
        "ham_seed": ham_seed,
        "workers": workers,
        "n_success": n_success,
        "success_rate": n_success / max(len(records), 1),
        "mean_rel_gap": float(np.mean([r["rel_gap"] for r in records])) if records else float("nan"),
        "mean_eta": float(np.mean([r["eta"] for r in records])) if records else float("nan"),
        "by_family": family_summary,
        "hamiltonians": ham_meta,
        "trials": records,
        "spsa": {
            "a": spsa_a,
            "c": spsa_c,
            "A": spsa_A,
            "alpha": spsa_alpha,
            "gamma": spsa_gamma,
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / MIXED_JSON
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    print(
        f"  exact-GS hits {n_success}/{len(records)}  "
        f"mean rel-gap {payload['mean_rel_gap']:.3f}  "
        f"mean η {payload['mean_eta']:.3f}",
        flush=True,
    )
    for k, s in family_summary.items():
        extra = ""
        if k == "knapsack":
            extra = f"  feas {s['n_feasible']}/{s['n']}"
        print(
            f"  {k}: {s['n_success']}/{s['n']}{extra}  "
            f"mean rel-gap {s['mean_rel_gap']:.3f}  mean η {s['mean_eta']:.3f}",
            flush=True,
        )
    return payload


def run_energy_trial(job: dict) -> dict:
    """Paper-style energy VQE: vacuum initial state, one SPSA start, cost = ⟨H⟩."""
    trial = int(job["trial"])
    ndepth = int(job["ndepth"])
    nfocks = (int(job["nfocks"][0]), int(job["nfocks"][1]))
    rng = np.random.default_rng(int(job["seed_base"]) + trial)
    x0 = random_parameters(ndepth, rng)
    energy_tensor = np.asarray(job["energy_tensor"], dtype=float)
    sim = HybridSimulator(
        ndepth=ndepth,
        nfocks=nfocks,
        cost_kind="energy",
        target_qnm=None,
        energy_tensor=energy_tensor,
    )
    opt = optimize_vqe(
        sim,
        x0,
        method="SPSA",
        maxiter=int(job["maxiter"]),
        rng=rng,
        bounds=paper_bounds(ndepth),
        record_every=0,
        spsa_a=float(job["spsa_a"]),
        spsa_c=float(job["spsa_c"]),
        spsa_A=float(job["spsa_A"]),
        spsa_alpha=float(job["spsa_alpha"]),
        spsa_gamma=float(job["spsa_gamma"]),
    )
    ev = sim.evaluate(opt.x).as_dict()
    rec = {
        "trial": trial,
        "method": "energy_spsa",
        "family": str(job.get("family", "custom")),
        "kind": str(job.get("kind", "custom")),
        "hamiltonian_id": int(job["hamiltonian_id"]),
        "x0": np.asarray(x0, dtype=float),
        "x": np.asarray(opt.x, dtype=float),
        "cost": float(opt.fun),
        "energy_physical": float(ev["energy_physical"]),
        "most_likely": [int(v) for v in ev["most_likely"]],
        "most_likely_bitstring": str(ev["most_likely_bitstring"]),
        "nit": int(opt.nit),
        "nfev": int(opt.nfev),
    }
    rec.update(_score_tensor(rec["most_likely"], energy_tensor))
    rec["ground_bitstring"] = bitstring_from_bits(bits_from_qnm(*rec["ground_qnm"]))
    inst_d = job.get("instance")
    if inst_d is not None:
        inst = BKPInstance(
            inst_d["values"], inst_d["weights"], inst_d["capacity"], inst_d["penalty"]
        )
        rec["packing"] = knapsack_packing_stats(bits_from_qnm(*rec["most_likely"]), inst)
    return rec


def _paired_compare(energy_recs: list[dict], gibbs_recs: list[dict]) -> dict:
    def key(r: dict) -> tuple:
        return (str(r.get("family", "")), int(r.get("hamiltonian_id", -1)), int(r["trial"]))

    gmap = {key(r): r for r in gibbs_recs}
    n = 0
    both = energy_only = gibbs_only = neither = 0
    by_family: dict[str, dict] = {}
    for rec in energy_recs:
        g = gmap.get(key(rec))
        if g is None:
            continue
        n += 1
        e_hit = bool(rec.get("success"))
        g_hit = bool(g.get("success"))
        if e_hit and g_hit:
            both += 1
        elif e_hit:
            energy_only += 1
        elif g_hit:
            gibbs_only += 1
        else:
            neither += 1
        fam = str(rec.get("family", "?"))
        slot = by_family.setdefault(
            fam, {"n": 0, "energy_hits": 0, "gibbs_hits": 0, "both": 0, "energy_only": 0, "gibbs_only": 0}
        )
        slot["n"] += 1
        slot["energy_hits"] += int(e_hit)
        slot["gibbs_hits"] += int(g_hit)
        slot["both"] += int(e_hit and g_hit)
        slot["energy_only"] += int(e_hit and not g_hit)
        slot["gibbs_only"] += int(g_hit and not e_hit)
    return {
        "n_paired": n,
        "energy_hits": both + energy_only,
        "gibbs_hits": both + gibbs_only,
        "both": both,
        "energy_only": energy_only,
        "gibbs_only": gibbs_only,
        "neither": neither,
        "by_family": by_family,
    }


def run_energy_spsa_baseline(
    *,
    ham_path: Path,
    outer_iter: int,
    spsa_iter: int,
    workers: int,
    seed_base: int,
    outdir: Path,
    ndepth: int,
    nfocks: tuple[int, int],
    spsa_a: float,
    spsa_c: float,
    spsa_A: float,
    spsa_alpha: float,
    spsa_gamma: float,
    gibbs_path: Path | None = None,
) -> dict:
    """Vacuum + ⟨H⟩ + one SPSA start on previously saved Hamiltonians."""
    ham_meta = json.loads(Path(ham_path).read_text(encoding="utf-8"))
    if not isinstance(ham_meta, list) or not ham_meta:
        raise ValueError(f"expected a non-empty Hamiltonian list in {ham_path}")
    maxiter = max(int(outer_iter) + int(spsa_iter), 1)
    jobs: list[dict] = []
    for meta in ham_meta:
        family = str(meta.get("family", meta.get("kind", "custom")))
        hid = int(meta["hamiltonian_id"])
        seed = seed_base + 100 * hid
        if family == "ising":
            seed = seed_base + 10_000 + 100 * hid
        job = {
            "trial": 0,
            "hamiltonian_id": hid,
            "family": family,
            "kind": str(meta.get("kind", family)),
            "seed_base": seed,
            "ndepth": ndepth,
            "nfocks": nfocks,
            "maxiter": maxiter,
            "energy_tensor": np.asarray(meta["energy_tensor"], dtype=float),
            "spsa_a": spsa_a,
            "spsa_c": spsa_c,
            "spsa_A": spsa_A,
            "spsa_alpha": spsa_alpha,
            "spsa_gamma": spsa_gamma,
        }
        if meta.get("instance") is not None:
            job["instance"] = meta["instance"]
        jobs.append(job)
    print(
        f"=== Paper-style energy SPSA (vacuum init, ⟨H⟩), {len(jobs)} Hamiltonians, "
        f"maxiter={maxiter}, workers={workers} ===",
        flush=True,
    )
    records: list[dict] = []
    if workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs, 1):
            rec = run_energy_trial(job)
            records.append(rec)
            _print_job_line(i, len(jobs), rec)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(run_energy_trial, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                done += 1
                _print_job_line(done, len(jobs), rec)
    records.sort(key=lambda r: (str(r.get("family", "")), int(r.get("hamiltonian_id", 0)), int(r["trial"])))
    by_family: dict[str, list[dict]] = {}
    for rec in records:
        by_family.setdefault(str(rec.get("family", "custom")), []).append(rec)
    family_summary = {k: _summarize_group(v) for k, v in by_family.items()}
    n_success = int(sum(bool(r.get("success")) for r in records))
    paired = None
    gpath = Path(gibbs_path) if gibbs_path is not None else outdir / MIXED_JSON
    if gpath.exists():
        gpayload = json.loads(gpath.read_text(encoding="utf-8"))
        paired = _paired_compare(records, list(gpayload.get("trials", [])))
    payload = {
        "method": "energy_spsa",
        "initial_state": "vacuum",
        "objective": "energy",
        "n_hamiltonians": len(jobs),
        "n_trials_per_hamiltonian": 1,
        "maxiter": maxiter,
        "ndepth": ndepth,
        "nfocks": list(nfocks),
        "seed_base": seed_base,
        "workers": workers,
        "n_success": n_success,
        "success_rate": n_success / max(len(records), 1),
        "mean_rel_gap": float(np.mean([r["rel_gap"] for r in records])) if records else float("nan"),
        "by_family": family_summary,
        "paired_vs_gibbs": paired,
        "hamiltonian_file": str(ham_path),
        "trials": records,
        "spsa": {
            "a": spsa_a,
            "c": spsa_c,
            "A": spsa_A,
            "alpha": spsa_alpha,
            "gamma": spsa_gamma,
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / ENERGY_JSON
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    print(
        f"  energy-SPSA exact-GS hits {n_success}/{len(records)}  "
        f"mean rel-gap {payload['mean_rel_gap']:.3f}",
        flush=True,
    )
    for k, s in family_summary.items():
        extra = ""
        if k == "knapsack":
            extra = f"  feas {s['n_feasible']}/{s['n']}"
        print(
            f"  {k}: {s['n_success']}/{s['n']}{extra}  mean rel-gap {s['mean_rel_gap']:.3f}",
            flush=True,
        )
    if paired is not None:
        print(
            f"  paired vs Gibbs: energy {paired['energy_hits']}/{paired['n_paired']}  "
            f"Gibbs {paired['gibbs_hits']}/{paired['n_paired']}  "
            f"Gibbs-only {paired['gibbs_only']}  energy-only {paired['energy_only']}  "
            f"both {paired['both']}  neither {paired['neither']}",
            flush=True,
        )
        for fam, s in paired["by_family"].items():
            print(
                f"    {fam}: energy {s['energy_hits']}/{s['n']}  "
                f"Gibbs {s['gibbs_hits']}/{s['n']}  "
                f"Gibbs-only {s['gibbs_only']}  energy-only {s['energy_only']}",
                flush=True,
            )
    return payload


def _pair_key(rec: dict) -> tuple:
    return (str(rec.get("family", "")), int(rec.get("hamiltonian_id", -1)), int(rec["trial"]))


def _schedule_venn(by_schedule: dict[str, list[dict]], names: list[str]) -> dict:
    maps = {n: {_pair_key(r): r for r in by_schedule.get(n, [])} for n in names}
    keys = sorted(set.intersection(*(set(m) for m in maps.values()))) if maps else []
    counts: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = {}
    for key in keys:
        bits = tuple(bool(maps[n][key].get("success")) for n in names)
        mask = "|".join(n for n, b in zip(names, bits) if b) or "none"
        counts[mask] = counts.get(mask, 0) + 1
        fam = key[0]
        slot = by_family.setdefault(fam, {})
        slot[mask] = slot.get(mask, 0) + 1
    pairwise: dict[str, dict] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = a_only = b_only = neither = 0
            for key in keys:
                ha = bool(maps[a][key].get("success"))
                hb = bool(maps[b][key].get("success"))
                if ha and hb:
                    both += 1
                elif ha:
                    a_only += 1
                elif hb:
                    b_only += 1
                else:
                    neither += 1
            pairwise[f"{a}_vs_{b}"] = {
                "both": both,
                f"{a}_only": a_only,
                f"{b}_only": b_only,
                "neither": neither,
            }
    return {"n_paired": len(keys), "masks": counts, "pairwise": pairwise, "by_family": by_family}


def _jobs_from_saved_starts(
    ham_meta: list,
    starts: dict,
    payload_seed: int,
    schedules: list[dict],
    *,
    ndepth: int,
    nfocks: tuple[int, int],
    spsa_a: float,
    spsa_c: float,
    spsa_A: float,
    spsa_alpha: float,
    spsa_gamma: float,
) -> tuple[list[dict], int]:
    ham_by_key = {
        (str(h.get("family", h.get("kind", "custom"))), int(h["hamiltonian_id"])): h for h in ham_meta
    }
    jobs: list[dict] = []
    missing = 0
    for (family, hid), meta in ham_by_key.items():
        start = starts.get((family, hid, 0))
        if start is None:
            missing += 1
            continue
        job_seed = payload_seed + (10_000 if family == "ising" else 0) + 100 * hid
        for sched in schedules:
            job = {
                "trial": 0,
                "hamiltonian_id": hid,
                "family": family,
                "kind": str(meta.get("kind", family)),
                "seed_base": job_seed,
                "prep0": vacuum_start()[1],
                "prep_unit": vacuum_start()[0],
                "x0": np.asarray(start["x0"], dtype=float),
                "energy_tensor": np.asarray(meta["energy_tensor"], dtype=float),
                "ndepth": ndepth,
                "nfocks": nfocks,
                "outer_iter": sched["outer_iter"],
                "spsa_iter": sched["spsa_iter"],
                "prep_step_scale": sched["prep_step_scale"],
                "schedule": sched["name"],
                "spsa_a": spsa_a,
                "spsa_c": spsa_c,
                "spsa_A": spsa_A,
                "spsa_alpha": spsa_alpha,
                "spsa_gamma": spsa_gamma,
            }
            if meta.get("instance") is not None:
                job["instance"] = meta["instance"]
            jobs.append(job)
    return jobs, missing


def _summarize_by_schedule(records: list[dict], names: list[str]) -> tuple[dict, dict]:
    by_schedule: dict[str, list[dict]] = {}
    for rec in records:
        by_schedule.setdefault(str(rec.get("schedule") or "unknown"), []).append(rec)
    schedule_summary = {}
    for name in names:
        recs = by_schedule.get(name, [])
        fams: dict[str, list[dict]] = {}
        for rec in recs:
            fams.setdefault(str(rec.get("family", "custom")), []).append(rec)
        schedule_summary[name] = {
            **_summarize_group(recs),
            "by_family": {k: _summarize_group(v) for k, v in fams.items()},
        }
    return by_schedule, schedule_summary


def run_prep_schedule_compare(
    *,
    ham_path: Path,
    starts_path: Path,
    workers: int,
    seed_base: int,
    outdir: Path,
    ndepth: int,
    nfocks: tuple[int, int],
    spsa_a: float,
    spsa_c: float,
    spsa_A: float,
    spsa_alpha: float,
    spsa_gamma: float,
    prep_step_scale: float,
    outer_iter: int,
    spsa_iter: int,
) -> dict:
    """Same 40 H and starts: joint-70 (current default) vs freeze 20+50 vs scaled-prep."""
    del outer_iter, spsa_iter  # schedules fix the budgets; CLI values are unused
    ham_meta = json.loads(Path(ham_path).read_text(encoding="utf-8"))
    if not isinstance(ham_meta, list) or not ham_meta:
        raise ValueError(f"expected a non-empty Hamiltonian list in {ham_path}")
    starts_payload = json.loads(Path(starts_path).read_text(encoding="utf-8"))
    start_rows = list(starts_payload.get("trials", []))
    starts = {_pair_key(r): r for r in start_rows}
    payload_seed = int(starts_payload.get("seed_base", seed_base))
    ham_by_key = {(str(h.get("family", h.get("kind", "custom"))), int(h["hamiltonian_id"])): h for h in ham_meta}

    schedules = [
        {"name": "joint70", "outer_iter": 70, "spsa_iter": 0, "prep_step_scale": 1.0},
        {"name": "freeze20_50", "outer_iter": 20, "spsa_iter": 50, "prep_step_scale": 1.0},
        {
            "name": "scaled_prep",
            "outer_iter": 20,
            "spsa_iter": 50,
            "prep_step_scale": float(prep_step_scale),
        },
    ]
    jobs: list[dict] = []
    missing = 0
    for (family, hid), meta in ham_by_key.items():
        start = starts.get((family, hid, 0))
        if start is None:
            missing += 1
            continue
        if family == "ising":
            job_seed = payload_seed + 10_000 + 100 * hid
        else:
            job_seed = payload_seed + 100 * hid
        for sched in schedules:
            job = {
                "trial": 0,
                "hamiltonian_id": hid,
                "family": family,
                "kind": str(meta.get("kind", family)),
                "seed_base": job_seed,
                "prep0": vacuum_start()[1],
                "prep_unit": vacuum_start()[0],
                "x0": np.asarray(start["x0"], dtype=float),
                "energy_tensor": np.asarray(meta["energy_tensor"], dtype=float),
                "ndepth": ndepth,
                "nfocks": nfocks,
                "outer_iter": sched["outer_iter"],
                "spsa_iter": sched["spsa_iter"],
                "prep_step_scale": sched["prep_step_scale"],
                "schedule": sched["name"],
                "spsa_a": spsa_a,
                "spsa_c": spsa_c,
                "spsa_A": spsa_A,
                "spsa_alpha": spsa_alpha,
                "spsa_gamma": spsa_gamma,
            }
            if meta.get("instance") is not None:
                job["instance"] = meta["instance"]
            jobs.append(job)
    if missing:
        raise ValueError(f"{missing} Hamiltonian(s) in {ham_path} have no matching start in {starts_path}")
    n_h = len(ham_by_key)
    print(
        f"=== Prep-schedule A/B/C on {n_h} Hamiltonians × {len(schedules)} schedules "
        f"({len(jobs)} jobs), workers={workers} ===",
        flush=True,
    )
    for sched in schedules:
        print(
            f"  {sched['name']}: joint {sched['outer_iter']} + ansatz {sched['spsa_iter']}  "
            f"prep_step_scale={sched['prep_step_scale']}",
            flush=True,
        )
    t0 = time.perf_counter()
    records = _run_jobs(jobs, workers)
    elapsed = time.perf_counter() - t0

    by_schedule: dict[str, list[dict]] = {}
    for rec in records:
        by_schedule.setdefault(str(rec.get("schedule") or "unknown"), []).append(rec)
    schedule_summary = {}
    names = [s["name"] for s in schedules]
    for name in names:
        recs = by_schedule.get(name, [])
        fams: dict[str, list[dict]] = {}
        for rec in recs:
            fams.setdefault(str(rec.get("family", "custom")), []).append(rec)
        schedule_summary[name] = {
            **_summarize_group(recs),
            "by_family": {k: _summarize_group(v) for k, v in fams.items()},
        }
    venn = _schedule_venn(by_schedule, names)
    payload = {
        "n_hamiltonians": n_h,
        "n_jobs": len(jobs),
        "eta_policy": "sampled_tail",
        "ndepth": ndepth,
        "nfocks": list(nfocks),
        "seed_base": payload_seed,
        "workers": workers,
        "elapsed_sec": elapsed,
        "prep_step_scale_scaled": float(prep_step_scale),
        "schedules": schedules,
        "by_schedule": schedule_summary,
        "venn": venn,
        "hamiltonian_file": str(ham_path),
        "starts_file": str(starts_path),
        "trials": records,
        "spsa": {
            "a": spsa_a,
            "c": spsa_c,
            "A": spsa_A,
            "alpha": spsa_alpha,
            "gamma": spsa_gamma,
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / SCHEDULE_JSON
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    print(f"  wall time {elapsed:.1f}s for {len(jobs)} jobs ({workers} workers)", flush=True)
    for name in names:
        s = schedule_summary[name]
        print(
            f"  {name}: {s['n_success']}/{s['n']}  mean rel-gap {s['mean_rel_gap']:.3f}  "
            f"mean η {s['mean_eta']:.3f}",
            flush=True,
        )
        for fam, fs in s["by_family"].items():
            extra = f"  feas {fs['n_feasible']}/{fs['n']}" if fam == "knapsack" else ""
            print(
                f"    {fam}: {fs['n_success']}/{fs['n']}{extra}  mean rel-gap {fs['mean_rel_gap']:.3f}",
                flush=True,
            )
    print("  pairwise exact-GS:", flush=True)
    for pair, s in venn["pairwise"].items():
        extras = "  ".join(f"{k}={v}" for k, v in s.items() if k != "both")
        print(f"    {pair}: both={s['both']}  {extras}", flush=True)
    print("  3-way masks:", flush=True)
    for mask, n in sorted(venn["masks"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {mask}: {n}", flush=True)
    return payload


def run_joint_step_sweep(
    *,
    ham_path: Path,
    starts_path: Path,
    workers: int,
    seed_base: int,
    outdir: Path,
    ndepth: int,
    nfocks: tuple[int, int],
    spsa_a: float,
    spsa_c: float,
    spsa_A: float,
    spsa_alpha: float,
    spsa_gamma: float,
    prep_step_scale: float,
    outer_iter: int,
    spsa_iter: int,
    budgets: tuple[int, ...] = JOINT_STEP_BUDGETS,
) -> dict:
    """Fully joint SPSA (prep never frozen) at several step budgets, same H and starts."""
    del outer_iter, spsa_iter, prep_step_scale
    ham_meta = json.loads(Path(ham_path).read_text(encoding="utf-8"))
    if not isinstance(ham_meta, list) or not ham_meta:
        raise ValueError(f"expected a non-empty Hamiltonian list in {ham_path}")
    starts_payload = json.loads(Path(starts_path).read_text(encoding="utf-8"))
    starts = {_pair_key(r): r for r in starts_payload.get("trials", [])}
    payload_seed = int(starts_payload.get("seed_base", seed_base))
    schedules = [
        {"name": f"joint{int(n)}", "outer_iter": int(n), "spsa_iter": 0, "prep_step_scale": 1.0}
        for n in budgets
    ]
    jobs, missing = _jobs_from_saved_starts(
        ham_meta,
        starts,
        payload_seed,
        schedules,
        ndepth=ndepth,
        nfocks=nfocks,
        spsa_a=spsa_a,
        spsa_c=spsa_c,
        spsa_A=spsa_A,
        spsa_alpha=spsa_alpha,
        spsa_gamma=spsa_gamma,
    )
    if missing:
        raise ValueError(f"{missing} Hamiltonian(s) in {ham_path} have no matching start in {starts_path}")
    n_h = len({(str(h.get("family", h.get("kind", "custom"))), int(h["hamiltonian_id"])) for h in ham_meta})
    print(
        f"=== Joint-step sweep on {n_h} Hamiltonians × {len(schedules)} budgets "
        f"({len(jobs)} jobs), workers={workers} ===",
        flush=True,
    )
    for sched in schedules:
        print(f"  {sched['name']}: {_budget_label(sched['outer_iter'], sched['spsa_iter'])}", flush=True)
    t0 = time.perf_counter()
    records = _run_jobs(jobs, workers)
    elapsed = time.perf_counter() - t0
    path = Path(outdir) / JOINT_STEPS_JSON
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        new_names = {s["name"] for s in schedules}
        kept = [r for r in old.get("trials", []) if r.get("schedule") not in new_names]
        if kept:
            records = kept + records
            elapsed += float(old.get("elapsed_sec") or 0.0)
            schedules = [s for s in old.get("schedules", []) if s.get("name") not in new_names] + schedules
            budgets = tuple(sorted({int(s["outer_iter"]) for s in schedules}))
            print(f"  merged {len(kept)} saved trials from {path}", flush=True)
    names = [s["name"] for s in sorted(schedules, key=lambda s: int(s["outer_iter"]))]
    by_schedule, schedule_summary = _summarize_by_schedule(records, names)

    saved70 = None
    abc_path = Path(outdir) / SCHEDULE_JSON
    if abc_path.exists():
        abc = json.loads(abc_path.read_text(encoding="utf-8"))
        joint70 = [r for r in abc.get("trials", []) if r.get("schedule") == "joint70"]
        if joint70:
            saved70 = joint70
            by_schedule["joint70"] = joint70
            fams: dict[str, list[dict]] = {}
            for rec in joint70:
                fams.setdefault(str(rec.get("family", "custom")), []).append(rec)
            schedule_summary["joint70"] = {
                **_summarize_group(joint70),
                "by_family": {k: _summarize_group(v) for k, v in fams.items()},
            }
            names = sorted(set(names) | {"joint70"}, key=lambda n: int(str(n).replace("joint", "")))

    venn = _schedule_venn(by_schedule, names)
    payload = {
        "n_hamiltonians": n_h,
        "n_jobs": len(jobs),
        "eta_policy": "sampled_tail",
        "ndepth": ndepth,
        "nfocks": list(nfocks),
        "seed_base": payload_seed,
        "workers": workers,
        "elapsed_sec": elapsed,
        "prep_frozen": False,
        "budgets": [int(n) for n in budgets],
        "schedules": schedules,
        "by_schedule": schedule_summary,
        "venn": venn,
        "joint70_from": str(abc_path) if saved70 is not None else None,
        "hamiltonian_file": str(ham_path),
        "starts_file": str(starts_path),
        "trials": records,
        "spsa": {
            "a": spsa_a,
            "c": spsa_c,
            "A": spsa_A,
            "alpha": spsa_alpha,
            "gamma": spsa_gamma,
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / JOINT_STEPS_JSON
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    print(f"  wall time {elapsed:.1f}s for {len(jobs)} jobs ({workers} workers)", flush=True)
    for name in names:
        s = schedule_summary[name]
        extra = "  [saved]" if name == "joint70" else ""
        print(
            f"  {name}: {s['n_success']}/{s['n']}  ({100.0 * s['success_rate']:.1f}%)  "
            f"mean rel-gap {s['mean_rel_gap']:.3f}  mean η {s['mean_eta']:.3f}{extra}",
            flush=True,
        )
        for fam, fs in s["by_family"].items():
            feas = f"  feas {fs['n_feasible']}/{fs['n']}" if fam == "knapsack" else ""
            print(
                f"    {fam}: {fs['n_success']}/{fs['n']}{feas}  mean rel-gap {fs['mean_rel_gap']:.3f}",
                flush=True,
            )
    print("  pairwise exact-GS:", flush=True)
    for pair, s in venn["pairwise"].items():
        extras = "  ".join(f"{k}={v}" for k, v in s.items() if k != "both")
        print(f"    {pair}: both={s['both']}  {extras}", flush=True)
    return payload


def run_experiment(
    *,
    n_trials: int,
    outer_iter: int,
    spsa_iter: int,
    workers: int,
    seed_base: int,
    outdir: Path,
    ndepth: int,
    nfocks: tuple[int, int],
    spsa_a: float,
    spsa_c: float,
    spsa_A: float,
    spsa_alpha: float,
    spsa_gamma: float,
) -> dict:
    jobs = []
    for t in range(n_trials):
        unit, prep0 = vacuum_start()
        jobs.append(
            {
                "trial": t,
                "seed_base": seed_base,
                "ndepth": ndepth,
                "nfocks": nfocks,
                "outer_iter": outer_iter,
                "spsa_iter": spsa_iter,
                "prep0": prep0,
                "prep_unit": unit,
                "spsa_a": spsa_a,
                "spsa_c": spsa_c,
                "spsa_A": spsa_A,
                "spsa_alpha": spsa_alpha,
                "spsa_gamma": spsa_gamma,
            }
        )
    print(
        f"=== Gibbs adaptive SPSA (sampled_tail η, noiseless): {n_trials} trials, "
        f"{_budget_label(outer_iter, spsa_iter)}, workers={workers} ===",
        flush=True,
    )
    records = _run_jobs(jobs, workers)
    records.sort(key=lambda r: r["trial"])
    payload = {
        "n_trials": n_trials,
        **_budget_fields(outer_iter, spsa_iter),
        "ndepth": ndepth,
        "nfocks": list(nfocks),
        "seed_base": seed_base,
        "workers": workers,
        "eta_policy": "sampled_tail",
        "mean_eta": float(np.mean([r["eta"] for r in records])) if records else float("nan"),
        "spsa": {
            "a": spsa_a,
            "c": spsa_c,
            "A": spsa_A,
            "alpha": spsa_alpha,
            "gamma": spsa_gamma,
        },
        "summary": summarize_trials(records),
        "trials": records,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / OUTPUT_JSON
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    s = payload["summary"]
    print(
        f"  mean Gibbs={s['mean_cost']:.4f}  mean E_qubo={s['mean_energy_qubo']:.4f}  "
        f"best Gibbs={s['best_cost']:.4f} (trial {s['best_trial']})  "
        f"mean η={payload['mean_eta']:.3f}",
        flush=True,
    )
    return payload


def _ham_meta_for_json(inst: dict) -> dict:
    return {k: v for k, v in inst.items() if k != "energy_tensor"}


def run_mixed_p_spin_suite(
    *,
    ham_dir: Path,
    n_trials: int,
    outer_iter: int,
    spsa_iter: int,
    workers: int,
    seed_base: int,
    outdir: Path,
    ndepths: tuple[int, ...] | list[int],
    nfocks: tuple[int, int],
    spsa_a: float,
    spsa_c: float,
    spsa_A: float,
    spsa_alpha: float,
    spsa_gamma: float,
    ansatz: str = "ecd",
    output: Path | None = None,
    max_hamiltonians: int | None = None,
) -> dict:
    """Layer sweep of joint prep+ansatz Gibbs SPSA on saved mixed p-spin NPZs."""
    ansatz = str(ansatz).lower()
    depths = tuple(int(d) for d in ndepths)
    if not depths or any(d < 1 for d in depths):
        raise ValueError(f"ndepths must be positive integers, got {ndepths}")
    instances = load_mixed_p_spin_instances(
        ham_dir, nfocks=nfocks, glob=MIXED_P_SPIN_NPZ_GLOB, max_hamiltonians=max_hamiltonians
    )
    if not instances:
        raise FileNotFoundError(f"no mixed p-spin instances in {ham_dir}")
    n_t = max(int(n_trials), 1)
    ham_meta = [_ham_meta_for_json(inst) for inst in instances]
    inventories = [ansatz_inventory(ansatz, d, nfocks, n_prep_params=N_PREP_PARAMS) for d in depths]
    print("=== Mixed p-spin instances (landscape, before VQE) ===", flush=True)
    print(f"{'H':>3}  {'file':<28}  {'Emin':>8}  {'spread':>8}  {'gap':>6}", flush=True)
    for meta in ham_meta:
        print(
            f"{int(meta['hamiltonian_id']):3d}  {str(meta['file']):<28}  "
            f"{float(meta['energy_min']):8.3f}  {float(meta['spread']):8.3f}  "
            f"{float(meta['gap']):6.3f}",
            flush=True,
        )
    print("=== Ansatz resource counts ===", flush=True)
    for inv in inventories:
        print(
            f"  {ansatz} L{inv['ndepth']}: {inv['n_ansatz_params']} ansatz params "
            f"(+{inv['n_prep_params']} prep = {inv['n_params']} total), "
            f"{inv['n_primitive_gates']} primitive gates "
            f"({inv['gates_per_layer']}/layer: {', '.join(inv['primitive_gates'])})",
            flush=True,
        )
        print(f"    {inv['description']}", flush=True)

    jobs: list[dict] = []
    spsa_fields = _spsa_job_fields(
        ndepth=depths[0],
        nfocks=nfocks,
        outer_iter=outer_iter,
        spsa_iter=spsa_iter,
        spsa_a=spsa_a,
        spsa_c=spsa_c,
        spsa_A=spsa_A,
        spsa_alpha=spsa_alpha,
        spsa_gamma=spsa_gamma,
    )
    for inst in instances:
        hid = int(inst["hamiltonian_id"])
        for depth in depths:
            for t in range(n_t):
                unit, prep0 = vacuum_start()
                job = {
                    **spsa_fields,
                    "trial": t,
                    "hamiltonian_id": hid,
                    "family": "mixed_p_spin",
                    "kind": "mixed_p_spin",
                    "file": inst["file"],
                    "ansatz": ansatz,
                    "ndepth": int(depth),
                    "seed_base": seed_base + 1000 * int(depth) + 100 * hid,
                    "prep0": prep0,
                    "prep_unit": unit,
                    "energy_tensor": inst["energy_tensor"],
                }
                jobs.append(job)

    print(
        f"=== Mixed p-spin {ansatz} Gibbs SPSA, {len(instances)} H × {n_t} trial(s) × "
        f"{len(depths)} depth(s), {_budget_label(outer_iter, spsa_iter)}, "
        f"workers={workers} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    records = _run_jobs(jobs, workers)
    elapsed = time.perf_counter() - t0

    by_ndepth: dict[str, list[dict]] = {}
    for rec in records:
        by_ndepth.setdefault(str(int(rec["ndepth"])), []).append(rec)
    depth_summary = {}
    for depth, inv in zip(depths, inventories):
        recs = by_ndepth.get(str(int(depth)), [])
        depth_summary[str(int(depth))] = {**inv, **_summarize_group(recs)}

    n_success = int(sum(bool(r.get("success")) for r in records))
    method = f"{ansatz}_gibbs_spsa"
    payload = {
        "method": method,
        "ansatz": ansatz,
        "objective": "gibbs",
        "eta_policy": "sampled_tail",
        "initial_state": "vacuum",
        "n_hamiltonians": len(instances),
        "n_trials_per_hamiltonian": n_t,
        "ndepths": [int(d) for d in depths],
        "inventories": inventories,
        "by_ndepth": depth_summary,
        **_budget_fields(outer_iter, spsa_iter),
        "maxiter": int(outer_iter) + int(spsa_iter),
        "nfocks": list(nfocks),
        "seed_base": seed_base,
        "workers": workers,
        "elapsed_sec": elapsed,
        "ham_dir": str(Path(ham_dir)),
        "n_success": n_success,
        "success_rate": n_success / max(len(records), 1),
        "mean_rel_gap": float(np.mean([r["rel_gap"] for r in records])) if records else float("nan"),
        "mean_eta": float(np.mean([r["eta"] for r in records])) if records else float("nan"),
        "mean_energy_physical": (
            float(np.mean([r["energy_physical"] for r in records])) if records else float("nan")
        ),
        "mean_nfev": float(np.mean([r["nfev"] for r in records])) if records else float("nan"),
        "hamiltonians": ham_meta,
        "trials": records,
        "spsa": {
            "a": spsa_a,
            "c": spsa_c,
            "A": spsa_A,
            "alpha": spsa_alpha,
            "gamma": spsa_gamma,
        },
    }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if output is None:
        path = outdir / (MIXED_P_SPIN_JSON if ansatz == "ecd" else f"gibbs_mixed_p_spin_{ansatz}.json")
    else:
        path = Path(output)
        if not path.is_absolute():
            path = outdir / path.name if path.parent == Path(".") else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    print(
        f"  {ansatz} exact-GS hits {n_success}/{len(records)}  "
        f"pooled rate {payload['success_rate']:.3f}  "
        f"mean rel-gap {payload['mean_rel_gap']:.3f}  "
        f"wall {elapsed:.1f}s",
        flush=True,
    )
    for depth in depths:
        s = depth_summary[str(int(depth))]
        print(
            f"  L{int(depth)}: {s['n_success']}/{s['n']}  "
            f"params={s['n_ansatz_params']}+{N_PREP_PARAMS}  "
            f"gates={s['n_primitive_gates']}  "
            f"success={s['success_rate']:.3f}  "
            f"rel-gap={s['mean_rel_gap']:.3f}  "
            f"⟨H⟩={s['mean_energy_physical']:.3f}  "
            f"nfev={s['mean_nfev']:.0f}  "
            f"{s['mean_elapsed_s']:.1f}s/trial",
            flush=True,
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument(
        "--outer-iter",
        type=int,
        default=None,
        help="Joint SPSA steps on (prep, ansatz). Default 70, or 200 with --mixed-p-spin.",
    )
    parser.add_argument(
        "--spsa-iter",
        type=int,
        default=SPSA_ITERATIONS,
        help="Optional ansatz-only steps after freezing prep. Default 0 (never freeze).",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--ndepth", type=int, default=NDEPTH)
    parser.add_argument(
        "--ndepths",
        type=int,
        nargs="+",
        default=None,
        help="Ansatz layer counts for --mixed-p-spin (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--mixed-p-spin",
        action="store_true",
        help="Sweep ECD depth on saved mixed p-spin NPZ Hamiltonians (200 joint SPSA steps).",
    )
    parser.add_argument("--ham-dir", type=Path, default=MIXED_P_SPIN_DIR)
    parser.add_argument(
        "--max-hamiltonians",
        type=int,
        default=None,
        help="Optional cap on how many mixed p-spin NPZ files to load.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON path for --mixed-p-spin (default: <outdir>/gibbs_mixed_p_spin_ecd.json).",
    )
    parser.add_argument(
        "--n-hamiltonians",
        type=int,
        default=N_HAMILTONIANS,
        help="If >0, sweep this many knapsack-like QUBOs (paper BKP + variants; 1 trial each by default).",
    )
    parser.add_argument("--ham-seed", type=int, default=HAM_SEED)
    parser.add_argument(
        "--n-knapsack",
        type=int,
        default=N_KNAPSACK,
        help="Random paper-encoding knapsacks (different values/weights/capacity).",
    )
    parser.add_argument(
        "--n-ising",
        type=int,
        default=N_ISING,
        help="Random 7-spin diagonal Ising Hamiltonians (Z + ZZ).",
    )
    parser.add_argument(
        "--energy-baseline",
        action="store_true",
        help="Paper-style vacuum + ⟨H⟩ + SPSA on saved Hamiltonians (budget = outer+spsa, 70 by default).",
    )
    parser.add_argument(
        "--hamiltonians",
        type=Path,
        default=None,
        help="Path to hamiltonians.json (default: <outdir>/hamiltonians.json).",
    )
    parser.add_argument(
        "--compare-prep-schedules",
        action="store_true",
        help="Joint-70 (current default) vs freeze 20+50 vs scaled-prep 20+50 on saved Hamiltonians and starts.",
    )
    parser.add_argument(
        "--joint-step-sweep",
        action="store_true",
        help="Fully joint SPSA at several step budgets on saved Hamiltonians and starts.",
    )
    parser.add_argument(
        "--joint-steps",
        type=int,
        nargs="+",
        default=None,
        help="Joint SPSA step budgets for --joint-step-sweep (default: 40 50 80 100 150 200).",
    )
    parser.add_argument(
        "--starts",
        type=Path,
        default=None,
        help="Path to gibbs_mixed_40.json with prep0/x0 (default: <outdir>/gibbs_mixed_40.json).",
    )
    parser.add_argument(
        "--prep-step-scale",
        type=float,
        default=PREP_STEP_SCALE,
        help="SPSA gain on the five prep coordinates for the scaled_prep schedule.",
    )
    args = parser.parse_args(argv)
    outer_iter = int(args.outer_iter) if args.outer_iter is not None else (
        MIXED_P_SPIN_STEPS if args.mixed_p_spin else OUTER_ITERATIONS
    )

    spsa_kw = dict(
        outer_iter=outer_iter,
        spsa_iter=args.spsa_iter,
        workers=args.workers,
        seed_base=args.seed_base,
        outdir=args.outdir,
        ndepth=args.ndepth,
        nfocks=NFOCKS,
        spsa_a=SPSA_A,
        spsa_c=SPSA_C,
        spsa_A=SPSA_A_STAB,
        spsa_alpha=SPSA_ALPHA,
        spsa_gamma=SPSA_GAMMA,
    )
    if args.mixed_p_spin:
        n_trials = args.n_trials if args.n_trials != N_TRIALS else 1
        run_mixed_p_spin_suite(
            ham_dir=args.ham_dir,
            n_trials=n_trials,
            outer_iter=outer_iter,
            spsa_iter=args.spsa_iter,
            workers=args.workers,
            seed_base=args.seed_base,
            outdir=args.outdir,
            ndepths=tuple(args.ndepths) if args.ndepths else MIXED_P_SPIN_ECD_DEPTHS,
            nfocks=NFOCKS,
            spsa_a=SPSA_A,
            spsa_c=SPSA_C,
            spsa_A=SPSA_A_STAB,
            spsa_alpha=SPSA_ALPHA,
            spsa_gamma=SPSA_GAMMA,
            ansatz="ecd",
            output=args.output,
            max_hamiltonians=args.max_hamiltonians,
        )
        return 0
    if args.energy_baseline:
        ham_path = args.hamiltonians or (args.outdir / HAMILTONIANS_JSON)
        run_energy_spsa_baseline(ham_path=ham_path, **spsa_kw)
        return 0
    if args.compare_prep_schedules:
        ham_path = args.hamiltonians or (args.outdir / HAMILTONIANS_JSON)
        starts_path = args.starts or (args.outdir / MIXED_JSON)
        run_prep_schedule_compare(
            ham_path=ham_path,
            starts_path=starts_path,
            prep_step_scale=args.prep_step_scale,
            **spsa_kw,
        )
        return 0
    if args.joint_step_sweep:
        ham_path = args.hamiltonians or (args.outdir / HAMILTONIANS_JSON)
        starts_path = args.starts or (args.outdir / MIXED_JSON)
        run_joint_step_sweep(
            ham_path=ham_path,
            starts_path=starts_path,
            prep_step_scale=args.prep_step_scale,
            budgets=tuple(args.joint_steps) if args.joint_steps else JOINT_STEP_BUDGETS,
            **spsa_kw,
        )
        return 0
    if args.n_knapsack > 0 or args.n_ising > 0:
        n_trials = args.n_trials if args.n_trials != N_TRIALS else 1
        run_mixed_suite(
            n_knapsack=args.n_knapsack,
            n_ising=args.n_ising,
            n_trials=n_trials,
            ham_seed=args.ham_seed,
            **spsa_kw,
        )
        return 0
    if args.n_hamiltonians > 0:
        n_trials = args.n_trials if args.n_trials != N_TRIALS else 1
        run_knapsack_hamiltonian_sweep(
            n_hamiltonians=args.n_hamiltonians,
            n_trials=n_trials,
            ham_seed=args.ham_seed,
            **spsa_kw,
        )
        return 0

    run_experiment(
        n_trials=args.n_trials,
        **spsa_kw,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
