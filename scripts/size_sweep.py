#!/usr/bin/env python3
"""HEA vs ECD Gibbs size sweep on the Dutta hybrid ladder (n=9, 11).

Does not overwrite results/hamiltonians.json or any stored n=7 JSON.
Does not edit src/qumode_vqe/eta.py. No QAOA. No energy-SPSA primary.

Protocols match the n=7 Gibbs suite: sampled_tail η, run_spsa a=0.2, c=0.15,
A=10, α=0.602, γ=0.101, 70 steps, one random start per H. Success = most-likely
computational bitstring is an exact ground (atol=1e-8).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from qumode_vqe.circuit import N_PREP_PARAMS, coherent_radius_max
from qumode_vqe.hamiltonian import (
    bits_from_qnm,
    bitstring_from_bits,
    bkp_instance_as_dict,
    bkp_instance_from_dict,
    energy_spectrum_stats,
    energy_tensor_from_z_terms,
    hybrid_energy_tensor,
    hybrid_ladder,
    ising_benchmark_terms,
    knapsack_benchmark_instance,
    knapsack_packing_stats,
    max_pauli_weight,
    qnm_from_bits,
    z_terms_as_list,
)
from qumode_vqe.params import n_parameters, random_parameters
from qumode_vqe.qaoa import (
    DEFAULT_HEA_LAYERS,
    energy_vector_from_tensor,
    n_hea_params,
    optimize_qubit_ansatz,
    random_hea_params,
    verify_energy_vector,
)
from qumode_vqe.vqe import HybridSimulator, optimize_gibbs_adaptive

def _limit_blas_threads(n: int = 1) -> None:
    """Avoid 4-worker × OpenMP oversubscription (QuTiP/NumPy matmul)."""
    n_s = str(int(n))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "QUTIP_NUM_PROCESSES",
    ):
        os.environ[key] = n_s
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(int(n))
    except Exception:
        pass


_limit_blas_threads(1)

OUTDIR = Path("results")
SEED_BASE = 4000
# Distinct from n=7 ham_seed=8000. Documented per n.
HAM_SEED = {7: 8000, 9: 9000, 11: 11000, 13: 13000}
HEA_PROTOCOL_OFFSET = 32_000  # same family offset as n=7 HEA Gibbs
PREP_SEED_OFFSET = 50_000  # same as Gibbs_and_adaptive_optim.py
NDEPTH = 5
HEA_LAYERS = DEFAULT_HEA_LAYERS
MAXITER = 70
N_KNAPSACK = 20
N_ISING = 20
N_TRIALS = 1
SPSA = dict(a=0.2, c=0.15, A=10.0, alpha=0.602, gamma=0.101)
KILL_SECONDS_PER_70 = 180.0  # skip ECD n=11 if extrapolated instance > 3 min
GROUND_ATOL = 1e-8


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


def _family_hid_seed(seed_base: int, family: str, hid: int) -> int:
    if str(family) == "ising":
        return int(seed_base) + 10_000 + 100 * int(hid)
    return int(seed_base) + 100 * int(hid)


def hea_trial_seed(seed_base: int, family: str, hid: int, trial: int) -> int:
    return _family_hid_seed(seed_base, family, hid) + HEA_PROTOCOL_OFFSET + int(trial)


def ecd_ansatz_seed(seed_base: int, family: str, hid: int, trial: int) -> int:
    return _family_hid_seed(seed_base, family, hid) + int(trial)


def ecd_prep_seed(seed_base: int, family: str, hid: int, trial: int) -> int:
    return _family_hid_seed(seed_base, family, hid) + PREP_SEED_OFFSET + int(trial)


def unit_to_prep(u: np.ndarray, nfocks: tuple[int, int]) -> np.ndarray:
    """Same map as scripts/Gibbs_and_adaptive_optim.py."""
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
    rng = np.random.default_rng(int(seed))
    unit = rng.random(N_PREP_PARAMS)
    return unit, unit_to_prep(unit, nfocks)


def p_ground_from_tensor(probs: np.ndarray, energy_tensor: np.ndarray) -> float:
    e = np.asarray(energy_tensor, dtype=float).reshape(-1)
    p = np.clip(np.asarray(probs, dtype=float).reshape(-1), 0.0, None)
    total = float(p.sum())
    if total > 0.0:
        p = p / total
    mask = np.isclose(e, float(np.min(e)), atol=GROUND_ATOL, rtol=0.0)
    return float(np.dot(p, mask.astype(float)))


def _summarize(recs: list[dict]) -> dict:
    n = len(recs)
    n_success = int(sum(bool(r.get("success")) for r in recs))
    n_feas = int(sum(bool(r.get("packing", {}).get("feasible")) for r in recs))
    etas = [float(r["eta"]) for r in recs if r.get("eta") is not None]
    pgs = [float(r["p_ground"]) for r in recs if r.get("p_ground") is not None]
    gaps = [float(r["gap_to_min"]) for r in recs if r.get("gap_to_min") is not None]
    return {
        "n": n,
        "n_success": n_success,
        "success_rate": n_success / max(n, 1),
        "n_feasible": n_feas,
        "mean_rel_gap": float(np.mean([r["rel_gap"] for r in recs])) if n else float("nan"),
        "mean_gap_to_min": float(np.mean(gaps)) if gaps else float("nan"),
        "mean_energy_diag": float(np.mean([r["energy_diag"] for r in recs])) if n else float("nan"),
        "mean_p_ground": float(np.mean(pgs)) if pgs else float("nan"),
        "mean_eta": float(np.mean(etas)) if etas else None,
    }


def _print_line(i: int, n: int, rec: dict) -> None:
    feas = rec.get("packing", {}).get("feasible")
    feas_s = "" if feas is None else f"  feas={feas}"
    eta = rec.get("eta")
    eta_s = "" if eta is None else f"  η={float(eta):.3f}"
    pgs = rec.get("p_ground")
    pgs_s = "" if pgs is None else f"  P(gs)={float(pgs):.3f}"
    print(
        f"[{i}/{n}] {rec.get('protocol')}  {rec.get('family')}/H{rec.get('hamiltonian_id')}  "
        f"cost={rec['cost']:.4f}  E={rec['energy_diag']:.3f} "
        f"(min {rec['energy_min']:.3f}, gap={rec['gap_to_min']:.3f})  "
        f"gs={rec['success']}{pgs_s}{feas_s}{eta_s}  bits={rec['most_likely_bitstring']}",
        flush=True,
    )


def generate_hamiltonians(n_qubits: int, ham_seed: int, n_knapsack: int, n_ising: int) -> list[dict]:
    spec = hybrid_ladder(n_qubits)
    part = spec["partition"]
    nfocks = spec["nfocks"]
    metas: list[dict] = []
    print(
        f"=== Generate n={n_qubits}  partition={part}  L={nfocks}  dim={spec['dim']}  "
        f"ham_seed={ham_seed} ===",
        flush=True,
    )
    print(f"{'fam':<9} {'H':>2}  {'Emin':>8}  {'spread':>8}  {'gap':>6}  extra", flush=True)
    for h in range(int(n_knapsack)):
        inst = knapsack_benchmark_instance(
            h, ham_seed, n_primary=spec["n_primary"], n_slack=spec["n_slack"]
        )
        tensor = hybrid_energy_tensor(nfocks, inst, part)
        stats = energy_spectrum_stats(tensor)
        gs_bits = bitstring_from_bits(bits_from_qnm(*stats["ground_qnm"], part))
        energies = energy_vector_from_tensor(tensor, part, n_qubits=n_qubits, ground_bitstring=gs_bits)
        verify_energy_vector(energies, tensor, part, ground_bitstring=gs_bits)
        gs_pack = knapsack_packing_stats(bits_from_qnm(*stats["ground_qnm"], part), inst)
        if not gs_pack["feasible"]:
            raise RuntimeError(f"n={n_qubits} knapsack H{h} QUBO ground is infeasible")
        meta = {
            "family": "knapsack",
            "kind": "knapsack",
            "hamiltonian_id": h,
            "n_qubits": n_qubits,
            "partition": list(part),
            "nfocks": list(nfocks),
            "instance": bkp_instance_as_dict(inst),
            "ground_bitstring": gs_bits,
            "ground_packing": gs_pack,
            "energy_tensor": np.asarray(tensor, dtype=float),
            **stats,
        }
        metas.append(meta)
        print(
            f"{'knapsack':<9} {h:2d}  {stats['energy_min']:8.2f}  {stats['spread']:8.1f}  "
            f"{stats['gap']:6.2f}  C={inst.capacity:.0f} λ={inst.penalty:.2f} "
            f"v={np.asarray(inst.values).tolist()} w={np.asarray(inst.weights).tolist()}",
            flush=True,
        )
    for h in range(int(n_ising)):
        terms = ising_benchmark_terms(h, ham_seed, n_qubits=n_qubits)
        tensor = energy_tensor_from_z_terms(terms, nfocks, part)
        stats = energy_spectrum_stats(tensor)
        gs_bits = bitstring_from_bits(bits_from_qnm(*stats["ground_qnm"], part))
        energies = energy_vector_from_tensor(tensor, part, n_qubits=n_qubits, ground_bitstring=gs_bits)
        verify_energy_vector(energies, tensor, part, ground_bitstring=gs_bits)
        meta = {
            "family": "ising",
            "kind": "ising",
            "hamiltonian_id": h,
            "n_qubits": n_qubits,
            "partition": list(part),
            "nfocks": list(nfocks),
            "terms": z_terms_as_list(terms),
            "max_pauli_weight": int(max_pauli_weight(terms)),
            "n_terms": len(terms),
            "ground_bitstring": gs_bits,
            "energy_tensor": np.asarray(tensor, dtype=float),
            **stats,
        }
        metas.append(meta)
        print(
            f"{'ising':<9} {h:2d}  {stats['energy_min']:8.2f}  {stats['spread']:8.1f}  "
            f"{stats['gap']:6.2f}  n_terms={len(terms)} weight≤{meta['max_pauli_weight']}",
            flush=True,
        )
    return metas


def save_hamiltonians(metas: list[dict], json_path: Path, npz_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_json_ready(metas), indent=2), encoding="utf-8")
    families = np.array([str(h["family"]) for h in metas])
    ids = np.array([int(h["hamiltonian_id"]) for h in metas], dtype=int)
    tensors = np.stack([np.asarray(h["energy_tensor"], dtype=float) for h in metas])
    np.savez_compressed(npz_path, family=families, hamiltonian_id=ids, energy_tensor=tensors)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {npz_path}", flush=True)


def load_hamiltonians(path: Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("hamiltonians", [])
    return raw


def _prepare_meta(meta: dict) -> dict:
    nq = int(meta.get("n_qubits") or (1 + int(meta["partition"][1]) + int(meta["partition"][2])))
    part = tuple(int(v) for v in meta["partition"])
    tensor = np.asarray(meta["energy_tensor"], dtype=float)
    ground = meta.get("ground_bitstring")
    energies = energy_vector_from_tensor(tensor, part, n_qubits=nq, ground_bitstring=ground)
    verify_energy_vector(energies, tensor, part, ground_bitstring=ground)
    return {
        **meta,
        "n_qubits": nq,
        "partition": part,
        "nfocks": (int(meta["nfocks"][0]), int(meta["nfocks"][1])),
        "energy_tensor": tensor,
        "energies": energies,
        "ground_bitstring": None if ground is None else str(ground),
    }


def run_hea_job(job: dict) -> dict:
    _limit_blas_threads(1)
    n_qubits = int(job["n_qubits"])
    n_layers = int(job["n_layers"])
    energies = np.asarray(job["energies"], dtype=float)
    rng = np.random.default_rng(int(job["seed"]))
    x0 = random_hea_params(n_qubits, n_layers, rng)
    result = optimize_qubit_ansatz(
        x0,
        energies,
        ansatz="hea",
        objective="gibbs",
        n_qubits=n_qubits,
        n_layers=n_layers,
        maxiter=int(job["maxiter"]),
        rng=rng,
        **{k: float(job[f"spsa_{k}"]) for k in ("a", "c", "A", "alpha", "gamma")},
    )
    rec = {
        "trial": int(job["trial"]),
        "protocol": "hea_gibbs",
        "ansatz": "hea",
        "objective": "gibbs",
        "family": str(job["family"]),
        "kind": str(job["kind"]),
        "hamiltonian_id": int(job["hamiltonian_id"]),
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_params": int(job["n_params"]),
        "initial_state": "zero",
        "cz_packing": "even_odd",
        "seed": int(job["seed"]),
        "x0": np.asarray(result.x0, dtype=float),
        "x": np.asarray(result.x, dtype=float),
        "eta_policy": result.eta_policy,
        "n_eta_clamps": int(result.n_eta_clamps),
        "n_eta_fallbacks": int(result.n_eta_fallbacks),
    }
    rec.update(result.as_score_dict())
    stored_gs = job.get("ground_bitstring")
    if stored_gs:
        rec["ground_bitstring"] = str(stored_gs)
    part = tuple(int(v) for v in job["partition"])
    bits = np.array([int(ch) for ch in rec["most_likely_bitstring"]], dtype=int)
    rec["most_likely"] = [int(v) for v in qnm_from_bits(bits, part)]
    gs_bits = np.array([int(ch) for ch in rec["ground_bitstring"]], dtype=int)
    rec["ground_qnm"] = [int(v) for v in qnm_from_bits(gs_bits, part)]
    inst_d = job.get("instance")
    if inst_d is not None:
        inst = bkp_instance_from_dict(inst_d)
        rec["packing"] = knapsack_packing_stats(bits, inst)
    return rec


def run_ecd_job(job: dict) -> dict:
    _limit_blas_threads(1)
    ndepth = int(job["ndepth"])
    nfocks = (int(job["nfocks"][0]), int(job["nfocks"][1]))
    part = tuple(int(v) for v in job["partition"])
    rng = np.random.default_rng(int(job["ansatz_seed"]))
    x0 = random_parameters(ndepth, rng)
    prep0 = np.asarray(job["prep0"], dtype=float)
    tensor = np.asarray(job["energy_tensor"], dtype=float)
    result = optimize_gibbs_adaptive(
        prep0,
        x0,
        ndepth=ndepth,
        nfocks=nfocks,
        outer_iter=int(job["maxiter"]),
        spsa_iter=0,
        rng=rng,
        a=float(job["spsa_a"]),
        c=float(job["spsa_c"]),
        A=float(job["spsa_A"]),
        alpha=float(job["spsa_alpha"]),
        gamma=float(job["spsa_gamma"]),
        prep_step_scale=1.0,
        energy_tensor=tensor,
        partition=part,
    )
    sim = HybridSimulator(
        ndepth=ndepth,
        nfocks=nfocks,
        cost_kind="gibbs",
        target_qnm=None,
        partition=part,
        energy_tensor=tensor,
        initial_state=None,
    )
    from qumode_vqe.circuit import prep_params_to_ket, project_prep_params

    prep = project_prep_params(result.prep, nfocks)
    sim.initial_state = prep_params_to_ket(prep, nfocks)
    ev = sim.evaluate(result.x)
    probs = np.asarray(ev.measurement.physical_probs, dtype=float)
    ml = [int(v) for v in ev.most_likely]
    found = float(tensor[tuple(ml)])
    emin = float(np.min(tensor))
    emax = float(np.max(tensor))
    spread = max(emax - emin, 1e-12)
    rec = {
        "trial": int(job["trial"]),
        "protocol": "ecd_gibbs_joint70",
        "ansatz": "ecd",
        "objective": "gibbs",
        "family": str(job["family"]),
        "kind": str(job["kind"]),
        "hamiltonian_id": int(job["hamiltonian_id"]),
        "n_qubits": int(job["n_qubits"]),
        "ndepth": ndepth,
        "nfocks": list(nfocks),
        "partition": list(part),
        "n_params": int(job["n_params"]),
        "n_ecd_params": int(n_parameters(ndepth)),
        "n_prep_params": N_PREP_PARAMS,
        "initial_state": "product_prep",
        "ansatz_seed": int(job["ansatz_seed"]),
        "prep_seed": int(job["prep_seed"]),
        "prep_unit": np.asarray(job["prep_unit"], dtype=float),
        "prep0": np.asarray(result.prep0, dtype=float),
        "prep": np.asarray(result.prep, dtype=float),
        "x0": np.asarray(result.x0, dtype=float),
        "x": np.asarray(result.x, dtype=float),
        "cost": float(result.fun),
        "energy_physical": float(ev.energy_physical),
        "energy_diag": found,
        "energy_min": emin,
        "energy_max": emax,
        "gap_to_min": found - emin,
        "rel_gap": (found - emin) / spread,
        "p_ground": p_ground_from_tensor(probs, tensor),
        "p_most_likely": float(probs[tuple(ml)]),
        "most_likely": ml,
        "most_likely_bitstring": str(ev.most_likely_bitstring),
        "ground_qnm": [int(v) for v in np.unravel_index(int(np.argmin(tensor)), tensor.shape)],
        "ground_bitstring": str(job["ground_bitstring"]),
        "success": bool(np.isclose(found, emin, atol=GROUND_ATOL, rtol=0.0)),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "eta_policy": "sampled_tail",
        "eta0": float(result.eta0),
        "eta": float(result.eta),
        "n_eta_clamps": int(result.n_eta_clamps),
        "n_eta_fallbacks": int(result.n_eta_fallbacks),
    }
    inst_d = job.get("instance")
    if inst_d is not None:
        inst = bkp_instance_from_dict(inst_d)
        rec["packing"] = knapsack_packing_stats(bits_from_qnm(*ml, part), inst)
    return rec


def _checkpoint(path: Path | None, records: list[dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(_json_ready({"n_done": len(records), "trials": records}), indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_partial_trials(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw.get("trials") or [])


def _run_pool(
    jobs: list[dict],
    worker,
    workers: int,
    checkpoint_path: Path | None = None,
    prior: list[dict] | None = None,
) -> list[dict]:
    records: list[dict] = list(prior or [])
    if not jobs:
        records.sort(key=lambda r: (str(r.get("family", "")), int(r.get("hamiltonian_id", 0)), int(r["trial"])))
        return records
    if workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs, 1):
            rec = worker(job)
            records.append(rec)
            _print_line(i, len(jobs), rec)
            if checkpoint_path is not None:
                _checkpoint(checkpoint_path, records)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(worker, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                done += 1
                _print_line(done, len(jobs), rec)
                if checkpoint_path is not None:
                    _checkpoint(checkpoint_path, records)
    records.sort(key=lambda r: (str(r.get("family", "")), int(r.get("hamiltonian_id", 0)), int(r["trial"])))
    return records


def _spsa_job_fields(maxiter: int) -> dict:
    return {
        "maxiter": int(maxiter),
        "spsa_a": SPSA["a"],
        "spsa_c": SPSA["c"],
        "spsa_A": SPSA["A"],
        "spsa_alpha": SPSA["alpha"],
        "spsa_gamma": SPSA["gamma"],
    }


def build_hea_jobs(instances: list[dict], *, seed_base: int, maxiter: int, n_layers: int) -> list[dict]:
    jobs = []
    for inst in instances:
        nq = int(inst["n_qubits"])
        for trial in range(N_TRIALS):
            jobs.append(
                {
                    "trial": trial,
                    "family": inst["family"],
                    "kind": inst["kind"],
                    "hamiltonian_id": inst["hamiltonian_id"],
                    "energies": inst["energies"],
                    "ground_bitstring": inst["ground_bitstring"],
                    "instance": inst.get("instance"),
                    "partition": list(inst["partition"]),
                    "n_qubits": nq,
                    "n_layers": n_layers,
                    "n_params": n_hea_params(nq, n_layers),
                    "seed": hea_trial_seed(seed_base, inst["family"], inst["hamiltonian_id"], trial),
                    **_spsa_job_fields(maxiter),
                }
            )
    return jobs


def build_ecd_jobs(instances: list[dict], *, seed_base: int, maxiter: int) -> list[dict]:
    jobs = []
    for inst in instances:
        nfocks = inst["nfocks"]
        for trial in range(N_TRIALS):
            pseed = ecd_prep_seed(seed_base, inst["family"], inst["hamiltonian_id"], trial)
            unit, prep0 = random_prep(nfocks, pseed)
            jobs.append(
                {
                    "trial": trial,
                    "family": inst["family"],
                    "kind": inst["kind"],
                    "hamiltonian_id": inst["hamiltonian_id"],
                    "energy_tensor": inst["energy_tensor"],
                    "ground_bitstring": inst["ground_bitstring"],
                    "instance": inst.get("instance"),
                    "partition": list(inst["partition"]),
                    "nfocks": list(nfocks),
                    "n_qubits": int(inst["n_qubits"]),
                    "ndepth": NDEPTH,
                    "n_params": n_parameters(NDEPTH) + N_PREP_PARAMS,
                    "ansatz_seed": ecd_ansatz_seed(seed_base, inst["family"], inst["hamiltonian_id"], trial),
                    "prep_seed": pseed,
                    "prep_unit": unit,
                    "prep0": prep0,
                    **_spsa_job_fields(maxiter),
                }
            )
    return jobs


def _write_payload(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    return payload


def run_hea_phase(
    instances: list[dict],
    *,
    n_qubits: int,
    seed_base: int,
    maxiter: int,
    workers: int,
    outpath: Path,
    n_layers: int = HEA_LAYERS,
    protocol: str = "hea_gibbs",
) -> dict:
    jobs = build_hea_jobs(instances, seed_base=seed_base, maxiter=maxiter, n_layers=n_layers)
    print(
        f"=== HEA L={n_layers} Gibbs-{maxiter}  n={n_qubits}  params={n_hea_params(n_qubits, n_layers)}  "
        f"jobs={len(jobs)}  workers={workers} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    records = _run_pool(jobs, run_hea_job, workers, checkpoint_path=outpath.with_suffix(".partial.json"))
    elapsed = time.perf_counter() - t0
    by_family: dict[str, list[dict]] = {}
    for rec in records:
        by_family.setdefault(str(rec["family"]), []).append(rec)
    family_summary = {k: _summarize(v) for k, v in by_family.items()}
    n_success = int(sum(bool(r.get("success")) for r in records))
    payload = {
        "method": protocol,
        "protocol": protocol,
        "ansatz": "hea",
        "objective": "gibbs",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_params": n_hea_params(n_qubits, n_layers),
        "cz_packing": "even_odd",
        "initial_state": "zero",
        "n_hamiltonians": len({(r["family"], r["hamiltonian_id"]) for r in records}),
        "n_trials_per_hamiltonian": N_TRIALS,
        "maxiter": maxiter,
        "seed_base": seed_base,
        "seed_protocol_offset": HEA_PROTOCOL_OFFSET,
        "seed_formula": "knapsack: seed_base + 100*hid + 32000 + trial; "
        "ising: seed_base + 10000 + 100*hid + 32000 + trial",
        "workers": workers,
        "elapsed_sec": elapsed,
        "eta_policy": "sampled_tail",
        "n_success": n_success,
        "success_rate": n_success / max(len(records), 1),
        "mean_rel_gap": float(np.mean([r["rel_gap"] for r in records])) if records else float("nan"),
        "mean_gap_to_min": float(np.mean([r["gap_to_min"] for r in records])) if records else float("nan"),
        "mean_p_ground": float(np.mean([r["p_ground"] for r in records])) if records else float("nan"),
        "by_family": family_summary,
        "trials": records,
        "spsa": SPSA,
        "param_init": "Ry angles ~U(0,2pi)",
    }
    _write_payload(outpath, payload)
    print(
        f"  HEA n={n_qubits} hits {n_success}/{len(records)}  "
        f"mean P(gs) {payload['mean_p_ground']:.3f}  "
        f"mean gap {payload['mean_gap_to_min']:.3f}  elapsed {elapsed:.1f}s",
        flush=True,
    )
    return payload


def _done_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    trials = raw.get("trials") or []
    return {(str(t["family"]), int(t["hamiltonian_id"]), int(t.get("trial", 0))) for t in trials}


def run_ecd_phase(
    instances: list[dict],
    *,
    n_qubits: int,
    seed_base: int,
    maxiter: int,
    workers: int,
    outpath: Path,
) -> dict:
    jobs = build_ecd_jobs(instances, seed_base=seed_base, maxiter=maxiter)
    partial = outpath.with_suffix(".partial.json")
    prior = _load_partial_trials(partial)
    done = {(str(t["family"]), int(t["hamiltonian_id"]), int(t.get("trial", 0))) for t in prior}
    if done:
        before = len(jobs)
        jobs = [j for j in jobs if (str(j["family"]), int(j["hamiltonian_id"]), int(j["trial"])) not in done]
        print(f"Resuming ECD: {len(done)} done, {len(jobs)} remaining (of {before}).", flush=True)
    print(
        f"=== ECD Nd={NDEPTH} Gibbs joint-{maxiter}  n={n_qubits}  "
        f"params={n_parameters(NDEPTH)}+{N_PREP_PARAMS}  jobs={len(jobs)}  workers={workers} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    records = _run_pool(
        jobs, run_ecd_job, workers, checkpoint_path=partial, prior=prior
    )
    elapsed = time.perf_counter() - t0
    by_family: dict[str, list[dict]] = {}
    for rec in records:
        by_family.setdefault(str(rec["family"]), []).append(rec)
    family_summary = {k: _summarize(v) for k, v in by_family.items()}
    n_success = int(sum(bool(r.get("success")) for r in records))
    payload = {
        "method": "ecd_gibbs_joint70",
        "protocol": "ecd_gibbs_joint70",
        "ansatz": "ecd",
        "objective": "gibbs",
        "n_qubits": n_qubits,
        "ndepth": NDEPTH,
        "nfocks": list(instances[0]["nfocks"]) if instances else None,
        "n_params": n_parameters(NDEPTH) + N_PREP_PARAMS,
        "n_ecd_params": n_parameters(NDEPTH),
        "n_prep_params": N_PREP_PARAMS,
        "initial_state": "product_prep",
        "n_hamiltonians": len({(r["family"], r["hamiltonian_id"]) for r in records}),
        "n_trials_per_hamiltonian": N_TRIALS,
        "maxiter": maxiter,
        "seed_base": seed_base,
        "prep_seed_offset": PREP_SEED_OFFSET,
        "seed_formula": (
            "ansatz knapsack: seed_base + 100*hid + trial; "
            "ansatz ising: seed_base + 10000 + 100*hid + trial; "
            "prep = ansatz_family_seed + 50000 + trial"
        ),
        "workers": workers,
        "elapsed_sec": elapsed,
        "eta_policy": "sampled_tail",
        "n_success": n_success,
        "success_rate": n_success / max(len(records), 1),
        "mean_rel_gap": float(np.mean([r["rel_gap"] for r in records])) if records else float("nan"),
        "mean_gap_to_min": float(np.mean([r["gap_to_min"] for r in records])) if records else float("nan"),
        "mean_p_ground": float(np.mean([r["p_ground"] for r in records])) if records else float("nan"),
        "by_family": family_summary,
        "trials": records,
        "spsa": SPSA,
    }
    _write_payload(outpath, payload)
    print(
        f"  ECD n={n_qubits} hits {n_success}/{len(records)}  "
        f"mean P(gs) {payload['mean_p_ground']:.3f}  "
        f"mean gap {payload['mean_gap_to_min']:.3f}  elapsed {elapsed:.1f}s",
        flush=True,
    )
    return payload


def _time_one_eval(kind: str, inst: dict, maxiter: int, seed_base: int) -> dict:
    """Time a short SPSA run and a single circuit evaluation."""
    nq = int(inst["n_qubits"])
    if kind == "hea":
        job = build_hea_jobs([inst], seed_base=seed_base, maxiter=maxiter, n_layers=HEA_LAYERS)[0]
        t0 = time.perf_counter()
        rec = run_hea_job(job)
        elapsed = time.perf_counter() - t0
        nfev = int(rec.get("nfev") or 1)
        t1 = time.perf_counter()
        rng = np.random.default_rng(int(job["seed"]))
        x0 = random_hea_params(nq, HEA_LAYERS, rng)
        from qumode_vqe.qaoa import hea_probs

        hea_probs(x0, nq, HEA_LAYERS)
        eval_s = time.perf_counter() - t1
        return {
            "ansatz": "hea",
            "n_qubits": nq,
            "maxiter": maxiter,
            "elapsed_sec": elapsed,
            "nfev": nfev,
            "sec_per_spsa_step": elapsed / max(maxiter, 1),
            "sec_per_eval": elapsed / max(nfev, 1),
            "sec_single_eval": eval_s,
            "extrapolated_70_sec": elapsed * (70.0 / max(maxiter, 1)),
        }
    job = build_ecd_jobs([inst], seed_base=seed_base, maxiter=maxiter)[0]
    t0 = time.perf_counter()
    rec = run_ecd_job(job)
    elapsed = time.perf_counter() - t0
    nfev = int(rec.get("nfev") or 1)
    t1 = time.perf_counter()
    from qumode_vqe.circuit import prep_params_to_ket

    sim = HybridSimulator(
        ndepth=NDEPTH,
        nfocks=inst["nfocks"],
        cost_kind="gibbs",
        target_qnm=None,
        partition=inst["partition"],
        energy_tensor=inst["energy_tensor"],
        initial_state=prep_params_to_ket(job["prep0"], inst["nfocks"]),
    )
    sim.evaluate(np.asarray(rec["x0"], dtype=float))
    eval_s = time.perf_counter() - t1
    return {
        "ansatz": "ecd",
        "n_qubits": nq,
        "maxiter": maxiter,
        "elapsed_sec": elapsed,
        "nfev": nfev,
        "sec_per_spsa_step": elapsed / max(maxiter, 1),
        "sec_per_eval": elapsed / max(nfev, 1),
        "sec_single_eval": eval_s,
        "extrapolated_70_sec": elapsed * (70.0 / max(maxiter, 1)),
        "hit": bool(rec.get("success")),
    }


def run_phase0(outdir: Path, seed_base: int, workers: int) -> dict:
    rows = []
    skip_ecd_n11 = False
    for n in (9, 11):
        ham_seed = HAM_SEED[n]
        metas = generate_hamiltonians(n, ham_seed, n_knapsack=1, n_ising=0)
        inst = _prepare_meta(metas[0])
        print(f"--- Phase 0 microbench n={n} knapsack H0, 5 SPSA steps ---", flush=True)
        hea = _time_one_eval("hea", inst, maxiter=5, seed_base=seed_base)
        print(
            f"  HEA n={n}: {hea['elapsed_sec']:.3f}s / 5 steps  "
            f"({hea['sec_per_eval']:.4f}s/eval, 70-step ~{hea['extrapolated_70_sec']:.2f}s)",
            flush=True,
        )
        ecd = _time_one_eval("ecd", inst, maxiter=5, seed_base=seed_base)
        print(
            f"  ECD n={n}: {ecd['elapsed_sec']:.3f}s / 5 steps  "
            f"({ecd['sec_per_eval']:.4f}s/eval, 70-step ~{ecd['extrapolated_70_sec']:.2f}s)",
            flush=True,
        )
        rows.append(hea)
        rows.append(ecd)
        if n == 11 and ecd["extrapolated_70_sec"] > KILL_SECONDS_PER_70:
            skip_ecd_n11 = True
            print(
                f"KILL SWITCH: ECD n=11 extrapolated {ecd['extrapolated_70_sec']:.1f}s "
                f"> {KILL_SECONDS_PER_70:.0f}s per 70-step instance. Skip Phase 4.",
                flush=True,
            )
    payload = {
        "phase": 0,
        "seed_base": seed_base,
        "microbench_steps": 5,
        "kill_seconds_per_70": KILL_SECONDS_PER_70,
        "skip_ecd_n11": skip_ecd_n11,
        "rows": rows,
        "workers_note": f"nproc={os.cpu_count()} requested_workers={workers}",
    }
    _write_payload(outdir / "size_sweep_microbench.json", payload)
    return payload


def n7_stored_row(outdir: Path) -> dict:
    """Cite stored n=7 numbers. HEA P(gs)/gap from hea_gibbs.json.

    ECD joint-70 hits/gap from gibbs_schedule_abc.json. P(gs) was not stored
    there; it is recomputed from the stored (prep, x) against stored tensors
    (one evaluation each, no re-optimization).
    """
    hea = json.loads((outdir / "hea_gibbs.json").read_text(encoding="utf-8"))
    sched = json.loads((outdir / "gibbs_schedule_abc.json").read_text(encoding="utf-8"))
    joint = sched["by_schedule"]["joint70"]
    trials = [t for t in sched.get("trials", []) if t.get("schedule") == "joint70"]
    hams = load_hamiltonians(outdir / "hamiltonians.json")
    ham_map = {(h["family"], int(h["hamiltonian_id"])): h for h in hams}
    pgs = []
    from qumode_vqe.circuit import prep_params_to_ket

    for t in trials:
        key = (str(t["family"]), int(t["hamiltonian_id"]))
        meta = ham_map[key]
        tensor = np.asarray(meta["energy_tensor"], dtype=float)
        sim = HybridSimulator(
            ndepth=5,
            nfocks=(8, 8),
            cost_kind="gibbs",
            target_qnm=None,
            partition=(1, 3, 3),
            energy_tensor=tensor,
            initial_state=prep_params_to_ket(np.asarray(t["prep"], dtype=float), (8, 8)),
        )
        ev = sim.evaluate(np.asarray(t["x"], dtype=float))
        pgs.append(p_ground_from_tensor(ev.measurement.physical_probs, tensor))
    hea_gaps = [float(t["gap_to_min"]) for t in hea.get("trials", [])]
    return {
        "n": 7,
        "partition": (1, 3, 3),
        "dim": 128,
        "hea": {
            "hits": f"{hea['n_success']}/{hea['n_hamiltonians']}",
            "n_success": int(hea["n_success"]),
            "n": int(hea["n_hamiltonians"]),
            "mean_p_ground": float(hea["mean_p_ground"]),
            "mean_gap_to_min": float(np.mean(hea_gaps)) if hea_gaps else float("nan"),
            "n_params": int(hea["n_params"]),
            "source": "results/hea_gibbs.json",
            "by_family": hea.get("by_family"),
        },
        "ecd": {
            "hits": f"{joint['n_success']}/{joint['n']}",
            "n_success": int(joint["n_success"]),
            "n": int(joint["n"]),
            "mean_p_ground": float(np.mean(pgs)) if pgs else float("nan"),
            "mean_p_ground_note": (
                "not stored in gibbs_schedule_abc.json; recomputed from stored "
                "(prep, x) vs stored energy tensors, one eval each, no re-opt"
            ),
            "mean_gap_to_min": float(joint["mean_gap_to_min"]),
            "n_params": 45,
            "source": "results/gibbs_schedule_abc.json joint70",
            "by_family": joint.get("by_family"),
        },
    }


def write_report(outdir: Path, *, skip_ecd_n11: bool, leftover: dict | None = None) -> Path:
    n7 = n7_stored_row(outdir)
    rows = [n7]
    for n, hea_name, ecd_name in (
        (9, "size_sweep_n9_hea.json", "size_sweep_n9_ecd.json"),
        (11, "size_sweep_n11_hea.json", "size_sweep_n11_ecd.json"),
    ):
        spec = hybrid_ladder(n)
        slot = {
            "n": n,
            "partition": spec["partition"],
            "dim": spec["dim"],
            "hea": None,
            "ecd": None,
        }
        hp = outdir / hea_name
        if hp.exists():
            hea = json.loads(hp.read_text(encoding="utf-8"))
            slot["hea"] = {
                "hits": f"{hea['n_success']}/{hea['n_hamiltonians']}",
                "n_success": int(hea["n_success"]),
                "n": int(hea["n_hamiltonians"]),
                "mean_p_ground": float(hea["mean_p_ground"]),
                "mean_gap_to_min": float(hea["mean_gap_to_min"]),
                "n_params": int(hea["n_params"]),
                "source": str(hp),
                "by_family": hea.get("by_family"),
                "elapsed_sec": hea.get("elapsed_sec"),
            }
        ep = outdir / ecd_name
        if ep.exists():
            ecd = json.loads(ep.read_text(encoding="utf-8"))
            slot["ecd"] = {
                "hits": f"{ecd['n_success']}/{ecd['n_hamiltonians']}",
                "n_success": int(ecd["n_success"]),
                "n": int(ecd["n_hamiltonians"]),
                "mean_p_ground": float(ecd["mean_p_ground"]),
                "mean_gap_to_min": float(ecd["mean_gap_to_min"]),
                "n_params": int(ecd["n_params"]),
                "source": str(ep),
                "by_family": ecd.get("by_family"),
                "elapsed_sec": ecd.get("elapsed_sec"),
            }
        elif n == 11 and skip_ecd_n11:
            slot["ecd"] = {"skipped": True, "reason": "Phase 0 kill switch"}
        rows.append(slot)

    def cell(slot, key, fmt=".3f"):
        if slot is None:
            return "—"
        if slot.get("skipped"):
            return "skipped"
        val = slot.get(key)
        if val is None:
            return "—"
        if isinstance(val, str):
            return val
        return f"{val:{fmt}}"

    lines = [
        "# HEA vs ECD Gibbs size sweep",
        "",
        r"Question: ECD \(N_d=5\) has a **fixed** 40 ECD + 5 prep parameters as the",
        r"Fock cutoff grows. HEA \(L=5\) has \(n(L+1)\) parameters. Does ECD catch",
        r"HEA as \(n\) grows on this diagonal knapsack / Ising class?",
        "",
        "No retuning. Same Gibbs + SPSA protocol as the n=7 suite. No QAOA.",
        "n=7 numbers are stored (not rerun). n=9/11 instances are **new** files",
        "(`hamiltonians_n9.json`, `hamiltonians_n11.json`), same recipes:",
        "20 random knapsacks (integer values/weights, slack-faithful λ) + 20",
        "diagonal Ising (all local Z + 12 random ZZ, RMS-normalized).",
        "",
        "## Protocol",
        "",
        "| Item | Setting |",
        "|---|---|",
        "| Embedding | Dutta ladder: n=7 L=8 (1,3,3); n=9 L=16 (1,4,4); n=11 L=32 (1,5,5). Qubit 0 = MSB. |",
        r"| Cost | Gibbs \(f=-\ln\langle e^{-\eta E}\rangle\), `sampled_tail` η (eta.py untouched) |",
        "| Optimizer | `run_spsa` a=0.2, c=0.15, A=10, α=0.602, γ=0.101, 70 steps, one start / H |",
        "| Success | most-likely bitstring is an exact ground (atol=1e-8) |",
        "| HEA | \\|0⟩^n, L=5, n(L+1) params, even-then-odd NN CZ, Ry ~ U(0,2π) |",
        "| ECD | Nd=5, 40 polar ECD + 5 product-state prep (joint-70), nfocks=(L,L) |",
        f"| Seeds | seed_base={SEED_BASE} (n=7 used 3000). ham_seed n=9→9000, n=11→11000 (n=7 used 8000). |",
        "| HEA seed | knapsack: seed_base+100·hid+32000; ising: seed_base+10000+100·hid+32000 |",
        "| ECD seed | ansatz: seed_base+family_offset+trial; prep: +50000 (same offsets as the n=7 mixed suite) |",
        "",
        "## Results",
        "",
        "| n | dim | HEA params | HEA hits | HEA mean P(gs) | HEA mean gap | ECD params | ECD hits | ECD mean P(gs) | ECD mean gap |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for slot in rows:
        hea, ecd = slot.get("hea"), slot.get("ecd")
        lines.append(
            "| {n} | {dim} | {hp} | {hh} | {hpg} | {hg} | {ep} | {eh} | {epg} | {eg} |".format(
                n=slot["n"],
                dim=slot["dim"],
                hp=cell(hea, "n_params", "d"),
                hh=cell(hea, "hits"),
                hpg=cell(hea, "mean_p_ground"),
                hg=cell(hea, "mean_gap_to_min"),
                ep=cell(ecd, "n_params", "d"),
                eh=cell(ecd, "hits"),
                epg=cell(ecd, "mean_p_ground"),
                eg=cell(ecd, "mean_gap_to_min"),
            )
        )
    lines.extend(["", "### By family", ""])
    for slot in rows:
        lines.append(f"**n={slot['n']}**")
        for name, block in (("HEA", slot.get("hea")), ("ECD", slot.get("ecd"))):
            if not block or block.get("skipped"):
                if block and block.get("skipped"):
                    lines.append(f"- {name}: skipped ({block.get('reason')})")
                continue
            by = block.get("by_family") or {}
            kn = by.get("knapsack") or {}
            iso = by.get("ising") or {}
            lines.append(
                f"- {name}: knapsack {kn.get('n_success', '?')}/{kn.get('n', '?')}, "
                f"Ising {iso.get('n_success', '?')}/{iso.get('n', '?')}"
            )
        lines.append("")

    hea_wins = True
    notes = []
    for slot in rows:
        hea, ecd = slot.get("hea"), slot.get("ecd")
        if not hea or not ecd or ecd.get("skipped"):
            continue
        if int(ecd["n_success"]) > int(hea["n_success"]):
            hea_wins = False
        notes.append((slot["n"], hea["n_success"], ecd["n_success"], hea.get("mean_p_ground"), ecd.get("mean_p_ground")))

    lines.append("## Verdict")
    lines.append("")
    if hea_wins:
        lines.append(
            "HEA still wins on this diagonal knapsack / Ising class as n grows. "
            "ECD's fixed 45-parameter joint-70 ansatz does not catch the growing "
            r"HEA \(L=5\) circuit at n=9 or n=11 under this protocol. "
            "Size does not create an ECD win here. "
            + (
                "Hit counts: "
                + "; ".join(
                    f"n={n} HEA {h}/{40 if n != 7 else 40} vs ECD {e}"
                    for n, h, e, *_ in notes
                )
                + "."
            )
        )
    else:
        lines.append(
            "ECD out-hit HEA at at least one size in this sweep. That is reported "
            "as measured; it is not a claim of ECD ≫ HEA in general, and it was "
            "not obtained by retuning."
        )
    if skip_ecd_n11:
        lines.append("")
        lines.append(
            "Phase 0 kill switch fired: ECD n=11 was slower than ~3 minutes per "
            "70-step instance (extrapolated from 5 SPSA steps). Phase 4 was skipped."
        )
    if leftover:
        lines.append("")
        lines.append("## Leftover")
        lines.append("")
        lines.append(json.dumps(_json_ready(leftover), indent=2))
    lines.extend(
        [
            "",
            "## Honesty",
            "",
            "All n=9/11 numbers are from this run. n=7 HEA is `results/hea_gibbs.json` "
            "(38/40, mean P(gs)=0.611). n=7 ECD joint-70 is `results/gibbs_schedule_abc.json` "
            "(26/40). n=7 ECD mean P(gs) was not stored; it is one eval of the stored "
            "parameters on the stored tensors, not a re-optimization.",
            "",
        ]
    )
    path = outdir / "size_sweep.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    (outdir / "size_sweep_table.json").write_text(
        json.dumps(_json_ready({"n7": n7, "rows": rows, "skip_ecd_n11": skip_ecd_n11, "leftover": leftover}), indent=2),
        encoding="utf-8",
    )
    return path


def default_workers() -> int:
    return max(1, min(7, int(os.cpu_count() or 4)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument(
        "--phase",
        choices=["0", "1", "2", "3", "4", "leftover", "report", "all"],
        default="all",
    )
    parser.add_argument("--force-ecd-n11", action="store_true", help="Ignore Phase 0 kill switch.")
    args = parser.parse_args(argv)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers)
    seed_base = int(args.seed_base)
    maxiter = int(args.maxiter)
    phase = args.phase

    skip_ecd_n11 = False
    mb_path = outdir / "size_sweep_microbench.json"
    if mb_path.exists():
        skip_ecd_n11 = bool(json.loads(mb_path.read_text())["skip_ecd_n11"])
    if args.force_ecd_n11:
        skip_ecd_n11 = False

    def ensure_hams(n: int) -> list[dict]:
        path = outdir / f"hamiltonians_n{n}.json"
        if path.exists():
            metas = load_hamiltonians(path)
            print(f"Loaded {len(metas)} Hamiltonians from {path}", flush=True)
        else:
            metas = generate_hamiltonians(n, HAM_SEED[n], N_KNAPSACK, N_ISING)
            save_hamiltonians(metas, path, outdir / f"hamiltonians_n{n}.npz")
        insts = [_prepare_meta(m) for m in metas]
        print(f"Encoding check passed for {len(insts)} n={n} instances.", flush=True)
        return insts

    leftover_info = None
    if phase in ("0", "all"):
        mb = run_phase0(outdir, seed_base, workers)
        skip_ecd_n11 = bool(mb["skip_ecd_n11"]) and not args.force_ecd_n11
    if phase in ("1", "all"):
        insts = ensure_hams(9)
        run_hea_phase(insts, n_qubits=9, seed_base=seed_base, maxiter=maxiter, workers=workers, outpath=outdir / "size_sweep_n9_hea.json")
    if phase in ("2", "all"):
        insts = ensure_hams(9)
        run_ecd_phase(insts, n_qubits=9, seed_base=seed_base, maxiter=maxiter, workers=workers, outpath=outdir / "size_sweep_n9_ecd.json")
    if phase in ("3", "all"):
        insts = ensure_hams(11)
        run_hea_phase(insts, n_qubits=11, seed_base=seed_base, maxiter=maxiter, workers=workers, outpath=outdir / "size_sweep_n11_hea.json")
    if phase in ("4", "all"):
        if skip_ecd_n11:
            print("Phase 4 skipped (Phase 0 kill switch).", flush=True)
        else:
            insts = ensure_hams(11)
            run_ecd_phase(insts, n_qubits=11, seed_base=seed_base, maxiter=maxiter, workers=workers, outpath=outdir / "size_sweep_n11_ecd.json")
    if phase == "leftover":
        # n=11 L=3 HEA (~44 params, closest to ECD's 45). Only after Phase 4 / skip-4.
        if not (outdir / "size_sweep_n11_hea.json").exists():
            raise SystemExit("leftover requires Phase 3 (size_sweep_n11_hea.json) first")
        insts = ensure_hams(11)
        leftover_run = run_hea_phase(
            insts,
            n_qubits=11,
            seed_base=seed_base + 1000,
            maxiter=maxiter,
            workers=workers,
            outpath=outdir / "size_sweep_n11_hea_L3.json",
            n_layers=3,
            protocol="hea_gibbs_L3",
        )
        leftover_info = {
            "note": "n=11 HEA L=3 (44 params), closest HEA to ECD's 45 coordinates. Distinct seed_base+1000.",
            "n_success": leftover_run["n_success"],
            "n": leftover_run["n_hamiltonians"],
            "mean_p_ground": leftover_run["mean_p_ground"],
            "mean_gap_to_min": leftover_run["mean_gap_to_min"],
            "n_params": leftover_run["n_params"],
        }
    if phase in ("report", "all", "leftover"):
        write_report(outdir, skip_ecd_n11=skip_ecd_n11, leftover=leftover_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
