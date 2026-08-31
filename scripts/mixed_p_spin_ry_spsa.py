#!/usr/bin/env python3
"""Product-state Gibbs baseline: one layer of R_y gates on mixed p-spin instances.

Does **not** use a known ground-state label during optimization. Each of the
20 saved mixed p-spin Hamiltonians gets one independent start per trial:

1. Draw seven angles uniformly from ``[0, 2π)``.
2. SPSA-minimize the same Gibbs cost as ``Gibbs_and_adaptive_optim_ECD.py``:
   ``f = −ln ⟨e^{−ηE}⟩`` with adaptive ``sampled_tail`` η.

The ansatz is a product state ``⊗_i R_y(θ_i)|0⟩``. The Born histogram over
the 2^N computational basis is the product of independent qubit
probabilities ``P(1) = sin²(θ_i/2)``. No ECD circuit is used.

Default budget is 200 SPSA steps with the Gibbs SPSA gains.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_sampled_tail_eta():
    """Load SampledTailEta without importing the qumode_vqe package."""
    import sys

    eta_path = Path(__file__).resolve().parents[1] / "src" / "qumode_vqe" / "eta.py"
    spec = importlib.util.spec_from_file_location("qumode_vqe_eta", eta_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load SampledTailEta from {eta_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SampledTailEta


SampledTailEta = _load_sampled_tail_eta()


def gibbs_objective(
    probs: np.ndarray,
    energies: np.ndarray,
    eta: float,
) -> float:
    """Gibbs objective f = −ln ⟨e^{−ηE}⟩, matching qumode_vqe.vqe.gibbs_objective."""
    p = np.asarray(probs, dtype=float).reshape(-1)
    e = np.asarray(energies, dtype=float).reshape(-1)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        return 0.0
    p = p / total
    emin = float(np.min(e))
    avg = float(np.dot(p, np.exp(-float(eta) * (e - emin))))
    return float(-np.log(max(avg, 1e-300)) + float(eta) * emin)


def run_spsa(
    fun,
    x0: np.ndarray,
    *,
    maxiter: int,
    rng: np.random.Generator,
    project,
    a: float,
    c: float,
    A: float,
    alpha: float,
    gamma: float,
    on_before_step,
):
    """Periodic-angle SPSA, matching qumode_vqe.vqe.run_spsa without box bounds."""
    x = np.asarray(project(np.asarray(x0, dtype=float).copy()), dtype=float)
    nfev = 0
    for k in range(1, int(maxiter) + 1):
        if on_before_step is not None:
            on_before_step(k, x)
        ak = a / (k + A) ** alpha
        ck = c / k**gamma
        delta = rng.choice([-1.0, 1.0], size=x.size)
        xp = np.asarray(project(x + ck * delta), dtype=float)
        xm = np.asarray(project(x - ck * delta), dtype=float)
        yp = float(fun(xp))
        ym = float(fun(xm))
        nfev += 2
        ghat = (yp - ym) / (2.0 * ck) * delta
        x = np.asarray(project(x - ak * ghat), dtype=float)
    return SimpleNamespace(
        fun=float(fun(x)),
        x=x,
        nfev=nfev,
        nit=int(maxiter),
    )

# ---------------------------------------------------------------------------
# Hyperparameters (SPSA gains match Gibbs_and_adaptive_optim_ECD.py)
# ---------------------------------------------------------------------------
OUTDIR = Path("results")
HAM_DIR = Path("Hamiltonians") / "mixed_p_spin"
OUTPUT_JSON = "mixed_p_spin_ry_spsa.json"
OUTPUT_JSON_ENERGY = "mixed_p_spin_ry_spsa_energy.json"
SEED_BASE = 3000
N_TRIALS = 1
MAXITER = 200
WORKERS = 7
SPSA_A = 0.2
SPSA_C = 0.15
SPSA_A_STAB = 10.0
SPSA_ALPHA = 0.602
SPSA_GAMMA = 0.101
NPZ_GLOB = "mixed_p_spin_p2-4_[0-9][0-9][0-9].npz"


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


def bit_table(n_spins: int) -> np.ndarray:
    """All 2^N bitstrings, shape (2^N, N), qubit 0 = MSB."""
    dim = 1 << n_spins
    index = np.arange(dim, dtype=np.uint32)[:, None]
    shifts = np.arange(n_spins - 1, -1, -1, dtype=np.uint32)
    return ((index >> shifts) & 1).astype(np.int8)


def enumerate_spectrum(
    n_spins: int,
    sites: np.ndarray,
    orders: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Exact Z-basis spectrum. Eigenvalue of a Z-string is (-1)^{sum of bits}."""
    bits = bit_table(n_spins)
    energies = np.zeros(bits.shape[0], dtype=float)
    for row, coeff in enumerate(coefficients):
        order = int(orders[row])
        parity = np.bitwise_xor.reduce(bits[:, sites[row, :order]], axis=1)
        energies += float(coeff) * (1.0 - 2.0 * parity)
    return energies


def bitstring_from_index(index: int, n_spins: int) -> str:
    return format(int(index), f"0{int(n_spins)}b")


def wrap_angles(theta: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(theta, dtype=float), 2.0 * np.pi)


def product_probs(theta: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Born probabilities of ⊗ R_y(θ_i)|0⟩ on the computational basis."""
    theta = wrap_angles(theta)
    p1 = np.sin(0.5 * theta) ** 2
    p0 = 1.0 - p1
    logp = np.where(
        bits == 1,
        np.log(np.clip(p1, 1e-300, 1.0)),
        np.log(np.clip(p0, 1e-300, 1.0)),
    )
    return np.exp(logp.sum(axis=1))


def histogram_stats(
    probs: np.ndarray,
    energies: np.ndarray,
    n_spins: int,
    eta: float | None = None,
) -> dict:
    p = np.asarray(probs, dtype=float).reshape(-1)
    e = np.asarray(energies, dtype=float).reshape(-1)
    total = float(p.sum())
    if total <= 0.0:
        p = np.full(e.shape, 1.0 / max(e.size, 1))
    else:
        p = p / total
    idx = int(np.argmax(p))
    e_min = float(np.min(e))
    e_max = float(np.max(e))
    spread = max(e_max - e_min, 1e-12)
    found = float(e[idx])
    ground_index = int(np.argmin(e))
    energy_physical = float(np.dot(p, e))
    rec = {
        "energy_physical": energy_physical,
        "energy_diag": found,
        "energy_min": e_min,
        "energy_max": e_max,
        "gap_to_min": found - e_min,
        "rel_gap": (found - e_min) / spread,
        "most_likely_index": idx,
        "most_likely_bitstring": bitstring_from_index(idx, n_spins),
        "ground_index": ground_index,
        "ground_bitstring": bitstring_from_index(ground_index, n_spins),
        "success": bool(np.isclose(found, e_min, atol=1e-8, rtol=0.0)),
        "eta": None if eta is None else float(eta),
    }
    if eta is not None:
        rec["gibbs_cost"] = float(gibbs_objective(p, e, eta))
    return rec


def load_instances(ham_dir: Path) -> list[dict]:
    ham_dir = Path(ham_dir)
    paths = sorted(ham_dir.glob(NPZ_GLOB))
    if not paths:
        raise FileNotFoundError(f"no mixed p-spin files matching {NPZ_GLOB} in {ham_dir}")
    instances: list[dict] = []
    n_spins: int | None = None
    n_terms: int | None = None
    for path in paths:
        data = np.load(path)
        required = ("sites", "orders", "coefficients", "num_spins")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"{path.name} is missing arrays: {missing}")
        n = int(data["num_spins"])
        sites = np.asarray(data["sites"])
        orders = np.asarray(data["orders"], dtype=np.int64)
        coefficients = np.asarray(data["coefficients"], dtype=float)
        if sites.ndim != 2 or orders.ndim != 1 or coefficients.ndim != 1:
            raise ValueError(f"{path.name} has unexpected array ranks.")
        if not (sites.shape[0] == orders.shape[0] == coefficients.shape[0]):
            raise ValueError(f"{path.name} term arrays have mismatched lengths.")
        if int(np.min(orders)) < 1 or int(np.max(orders)) > sites.shape[1]:
            raise ValueError(f"{path.name} body orders do not fit the padded site array.")
        if n_spins is None:
            n_spins = n
            n_terms = int(orders.size)
        elif n != n_spins:
            raise ValueError(f"{path.name} has num_spins={n}, expected {n_spins}.")
        elif int(orders.size) != n_terms:
            raise ValueError(f"{path.name} has {orders.size} terms, expected {n_terms}.")
        energies = enumerate_spectrum(n, sites, orders, coefficients)
        order = np.argsort(energies)
        e_min = float(energies[order[0]])
        deg = int(np.sum(np.isclose(energies, e_min, atol=1e-10, rtol=0.0)))
        gap = float(energies[order[deg]] - e_min) if deg < energies.size else 0.0
        hid = int(path.stem.rsplit("_", 1)[-1])
        instances.append(
            {
                "file": path.name,
                "path": str(path),
                "hamiltonian_id": hid,
                "num_spins": n,
                "n_terms": int(orders.size),
                "min_body_order": int(np.min(orders)),
                "max_body_order": int(np.max(orders)),
                "terms_by_order": {
                    str(int(order_p)): int(np.count_nonzero(orders == order_p))
                    for order_p in sorted(set(int(v) for v in orders))
                },
                "energy_min": e_min,
                "energy_max": float(np.max(energies)),
                "spread": float(np.max(energies) - e_min),
                "gap": gap,
                "ground_degeneracy": deg,
                "ground_index": int(order[0]),
                "ground_bitstring": bitstring_from_index(int(order[0]), n),
                "energies": energies,
            }
        )
    instances.sort(key=lambda rec: int(rec["hamiltonian_id"]))
    return instances


def run_trial(job: dict) -> dict:
    t0 = time.perf_counter()
    n_spins = int(job["num_spins"])
    energies = np.asarray(job["energies"], dtype=float)
    bits = bit_table(n_spins)
    maxiter = int(job["maxiter"])
    objective = str(job.get("objective", "gibbs"))
    rng = np.random.default_rng(int(job["seed"]))
    theta0 = wrap_angles(rng.uniform(0.0, 2.0 * np.pi, size=n_spins))
    policy = None
    eta0 = None
    if objective == "gibbs":
        policy = SampledTailEta()
        eta0 = float(policy.initialize(energies, product_probs(theta0, bits)).eta)

        def fun(theta: np.ndarray) -> float:
            return gibbs_objective(product_probs(theta, bits), energies, float(policy.eta))

        def before_step(k: int, theta: np.ndarray) -> None:
            policy.maybe_update(k, maxiter, energies, product_probs(theta, bits))

    elif objective == "energy":

        def fun(theta: np.ndarray) -> float:
            probs = product_probs(theta, bits)
            total = float(probs.sum())
            if total <= 0.0:
                return 0.0
            return float(np.dot(probs / total, energies))

        def before_step(k: int, theta: np.ndarray) -> None:
            del k, theta

    else:
        raise ValueError(f"unknown objective {objective!r}")

    opt = run_spsa(
        fun,
        theta0,
        maxiter=maxiter,
        rng=rng,
        project=wrap_angles,
        a=float(job["spsa_a"]),
        c=float(job["spsa_c"]),
        A=float(job["spsa_A"]),
        alpha=float(job["spsa_alpha"]),
        gamma=float(job["spsa_gamma"]),
        on_before_step=before_step,
    )
    theta = wrap_angles(opt.x)
    eta_final = None if policy is None else float(policy.eta)
    init = histogram_stats(product_probs(theta0, bits), energies, n_spins, eta=eta0)
    final = histogram_stats(product_probs(theta, bits), energies, n_spins, eta=eta_final)
    if objective == "energy":
        init["cost"] = float(init["energy_physical"])
        final["cost"] = float(final["energy_physical"])
        method = "ry_product_energy_spsa"
        eta_policy = None
        snap = {"n_clamps": 0, "n_fallbacks": 0}
    else:
        init["cost"] = float(init["gibbs_cost"])
        final["cost"] = float(final["gibbs_cost"])
        method = "ry_product_gibbs_spsa"
        eta_policy = "sampled_tail"
        snap = policy.snapshot()
    return {
        "trial": int(job["trial"]),
        "hamiltonian_id": int(job["hamiltonian_id"]),
        "file": str(job["file"]),
        "family": "mixed_p_spin",
        "kind": "mixed_p_spin",
        "method": method,
        "objective": objective,
        "num_spins": n_spins,
        "seed": int(job["seed"]),
        "theta0": theta0,
        "theta": theta,
        "init": init,
        "nit": int(opt.nit),
        "nfev": int(opt.nfev),
        "eta_policy": eta_policy,
        "eta0": eta0,
        "n_eta_clamps": int(snap["n_clamps"]),
        "n_eta_fallbacks": int(snap["n_fallbacks"]),
        "elapsed_s": float(time.perf_counter() - t0),
        **final,
    }


def _print_job_line(i: int, n: int, rec: dict) -> None:
    eta = rec.get("eta")
    eta_s = "" if eta is None else f"  η={float(eta):.3f}"
    label = "⟨H⟩" if rec.get("objective") == "energy" else "Gibbs"
    print(
        f"[{i}/{n}] H{rec['hamiltonian_id']:03d} t{rec['trial']}  {label}={rec['cost']:.4f}"
        f"{eta_s}  ⟨H⟩={rec['energy_physical']:.3f}  "
        f"E={rec['energy_diag']:.3f} (min {rec['energy_min']:.3f}, "
        f"rel={rec['rel_gap']:.3f})  gs={rec['success']}  "
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
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(run_trial, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                done += 1
                _print_job_line(done, len(jobs), rec)
    records.sort(key=lambda r: (int(r["hamiltonian_id"]), int(r["trial"])))
    return records


def _summarize(records: list[dict]) -> dict:
    n = len(records)
    n_success = int(sum(bool(r.get("success")) for r in records))
    etas = [float(r["eta"]) for r in records if r.get("eta") is not None]
    return {
        "n": n,
        "n_success": n_success,
        "success_rate": n_success / max(n, 1),
        "mean_rel_gap": float(np.mean([r["rel_gap"] for r in records])) if n else float("nan"),
        "mean_gap_to_min": float(np.mean([r["gap_to_min"] for r in records])) if n else float("nan"),
        "mean_energy_diag": float(np.mean([r["energy_diag"] for r in records])) if n else float("nan"),
        "mean_energy_physical": float(np.mean([r["energy_physical"] for r in records])) if n else float("nan"),
        "mean_cost": float(np.mean([r["cost"] for r in records])) if n else float("nan"),
        "mean_eta": float(np.mean(etas)) if etas else float("nan"),
        "mean_nfev": float(np.mean([r["nfev"] for r in records])) if n else float("nan"),
        "mean_elapsed_s": float(np.mean([r["elapsed_s"] for r in records])) if n else float("nan"),
    }


def _repeat_summary(records: list[dict]) -> dict:
    by_trial: dict[int, list[dict]] = {}
    for rec in records:
        by_trial.setdefault(int(rec["trial"]), []).append(rec)
    repeats = []
    for trial in sorted(by_trial):
        group = by_trial[trial]
        n = len(group)
        n_success = int(sum(bool(r.get("success")) for r in group))
        repeats.append(
            {
                "trial": trial,
                "n": n,
                "n_success": n_success,
                "success_rate": n_success / max(n, 1),
                "mean_rel_gap": float(np.mean([r["rel_gap"] for r in group])) if n else float("nan"),
            }
        )
    rates = [float(r["success_rate"]) for r in repeats]
    return {
        "n_repeats": len(repeats),
        "success_rates": rates,
        "mean_success_rate": float(np.mean(rates)) if rates else float("nan"),
        "std_success_rate": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
        "repeats": repeats,
    }


def run_experiment(
    *,
    ham_dir: Path,
    outdir: Path,
    output: Path | None,
    seed_base: int,
    n_trials: int,
    maxiter: int,
    workers: int,
    objective: str,
    spsa_a: float,
    spsa_c: float,
    spsa_A: float,
    spsa_alpha: float,
    spsa_gamma: float,
) -> dict:
    instances = load_instances(ham_dir)
    n_spins = int(instances[0]["num_spins"])
    print("=== Mixed p-spin instances (landscape, before VQE) ===", flush=True)
    print(f"{'H':>3}  {'file':<28}  {'Emin':>8}  {'spread':>8}  {'gap':>6}", flush=True)
    jobs: list[dict] = []
    ham_meta: list[dict] = []
    for inst in instances:
        meta = {k: v for k, v in inst.items() if k != "energies"}
        ham_meta.append(meta)
        print(
            f"{inst['hamiltonian_id']:3d}  {inst['file']:<28}  "
            f"{inst['energy_min']:8.3f}  {inst['spread']:8.3f}  {inst['gap']:6.3f}",
            flush=True,
        )
        n_t = max(int(n_trials), 1)
        for trial in range(n_t):
            jobs.append(
                {
                    "trial": trial,
                    "hamiltonian_id": int(inst["hamiltonian_id"]),
                    "file": inst["file"],
                    "num_spins": n_spins,
                    "energies": inst["energies"],
                    "seed": int(seed_base) + 100 * int(inst["hamiltonian_id"]) + trial,
                    "objective": objective,
                    "maxiter": int(maxiter),
                    "spsa_a": spsa_a,
                    "spsa_c": spsa_c,
                    "spsa_A": spsa_A,
                    "spsa_alpha": spsa_alpha,
                    "spsa_gamma": spsa_gamma,
                }
            )
    n_t = max(int(n_trials), 1)
    objective = str(objective)
    if objective not in ("gibbs", "energy"):
        raise ValueError(f"unknown objective {objective!r}")
    cost_label = "energy ⟨H⟩" if objective == "energy" else "Gibbs"
    print(
        f"=== R_y product {cost_label} SPSA, {len(ham_meta)} Hamiltonians × {n_t} trial(s), "
        f"maxiter={maxiter}, workers={workers} ===",
        flush=True,
    )
    records = _run_jobs(jobs, workers)
    finite = all(math.isfinite(float(r["cost"])) for r in records)
    steps_ok = all(int(r["nit"]) == int(maxiter) for r in records)
    meta_by_id = {int(m["hamiltonian_id"]): m for m in ham_meta}
    gs_ok = all(
        np.isclose(
            float(r["energy_min"]),
            float(meta_by_id[int(r["hamiltonian_id"])]["energy_min"]),
            atol=1e-10,
            rtol=0.0,
        )
        for r in records
    )
    if not finite:
        raise RuntimeError("non-finite cost in at least one trial")
    if not steps_ok:
        raise RuntimeError("a trial did not use the requested SPSA step count")
    if not gs_ok:
        raise RuntimeError("saved ground energies do not match exact enumeration")
    summary = _summarize(records)
    repeats = _repeat_summary(records)
    payload = {
        "method": "ry_product_energy_spsa" if objective == "energy" else "ry_product_gibbs_spsa",
        "ansatz": "one_layer_ry",
        "objective": objective,
        "eta_policy": None if objective == "energy" else "sampled_tail",
        "initial_state": "|0>^n",
        "n_hamiltonians": len(ham_meta),
        "n_trials_per_hamiltonian": n_t,
        "num_spins": n_spins,
        "maxiter": int(maxiter),
        "seed_base": int(seed_base),
        "workers": int(workers),
        "ham_dir": str(Path(ham_dir)),
        "n_success": summary["n_success"],
        "success_rate": summary["success_rate"],
        "mean_success_rate": repeats["mean_success_rate"],
        "std_success_rate": repeats["std_success_rate"],
        "mean_rel_gap": summary["mean_rel_gap"],
        "mean_eta": summary["mean_eta"],
        "summary": summary,
        "repeats": repeats,
        "hamiltonians": ham_meta,
        "trials": records,
        "spsa": {
            "a": spsa_a,
            "c": spsa_c,
            "A": spsa_A,
            "alpha": spsa_alpha,
            "gamma": spsa_gamma,
        },
        "checks": {
            "finite_costs": finite,
            "requested_steps": steps_ok,
            "ground_energy_match": gs_ok,
        },
    }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = Path(output) if output is not None else outdir / (
        OUTPUT_JSON_ENERGY if objective == "energy" else OUTPUT_JSON
    )
    if not path.is_absolute():
        path = outdir / path.name if path.parent == Path(".") else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    print(
        f"  R_y {cost_label} exact-GS hits {summary['n_success']}/{summary['n']}  "
        f"pooled rate {summary['success_rate']:.3f}  "
        f"mean of {repeats['n_repeats']} runs {repeats['mean_success_rate']:.3f}"
        + (
            f" ± {repeats['std_success_rate']:.3f}"
            if repeats["n_repeats"] > 1
            else ""
        )
        + f"  mean rel-gap {summary['mean_rel_gap']:.3f}  mean η={summary['mean_eta']:.3f}",
        flush=True,
    )
    for rep in repeats["repeats"]:
        print(
            f"    run {rep['trial']}: {rep['n_success']}/{rep['n']}  "
            f"success_rate={rep['success_rate']:.3f}  mean rel-gap={rep['mean_rel_gap']:.3f}",
            flush=True,
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ham-dir", type=Path, default=HAM_DIR)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON filename or path (default: <outdir>/mixed_p_spin_ry_spsa.json).",
    )
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help="Independent random starts per Hamiltonian.",
    )
    parser.add_argument("--maxiter", type=int, default=MAXITER, help="SPSA steps per Hamiltonian.")
    parser.add_argument(
        "--objective",
        choices=("gibbs", "energy"),
        default="gibbs",
        help="SPSA cost: Gibbs −ln⟨e^{−ηE}⟩ or energy ⟨H⟩.",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--spsa-a", type=float, default=SPSA_A)
    parser.add_argument("--spsa-c", type=float, default=SPSA_C)
    parser.add_argument("--spsa-A", type=float, default=SPSA_A_STAB)
    parser.add_argument("--spsa-alpha", type=float, default=SPSA_ALPHA)
    parser.add_argument("--spsa-gamma", type=float, default=SPSA_GAMMA)
    args = parser.parse_args(argv)
    run_experiment(
        ham_dir=args.ham_dir,
        outdir=args.outdir,
        output=args.output,
        seed_base=args.seed_base,
        n_trials=args.n_trials,
        maxiter=args.maxiter,
        workers=args.workers,
        objective=args.objective,
        spsa_a=args.spsa_a,
        spsa_c=args.spsa_c,
        spsa_A=args.spsa_A,
        spsa_alpha=args.spsa_alpha,
        spsa_gamma=args.spsa_gamma,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
