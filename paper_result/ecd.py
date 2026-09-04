"""ECD-VQE trials matching bkp_ecd_vqe.ipynb initialization.

Notebook guess (``ecd_opt_vqe`` when ``Xvec`` is empty)::

    beta_mag ~ Uniform(0, 3)
    beta_arg, theta, phi ~ Uniform(0, π)

packed as ``[β_mag, β_arg, θ, φ]`` with shape ``(N_d, 2)`` each. That is the
same recipe as ``qumode_vqe.params.random_parameters``.
"""

from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import scipy.optimize as sciopt

from qumode_vqe.circuit import (
    N_PREP_PARAMS,
    coherent_radius_max,
    prep_params_to_ket,
)
from qumode_vqe.eta import SampledTailEta
from qumode_vqe.hamiltonian import TARGET_QNM
from qumode_vqe.params import n_parameters, n_snap_parameters, random_parameters, random_snap_parameters
from qumode_vqe.vqe import HybridSimulator, OptimizeResult, _history_record, optimize_vqe


def notebook_ecd_x0(ndepth: int, rng: np.random.Generator) -> np.ndarray:
    """Initial ECD vector from the published knapsack notebook."""
    return random_parameters(int(ndepth), rng)


def notebook_snap_x0(
    ndepth: int,
    rng: np.random.Generator,
    nfocks: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """SNAP+D start with the same Uniform ranges as the ECD notebook guess."""
    return random_snap_parameters(int(ndepth), nfocks, rng)


def notebook_x0(
    ansatz: str,
    ndepth: int,
    rng: np.random.Generator,
    nfocks: tuple[int, int] = (8, 8),
) -> np.ndarray:
    kind = str(ansatz).lower()
    if kind == "snap":
        return notebook_snap_x0(ndepth, rng, nfocks)
    if kind == "ecd":
        return notebook_ecd_x0(ndepth, rng)
    raise ValueError(f"unknown ansatz {ansatz!r}")


def unit_to_prep(u: np.ndarray, nfocks: tuple[int, int]) -> np.ndarray:
    """Map a unit-cube point to ``(θ, Re α₁, Im α₁, Re α₂, Im α₂)``.

    Same covering as ``scripts/Gibbs_and_adaptive_optim_ECD.py``: θ ~ U[0, π],
    each coherent amplitude uniform in area on the disk |α| ≤ √(N−1).
    """
    u = np.asarray(u, dtype=float).reshape(N_PREP_PARAMS)
    theta = float(u[0] * np.pi)
    r1 = coherent_radius_max(nfocks[0])
    r2 = coherent_radius_max(nfocks[1])
    mag1 = r1 * np.sqrt(float(u[1]))
    mag2 = r2 * np.sqrt(float(u[3]))
    a1 = mag1 * np.exp(1j * (2.0 * np.pi * float(u[2])))
    a2 = mag2 * np.exp(1j * (2.0 * np.pi * float(u[4])))
    return np.array([theta, a1.real, a1.imag, a2.real, a2.imag], dtype=float)


def random_product_prep(
    nfocks: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Random product state Ry(θ)|0⟩ ⊗ |α₁⟩ ⊗ |α₂⟩, frozen during ECD BFGS."""
    return unit_to_prep(rng.random(N_PREP_PARAMS), nfocks)


def _history_lite(history: list[dict]) -> list[dict]:
    lite = []
    for h in history:
        rec = {
            "iteration": int(h["iteration"]),
            "energy_physical": float(h["energy_physical"]),
            "target_prob_physical": float(h.get("target_prob_physical", float("nan"))),
            "most_likely": h.get("most_likely"),
            "most_likely_bitstring": h.get("most_likely_bitstring"),
            "cost": h.get("cost"),
            "eta": h.get("eta"),
        }
        lite.append(rec)
    return lite


def optimize_ecd_bfgs(
    sim: HybridSimulator,
    x0: np.ndarray,
    *,
    objective: str,
    maxiter: int,
    record_every: int,
    adaptive_eta: bool,
    verbose: bool,
):
    """BFGS on energy or Gibbs. Same recording as ``optimize_vqe`` plus η updates."""
    objective = str(objective)
    x0 = np.asarray(x0, dtype=float)
    if objective == "energy":
        return optimize_vqe(
            sim,
            x0,
            method="BFGS",
            maxiter=maxiter,
            record_every=record_every,
            verbose=verbose,
            objective="energy",
        )

    policy = SampledTailEta()
    ev0 = sim.evaluate(x0)
    st0 = policy.initialize(sim.energy_tensor, ev0.measurement.physical_probs)
    sim.gibbs_eta = float(st0.eta)
    sim.cost_kind = "gibbs"
    history: list[dict] = []
    iteration = {"k": 0}

    def fun(x):
        return sim.cost(x, objective="gibbs", gibbs_eta=float(policy.eta))

    def callback(xk, *args):
        iteration["k"] += 1
        k = iteration["k"]
        if adaptive_eta:
            probs = np.asarray(sim.evaluate(xk).measurement.physical_probs, dtype=float)
            st = policy.maybe_update(k, int(maxiter), sim.energy_tensor, probs)
            sim.gibbs_eta = float(st.eta)
        if record_every > 0 and k % int(record_every) == 0:
            rec = _history_record(sim, xk, k)
            rec["eta"] = float(policy.eta)
            rec["cost"] = float(sim.cost(xk, objective="gibbs", gibbs_eta=float(policy.eta)))
            history.append(rec)
            if verbose:
                print(
                    f"iter {k}: Gibbs={rec['cost']:.6f}  E={rec['energy_physical']:.6f}  "
                    f"P*={rec['target_prob_physical']:.4f}  η={policy.eta:.3f}",
                    flush=True,
                )

    result = sciopt.minimize(
        fun,
        x0,
        method="BFGS",
        callback=callback,
        options={"maxiter": int(maxiter), "disp": bool(verbose)},
    )
    if not history or history[-1]["iteration"] != iteration["k"]:
        rec = _history_record(sim, result.x, iteration["k"])
        rec["eta"] = float(policy.eta)
        rec["cost"] = float(result.fun)
        history.append(rec)
    out = OptimizeResult(
        fun=float(result.fun),
        x=np.asarray(result.x, dtype=float),
        history=history,
        nfev=int(result.nfev),
        nit=int(result.nit),
        message=str(result.message),
        success=bool(result.success),
    )
    out.eta = float(policy.eta)
    out.eta0 = float(st0.eta)
    return out


def run_ecd_trial(job: dict) -> dict:
    """Picklable worker: one notebook start, energy or Gibbs BFGS.

    ``job["prep"]`` is optional. If omitted, the hybrid vacuum is used.
    If present, it is the frozen 5-vector of Ry(θ)|0⟩ ⊗ |α₁⟩ ⊗ |α₂⟩.
    ``job["ansatz"]`` is ``"ecd"`` (default) or ``"snap"``.
    """
    x0 = np.asarray(job["x0"], dtype=float)
    objective = str(job.get("objective", "energy"))
    ansatz = str(job.get("ansatz", "ecd")).lower()
    ndepth = int(job.get("ndepth", 5))
    prep = job.get("prep")
    init = "random_product" if prep is not None else "vacuum"
    sim = HybridSimulator(ndepth=ndepth, cost_kind=objective, ansatz=ansatz)
    expected = n_snap_parameters(ndepth, sim.nfocks) if ansatz == "snap" else n_parameters(ndepth)
    if x0.size != expected:
        raise ValueError(f"Expected {expected} {ansatz} parameters for ndepth={ndepth}, got {x0.size}.")
    if prep is not None:
        sim.initial_state = prep_params_to_ket(np.asarray(prep, dtype=float), sim.nfocks)
    ev0 = sim.evaluate(x0)
    opt = optimize_ecd_bfgs(
        sim,
        x0,
        objective=objective,
        maxiter=int(job["maxiter"]),
        record_every=int(job.get("record_every", 10)),
        adaptive_eta=bool(job.get("adaptive_eta", True)) and objective == "gibbs",
        verbose=False,
    )
    final = sim.evaluate(opt.x)
    rec = {
        "trial": int(job["trial"]),
        "seed": int(job["seed"]),
        "ansatz": ansatz,
        "ndepth": ndepth,
        "n_parameters": int(sim.n_ansatz_params()),
        "objective": objective,
        "init": init,
        "prep": None if prep is None else np.asarray(prep, dtype=float),
        "nit": int(opt.nit),
        "nfev": int(opt.nfev),
        "fun": float(opt.fun),
        "energy_physical": float(final.energy_physical),
        "pstar": float(final.target_prob_physical),
        "pstar0": float(ev0.target_prob_physical),
        "energy0": float(ev0.energy_physical),
        "most_likely": list(final.most_likely),
        "most_likely_bitstring": str(final.most_likely_bitstring),
        "success": tuple(int(v) for v in final.most_likely) == TARGET_QNM,
        "eta": getattr(opt, "eta", None),
        "eta0": getattr(opt, "eta0", None),
        "history": _history_lite(opt.history),
        "x0": x0,
        "x": np.asarray(opt.x, dtype=float),
    }
    return rec
