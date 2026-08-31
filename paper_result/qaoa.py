"""Qubit-only QAOA matching Dutta et al. (arXiv:2501.11735) Fig. 6–7.

The published notebook (QAOA_QuTiP_PaperCode.ipynb) uses QuTiP ``expm`` of
the full 128×128 cost and mixer operators. That is equivalent to, and much
slower than, the diagonal cost phase and single-qubit X mixers implemented
here. Circuit details copied from the notebook:

* initial state ``|+>^{⊗7}``
* layer ``k`` applies ``exp(-i |γ_k| H_P)`` then ``exp(-i |β_k| H_M)``
* ``H_P`` is the Eq. (25) knapsack Hamiltonian (unshifted; a constant
  identity shift does not change the energy landscape)
* ``H_M = Σ_j X_j``
* BFGS on ``(γ, β) ∈ [0, π)^{2p}`` random starts, ``p = 20`` for Fig. 7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import scipy.optimize as sciopt

from qumode_vqe.eta import SampledTailEta
from qumode_vqe.hamiltonian import (
    EXACT_GROUND_ENERGY,
    N_QUBITS,
    TARGET_BITSTRING,
    bits_from_qubit_index,
    qubo_energy,
)
from qumode_vqe.vqe import gibbs_objective

N = N_QUBITS
DIM = 1 << N
# Published Fig. 7 / Fig. 6 p=20 squared overlap with |0110000>.
PUBLISHED_PSTAR = 0.12624853432076624
PUBLISHED_LAYERS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20)


def bitstring_from_index(index: int, n: int = N) -> str:
    return format(int(index), f"0{int(n)}b")


def index_from_bitstring(bits: str) -> int:
    return int(bits, 2)


def qubo_spectrum(n: int = N) -> np.ndarray:
    """Diagonal of Eq. (25) in QuTiP tensor order (qubit 0 = MSB)."""
    energies = np.empty(1 << int(n), dtype=float)
    for idx in range(energies.size):
        energies[idx] = qubo_energy(bits_from_qubit_index(idx, n))
    return energies


def plus_state(n: int = N) -> np.ndarray:
    dim = 1 << int(n)
    return np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)


def apply_mixer(psi: np.ndarray, beta: float, n: int = N) -> np.ndarray:
    """Apply ⊗_j exp(-i |β| X_j) by rotating each qubit in place."""
    b = abs(float(beta))
    c = np.cos(b)
    s = -1j * np.sin(b)
    t = np.asarray(psi, dtype=complex).reshape((2,) * int(n))
    for q in range(int(n)):
        t0 = np.take(t, 0, axis=q)
        t1 = np.take(t, 1, axis=q)
        t = np.stack((c * t0 + s * t1, s * t0 + c * t1), axis=q)
    return t.reshape(-1)


def qaoa_state(params: np.ndarray, energies: np.ndarray, n: int = N) -> np.ndarray:
    """Paper Eq. (27) with the notebook's ``abs(γ)``, ``abs(β)`` wrapping."""
    x = np.asarray(params, dtype=float).reshape(-1)
    if x.size % 2:
        raise ValueError(f"QAOA parameter length must be even, got {x.size}.")
    p = x.size // 2
    gamma = x[:p]
    beta = x[p:]
    psi = plus_state(n)
    e = np.asarray(energies, dtype=float).reshape(-1)
    for k in range(p):
        psi = psi * np.exp(-1j * abs(float(gamma[k])) * e)
        psi = apply_mixer(psi, float(beta[k]), n)
    return psi


def born_probs(psi: np.ndarray) -> np.ndarray:
    p = np.abs(np.asarray(psi, dtype=complex).reshape(-1)) ** 2
    total = float(p.sum())
    if total <= 0.0:
        return np.full(p.shape, 1.0 / max(p.size, 1))
    return p.real / total


def random_qaoa_params(p: int, rng: np.random.Generator) -> np.ndarray:
    """Notebook initialization: Uniform[0, π) for every γ and β."""
    return rng.random(2 * int(p)) * np.pi


def histogram_stats(probs: np.ndarray, energies: np.ndarray, eta: float | None = None) -> dict:
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
    gs = int(np.argmin(e))
    energy_physical = float(np.dot(p, e))
    rec = {
        "energy_physical": energy_physical,
        "energy_diag": found,
        "energy_min": e_min,
        "energy_max": e_max,
        "gap_to_min": found - e_min,
        "rel_gap": (found - e_min) / spread,
        "pstar": float(p[gs]),
        "most_likely_index": idx,
        "most_likely_bitstring": bitstring_from_index(idx),
        "ground_index": gs,
        "ground_bitstring": bitstring_from_index(gs),
        "success": bool(np.isclose(found, e_min, atol=1e-8, rtol=0.0)),
        "eta": None if eta is None else float(eta),
    }
    if eta is not None:
        rec["gibbs_cost"] = float(gibbs_objective(p, e, eta))
    return rec


@dataclass
class QAOAResult:
    x: np.ndarray
    x0: np.ndarray
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str
    objective: str
    p_layers: int
    eta: float | None
    eta0: float | None
    history: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)
    probs: np.ndarray | None = None


def optimize_qaoa(
    x0: np.ndarray,
    energies: np.ndarray,
    *,
    objective: str = "energy",
    maxiter: int = 200,
    record_every: int = 10,
    adaptive_eta: bool = True,
    n: int = N,
) -> QAOAResult:
    """BFGS on ⟨H⟩ or on the Gibbs cost −ln⟨e^{−ηE}⟩, same QAOA circuit."""
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    e = np.asarray(energies, dtype=float).reshape(-1)
    p_layers = x0.size // 2
    objective = str(objective)
    if objective not in ("energy", "gibbs"):
        raise ValueError(f"unknown objective {objective!r}")

    policy = None
    eta0 = None
    if objective == "gibbs":
        policy = SampledTailEta()
        eta0 = float(policy.initialize(e, born_probs(qaoa_state(x0, e, n))).eta)

        def fun(params: np.ndarray) -> float:
            p = born_probs(qaoa_state(params, e, n))
            return gibbs_objective(p, e, float(policy.eta))

    else:

        def fun(params: np.ndarray) -> float:
            p = born_probs(qaoa_state(params, e, n))
            return float(np.dot(p, e))

    history: list[dict] = []
    iteration = {"k": 0}

    def callback(xk, *args):
        iteration["k"] += 1
        k = iteration["k"]
        if policy is not None and adaptive_eta:
            p_now = born_probs(qaoa_state(xk, e, n))
            policy.maybe_update(k, int(maxiter), e, p_now)
        if record_every > 0 and k % int(record_every) == 0:
            eta = None if policy is None else float(policy.eta)
            rec = histogram_stats(born_probs(qaoa_state(xk, e, n)), e, eta=eta)
            rec["iteration"] = k
            rec["cost"] = float(fun(xk))
            history.append(rec)

    options = {"maxiter": int(maxiter), "disp": False}
    result = sciopt.minimize(fun, x0, method="BFGS", callback=callback, options=options)
    x = np.asarray(result.x, dtype=float)
    eta = None if policy is None else float(policy.eta)
    probs = born_probs(qaoa_state(x, e, n))
    final = histogram_stats(probs, e, eta=eta)
    final["cost"] = float(result.fun)
    final["iteration"] = int(result.nit)
    if not history or history[-1]["iteration"] != int(result.nit):
        history.append(dict(final))
    return QAOAResult(
        x=x,
        x0=x0,
        fun=float(result.fun),
        nit=int(result.nit),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        objective=objective,
        p_layers=p_layers,
        eta=eta,
        eta0=eta0,
        history=history,
        final=final,
        probs=probs,
    )


def run_qaoa_trial(job: dict) -> dict:
    """Picklable worker: one random (or supplied) start on one layer count."""
    p = int(job["p_layers"])
    energies = np.asarray(job["energies"], dtype=float)
    rng = np.random.default_rng(int(job["seed"]))
    x0 = np.asarray(job["x0"], dtype=float) if job.get("x0") is not None else random_qaoa_params(p, rng)
    opt = optimize_qaoa(
        x0,
        energies,
        objective=str(job.get("objective", "energy")),
        maxiter=int(job["maxiter"]),
        record_every=int(job.get("record_every", 10)),
        adaptive_eta=bool(job.get("adaptive_eta", True)),
    )
    rec = {
        "trial": int(job["trial"]),
        "seed": int(job["seed"]),
        "p_layers": p,
        "objective": opt.objective,
        "x0": opt.x0,
        "x": opt.x,
        "fun": opt.fun,
        "nit": opt.nit,
        "nfev": opt.nfev,
        "optimizer_success": opt.success,
        "eta0": opt.eta0,
        "eta": opt.eta,
        "probs": opt.probs,
        "history": opt.history,
        **opt.final,
    }
    return rec


def check_spectrum(energies: np.ndarray | None = None) -> dict:
    """Sanity checks against the paper knapsack instance."""
    e = qubo_spectrum() if energies is None else np.asarray(energies, dtype=float)
    gs = int(np.argmin(e))
    return {
        "dim": int(e.size),
        "energy_min": float(np.min(e)),
        "mean_energy": float(np.mean(e)),
        "ground_bitstring": bitstring_from_index(gs),
        "target_bitstring": TARGET_BITSTRING,
        "ground_is_target": bitstring_from_index(gs) == TARGET_BITSTRING,
        "ground_energy_ok": bool(np.isclose(float(np.min(e)), EXACT_GROUND_ENERGY)),
        "uniform_energy_is_identity": bool(np.isclose(float(np.mean(e)), 41.75, atol=1e-8)),
    }
