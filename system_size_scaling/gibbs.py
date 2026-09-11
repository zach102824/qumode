"""Noiseless Gibbs cost and joint prep+ECD SPSA (sampled_tail η)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .config import (
    OUTER_ITER,
    PREP_STEP_SCALE,
    SPSA_A,
    SPSA_A_STAB,
    SPSA_ALPHA,
    SPSA_C,
    SPSA_GAMMA,
)
from .ecd import (
    apply_ecd_ansatz,
    ecd_bounds,
    n_ecd_parameters,
    prep_bounds,
    prep_to_ket,
    project_prep,
    vacuum_prep,
)
from .embedding import Embedding
from .eta import SampledTailEta


def gibbs_objective(probs: np.ndarray, energies: np.ndarray, eta: float) -> float:
    """f = −ln ⟨e^{−ηE}⟩, numerically shifted by min E."""
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


def _clip_bounds(x: np.ndarray, bounds: Sequence[tuple[float, float]] | None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if bounds is None:
        return x
    lo = np.fromiter((b[0] for b in bounds), dtype=float, count=len(bounds))
    hi = np.fromiter((b[1] for b in bounds), dtype=float, count=len(bounds))
    if lo.size != x.size or hi.size != x.size:
        raise ValueError(f"bounds length {lo.size} does not match parameter length {x.size}.")
    return np.clip(x, lo, hi)


def run_spsa(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    maxiter: int,
    rng: np.random.Generator,
    bounds: Sequence[tuple[float, float]] | None = None,
    project: Callable[[np.ndarray], np.ndarray] | None = None,
    a: float = SPSA_A,
    c: float = SPSA_C,
    A: float = SPSA_A_STAB,
    alpha: float = SPSA_ALPHA,
    gamma: float = SPSA_GAMMA,
    on_before_step: Callable[[int, np.ndarray], None] | None = None,
    step_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    """Box-constrained SPSA. Returns (x, fun(x), nfev)."""

    def apply_project(vec: np.ndarray) -> np.ndarray:
        out = _clip_bounds(vec, bounds)
        if project is not None:
            out = np.asarray(project(out), dtype=float)
        return out

    x = apply_project(np.asarray(x0, dtype=float).copy())
    scale = None if step_scale is None else np.asarray(step_scale, dtype=float).reshape(-1)
    if scale is not None and scale.size != x.size:
        raise ValueError(f"step_scale length {scale.size} does not match parameter length {x.size}.")
    nfev = 0
    for k in range(1, int(maxiter) + 1):
        if on_before_step is not None:
            on_before_step(k, x)
        ak = a / (k + A) ** alpha
        ck = c / k**gamma
        delta = rng.choice([-1.0, 1.0], size=x.size)
        xp = apply_project(x + ck * delta)
        xm = apply_project(x - ck * delta)
        yp = float(fun(xp))
        ym = float(fun(xm))
        nfev += 2
        ghat = (yp - ym) / (2.0 * ck) * delta
        step = ak * ghat
        if scale is not None:
            step = step * scale
        x = apply_project(x - step)
    fun_x = float(fun(x))
    nfev += 1
    return x, fun_x, nfev


@dataclass
class EvalRecord:
    most_likely_bitstring: str
    ground_bitstring: str
    success: bool
    gibbs_cost: float
    energy: float
    p_most_likely: float
    p_ground: float
    eta: float
    most_likely_occupations: list[int]


@dataclass
class AdaptiveGibbsResult:
    prep: np.ndarray
    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    eval_final: EvalRecord
    eta: float
    eta0: float
    eta_policy: str = "sampled_tail"
    n_eta_clamps: int = 0
    n_eta_fallbacks: int = 0
    eta_history: list[dict] = field(default_factory=list)


class EcdGibbsSim:
    """Statevector ECD + diagonal Gibbs cost on a 5-mode (or subspace) register."""

    def __init__(
        self,
        emb: Embedding,
        energy_tensor: np.ndarray,
        ndepth: int,
        ground_bitstring: str,
    ) -> None:
        self.emb = emb
        self.energy_tensor = np.asarray(energy_tensor, dtype=float).reshape(emb.dims)
        if self.energy_tensor.shape != emb.dims:
            raise ValueError(f"energy tensor {self.energy_tensor.shape} != dims {emb.dims}")
        self.ndepth = int(ndepth)
        self.ground_bitstring = str(ground_bitstring)
        self.prep = vacuum_prep(emb)
        self.eta = 1.0
        self._n_ansatz = n_ecd_parameters(self.ndepth, emb.n_pairs)

    def statevector(self, xvec: np.ndarray) -> np.ndarray:
        psi0 = prep_to_ket(self.prep, self.emb)
        return apply_ecd_ansatz(psi0, xvec, self.emb, self.ndepth)

    def probabilities(self, xvec: np.ndarray) -> np.ndarray:
        psi = self.statevector(xvec)
        return (np.abs(psi) ** 2).reshape(self.emb.dims)

    def evaluate(self, xvec: np.ndarray) -> EvalRecord:
        probs = self.probabilities(xvec)
        flat_p = probs.reshape(-1)
        ml = int(np.argmax(flat_p))
        occ = [int(v) for v in np.unravel_index(ml, self.emb.dims)]
        bits = self.emb.decode_occupations(occ)
        label = self.emb.bitstring(bits)
        ground_occ = self.emb.encode_bits([int(c) for c in self.ground_bitstring])
        p_ground = float(probs[ground_occ])
        energy = float(np.sum(probs * self.energy_tensor))
        cost = gibbs_objective(probs, self.energy_tensor, self.eta)
        return EvalRecord(
            most_likely_bitstring=label,
            ground_bitstring=self.ground_bitstring,
            success=label == self.ground_bitstring,
            gibbs_cost=cost,
            energy=energy,
            p_most_likely=float(flat_p[ml]),
            p_ground=p_ground,
            eta=float(self.eta),
            most_likely_occupations=occ,
        )

    def cost(self, xvec: np.ndarray) -> float:
        probs = self.probabilities(xvec)
        return gibbs_objective(probs, self.energy_tensor, self.eta)


def optimize_gibbs_adaptive(
    emb: Embedding,
    energy_tensor: np.ndarray,
    ground_bitstring: str,
    x0: np.ndarray,
    *,
    ndepth: int,
    prep0: np.ndarray | None = None,
    outer_iter: int = OUTER_ITER,
    rng: np.random.Generator | None = None,
    a: float = SPSA_A,
    c: float = SPSA_C,
    A: float = SPSA_A_STAB,
    alpha: float = SPSA_ALPHA,
    gamma: float = SPSA_GAMMA,
    prep_step_scale: float = PREP_STEP_SCALE,
) -> AdaptiveGibbsResult:
    """Joint prep+ansatz SPSA with sampled_tail η (production protocol)."""
    rng = rng or np.random.default_rng()
    sim = EcdGibbsSim(emb, energy_tensor, ndepth, ground_bitstring)
    if prep0 is None:
        prep0 = vacuum_prep(emb)
    else:
        prep0 = project_prep(prep0, emb)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.size != sim._n_ansatz:
        raise ValueError(f"expected {sim._n_ansatz} ansatz params, got {x0.size}")

    ansatz_b = ecd_bounds(ndepth, emb.n_pairs)
    x0 = _clip_bounds(x0, ansatz_b)
    joint_bounds = list(prep_bounds(emb)) + ansatz_b
    n_prep = emb.n_prep_params

    policy = SampledTailEta()
    sim.prep = prep0
    init_probs = sim.probabilities(x0)
    st0 = policy.initialize(sim.energy_tensor, init_probs)
    sim.eta = float(st0.eta)
    eta0 = float(st0.eta)

    def project_joint(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).copy()
        z[:n_prep] = project_prep(z[:n_prep], emb)
        return z

    def joint_fun(z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        sim.prep = project_prep(z[:n_prep], emb)
        return sim.cost(z[n_prep:])

    def before_joint(k: int, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float)
        sim.prep = project_prep(z[:n_prep], emb)
        probs = sim.probabilities(z[n_prep:])
        st = policy.maybe_update(k, int(outer_iter), sim.energy_tensor, probs)
        sim.eta = float(st.eta)

    z0 = np.concatenate([prep0, x0])
    joint_scale = np.ones(z0.size, dtype=float)
    joint_scale[:n_prep] = float(prep_step_scale)
    if int(outer_iter) > 0:
        z_final, fun, nfev = run_spsa(
            joint_fun,
            z0,
            maxiter=int(outer_iter),
            rng=rng,
            bounds=joint_bounds,
            project=project_joint,
            a=a,
            c=c,
            A=A,
            alpha=alpha,
            gamma=gamma,
            on_before_step=before_joint,
            step_scale=joint_scale,
        )
        prep = project_prep(z_final[:n_prep], emb)
        x_final = _clip_bounds(z_final[n_prep:], ansatz_b)
        nit = int(outer_iter)
    else:
        prep = prep0
        x_final = x0
        sim.prep = prep
        fun = float(sim.cost(x_final))
        nfev = 1
        nit = 0

    sim.prep = prep
    ev = sim.evaluate(x_final)
    snap = policy.snapshot()
    return AdaptiveGibbsResult(
        prep=prep,
        x=x_final,
        fun=float(fun),
        nfev=int(nfev),
        nit=nit,
        eval_final=ev,
        eta=float(policy.eta),
        eta0=eta0,
        eta_policy=str(snap["name"]),
        n_eta_clamps=int(snap["n_clamps"]),
        n_eta_fallbacks=int(snap["n_fallbacks"]),
        eta_history=list(snap["history"]),
    )
