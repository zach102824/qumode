"""ECD-VQE cost functions, circuit evolution, and classical optimizers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import qutip as qt
import scipy.optimize as sciopt

from .channels import apply_full_unitary, density_physicality, ket_to_dm
from .circuit import (
    N_PREP_PARAMS,
    as_hybrid_ket,
    ecd_ansatz_unitary,
    ecd_rotation_pair,
    apply_coherent_errors,
    prep_bounds,
    prep_params_to_ket,
    project_prep_params,
    snap_ansatz_unitary,
    snap_displacement_pair,
    uer_layer,
    vacuum,
)
from .eta import SampledTailEta
from .hamiltonian import (
    DEFAULT_NFOCKS,
    EXACT_GROUND_ENERGY,
    TARGET_QNM,
    diagonal_hybrid_hamiltonian,
    hybrid_energy_tensor,
    hybrid_hamiltonian,
)
from .measurement import MeasurementConfig, MeasurementResult, joint_probabilities, measure
from .noise import ChannelCache, LossModel, NoiseConfig, TimingMode
from .params import (
    ParamLayout,
    cartesian_bounds,
    n_parameters,
    n_snap_parameters,
    paper_bounds,
    random_parameters,
    snap_bounds,
    unpack,
    unpack_snap,
)

# Fallback η for HybridSimulator.cost when Gibbs is evaluated outside
# optimize_gibbs_adaptive (which always sets η via sampled_tail).
DEFAULT_GIBBS_ETA = 1.0 / (0.05 * abs(EXACT_GROUND_ENERGY))


def gibbs_objective(
    probs: np.ndarray,
    energies: np.ndarray,
    eta: float = DEFAULT_GIBBS_ETA,
) -> float:
    """Gibbs objective f = −ln ⟨e^{−ηE}⟩, numerically shifted by min E."""
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


@dataclass
class EvalResult:
    energy_ideal: float | None
    energy_physical: float
    energy_observed: float
    target_prob_physical: float
    target_prob_observed: float
    most_likely: tuple[int, int, int]
    most_likely_bitstring: str
    trace: float
    purity: float
    n_shots: int | None
    physicality: dict[str, float]
    measurement: MeasurementResult

    def as_dict(self) -> dict:
        q, n, m = self.most_likely
        return {
            "energy_ideal": self.energy_ideal,
            "energy_physical": self.energy_physical,
            "energy_observed": self.energy_observed,
            "target_prob_physical": self.target_prob_physical,
            "target_prob_observed": self.target_prob_observed,
            "most_likely": [q, n, m],
            "most_likely_bitstring": self.most_likely_bitstring,
            "trace": self.trace,
            "purity": self.purity,
            "n_shots": self.n_shots,
        }


class HybridSimulator:
    """Statevector (ideal) and density-matrix (noisy) simulator."""

    def __init__(
        self,
        ndepth: int = 5,
        nfocks: Sequence[int] = DEFAULT_NFOCKS,
        noise: NoiseConfig | None = None,
        measurement: MeasurementConfig | None = None,
        layout: ParamLayout = ParamLayout.PAPER,
        hamiltonian: qt.Qobj | None = None,
        energy_tensor: np.ndarray | None = None,
        target_qnm: tuple[int, int, int] | None = TARGET_QNM,
        partition: tuple[int, int, int] = (1, 3, 3),
        cost_kind: str = "energy",
        gibbs_eta: float | None = None,
        pair_mask: np.ndarray | None = None,
        initial_state: qt.Qobj | np.ndarray | None = None,
        ansatz: str = "ecd",
    ) -> None:
        kind = str(ansatz).lower()
        if kind not in ("ecd", "snap"):
            raise ValueError(f"Unknown ansatz {ansatz!r}")
        self.ansatz = kind
        self.ndepth = int(ndepth)
        self.nfocks = (int(nfocks[0]), int(nfocks[1]))
        self.dims = (2, self.nfocks[0], self.nfocks[1])
        self.layout = layout
        self.target_qnm = None if target_qnm is None else tuple(int(v) for v in target_qnm)
        self.partition = partition
        self.noise = noise or NoiseConfig(loss_model=LossModel.NONE, dims=self.dims)
        if self.noise.dims != self.dims:
            self.noise.dims = self.dims
        self.measurement = measurement or MeasurementConfig()
        if hamiltonian is not None:
            self.hamiltonian = hamiltonian
            if energy_tensor is None:
                diag = np.real(np.diag(np.asarray(self.hamiltonian.full())))
                self.energy_tensor = diag.reshape(self.dims)
            else:
                self.energy_tensor = np.asarray(energy_tensor, dtype=float)
        else:
            self.hamiltonian = hybrid_hamiltonian(self.nfocks)
            self.energy_tensor = hybrid_energy_tensor(self.nfocks)
        self._h_np = np.asarray(self.hamiltonian.full(), dtype=complex)
        self._channel_cache: ChannelCache | None = None
        self._needs_dm = not self.noise.is_identity()
        if cost_kind not in ("energy", "gibbs"):
            raise ValueError(f"Unknown cost_kind {cost_kind}")
        self.cost_kind = cost_kind
        self.gibbs_eta = float(gibbs_eta) if gibbs_eta is not None else DEFAULT_GIBBS_ETA
        self.pair_mask = None if pair_mask is None else np.asarray(pair_mask, dtype=bool)
        self._set_initial_state(initial_state)

    def _set_initial_state(self, ket: qt.Qobj | np.ndarray | None) -> None:
        if ket is None:
            self._initial_ket = vacuum(self.nfocks)
        else:
            self._initial_ket = as_hybrid_ket(ket, self.nfocks)

    @property
    def initial_state(self) -> qt.Qobj:
        return self._initial_ket

    @initial_state.setter
    def initial_state(self, ket: qt.Qobj | np.ndarray | None) -> None:
        self._set_initial_state(ket)

    @property
    def channel_cache(self) -> ChannelCache:
        if self._channel_cache is None:
            self._channel_cache = ChannelCache(self.noise)
        return self._channel_cache

    def unpack(self, xvec: np.ndarray):
        if self.ansatz == "snap":
            return unpack_snap(xvec, self.ndepth, self.nfocks)
        return unpack(xvec, self.ndepth, self.layout)

    def n_ansatz_params(self) -> int:
        if self.ansatz == "snap":
            return n_snap_parameters(self.ndepth, self.nfocks)
        return n_parameters(self.ndepth)

    def _errored_params(self, xvec: np.ndarray):
        p = self.unpack(xvec)
        amp = self.noise.ecd_amp_rel_error
        phase = self.noise.ecd_phase_error
        rot = self.noise.rotation_rel_error
        if amp == 0.0 and phase == 0.0 and rot == 0.0:
            return p.beta, p.theta, p.phi
        beta = np.empty_like(p.beta)
        theta = np.empty_like(p.theta)
        phi = p.phi.copy()
        for i in range(self.ndepth):
            for k in range(2):
                b, th, ph = apply_coherent_errors(
                    p.beta[i, k], p.theta[i, k], p.phi[i, k], amp, phase, rot
                )
                beta[i, k] = b
                theta[i, k] = th
                phi[i, k] = ph
        return beta, theta, phi

    def statevector(self, xvec: np.ndarray) -> qt.Qobj:
        if self.ansatz == "snap":
            params = unpack_snap(xvec, self.ndepth, self.nfocks)
            return snap_ansatz_unitary(params, self.nfocks) * self._initial_ket
        beta, theta, phi = self._errored_params(xvec)
        from .params import UnpackedParams

        params = UnpackedParams(beta=beta, theta=theta, phi=phi)
        return ecd_ansatz_unitary(params, self.nfocks, pair_mask=self.pair_mask) * self._initial_ket

    def density_matrix(self, xvec: np.ndarray) -> np.ndarray:
        if self.ansatz == "snap":
            return self._snap_density_matrix(xvec)
        beta, theta, phi = self._errored_params(xvec)
        rho = ket_to_dm(self._initial_ket)
        cache = self.channel_cache
        apply_noise = not self.noise.is_identity() or self.noise.loss_model is not LossModel.NONE
        # Even identity paper kraus with kappa_tau=0 is identity; skip if truly none.
        do_channel = (
            self.noise.loss_model is not LossModel.NONE
            or self.noise.enable_transmon
            or abs(self.noise.kerr)
            or abs(self.noise.cross_kerr)
            or abs(self.noise.chi_dispersive)
        )
        mask = self.pair_mask
        use_mask = mask is not None and not np.all(mask)
        if not use_mask and self.noise.timing is TimingMode.PER_UER_LAYER:
            for i in range(self.ndepth):
                u = np.asarray(
                    uer_layer(beta[i], theta[i], phi[i], self.nfocks).full(), dtype=complex
                )
                rho = apply_full_unitary(rho, u)
                if do_channel:
                    rho = cache.apply(rho)
        else:
            for i in range(self.ndepth):
                for cind in (0, 1):
                    if use_mask and not bool(mask[i, cind]):
                        continue
                    u = np.asarray(
                        ecd_rotation_pair(
                            beta[i, cind], theta[i, cind], phi[i, cind], cind, self.nfocks
                        ).full(),
                        dtype=complex,
                    )
                    rho = apply_full_unitary(rho, u)
                    if do_channel and self.noise.timing is TimingMode.PER_ECD_PAIR:
                        rho = cache.apply(rho)
                if do_channel and self.noise.timing is TimingMode.PER_UER_LAYER:
                    rho = cache.apply(rho)
        return rho

    def _snap_density_matrix(self, xvec: np.ndarray) -> np.ndarray:
        params = unpack_snap(xvec, self.ndepth, self.nfocks)
        rho = ket_to_dm(self._initial_ket)
        cache = self.channel_cache
        do_channel = (
            self.noise.loss_model is not LossModel.NONE
            or self.noise.enable_transmon
            or abs(self.noise.kerr)
            or abs(self.noise.cross_kerr)
            or abs(self.noise.chi_dispersive)
        )
        l1, l2 = self.nfocks
        for i in range(self.ndepth):
            for cind, n_fock in ((0, l1), (1, l2)):
                u = np.asarray(
                    snap_displacement_pair(
                        params.alpha[i, cind],
                        params.phases[i, cind, :n_fock],
                        cind,
                        self.nfocks,
                    ).full(),
                    dtype=complex,
                )
                rho = apply_full_unitary(rho, u)
                if do_channel and self.noise.timing is TimingMode.PER_ECD_PAIR:
                    rho = cache.apply(rho)
            if do_channel and self.noise.timing is TimingMode.PER_UER_LAYER:
                rho = cache.apply(rho)
        return rho

    def evaluate(self, xvec: np.ndarray, *, include_ideal: bool = False) -> EvalResult:
        xvec = np.asarray(xvec, dtype=float)
        energy_ideal = None
        if include_ideal or not self._needs_dm:
            psi = self.statevector(xvec)
            if not self._needs_dm:
                rho = ket_to_dm(psi)
                energy_ideal = float(np.real(qt.expect(self.hamiltonian, psi)))
            else:
                energy_ideal = float(np.real(qt.expect(self.hamiltonian, psi)))
                rho = self.density_matrix(xvec)
        else:
            rho = self.density_matrix(xvec)

        phys = density_physicality(rho)
        physical_probs = joint_probabilities(rho, self.dims)
        meas = measure(
            physical_probs,
            self.measurement,
            self.energy_tensor,
            self.target_qnm,
            self.partition,
        )
        energy_physical = float(np.real(np.trace(self._h_np @ rho)))
        return EvalResult(
            energy_ideal=energy_ideal,
            energy_physical=energy_physical,
            energy_observed=meas.energy_observed,
            target_prob_physical=meas.target_prob_physical,
            target_prob_observed=meas.target_prob_observed,
            most_likely=meas.most_likely,
            most_likely_bitstring=meas.most_likely_bitstring,
            trace=phys["trace_real"],
            purity=phys["purity"],
            n_shots=meas.n_shots,
            physicality=phys,
            measurement=meas,
        )

    def cost(
        self,
        xvec: np.ndarray,
        observed: bool = False,
        objective: str | None = None,
        gibbs_eta: float | None = None,
    ) -> float:
        """Deterministic VQE objective.

        ``energy`` is Tr(H ρ) (or the readout-confused histogram energy if
        ``observed``). ``gibbs`` is −ln ⟨e^{−ηE}⟩ on the same histogram
        (Li et al., Phys. Rev. Research 2, 023074). Defaults follow
        ``self.cost_kind`` / ``self.gibbs_eta``.
        """
        result = self.evaluate(xvec)
        kind = self.cost_kind if objective is None else objective
        eta = self.gibbs_eta if gibbs_eta is None else float(gibbs_eta)
        if kind == "gibbs":
            probs = result.measurement.observed_probs if observed else result.measurement.physical_probs
            return gibbs_objective(probs, self.energy_tensor, eta)
        if kind != "energy":
            raise ValueError(f"Unknown objective {kind!r}")
        return result.energy_observed if observed else result.energy_physical


@dataclass
class OptimizeResult:
    fun: float
    x: np.ndarray
    history: list[dict] = field(default_factory=list)
    nfev: int = 0
    nit: int = 0
    message: str = ""
    success: bool = True


@dataclass
class AdaptiveGibbsResult:
    """Joint (and optional freeze) Gibbs SPSA result.

    With the default budget, ``spsa_iter=0`` so ``x`` equals ``x_warmup``:
    all 45 coordinates (5 prep + ECD ansatz) stay live for every step.
    ``x_warmup`` is the joint-stage endpoint; ``x`` is after the optional
    ansatz-only continuation.
    """

    prep: np.ndarray
    x: np.ndarray
    x_warmup: np.ndarray
    prep0: np.ndarray
    x0: np.ndarray
    fun_warmup: float
    fun: float
    nfev_warmup: int
    nfev_ansatz: int
    nfev: int
    nit_warmup: int
    nit: int
    eval_warmup: dict
    eval_final: dict
    eta_policy: str = "sampled_tail"
    eta0: float = float("nan")
    eta: float = float("nan")
    eta_history: list[dict] = field(default_factory=list)
    n_eta_clamps: int = 0
    n_eta_fallbacks: int = 0


def _history_record(sim: HybridSimulator, x: np.ndarray, iteration: int) -> dict:
    ev = sim.evaluate(x, include_ideal=False)
    rec = ev.as_dict()
    rec["iteration"] = iteration
    rec["x"] = np.asarray(x, dtype=float).copy()
    rec["probs"] = np.asarray(ev.measurement.physical_probs, dtype=float).copy()
    return rec


def optimize_vqe(
    sim: HybridSimulator,
    x0: np.ndarray | None = None,
    *,
    method: str = "BFGS",
    maxiter: int = 200,
    tol: float = 1e-12,
    observed: bool = False,
    objective: str | None = None,
    gibbs_eta: float | None = None,
    rng: np.random.Generator | None = None,
    record_every: int = 1,
    verbose: bool = False,
    bounds: Sequence[tuple[float, float]] | None = None,
    spsa_a: float = 0.2,
    spsa_c: float = 0.15,
    spsa_A: float = 10.0,
    spsa_alpha: float = 0.602,
    spsa_gamma: float = 0.101,
) -> OptimizeResult:
    """Minimize the VQE cost.

    BFGS / L-BFGS-B are allowed only for deterministic objectives. Finite-shot
    stochastic costs must use COBYLA or SPSA.
    """
    rng = rng or np.random.default_rng()
    if x0 is None:
        x0 = random_parameters(sim.ndepth, rng, sim.layout)
    x0 = np.asarray(x0, dtype=float)

    stochastic = sim.measurement.n_shots is not None and observed
    method_u = method.upper()
    if stochastic and method_u in {"BFGS", "L-BFGS-B"}:
        raise ValueError("Finite-shot objectives are stochastic; use COBYLA or SPSA, not BFGS.")

    cost_kw = {"observed": observed, "objective": objective, "gibbs_eta": gibbs_eta}

    if method_u == "SPSA":
        return _spsa(
            sim,
            x0,
            maxiter=maxiter,
            rng=rng,
            record_every=record_every,
            bounds=bounds,
            a=spsa_a,
            c=spsa_c,
            A=spsa_A,
            alpha=spsa_alpha,
            gamma=spsa_gamma,
            **cost_kw,
        )

    opt_bounds = bounds
    if opt_bounds is None and method_u == "L-BFGS-B":
        opt_bounds = paper_bounds(sim.ndepth) if sim.layout is ParamLayout.PAPER else cartesian_bounds(sim.ndepth)

    history: list[dict] = []
    nfev_counter = {"n": 0}

    def fun(x):
        nfev_counter["n"] += 1
        return sim.cost(x, **cost_kw)

    iteration = {"k": 0}

    def callback(xk, *args):
        iteration["k"] += 1
        if record_every > 0 and iteration["k"] % record_every == 0:
            rec = _history_record(sim, xk, iteration["k"])
            history.append(rec)
            if verbose:
                pstar = rec["target_prob_physical"]
                extra = f"  P*={pstar:.4f}" if np.isfinite(pstar) else ""
                print(f"iter {iteration['k']}: E={rec['energy_physical']:.6f}{extra}")

    options = {"maxiter": int(maxiter), "disp": bool(verbose)}
    result = sciopt.minimize(
        fun,
        x0,
        method=method,
        bounds=opt_bounds,
        tol=tol,
        callback=callback,
        options=options,
    )
    if not history or history[-1]["iteration"] != iteration["k"]:
        history.append(_history_record(sim, result.x, iteration["k"]))
    return OptimizeResult(
        fun=float(result.fun),
        x=np.asarray(result.x, dtype=float),
        history=history,
        nfev=int(result.nfev),
        nit=int(result.nit),
        message=str(result.message),
        success=bool(result.success),
    )


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
    a: float = 0.2,
    c: float = 0.15,
    A: float = 10.0,
    alpha: float = 0.602,
    gamma: float = 0.101,
    on_iterate: Callable[[int, np.ndarray, float], None] | None = None,
    on_before_step: Callable[[int, np.ndarray], None] | None = None,
    step_scale: np.ndarray | None = None,
) -> OptimizeResult:
    """Box-constrained SPSA on an arbitrary scalar objective.

    ``step_scale`` multiplies the parameter update per coordinate (e.g. a
    smaller gain on preparation variables). Perturbations still use ``c_k``.
    """

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
    last_fun = 0.0
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
        last_fun = 0.5 * (yp + ym)
        if on_iterate is not None:
            on_iterate(k, x, last_fun)
    return OptimizeResult(
        fun=float(fun(x)),
        x=x,
        nfev=nfev,
        nit=int(maxiter),
        message="SPSA completed",
        success=True,
    )


def _spsa(
    sim: HybridSimulator,
    x0: np.ndarray,
    *,
    maxiter: int,
    observed: bool,
    objective: str | None,
    gibbs_eta: float | None,
    rng: np.random.Generator,
    record_every: int,
    bounds: Sequence[tuple[float, float]] | None = None,
    a: float = 0.2,
    c: float = 0.15,
    A: float = 10.0,
    alpha: float = 0.602,
    gamma: float = 0.101,
) -> OptimizeResult:
    history: list[dict] = []
    cost_kw = {"observed": observed, "objective": objective, "gibbs_eta": gibbs_eta}

    def fun(x: np.ndarray) -> float:
        return sim.cost(x, **cost_kw)

    def on_iterate(k: int, x: np.ndarray, last_fun: float) -> None:
        if record_every > 0 and k % record_every == 0:
            rec = _history_record(sim, x, k)
            rec["spsa_mean_cost"] = last_fun
            history.append(rec)

    result = run_spsa(
        fun,
        x0,
        maxiter=maxiter,
        rng=rng,
        bounds=bounds,
        a=a,
        c=c,
        A=A,
        alpha=alpha,
        gamma=gamma,
        on_iterate=on_iterate,
    )
    history.append(_history_record(sim, result.x, int(maxiter)))
    result.history = history
    return result


def _ansatz_bounds(sim: HybridSimulator) -> list[tuple[float, float]]:
    if sim.ansatz == "snap":
        return list(snap_bounds(sim.ndepth, sim.nfocks))
    if sim.layout is ParamLayout.PAPER:
        return list(paper_bounds(sim.ndepth))
    return list(cartesian_bounds(sim.ndepth))


# Default Gibbs budget: SPSA on all 45 coordinates (5 prep + ECD) for 70
# steps. ``DEFAULT_ANSATZ_STEPS = 0`` means prep is never frozen.
DEFAULT_JOINT_STEPS = 70
DEFAULT_ANSATZ_STEPS = 0


def optimize_gibbs_adaptive(
    prep0: np.ndarray,
    x0: np.ndarray,
    *,
    ndepth: int = 5,
    nfocks: Sequence[int] = DEFAULT_NFOCKS,
    outer_iter: int = DEFAULT_JOINT_STEPS,
    spsa_iter: int = DEFAULT_ANSATZ_STEPS,
    rng: np.random.Generator | None = None,
    noise=None,
    measurement: MeasurementConfig | None = None,
    layout: ParamLayout = ParamLayout.PAPER,
    a: float = 0.2,
    c: float = 0.15,
    A: float = 10.0,
    alpha: float = 0.602,
    gamma: float = 0.101,
    prep_step_scale: float = 1.0,
    energy_tensor: np.ndarray | None = None,
    hamiltonian: qt.Qobj | None = None,
    ansatz: str = "ecd",
) -> AdaptiveGibbsResult:
    """Joint prep+ansatz SPSA (default: 70 steps, prep never frozen).

    Default ``outer_iter=70``, ``spsa_iter=0``: one SPSA trajectory on the
    full 45-vector (5 preparation coordinates + ECD ansatz). Prep stays live
    for the whole budget. ``spsa_iter>0`` is an optional ablation that
    freezes prep after the joint stage and continues the **same** ansatz
    vector (not a new seed) for ``spsa_iter`` more steps.

    ``ansatz`` is ``"ecd"`` (8 parameters per UER layer) or ``"snap"``
    (gauge-fixed SNAP+displacement, 18 parameters per layer at nfocks=(8,8)).

    ``prep_step_scale`` multiplies the SPSA update on the five preparation
    coordinates during the joint stage (ansatz coordinates keep gain 1).

    Gibbs η is always ``sampled_tail``: probability-weighted 5%/25% energy
    quantiles of the current histogram, EMA-smoothed, refreshed at the
    unperturbed iterate and held fixed for both SPSA probes of that step.
    """
    rng = rng or np.random.default_rng()
    nfocks = (int(nfocks[0]), int(nfocks[1]))
    prep0 = project_prep_params(prep0, nfocks)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    ansatz = str(ansatz).lower()
    expected = n_snap_parameters(ndepth, nfocks) if ansatz == "snap" else n_parameters(ndepth)
    if x0.size != expected:
        raise ValueError(
            f"Expected {expected} {ansatz} ansatz parameters for ndepth={ndepth}, got {x0.size}."
        )
    ham_kw: dict = {}
    if energy_tensor is not None:
        tensor = np.asarray(energy_tensor, dtype=float)
        ham_kw["energy_tensor"] = tensor
        ham_kw["hamiltonian"] = hamiltonian if hamiltonian is not None else diagonal_hybrid_hamiltonian(tensor)
    elif hamiltonian is not None:
        ham_kw["hamiltonian"] = hamiltonian
    sim = HybridSimulator(
        ndepth=ndepth,
        nfocks=nfocks,
        noise=noise,
        measurement=measurement,
        layout=layout,
        cost_kind="gibbs",
        target_qnm=None,
        initial_state=prep_params_to_ket(prep0, nfocks),
        ansatz=ansatz,
        **ham_kw,
    )
    policy = SampledTailEta()

    ansatz_bounds = _ansatz_bounds(sim)
    x0 = _clip_bounds(x0, ansatz_bounds)
    joint_bounds = list(prep_bounds(nfocks)) + ansatz_bounds
    total_steps = max(int(outer_iter) + int(spsa_iter), 1)

    def current_probs(x_ansatz: np.ndarray) -> np.ndarray:
        ev = sim.evaluate(x_ansatz)
        return np.asarray(ev.measurement.physical_probs, dtype=float)

    init_probs = current_probs(x0)
    st0 = policy.initialize(sim.energy_tensor, init_probs)
    sim.gibbs_eta = float(st0.eta)
    eta0 = float(st0.eta)

    def project_joint(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).copy()
        z[:N_PREP_PARAMS] = project_prep_params(z[:N_PREP_PARAMS], nfocks)
        return z

    def joint_fun(z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        sim.initial_state = prep_params_to_ket(z[:N_PREP_PARAMS], nfocks)
        return sim.cost(z[N_PREP_PARAMS:], objective="gibbs", gibbs_eta=float(policy.eta))

    def before_joint(k: int, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float)
        sim.initial_state = prep_params_to_ket(z[:N_PREP_PARAMS], nfocks)
        probs = current_probs(z[N_PREP_PARAMS:])
        st = policy.maybe_update(k, total_steps, sim.energy_tensor, probs)
        sim.gibbs_eta = float(st.eta)

    z0 = np.concatenate([prep0, x0])
    joint_scale = np.ones(z0.size, dtype=float)
    joint_scale[:N_PREP_PARAMS] = float(prep_step_scale)
    if int(outer_iter) > 0:
        warm = run_spsa(
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
        prep = project_prep_params(warm.x[:N_PREP_PARAMS], nfocks)
        x_warmup = _clip_bounds(np.asarray(warm.x[N_PREP_PARAMS:], dtype=float), ansatz_bounds)
        nfev_warmup = int(warm.nfev)
        nit_warmup = int(warm.nit)
    else:
        prep = prep0
        x_warmup = x0
        nfev_warmup = 0
        nit_warmup = 0

    sim.initial_state = prep_params_to_ket(prep, nfocks)
    fun_warmup = float(sim.cost(x_warmup, objective="gibbs", gibbs_eta=float(policy.eta)))
    eval_warmup = sim.evaluate(x_warmup).as_dict()
    nfev_warmup = max(nfev_warmup, 1)

    def ansatz_fun(x: np.ndarray) -> float:
        return sim.cost(x, objective="gibbs", gibbs_eta=float(policy.eta))

    def before_ansatz(k: int, x: np.ndarray) -> None:
        probs = current_probs(x)
        st = policy.maybe_update(int(outer_iter) + k, total_steps, sim.energy_tensor, probs)
        sim.gibbs_eta = float(st.eta)

    if int(spsa_iter) > 0:
        opt = run_spsa(
            ansatz_fun,
            x_warmup,
            maxiter=int(spsa_iter),
            rng=rng,
            bounds=ansatz_bounds,
            a=a,
            c=c,
            A=A,
            alpha=alpha,
            gamma=gamma,
            on_before_step=before_ansatz,
        )
        x_final = np.asarray(opt.x, dtype=float)
        fun_final = float(opt.fun)
        nfev_ansatz = int(opt.nfev)
        nit = int(opt.nit)
    else:
        x_final = np.asarray(x_warmup, dtype=float)
        fun_final = float(fun_warmup)
        nfev_ansatz = 0
        nit = 0

    eval_final = sim.evaluate(x_final).as_dict()
    snap = policy.snapshot()
    return AdaptiveGibbsResult(
        prep=prep,
        x=x_final,
        x_warmup=x_warmup,
        prep0=prep0,
        x0=x0,
        fun_warmup=fun_warmup,
        fun=fun_final,
        nfev_warmup=nfev_warmup,
        nfev_ansatz=nfev_ansatz,
        nfev=nfev_warmup + nfev_ansatz,
        nit_warmup=nit_warmup,
        nit=nit,
        eval_warmup=eval_warmup,
        eval_final=eval_final,
        eta_policy=str(snap["name"]),
        eta0=eta0,
        eta=float(policy.eta),
        eta_history=list(snap["history"]),
        n_eta_clamps=int(snap["n_clamps"]),
        n_eta_fallbacks=int(snap["n_fallbacks"]),
    )


def greedy_pair_aas(
    sim: HybridSimulator,
    xvec: np.ndarray,
    *,
    max_remove: int = 3,
    tol: float = 1e-6,
) -> np.ndarray:
    """Greedy ECD-pair dropout, analog of Li et al. ansatz architecture search.

    Starts from the current ``pair_mask`` (or all-on) and, using *fixed*
    parameters, drops one ECD–rotation pair at a time while the objective
    keeps improving. This is the paper's inexpensive "fixed-parameter"
    scoring, not a nested reoptimization at every candidate.
    """
    nd = sim.ndepth
    mask = (
        np.ones((nd, 2), dtype=bool)
        if sim.pair_mask is None
        else np.asarray(sim.pair_mask, dtype=bool).reshape(nd, 2).copy()
    )
    orig = None if sim.pair_mask is None else np.asarray(sim.pair_mask, dtype=bool).copy()
    try:
        sim.pair_mask = mask
        best = float(sim.cost(xvec))
        for _ in range(int(max_remove)):
            on = [(int(i), int(c)) for i, c in zip(*np.where(mask))]
            if len(on) <= 1:
                break
            best_cand = None
            best_score = best
            for i, cind in on:
                trial = mask.copy()
                trial[i, cind] = False
                sim.pair_mask = trial
                score = float(sim.cost(xvec))
                if score < best_score - tol:
                    best_score = score
                    best_cand = trial
            if best_cand is None:
                break
            mask = best_cand
            best = best_score
            sim.pair_mask = mask
        return mask
    finally:
        sim.pair_mask = orig


def evaluate_fixed_parameters(
    xvec: np.ndarray,
    noise: NoiseConfig,
    measurement: MeasurementConfig | None = None,
    ndepth: int = 5,
    nfocks: Sequence[int] = DEFAULT_NFOCKS,
    layout: ParamLayout = ParamLayout.PAPER,
) -> EvalResult:
    sim = HybridSimulator(ndepth, nfocks, noise, measurement, layout)
    return sim.evaluate(xvec, include_ideal=True)
