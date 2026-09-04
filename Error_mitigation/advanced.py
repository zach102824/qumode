"""Regularized GDR, hybrid ZNE+readout, and identity-aware unfolding.

These sit on top of :mod:`Error_mitigation.mitigation` and do not change
the original ``gdr_param`` / ``zne_idle`` behaviour.  Mild-loss overfit is
the main target: at κτ=0.003 the physical TVD is ~0.02 while 4k-shot TVD
is ~0.06, so unconstrained 11-parameter MLE plus aggressive Richardson–
Lucy unfolding over-corrects shot noise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import optimize

from qumode_vqe.measurement import energy_from_histogram

from .metrics import total_variation
from .mitigation import (
    EPS_PROB,
    PARAM_BOUNDS,
    PARAM_NAMES,
    apply_transfer,
    confusion_from_measurement,
    factorial_moments,
    initial_theta,
    multinomial_nll,
    params_to_kernels,
    richardson_lucy,
    run_readout_only,
    unfold,
    zne_histogram,
)
from .noise_models import ReadoutSpec, is_trivial_readout, readout_config

# Indices into PARAM_NAMES / PARAM_BOUNDS.
_I_ETA1, _I_ETA2 = 0, 1
_I_NTH1, _I_NTH2 = 2, 3
_I_DOWN, _I_UP, _I_EPS = 4, 5, 6
_I_P01, _I_P10 = 7, 8
_I_PNN1, _I_PNN2 = 9, 10
_EXTRA_IDX = (_I_DOWN, _I_UP, _I_EPS)
_NTH_IDX = (_I_NTH1, _I_NTH2)
_READOUT_IDX = (_I_P01, _I_P10, _I_PNN1, _I_PNN2)
_ETA_IDX = (_I_ETA1, _I_ETA2)

# Default CV grid for the η→1 ridge (units: added to NLL, so scale with shots).
DEFAULT_L2_ETA_GRID = (0.0, 20.0, 80.0, 320.0, 1280.0)


@dataclass
class GdrRegConfig:
    """Regularization / constraints for parametric GDR.

    ``l2_*`` weights multiply ``n_shots * n_twins * (param - target)^2``.
    ``freeze_*`` pins those coordinates to the initializer (true readout
    spec, nth from the noise config, extras at 0).
    """

    l2_eta: float = 80.0
    eta_prior: str = "unity"  # "unity" | "init"
    l2_nth: float = 40.0
    l2_extra: float = 400.0
    l2_readout: float = 200.0
    freeze_readout: bool = False
    freeze_extra: bool = False
    freeze_nth: bool = False
    holdout_frac: float = 0.25
    moment_weight: float = 0.0
    energy_weight: float = 0.0
    energy_tensor: np.ndarray | None = None
    cv_l2_eta: tuple[float, ...] | None = None
    reduced: str | None = None  # None | "eta" | "eta_nth" | "eta_nth_extra"
    unfold_mode: str = "safe"  # "rl" | "safe" | "shrink"
    rng_seed: int = 0


def _apply_reduced(cfg_reg: GdrRegConfig) -> GdrRegConfig:
    mode = cfg_reg.reduced
    if mode is None:
        return cfg_reg
    if mode == "eta":
        return replace(cfg_reg, freeze_readout=True, freeze_extra=True, freeze_nth=True)
    if mode == "eta_nth":
        return replace(cfg_reg, freeze_readout=True, freeze_extra=True, freeze_nth=False)
    if mode == "eta_nth_extra":
        return replace(cfg_reg, freeze_readout=True, freeze_extra=False, freeze_nth=False)
    raise ValueError(f"unknown reduced model {mode!r}")


def _frozen_mask(reg: GdrRegConfig) -> np.ndarray:
    freeze = np.zeros(len(PARAM_NAMES), dtype=bool)
    if reg.freeze_readout:
        freeze[list(_READOUT_IDX)] = True
    if reg.freeze_extra:
        freeze[list(_EXTRA_IDX)] = True
    if reg.freeze_nth:
        freeze[list(_NTH_IDX)] = True
    return freeze


def _bounds_with_freeze(x0: np.ndarray, freeze: np.ndarray) -> list[tuple[float, float]]:
    bounds = []
    for i, (lo, hi) in enumerate(PARAM_BOUNDS):
        if freeze[i]:
            v = float(np.clip(x0[i], lo, hi))
            bounds.append((v, v))
        else:
            bounds.append((lo, hi))
    return bounds


def _eta_target(x0: np.ndarray, reg: GdrRegConfig) -> float:
    if reg.eta_prior == "init":
        return 0.5 * (float(x0[_I_ETA1]) + float(x0[_I_ETA2]))
    return 1.0


def _regularizer(
    theta: np.ndarray,
    x0: np.ndarray,
    reg: GdrRegConfig,
    n_shots: int,
    n_twins: int,
    freeze: np.ndarray,
) -> float:
    t = np.asarray(theta, dtype=float).reshape(-1)
    scale = max(float(n_shots), 1.0) * max(int(n_twins), 1)
    eta_tgt = _eta_target(x0, reg)
    pen = 0.0
    if not freeze[_I_ETA1]:
        pen += reg.l2_eta * scale * ((t[_I_ETA1] - eta_tgt) ** 2 + (t[_I_ETA2] - eta_tgt) ** 2)
    if not freeze[_I_NTH1]:
        pen += reg.l2_nth * scale * ((t[_I_NTH1] - x0[_I_NTH1]) ** 2 + (t[_I_NTH2] - x0[_I_NTH2]) ** 2)
    if not freeze[_I_DOWN]:
        pen += reg.l2_extra * scale * (t[_I_DOWN] ** 2 + t[_I_UP] ** 2 + t[_I_EPS] ** 2)
    if not freeze[_I_P01]:
        pen += reg.l2_readout * scale * (
            (t[_I_P01] - x0[_I_P01]) ** 2
            + (t[_I_P10] - x0[_I_P10]) ** 2
            + (t[_I_PNN1] - x0[_I_PNN1]) ** 2
            + (t[_I_PNN2] - x0[_I_PNN2]) ** 2
        )
    return float(pen)


def _moment_penalty(
    theta: np.ndarray,
    p_ideals: list[np.ndarray],
    q_obs: list[np.ndarray],
    dims: tuple[int, int, int],
    weight: float,
    n_shots: int,
) -> float:
    if weight <= 0.0:
        return 0.0
    cq, c1, c2 = params_to_kernels(theta, dims)
    eta = 0.5 * (float(theta[_I_ETA1]) + float(theta[_I_ETA2]))
    pen = 0.0
    for p, q in zip(p_ideals, q_obs):
        gn = factorial_moments(q, 2)
        gi = factorial_moments(p, 2)
        for mode in ("mode1", "mode2"):
            for k, (a, b) in enumerate(zip(gn[mode], gi[mode]), start=1):
                if abs(b) < 1e-10:
                    continue
                pen += (float(a / b) - eta**k) ** 2
    return float(weight) * max(float(n_shots), 1.0) * pen


def _energy_penalty(
    theta: np.ndarray,
    p_ideals: list[np.ndarray],
    q_obs: list[np.ndarray],
    dims: tuple[int, int, int],
    tensor: np.ndarray | None,
    weight: float,
    n_shots: int,
) -> float:
    if weight <= 0.0 or tensor is None:
        return 0.0
    cq, c1, c2 = params_to_kernels(theta, dims)
    pen = 0.0
    for p, q in zip(p_ideals, q_obs):
        pred = apply_transfer(p, cq, c1, c2)
        e_pred = energy_from_histogram(pred, tensor)
        e_obs = energy_from_histogram(q, tensor)
        pen += (e_pred - e_obs) ** 2
    return float(weight) * max(float(n_shots), 1.0) * pen


def regularized_nll(
    theta: np.ndarray,
    p_ideals: list[np.ndarray],
    q_obs: list[np.ndarray],
    n_shots: int,
    dims: tuple[int, int, int],
    x0: np.ndarray,
    reg: GdrRegConfig,
    freeze: np.ndarray,
) -> float:
    nll = multinomial_nll(theta, p_ideals, q_obs, n_shots, dims)
    nll += _regularizer(theta, x0, reg, n_shots, len(p_ideals), freeze)
    nll += _moment_penalty(theta, p_ideals, q_obs, dims, reg.moment_weight, n_shots)
    nll += _energy_penalty(theta, p_ideals, q_obs, dims, reg.energy_tensor, reg.energy_weight, n_shots)
    return float(nll)


def _split_holdout(
    p_ideals: list[np.ndarray],
    q_obs: list[np.ndarray],
    frac: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n = len(p_ideals)
    if frac <= 0.0 or n < 4:
        return p_ideals, q_obs, [], []
    n_hold = max(1, int(round(frac * n)))
    n_hold = min(n_hold, n - 2)
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    hold = idx[:n_hold]
    train = idx[n_hold:]
    p_tr = [p_ideals[i] for i in train]
    q_tr = [q_obs[i] for i in train]
    p_ho = [p_ideals[i] for i in hold]
    q_ho = [q_obs[i] for i in hold]
    return p_tr, q_tr, p_ho, q_ho


def _fit_once(
    p_ideals: list[np.ndarray],
    q_obs: list[np.ndarray],
    x0: np.ndarray,
    spec: ReadoutSpec,
    dims: tuple[int, int, int],
    reg: GdrRegConfig,
    freeze: np.ndarray,
    maxiter: int,
) -> tuple[np.ndarray, dict]:
    bounds = _bounds_with_freeze(x0, freeze)
    result = optimize.minimize(
        regularized_nll,
        x0,
        args=(p_ideals, q_obs, spec.n_shots, dims, x0, reg, freeze),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-10},
    )
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    theta = np.clip(np.asarray(result.x, dtype=float), lo, hi)
    info = {
        "success": bool(result.success),
        "nll": float(result.fun),
        "nfev": int(result.nfev),
        "message": str(result.message),
    }
    return theta, info


def _holdout_nll(
    theta: np.ndarray,
    p_hold: list[np.ndarray],
    q_hold: list[np.ndarray],
    n_shots: int,
    dims: tuple[int, int, int],
) -> float:
    if not p_hold:
        return float("nan")
    return float(multinomial_nll(theta, p_hold, q_hold, n_shots, dims))


def _fit_info(theta: np.ndarray, cfg, spec: ReadoutSpec, ndepth: int, extra: dict) -> dict:
    fitted = {name: float(theta[i]) for i, name in enumerate(PARAM_NAMES)}
    true_eta = float(np.exp(-cfg.cumulative_kappa_t(int(ndepth))))
    info = {
        "fitted": fitted,
        "true_eta": true_eta,
        "true_nth": float(cfg.nth_cav),
        "true_p01": float(spec.p01),
        "true_p10": float(spec.p10),
        "true_p_nn": float(spec.p_nn),
        "d_eta1": abs(fitted["eta1"] - true_eta),
        "d_eta2": abs(fitted["eta2"] - true_eta),
        "d_p01": abs(fitted["p01"] - spec.p01),
        "d_p10": abs(fitted["p10"] - spec.p10),
        "d_p_nn1": abs(fitted["p_nn1"] - spec.p_nn),
        "d_p_nn2": abs(fitted["p_nn2"] - spec.p_nn),
    }
    info.update(extra)
    return info


def fit_gdr_param_reg(
    p_ideals: list[np.ndarray],
    q_obs: list[np.ndarray],
    cfg,
    spec: ReadoutSpec,
    ndepth: int,
    dims: tuple[int, int, int],
    *,
    maxiter: int = 200,
    reg: GdrRegConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Parametric GDR with L2 / freeze / holdout-CV.

    When ``reg.cv_l2_eta`` is set and a holdout split exists, pick the
    η-ridge that minimises holdout multinomial NLL (early-stopping
    analogue for L-BFGS-B).
    """
    reg = _apply_reduced(reg if reg is not None else GdrRegConfig())
    x0 = initial_theta(cfg, spec, ndepth)
    if reg.freeze_extra:
        x0[_I_DOWN] = 0.0
        x0[_I_UP] = 0.0
        x0[_I_EPS] = 0.0
    freeze = _frozen_mask(reg)
    p_tr, q_tr, p_ho, q_ho = _split_holdout(p_ideals, q_obs, reg.holdout_frac, reg.rng_seed)

    grid = reg.cv_l2_eta
    if grid and p_ho:
        best = None
        scanned = []
        for lam in grid:
            reg_l = replace(reg, l2_eta=float(lam))
            theta_l, fit_l = _fit_once(p_tr, q_tr, x0, spec, dims, reg_l, freeze, maxiter)
            ho = _holdout_nll(theta_l, p_ho, q_ho, spec.n_shots, dims)
            scanned.append({"l2_eta": float(lam), "holdout_nll": ho, "train_nll": fit_l["nll"]})
            if best is None or ho < best[0]:
                best = (ho, theta_l, fit_l, float(lam))
        theta, fit_l, chosen = best[1], best[2], best[3]
        extra = {**fit_l, "holdout_nll": best[0], "chosen_l2_eta": chosen, "cv": scanned, "reg": _reg_as_dict(reg)}
        return theta, _fit_info(theta, cfg, spec, ndepth, extra)

    # Refit on all twins with the configured λ (holdout used only as a report).
    theta, fit_l = _fit_once(p_ideals, q_obs, x0, spec, dims, reg, freeze, maxiter)
    ho = _holdout_nll(theta, p_ho, q_ho, spec.n_shots, dims) if p_ho else float("nan")
    extra = {**fit_l, "holdout_nll": ho, "chosen_l2_eta": float(reg.l2_eta), "reg": _reg_as_dict(reg)}
    return theta, _fit_info(theta, cfg, spec, ndepth, extra)


def _reg_as_dict(reg: GdrRegConfig) -> dict:
    return {
        "l2_eta": float(reg.l2_eta),
        "eta_prior": str(reg.eta_prior),
        "l2_nth": float(reg.l2_nth),
        "l2_extra": float(reg.l2_extra),
        "l2_readout": float(reg.l2_readout),
        "freeze_readout": bool(reg.freeze_readout),
        "freeze_extra": bool(reg.freeze_extra),
        "freeze_nth": bool(reg.freeze_nth),
        "holdout_frac": float(reg.holdout_frac),
        "moment_weight": float(reg.moment_weight),
        "energy_weight": float(reg.energy_weight),
        "reduced": reg.reduced,
        "unfold_mode": str(reg.unfold_mode),
        "cv_l2_eta": None if reg.cv_l2_eta is None else [float(x) for x in reg.cv_l2_eta],
    }


def kernel_identity_tvd(cq: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> float:
    """Mean per-register TVD of each kernel from the identity (0 = no noise)."""

    def _tvd_I(c: np.ndarray) -> float:
        eye = np.eye(c.shape[0], dtype=float)
        # Average column TVD.
        return 0.5 * float(np.abs(c - eye).sum()) / c.shape[1]

    return float((_tvd_I(cq) + _tvd_I(c1) + _tvd_I(c2)) / 3.0)


def unfold_safe(
    q: np.ndarray,
    cq: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    *,
    n_iter: int = 80,
    ident_floor: float = 0.008,
) -> np.ndarray:
    """Richardson–Lucy with fewer iterations when M ≈ I (mild noise).

    Full 80-step RL on a near-identity kernel fits shot noise.  Scale the
    iteration count with how far the kernel is from the identity.
    """
    ident = kernel_identity_tvd(cq, c1, c2)
    if ident <= ident_floor:
        n_use = max(4, int(round(n_iter * ident / max(ident_floor, 1e-12))))
    else:
        n_use = int(n_iter)
    return richardson_lucy(q, cq, c1, c2, n_iter=n_use)


def unfold_shrink(
    q: np.ndarray,
    cq: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    spec: ReadoutSpec,
    dims: tuple[int, int, int],
    *,
    n_iter: int = 80,
) -> np.ndarray:
    """Mix full unfolding with readout-only (or raw) using kernel distance.

    α = clip(ident / 0.04, 0, 1): near-identity circuit noise → trust the
    readout-inverted histogram; stronger kernels → full GDR unfold.
    """
    p_full = unfold_safe(q, cq, c1, c2, n_iter=n_iter)
    if is_trivial_readout(spec):
        p_base = np.clip(np.asarray(q, dtype=float), 0.0, None)
        s = float(p_base.sum())
        p_base = p_base / s if s > 0.0 else np.full(p_base.shape, 1.0 / p_base.size)
    else:
        ro = run_readout_only(q, spec, dims)
        p_base = ro.histogram if ro is not None else p_full
    ident = kernel_identity_tvd(cq, c1, c2)
    alpha = float(np.clip(ident / 0.04, 0.0, 1.0))
    mixed = (1.0 - alpha) * p_base + alpha * p_full
    mixed = np.clip(mixed, 0.0, None)
    tot = float(mixed.sum())
    return mixed / tot if tot > 0.0 else p_full


def unfold_configured(
    q: np.ndarray,
    cq: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    spec: ReadoutSpec,
    dims: tuple[int, int, int],
    mode: str = "safe",
) -> np.ndarray:
    if mode == "rl":
        return unfold(q, cq, c1, c2, method="rl")
    if mode == "shrink":
        return unfold_shrink(q, cq, c1, c2, spec, dims)
    return unfold_safe(q, cq, c1, c2)


def run_gdr_variant(
    q_obs: np.ndarray,
    p_twin_ideal: list[np.ndarray],
    q_twin_obs: list[np.ndarray],
    cfg,
    spec: ReadoutSpec,
    ndepth: int,
    dims: tuple[int, int, int],
    energy_tensor: np.ndarray,
    *,
    name: str,
    maxiter: int,
    reg: GdrRegConfig,
) -> dict:
    theta, fit_info = fit_gdr_param_reg(
        p_twin_ideal, q_twin_obs, cfg, spec, ndepth, dims, maxiter=maxiter, reg=reg
    )
    cq, c1, c2 = params_to_kernels(theta, dims)
    hist = unfold_configured(q_obs, cq, c1, c2, spec, dims, mode=reg.unfold_mode)
    from .mitigation import oracle_residual

    return {
        "hist": hist,
        "energy": energy_from_histogram(hist, energy_tensor),
        "fit": fit_info,
        "residual_tvd": oracle_residual(p_twin_ideal[0], q_twin_obs[0], cq, c1, c2) if p_twin_ideal else None,
        "identity_tvd": kernel_identity_tvd(cq, c1, c2),
        "name": name,
        "cq": cq,
        "c1": c1,
        "c2": c2,
        "theta": theta,
    }


def fit_gdr_two_stage(
    p_ideals: list[np.ndarray],
    q_obs: list[np.ndarray],
    t_free: list[int],
    cfg,
    spec: ReadoutSpec,
    ndepth: int,
    dims: tuple[int, int, int],
    *,
    maxiter: int = 200,
) -> tuple[np.ndarray, dict]:
    """Gaussian-exact kernel on t_free=0, then a leak/hop residual on the rest.

    Stage 1: reduced η (+nth) with frozen readout on Gaussian twins.
    Stage 2: un-freeze extras, fit on t_free>0 twins (or all, if none).
    Composing two binomials is still a binomial, so the residual is the
    n-independent hop/leak piece that Gaussian twins cannot see.
    """
    gauss_p = [p for p, t in zip(p_ideals, t_free) if int(t) == 0]
    gauss_q = [q for q, t in zip(q_obs, t_free) if int(t) == 0]
    ng_p = [p for p, t in zip(p_ideals, t_free) if int(t) > 0]
    ng_q = [q for q, t in zip(q_obs, t_free) if int(t) > 0]
    if len(gauss_p) < 2:
        gauss_p, gauss_q = p_ideals, q_obs
    stage1 = GdrRegConfig(reduced="eta_nth", freeze_readout=True, freeze_extra=True, holdout_frac=0.0, unfold_mode="safe")
    theta, info1 = fit_gdr_param_reg(gauss_p, gauss_q, cfg, spec, ndepth, dims, maxiter=maxiter, reg=stage1)
    if not ng_p:
        info1["two_stage"] = "no_ng_twins"
        return theta, info1
    stage2 = GdrRegConfig(
        reduced="eta_nth_extra",
        freeze_readout=True,
        freeze_extra=False,
        freeze_nth=False,
        holdout_frac=0.0,
        l2_extra=80.0,
        l2_eta=40.0,
        unfold_mode="safe",
    )
    # Warm start: extras at 0, η from stage 1.
    theta2, info2 = fit_gdr_param_reg(ng_p, ng_q, cfg, spec, ndepth, dims, maxiter=maxiter, reg=stage2)
    # Keep stage-1 η if stage-2 extras are the only new d.o.f. we trust.
    # Blend: η from Gaussian twins (better identified), extras from NG twins.
    mixed = np.array(theta, dtype=float)
    mixed[_I_DOWN] = theta2[_I_DOWN]
    mixed[_I_UP] = theta2[_I_UP]
    mixed[_I_EPS] = theta2[_I_EPS]
    extra = {
        "stage1": info1,
        "stage2": info2,
        "two_stage": "gauss_eta_ng_extra",
        "success": True,
        "nll": info2.get("nll"),
        "nfev": int(info1.get("nfev", 0)) + int(info2.get("nfev", 0)),
        "message": "two-stage",
    }
    return mixed, _fit_info(mixed, cfg, spec, ndepth, extra)


def gdr_independent_registers(
    p_ideals: list[np.ndarray],
    q_obs: list[np.ndarray],
    cfg,
    spec: ReadoutSpec,
    ndepth: int,
    dims: tuple[int, int, int],
    *,
    maxiter: int = 200,
) -> tuple[np.ndarray, dict]:
    """gdr_param_reg with independent η, nth, p_nn per mode (already in the
    11-parameter model) but extras/readout frozen — the structured middle
    ground between a single-η binomial and gdr_full.
    """
    reg = GdrRegConfig(reduced="eta_nth", holdout_frac=0.0, unfold_mode="safe", l2_eta=20.0)
    return fit_gdr_param_reg(p_ideals, q_obs, cfg, spec, ndepth, dims, maxiter=maxiter, reg=reg)


def readout_then_zne(
    hist_by_scale: dict[int, np.ndarray],
    spec: ReadoutSpec,
    dims: tuple[int, int, int],
    *,
    degree: int = 2,
) -> np.ndarray:
    """Invert calibrated readout at each idle-time scale, then extrapolate.

    Detector errors do not stretch with idle time; this undoes them before
    Richardson extrapolation so the polynomial sees only circuit noise.
    """
    if is_trivial_readout(spec):
        return zne_histogram(hist_by_scale, degree=degree)
    meas = readout_config(spec, n_fock=int(dims[1]))
    cq, c1, c2 = confusion_from_measurement(meas, dims)
    unfolded = {int(s): unfold(h, cq, c1, c2, method="rl") for s, h in hist_by_scale.items()}
    return zne_histogram(unfolded, degree=degree)


def zne_then_readout(
    hist_by_scale: dict[int, np.ndarray],
    spec: ReadoutSpec,
    dims: tuple[int, int, int],
    *,
    degree: int = 2,
) -> np.ndarray:
    """Extrapolate shot histograms, then invert readout on the result."""
    extra = zne_histogram(hist_by_scale, degree=degree)
    if is_trivial_readout(spec):
        return extra
    ro = run_readout_only(extra, spec, dims)
    return extra if ro is None else ro.histogram


def per_layer_eta(cfg, ndepth: int) -> float:
    """Single-application η (product of n_app copies is the cumulative η)."""
    n_app = max(int(np.round(cfg.cumulative_kappa_t(int(ndepth)) / max(cfg.kappa_tau_used(), 1e-30))), 1)
    return float(np.exp(-cfg.kappa_tau_used()))


DEFAULT_VARIANTS: dict[str, GdrRegConfig] = {
    "gdr_param_reg": GdrRegConfig(
        l2_eta=80.0,
        l2_extra=400.0,
        l2_readout=200.0,
        holdout_frac=0.25,
        cv_l2_eta=DEFAULT_L2_ETA_GRID,
        unfold_mode="shrink",
    ),
    "gdr_eta": GdrRegConfig(reduced="eta", holdout_frac=0.0, unfold_mode="safe", l2_eta=40.0),
    "gdr_eta_nth": GdrRegConfig(reduced="eta_nth", holdout_frac=0.0, unfold_mode="safe", l2_eta=40.0),
    "gdr_energy": GdrRegConfig(
        reduced="eta_nth",
        holdout_frac=0.0,
        energy_weight=2.0,
        unfold_mode="safe",
        l2_eta=40.0,
    ),
}
