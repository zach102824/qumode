"""Histogram-level error mitigation for hybrid qubit-qumode circuits.

Methods
-------
raw
    Noisy shot histogram. Baseline.
oracle_binomial
    Known-model end-of-circuit kernel: thermal-loss birth-death with the
    true cumulative η = exp(−Σ κτ), composed with the true readout
    confusion. No learning. Residual TVD(M p_ideal, q_noisy) measures
    the interleaved-vs-end-of-circuit approximation (plus shot noise).
readout_only
    Invert only the MeasurementConfig confusion (Maciejewski et al.,
    Quantum 4, 257). Skipped when readout is ideal.
gdr_param
    Parametric Gaussian Data Regression. Fit a few-parameter Kronecker
    transfer (thermal loss × extra hops × leak × readout) by multinomial
    MLE on Gaussian twins, then unfold the target with Richardson-Lucy.
gdr_full
    Same twins, unstructured column-stochastic Cq ⊗ C1 ⊗ C2 fitted by
    alternating least squares (initialized from gdr_param).
scalar_cdr
    Classical CDR on the energy only: E_ideal ≈ a1 E_noisy + a0.
zne_idle
    Idle-time stretching of circuit noise (readout not scaled);
    Richardson extrapolation of each histogram bin.

Unfolding is Richardson-Lucy EM on the probability simplex, with NNLS
as a cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np
from scipy import linalg, optimize

from qumode_vqe.measurement import (
    apply_confusion,
    energy_from_histogram,
    identity_confusion,
    is_column_stochastic,
    nearest_neighbor_fock_confusion,
    qubit_bitflip_confusion,
)
from qumode_vqe.noise import NoiseConfig, TimingMode

from .metrics import total_variation
from .noise_models import ReadoutSpec, is_trivial_readout

PARAM_NAMES = (
    "eta1",
    "eta2",
    "nth1",
    "nth2",
    "p_down",
    "p_up",
    "eps",
    "p01",
    "p10",
    "p_nn1",
    "p_nn2",
)

PARAM_BOUNDS = [
    (0.15, 1.0),  # eta1
    (0.15, 1.0),  # eta2
    (0.0, 0.5),  # nth1
    (0.0, 0.5),  # nth2
    (0.0, 0.3),  # p_down
    (0.0, 0.3),  # p_up
    (0.0, 0.3),  # eps
    (0.0, 0.25),  # p01
    (0.0, 0.25),  # p10
    (0.0, 0.4),  # p_nn1
    (0.0, 0.4),  # p_nn2
]

EPS_PROB = 1e-15


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    m = np.clip(m, 0.0, None)
    col = m.sum(axis=0, keepdims=True)
    col = np.where(col <= 0.0, 1.0, col)
    return m / col


def binomial_loss_kernel(eta: float, n_fock: int) -> np.ndarray:
    """B[m, n] = C(n, m) η^m (1−η)^{n−m} for m ≤ n (column = true n)."""
    dim = int(n_fock)
    eta = float(np.clip(eta, 0.0, 1.0))
    b = np.zeros((dim, dim), dtype=float)
    if eta >= 1.0 - 1e-15:
        return np.eye(dim, dtype=float)
    for n in range(dim):
        for m in range(n + 1):
            b[m, n] = comb(n, m) * (eta**m) * ((1.0 - eta) ** (n - m))
    return _normalize_columns(b)


def thermal_loss_kernel(eta: float, nth: float, n_fock: int) -> np.ndarray:
    """Population transfer exp(G) of a truncated thermal-loss generator.

    Down-rate of |n⟩: (−ln η) (nth+1) n.
    Up-rate of |n⟩:   (−ln η) nth (n+1), truncated at n = L−1.

    Pure loss (nth = 0) recovers the binomial kernel up to truncation at
    the top Fock state. Number dephasing does not enter: it is diagonal
    in n and leaves populations alone.
    """
    dim = int(n_fock)
    eta = float(np.clip(eta, EPS_PROB, 1.0))
    nth = float(max(nth, 0.0))
    if abs(eta - 1.0) < 1e-15 and nth == 0.0:
        return np.eye(dim, dtype=float)
    kappa_t = -np.log(eta)
    g = np.zeros((dim, dim), dtype=float)
    for n in range(dim):
        down = kappa_t * (nth + 1.0) * n
        up = kappa_t * nth * (n + 1.0) if n < dim - 1 else 0.0
        g[n, n] -= down + up
        if n > 0:
            g[n - 1, n] += down
        if n < dim - 1:
            g[n + 1, n] += up
    return _normalize_columns(np.real(linalg.expm(g)))


def shift_kernel(n_fock: int, p_shift: float, direction: int) -> np.ndarray:
    """n-independent hop: with probability p, n → n+direction (clipped)."""
    dim = int(n_fock)
    p = float(np.clip(p_shift, 0.0, 1.0))
    c = np.zeros((dim, dim), dtype=float)
    for n in range(dim):
        dest = n + int(direction)
        if p <= 0.0 or dest < 0 or dest >= dim:
            c[n, n] = 1.0
        else:
            c[dest, n] = p
            c[n, n] = 1.0 - p
    return c


def leak_kernel(matrix: np.ndarray, eps: float) -> np.ndarray:
    dim = matrix.shape[0]
    e = float(np.clip(eps, 0.0, 1.0))
    if e <= 0.0:
        return matrix
    uniform = np.full((dim, dim), 1.0 / dim, dtype=float)
    return (1.0 - e) * matrix + e * uniform


def fock_kernel(
    eta: float,
    nth: float,
    p_down: float,
    p_up: float,
    eps: float,
    p_nn: float,
    n_fock: int,
) -> np.ndarray:
    """Loss then extra hops then leak then nearest-neighbour readout."""
    k = thermal_loss_kernel(eta, nth, n_fock)
    k = shift_kernel(n_fock, p_down, -1) @ k
    k = shift_kernel(n_fock, p_up, +1) @ k
    k = leak_kernel(k, eps)
    k = nearest_neighbor_fock_confusion(n_fock, p_nn) @ k
    return _normalize_columns(k)


def qubit_kernel(p01: float, p10: float) -> np.ndarray:
    return qubit_bitflip_confusion(float(p01), float(p10))


def params_to_kernels(theta: np.ndarray, dims: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(theta, dtype=float).reshape(-1)
    if t.size != len(PARAM_NAMES):
        raise ValueError(f"expected {len(PARAM_NAMES)} parameters, got {t.size}")
    lq, l1, l2 = (int(d) for d in dims)
    cq = qubit_kernel(t[7], t[8])
    c1 = fock_kernel(t[0], t[2], t[4], t[5], t[6], t[9], l1)
    c2 = fock_kernel(t[1], t[3], t[4], t[5], t[6], t[10], l2)
    if lq != 2:
        raise ValueError(f"expected a qubit, got dim {lq}")
    return cq, c1, c2


def apply_transfer(
    probs: np.ndarray,
    cq: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
) -> np.ndarray:
    return apply_confusion(probs, cq, c1, c2)


def apply_transfer_T(
    probs: np.ndarray,
    cq: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
) -> np.ndarray:
    """M† acting on an observed-space histogram."""
    return apply_confusion(probs, cq.T, c1.T, c2.T)


def kron_matrix(cq: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    return np.kron(cq, np.kron(c1, c2))


def richardson_lucy(
    q: np.ndarray,
    cq: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    *,
    n_iter: int = 80,
    eps: float = EPS_PROB,
) -> np.ndarray:
    """Nonnegative simplex unfolding of q ≈ M p."""
    qn = np.clip(np.asarray(q, dtype=float), 0.0, None)
    total = float(qn.sum())
    if total <= 0.0:
        return np.full(qn.shape, 1.0 / qn.size)
    qn = qn / total
    p = np.full(qn.shape, 1.0 / qn.size, dtype=float)
    for _ in range(int(n_iter)):
        mp = np.clip(apply_transfer(p, cq, c1, c2), eps, None)
        p = p * apply_transfer_T(qn / mp, cq, c1, c2)
        p = np.clip(p, 0.0, None)
        s = float(p.sum())
        p = p / s if s > 0.0 else np.full(qn.shape, 1.0 / qn.size)
    return p


def nnls_unfold(
    q: np.ndarray,
    cq: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
) -> np.ndarray:
    m = kron_matrix(cq, c1, c2)
    qv = np.clip(np.asarray(q, dtype=float).reshape(-1), 0.0, None)
    x, _ = optimize.nnls(m, qv)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s <= 0.0:
        x = np.ones_like(x) / x.size
    else:
        x = x / s
    return x.reshape(q.shape)


def unfold(
    q: np.ndarray,
    cq: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    *,
    method: str = "rl",
    n_iter: int = 80,
) -> np.ndarray:
    if method == "nnls":
        return nnls_unfold(q, cq, c1, c2)
    return richardson_lucy(q, cq, c1, c2, n_iter=n_iter)


def confusion_from_measurement(measurement, dims: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lq, l1, l2 = (int(d) for d in dims)
    cq = identity_confusion(lq) if measurement is None or measurement.qubit_c is None else np.asarray(measurement.qubit_c, dtype=float)
    c1 = identity_confusion(l1) if measurement is None or measurement.fock1_c is None else np.asarray(measurement.fock1_c, dtype=float)
    c2 = identity_confusion(l2) if measurement is None or measurement.fock2_c is None else np.asarray(measurement.fock2_c, dtype=float)
    return cq, c1, c2


def transmon_bitflip(cfg: NoiseConfig, ndepth: int) -> tuple[float, float]:
    """End-of-circuit amplitude-damping bit-flip from transmon T1."""
    if not cfg.enable_transmon:
        return 0.0, 0.0
    n_app = ndepth if cfg.timing is TimingMode.PER_UER_LAYER else 2 * ndepth
    tau = n_app * cfg.tau_application
    gamma = 1.0 - float(np.exp(-tau / cfg.t1_q)) if cfg.t1_q > 0.0 else 0.0
    p10 = gamma  # P(measure 0 | true 1) from T1
    p01 = gamma * float(cfg.nth_q) / (1.0 + float(cfg.nth_q)) if cfg.nth_q > 0.0 else 0.0
    return float(np.clip(p01, 0.0, 1.0)), float(np.clip(p10, 0.0, 1.0))


def oracle_kernels(
    cfg: NoiseConfig,
    spec: ReadoutSpec,
    ndepth: int,
    dims: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Known-model end-of-circuit loss composed with the true readout."""
    eta = float(np.exp(-cfg.cumulative_kappa_t(int(ndepth))))
    nth = float(cfg.nth_cav)
    l1, l2 = int(dims[1]), int(dims[2])
    b1 = thermal_loss_kernel(eta, nth, l1)
    b2 = thermal_loss_kernel(eta, nth, l2)
    p01_t, p10_t = transmon_bitflip(cfg, ndepth)
    # Circuit transmon damping then readout bit-flip.
    cq_circ = qubit_kernel(p01_t, p10_t)
    cq_ro = qubit_kernel(spec.p01, spec.p10)
    cq = cq_ro @ cq_circ
    c1 = nearest_neighbor_fock_confusion(l1, spec.p_nn) @ b1
    c2 = nearest_neighbor_fock_confusion(l2, spec.p_nn) @ b2
    return _normalize_columns(cq), _normalize_columns(c1), _normalize_columns(c2)


def oracle_residual(p_ideal: np.ndarray, q_noisy: np.ndarray, cq, c1, c2) -> float:
    return total_variation(apply_transfer(p_ideal, cq, c1, c2), q_noisy)


def initial_theta(cfg: NoiseConfig, spec: ReadoutSpec, ndepth: int) -> np.ndarray:
    eta = float(np.clip(np.exp(-cfg.cumulative_kappa_t(int(ndepth))), 0.15, 1.0))
    nth = float(np.clip(cfg.nth_cav, 0.0, 0.5))
    return np.array(
        [eta, eta, nth, nth, 0.0, 0.0, 0.0, spec.p01, spec.p10, spec.p_nn, spec.p_nn],
        dtype=float,
    )


def multinomial_nll(
    theta: np.ndarray,
    p_ideals: list[np.ndarray],
    q_obs: list[np.ndarray],
    n_shots: int,
    dims: tuple[int, int, int],
) -> float:
    cq, c1, c2 = params_to_kernels(theta, dims)
    nll = 0.0
    shots = max(int(n_shots), 1)
    for p, q in zip(p_ideals, q_obs):
        pred = np.clip(apply_transfer(p, cq, c1, c2), EPS_PROB, None)
        pred = pred / pred.sum()
        counts = np.clip(np.asarray(q, dtype=float), 0.0, None)
        counts = counts / max(float(counts.sum()), EPS_PROB) * shots
        nll -= float(np.sum(counts * np.log(pred)))
    return nll


def fit_gdr_param(
    p_ideals: list[np.ndarray],
    q_obs: list[np.ndarray],
    cfg: NoiseConfig,
    spec: ReadoutSpec,
    ndepth: int,
    dims: tuple[int, int, int],
    *,
    maxiter: int = 200,
) -> tuple[np.ndarray, dict]:
    x0 = initial_theta(cfg, spec, ndepth)
    result = optimize.minimize(
        multinomial_nll,
        x0,
        args=(p_ideals, q_obs, spec.n_shots, dims),
        method="L-BFGS-B",
        bounds=PARAM_BOUNDS,
        options={"maxiter": int(maxiter), "ftol": 1e-10},
    )
    theta = np.clip(np.asarray(result.x, dtype=float), [b[0] for b in PARAM_BOUNDS], [b[1] for b in PARAM_BOUNDS])
    fitted = {name: float(theta[i]) for i, name in enumerate(PARAM_NAMES)}
    true_eta = float(np.exp(-cfg.cumulative_kappa_t(int(ndepth))))
    info = {
        "success": bool(result.success),
        "nll": float(result.fun),
        "nfev": int(result.nfev),
        "message": str(result.message),
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
    return theta, info


def _project_stoch(matrix: np.ndarray) -> np.ndarray:
    return _normalize_columns(np.clip(matrix, 0.0, None))


def fit_gdr_full(
    p_ideals: list[np.ndarray],
    q_obs: list[np.ndarray],
    cq0: np.ndarray,
    c10: np.ndarray,
    c20: np.ndarray,
    *,
    n_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternating least-squares fit of Cq ⊗ C1 ⊗ C2, column-stochastic."""
    cq = _project_stoch(cq0)
    c1 = _project_stoch(c10)
    c2 = _project_stoch(c20)
    lq, l1, l2 = p_ideals[0].shape

    def _stack_true_obs(c_left, c_mid, c_right, axis: int):
        """Build Q ≈ C_axis @ T by contracting the other two kernels."""
        t_cols = []
        q_cols = []
        for p, q in zip(p_ideals, q_obs):
            if axis == 0:
                t = np.einsum("bn,cm,qnm->qbc", c_mid, c_right, p, optimize=True)
                t_cols.append(t.reshape(lq, -1))
                q_cols.append(np.asarray(q).reshape(lq, -1))
            elif axis == 1:
                t = np.einsum("aq,cm,qnm->anm", c_left, c_right, p, optimize=True)
                t_cols.append(np.moveaxis(t, 1, 0).reshape(l1, -1))
                qq = np.moveaxis(np.asarray(q), 1, 0)
                q_cols.append(qq.reshape(l1, -1))
            else:
                t = np.einsum("aq,bn,qnm->abm", c_left, c_mid, p, optimize=True)
                t_cols.append(np.moveaxis(t, 2, 0).reshape(l2, -1))
                qq = np.moveaxis(np.asarray(q), 2, 0)
                q_cols.append(qq.reshape(l2, -1))
        return np.concatenate(q_cols, axis=1), np.concatenate(t_cols, axis=1)

    for _ in range(int(n_iter)):
        q_mat, t_mat = _stack_true_obs(cq, c1, c2, 0)
        cq = _project_stoch(q_mat @ np.linalg.pinv(t_mat))
        q_mat, t_mat = _stack_true_obs(cq, c1, c2, 1)
        c1 = _project_stoch(q_mat @ np.linalg.pinv(t_mat))
        q_mat, t_mat = _stack_true_obs(cq, c1, c2, 2)
        c2 = _project_stoch(q_mat @ np.linalg.pinv(t_mat))
    return cq, c1, c2


def fit_scalar_cdr(
    e_ideal: np.ndarray,
    e_noisy: np.ndarray,
) -> tuple[float, float]:
    """E_ideal ≈ a1 E_noisy + a0."""
    y = np.asarray(e_ideal, dtype=float).reshape(-1)
    x = np.asarray(e_noisy, dtype=float).reshape(-1)
    a = np.column_stack([x, np.ones(x.size)])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_scalar_cdr(e_noisy: float, a1: float, a0: float) -> float:
    return float(a1) * float(e_noisy) + float(a0)


def richardson_extrapolate(values: dict[int, np.ndarray], degree: int) -> np.ndarray:
    """Lagrange interpolation to scale=0 using integer scales 1..degree+1."""
    deg = int(degree)
    scales = list(range(1, deg + 2))
    acc = None
    for s in scales:
        lag = 1.0
        for t in scales:
            if t == s:
                continue
            lag *= (0.0 - t) / (s - t)
        term = lag * np.asarray(values[s], dtype=float)
        acc = term if acc is None else acc + term
    return acc


def zne_histogram(
    hist_by_scale: dict[int, np.ndarray],
    *,
    degree: int = 1,
) -> np.ndarray:
    raw = richardson_extrapolate(hist_by_scale, degree)
    p = np.clip(raw, 0.0, None)
    s = float(p.sum())
    return p / s if s > 0.0 else np.full(p.shape, 1.0 / p.size)


def falling_factorial(n: np.ndarray, k: int) -> np.ndarray:
    out = np.ones_like(n, dtype=float)
    for i in range(int(k)):
        out *= n - i
    return out


def factorial_moments(probs: np.ndarray, k_max: int = 3) -> dict[str, list[float]]:
    """Per-mode factorial moments of a |q,n,m> histogram."""
    p = np.clip(np.asarray(probs, dtype=float), 0.0, None)
    p = p / max(float(p.sum()), EPS_PROB)
    pn = p.sum(axis=(0, 2))
    pm = p.sum(axis=(0, 1))
    n = np.arange(pn.size, dtype=float)
    m = np.arange(pm.size, dtype=float)
    g1 = [float(np.dot(pn, falling_factorial(n, k))) for k in range(1, k_max + 1)]
    g2 = [float(np.dot(pm, falling_factorial(m, k))) for k in range(1, k_max + 1)]
    return {"mode1": g1, "mode2": g2}


def moment_ratios(
    p_noisy: np.ndarray,
    p_ideal: np.ndarray,
    eta: float,
    k_max: int = 3,
) -> dict:
    """g_k^{noisy} / g_k^{ideal} vs η^k (exact for pure loss, not heating)."""
    gn = factorial_moments(p_noisy, k_max)
    gi = factorial_moments(p_ideal, k_max)
    eta = float(eta)
    out: dict = {"eta": eta, "eta_k": [eta**k for k in range(1, k_max + 1)]}
    for mode in ("mode1", "mode2"):
        ratios = []
        for k, (a, b) in enumerate(zip(gn[mode], gi[mode]), start=1):
            ratios.append(None if abs(b) < 1e-12 else float(a / b))
        out[mode] = {"noisy": gn[mode], "ideal": gi[mode], "ratio": ratios}
    return out


def sample_shots_from(probs: np.ndarray, n_shots: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    flat = np.clip(np.asarray(probs, dtype=float).reshape(-1), 0.0, None)
    total = float(flat.sum())
    if total <= 0.0:
        raise ValueError("cannot sample from a zero histogram")
    flat = flat / total
    counts = rng.multinomial(int(n_shots), flat)
    return (counts / float(n_shots)).reshape(probs.shape)


def observe_histogram(physical: np.ndarray, spec: ReadoutSpec, dims: tuple[int, int, int], seed: int) -> np.ndarray:
    """Apply readout confusion then multinomial shots."""
    from .noise_models import readout_config

    cfg = readout_config(spec, n_fock=int(dims[1]))
    cq, c1, c2 = confusion_from_measurement(cfg, dims)
    blurred = apply_transfer(physical, cq, c1, c2)
    if spec.n_shots is None or spec.n_shots <= 0:
        return blurred
    return sample_shots_from(blurred, spec.n_shots, seed)


@dataclass
class MethodResult:
    name: str
    histogram: np.ndarray | None
    energy: float | None
    extra: dict = field(default_factory=dict)


def run_readout_only(
    q_obs: np.ndarray,
    spec: ReadoutSpec,
    dims: tuple[int, int, int],
) -> MethodResult | None:
    if is_trivial_readout(spec):
        return None
    from .noise_models import readout_config

    meas = readout_config(spec, n_fock=int(dims[1]))
    cq, c1, c2 = confusion_from_measurement(meas, dims)
    hist = unfold(q_obs, cq, c1, c2, method="rl")
    return MethodResult("readout_only", hist, None, extra={"skipped": False})
