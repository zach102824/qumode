"""Gaussian and near-Gaussian training twins of ECD / SNAP circuits.

Clifford Data Regression uses classically simulable (Clifford) siblings of
the target circuit to learn a noise map. Here the easy set is Gaussian:
displacements and rotations, whose photon histogram is a (truncated) Poisson.

ECD twins
    Snap each qubit angle θ onto {0, π}. Then every R maps computational
    basis to computational basis, every ECD is an unconditional displacement
    plus a bit flip, and the ideal state is |q⟩|α₁⟩|α₂⟩.

SNAP twins
    Replace each SNAP phase vector θ_n by its least-squares affine fit
    b·n (θ_0 is already gauge-fixed to 0). Affine SNAP is a rotation, so
    the circuit is displacements-and-rotations only.

``t_free`` gates are left at their true (non-Gaussian) values so the
training set interpolates from rank-1 Gaussian to near-target circuits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import qutip as qt
from scipy.special import gammaln

from qumode_vqe.circuit import snap_operator
from qumode_vqe.measurement import probabilities_from_ket
from qumode_vqe.params import (
    pack,
    pack_snap,
    unpack,
    unpack_snap,
)

from .metrics import total_variation

# Product-state simulation (truncated 1-mode D / SNAP / ECD-as-displace)
# must match the full statevector at this TVD for every t_free=0 twin.
PRODUCT_TVD_TOL = 1e-6
# Keep tracked |α| inside the grid so twins stay identifiable but not
# saturating the cutoff. Poisson vs truncated D is a diagnostic, not the
# assertion: truncated D(α)|0⟩ is not an ideal coherent state at large |α|.
ALPHA_MARGIN = 2
POISSON_TVD_TOL = 1e-6  # kept as an alias for tests that import it



@dataclass
class Twin:
    x: np.ndarray
    t_free: int
    p_ideal: np.ndarray
    p_analytic: np.ndarray | None
    poisson_tvd: float | None
    product_tvd: float | None
    qubit: int | None
    alpha: tuple[complex, complex] | None
    free_gates: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "t_free": int(self.t_free),
            "poisson_tvd": None if self.poisson_tvd is None else float(self.poisson_tvd),
            "product_tvd": None if self.product_tvd is None else float(self.product_tvd),
            "qubit": self.qubit,
            "alpha": None
            if self.alpha is None
            else [complex(self.alpha[0]).real, complex(self.alpha[0]).imag,
                  complex(self.alpha[1]).real, complex(self.alpha[1]).imag],
            "free_gates": [int(i) for i in self.free_gates],
        }


def truncated_poisson(mean: float, n_fock: int) -> np.ndarray:
    """Renormalized Poisson on {0, …, n_fock−1}."""
    mu = max(float(mean), 0.0)
    n = np.arange(int(n_fock), dtype=float)
    if mu == 0.0:
        p = np.zeros(int(n_fock), dtype=float)
        p[0] = 1.0
        return p
    logp = -mu + n * np.log(mu) - gammaln(n + 1.0)
    logp -= np.max(logp)
    p = np.exp(logp)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    return p / total if total > 0.0 else p


def product_coherent_histogram(
    qubit: int,
    alpha1: complex,
    alpha2: complex,
    dims: tuple[int, int, int] = (2, 8, 8),
) -> np.ndarray:
    """δ_q ⊗ truncated-Poisson(|α₁|²) ⊗ truncated-Poisson(|α₂|²)."""
    lq, l1, l2 = dims
    p = np.zeros((lq, l1, l2), dtype=float)
    q = int(qubit) % lq
    p[q] = np.outer(truncated_poisson(abs(alpha1) ** 2, l1), truncated_poisson(abs(alpha2) ** 2, l2))
    total = float(p.sum())
    if total > 0.0:
        p /= total
    return p


def _snap_theta(theta: float, rng: np.random.Generator) -> float:
    t = float(theta) % (2.0 * np.pi)
    d0 = min(abs(t), abs(t - 2.0 * np.pi))
    d_pi = abs(t - np.pi)
    # Prefer the nearer computational-basis rotation; break ties at random.
    if abs(d0 - d_pi) < 1e-12:
        return 0.0 if rng.random() < 0.5 else np.pi
    return 0.0 if d0 < d_pi else np.pi


def _affine_slope(phases: np.ndarray) -> float:
    """Least-squares b in θ_n ≈ b n, n = 0 … L−1 (θ_0 = 0)."""
    th = np.asarray(phases, dtype=float).reshape(-1)
    n = np.arange(th.size, dtype=float)
    denom = float(np.dot(n, n))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(n, th) / denom)


def _affine_phases(phases: np.ndarray) -> np.ndarray:
    th = np.asarray(phases, dtype=float).reshape(-1)
    b = _affine_slope(th)
    return b * np.arange(th.size, dtype=float)


def track_ecd_gaussian(xvec: np.ndarray, ndepth: int) -> tuple[int, complex, complex]:
    """Follow the computational-basis ECD trajectory (θ ∈ {0, π} only)."""
    p = unpack(xvec, ndepth)
    q = 0
    alpha = [0.0 + 0.0j, 0.0 + 0.0j]
    for i in range(int(ndepth)):
        for cind in (0, 1):
            th = float(p.theta[i, cind])
            # θ=0: identity. θ=π: bit flip. Anything else is not Gaussian.
            if abs((th % (2.0 * np.pi)) - np.pi) < 1e-8 or abs(th - np.pi) < 1e-8:
                q = 1 - q
            elif abs(th % (2.0 * np.pi)) > 1e-8 and abs(th) > 1e-8:
                raise ValueError(f"ECD twin has a non-basis rotation θ={th} at layer {i} mode {cind}")
            beta = complex(p.beta[i, cind])
            if q == 0:
                alpha[cind] += 0.5 * beta
                q = 1
            else:
                alpha[cind] -= 0.5 * beta
                q = 0
    return q, alpha[0], alpha[1]


def track_snap_gaussian(
    xvec: np.ndarray,
    ndepth: int,
    nfocks: tuple[int, int] = (8, 8),
) -> tuple[int, complex, complex]:
    """Follow the coherent amplitude through affine-SNAP + displacement."""
    p = unpack_snap(xvec, ndepth, nfocks)
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    alpha = [0.0 + 0.0j, 0.0 + 0.0j]
    for i in range(int(ndepth)):
        for cind, n_fock in ((0, l1), (1, l2)):
            b = _affine_slope(p.phases[i, cind, :n_fock])
            # SNAP(θ) D(α): displace first, then rotate.
            alpha[cind] = np.exp(1j * b) * (alpha[cind] + complex(p.alpha[i, cind]))
    return 0, alpha[0], alpha[1]


def _cap_scale(alphas: list[complex], nfocks: tuple[int, int]) -> float:
    scale = 1.0
    for a, n_fock in zip(alphas, nfocks):
        rmax = float(np.sqrt(max(int(n_fock) - ALPHA_MARGIN, 1)))
        mag = abs(a)
        if mag > rmax:
            scale = min(scale, rmax / mag)
    return float(scale)


def make_ecd_twin(
    x_target: np.ndarray,
    ndepth: int,
    rng: np.random.Generator,
    *,
    t_free: int = 0,
    mag_scale_range: tuple[float, float] = (0.5, 1.0),
) -> np.ndarray:
    p = unpack(x_target, ndepth)
    n_gates = int(ndepth) * 2
    t_free = int(max(0, min(int(t_free), n_gates)))
    free = set()
    if t_free:
        free = set(int(i) for i in rng.choice(n_gates, size=t_free, replace=False))
    theta = np.array(p.theta, dtype=float, copy=True)
    mag_scale = rng.uniform(mag_scale_range[0], mag_scale_range[1], size=p.beta.shape)
    beta = np.asarray(p.beta) * mag_scale
    idx = 0
    for i in range(int(ndepth)):
        for cind in (0, 1):
            if idx not in free:
                theta[i, cind] = _snap_theta(theta[i, cind], rng)
            idx += 1
    x = pack(beta=beta, theta=theta, phi=p.phi)
    if t_free == 0:
        _, a1, a2 = track_ecd_gaussian(x, ndepth)
        cap = _cap_scale([a1, a2], (8, 8))
        if cap < 1.0 - 1e-12:
            x = pack(beta=beta * cap, theta=theta, phi=p.phi)
    return x


def make_snap_twin(
    x_target: np.ndarray,
    ndepth: int,
    rng: np.random.Generator,
    *,
    t_free: int = 0,
    nfocks: tuple[int, int] = (8, 8),
    mag_scale_range: tuple[float, float] = (0.5, 1.0),
) -> np.ndarray:
    p = unpack_snap(x_target, ndepth, nfocks)
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    n_gates = int(ndepth) * 2
    t_free = int(max(0, min(int(t_free), n_gates)))
    free = set()
    if t_free:
        free = set(int(i) for i in rng.choice(n_gates, size=t_free, replace=False))
    phases = np.array(p.phases, dtype=float, copy=True)
    mag_scale = rng.uniform(mag_scale_range[0], mag_scale_range[1], size=p.alpha.shape)
    alpha = np.asarray(p.alpha) * mag_scale
    idx = 0
    for i in range(int(ndepth)):
        for cind, n_fock in ((0, l1), (1, l2)):
            if idx not in free:
                phases[i, cind, :n_fock] = _affine_phases(phases[i, cind, :n_fock])
            idx += 1
    x = pack_snap(alpha, phases, nfocks)
    if t_free == 0:
        _, a1, a2 = track_snap_gaussian(x, ndepth, nfocks)
        cap = _cap_scale([a1, a2], nfocks)
        if cap < 1.0 - 1e-12:
            x = pack_snap(alpha * cap, phases, nfocks)
    return x


def analytic_histogram(ansatz: str, xvec: np.ndarray, ndepth: int, dims: tuple[int, int, int]) -> tuple[np.ndarray, int, tuple[complex, complex]]:
    if ansatz == "ecd":
        q, a1, a2 = track_ecd_gaussian(xvec, ndepth)
    elif ansatz == "snap":
        q, a1, a2 = track_snap_gaussian(xvec, ndepth, (dims[1], dims[2]))
    else:
        raise ValueError(f"unknown ansatz {ansatz!r}")
    hist = product_coherent_histogram(q, a1, a2, dims)
    return hist, q, (a1, a2)


def _ket_to_p(ket: qt.Qobj, n_fock: int) -> np.ndarray:
    vec = np.asarray(ket.full(), dtype=complex).reshape(-1)
    p = np.abs(vec) ** 2
    p = p[: int(n_fock)]
    total = float(p.sum())
    return p / total if total > 0.0 else p


def product_ecd_histogram(xvec: np.ndarray, ndepth: int, dims: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    """Exact truncated product evolution for a computational-basis ECD twin."""
    p = unpack(xvec, ndepth)
    lq, l1, l2 = (int(d) for d in dims)
    q = 0
    psi = [qt.basis(l1, 0), qt.basis(l2, 0)]
    nf = (l1, l2)
    for i in range(int(ndepth)):
        for cind in (0, 1):
            th = float(p.theta[i, cind])
            if abs((th % (2.0 * np.pi)) - np.pi) < 1e-8 or abs(th - np.pi) < 1e-8:
                q = 1 - q
            elif abs(th % (2.0 * np.pi)) > 1e-8 and abs(th) > 1e-8:
                raise ValueError("product ECD histogram requires θ ∈ {0, π}")
            beta = complex(p.beta[i, cind])
            disp = 0.5 * beta if q == 0 else -0.5 * beta
            psi[cind] = qt.displace(nf[cind], disp) * psi[cind]
            q = 1 - q
    hist = np.zeros((lq, l1, l2), dtype=float)
    hist[int(q) % lq] = np.outer(_ket_to_p(psi[0], l1), _ket_to_p(psi[1], l2))
    return hist, q


def product_snap_histogram(xvec: np.ndarray, ndepth: int, dims: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    """Exact truncated 1-mode SNAP+D evolution (always a product state)."""
    lq, l1, l2 = (int(d) for d in dims)
    p = unpack_snap(xvec, ndepth, (l1, l2))
    psi = [qt.basis(l1, 0), qt.basis(l2, 0)]
    nf = (l1, l2)
    for i in range(int(ndepth)):
        for cind, n_fock in ((0, l1), (1, l2)):
            psi[cind] = qt.displace(n_fock, complex(p.alpha[i, cind])) * psi[cind]
            psi[cind] = snap_operator(p.phases[i, cind, :n_fock], n_fock) * psi[cind]
    hist = np.zeros((lq, l1, l2), dtype=float)
    hist[0] = np.outer(_ket_to_p(psi[0], l1), _ket_to_p(psi[1], l2))
    return hist, 0


def product_histogram(ansatz: str, xvec: np.ndarray, ndepth: int, dims: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    if ansatz == "ecd":
        return product_ecd_histogram(xvec, ndepth, dims)
    if ansatz == "snap":
        return product_snap_histogram(xvec, ndepth, dims)
    raise ValueError(f"unknown ansatz {ansatz!r}")


def statevector_histogram(sim, xvec: np.ndarray) -> np.ndarray:
    psi = sim.statevector(np.asarray(xvec, dtype=float))
    return probabilities_from_ket(np.asarray(psi.full()).reshape(-1), sim.dims)


def build_twins(
    sim,
    x_target: np.ndarray,
    rng: np.random.Generator,
    *,
    n_train: int = 40,
    n_rank2: int | None = None,
    mag_scale_range: tuple[float, float] = (0.5, 1.0),
    product_tol: float = PRODUCT_TVD_TOL,
) -> list[Twin]:
    """Build ``n_train`` twins of ``x_target`` on a noiseless ``sim``.

    For ``t_free=0``, the classically easy product-state histogram (truncated
    1-mode displacements / SNAPs, ECD reduced to unconditional D(±β/2)) is
    asserted against ``sim.statevector`` at TVD ``product_tol``. A Poisson
    formula is stored as a diagnostic; it need not match at 1e-6 once
    truncated D(α) departs from an ideal coherent state.
    """
    ansatz = str(sim.ansatz).lower()
    ndepth = int(sim.ndepth)
    dims = tuple(int(d) for d in sim.dims)
    n_train = int(n_train)
    if n_rank2 is None:
        n_rank2 = max(0, n_train // 4)
    n_gauss = n_train - int(n_rank2)
    t_list = [0] * n_gauss + [min(2, ndepth * 2)] * int(n_rank2)
    twins: list[Twin] = []
    for t_free in t_list:
        if ansatz == "ecd":
            x = make_ecd_twin(x_target, ndepth, rng, t_free=t_free, mag_scale_range=mag_scale_range)
        else:
            x = make_snap_twin(
                x_target, ndepth, rng, t_free=t_free, nfocks=sim.nfocks, mag_scale_range=mag_scale_range
            )
        p_sv = statevector_histogram(sim, x)
        p_analytic = None
        poisson_tvd = None
        product_tvd = None
        qubit = None
        alpha = None
        if t_free == 0:
            p_poisson, qubit, alpha = analytic_histogram(ansatz, x, ndepth, dims)
            p_analytic, qubit_p = product_histogram(ansatz, x, ndepth, dims)
            if qubit is None:
                qubit = qubit_p
            poisson_tvd = total_variation(p_poisson, p_sv)
            product_tvd = total_variation(p_analytic, p_sv)
            if product_tvd > product_tol:
                raise AssertionError(
                    f"{ansatz} Gaussian twin product-state TVD={product_tvd:.3e} exceeds {product_tol:.1e}. "
                    "The twin is not a classically easy product Gaussian state."
                )
        twins.append(
            Twin(
                x=np.asarray(x, dtype=float),
                t_free=int(t_free),
                p_ideal=p_sv,
                p_analytic=p_analytic,
                poisson_tvd=poisson_tvd,
                product_tvd=product_tvd,
                qubit=qubit,
                alpha=alpha,
            )
        )
    return twins


