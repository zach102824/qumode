"""Local CPTP maps on the 128-dimensional hybrid space.

Channels are applied by contracting Kraus operators into the tensor-product
structure of ρ, never by building the 16,384 × 16,384 Liouvillian of the
full system. QuTiP ``mesolve`` is used only for independent validation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import qutip as qt


def destroy_matrix(n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=complex)
    for k in range(1, n):
        a[k - 1, k] = np.sqrt(k)
    return a


def create_matrix(n: int) -> np.ndarray:
    return destroy_matrix(n).conj().T


def num_matrix(n: int) -> np.ndarray:
    return np.diag(np.arange(n, dtype=float)).astype(complex)


def hermitian_sqrt(matrix: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Positive-semidefinite square root of a numerically Hermitian matrix."""
    h = 0.5 * (matrix + matrix.conj().T)
    evals, evecs = np.linalg.eigh(h)
    evals = np.clip(evals, 0.0, None)
    small = evals < tol
    evals = np.where(small, 0.0, evals)
    return (evecs * np.sqrt(evals)) @ evecs.conj().T


def paper_amplitude_damping_kraus(
    kappa_tau: float,
    n_fock: int,
    threshold: float = 1e-15,
) -> list[np.ndarray]:
    """Truncated photon-loss Kraus operators, Eqs. (37)–(39).

    K_j = sqrt((1-e^{-κτ})^j / j!) exp(-κτ n / 2) a^j  for j = 1, …, L-1,
    with K_0 replaced by the residual square root that restores Σ K†K = I.
    """
    n_fock = int(n_fock)
    ident = np.eye(n_fock, dtype=complex)
    if abs(kappa_tau) < threshold:
        return [ident]

    gamma = 1.0 - np.exp(-kappa_tau)
    damp_diag = np.exp(-0.5 * kappa_tau * np.arange(n_fock, dtype=float))
    damp = np.diag(damp_diag.astype(complex))
    a = destroy_matrix(n_fock)
    a_power = ident.copy()
    kraus: list[np.ndarray] = []
    completeness = np.zeros((n_fock, n_fock), dtype=complex)
    fact = 1.0
    for j in range(1, n_fock):
        a_power = a_power @ a
        fact *= j
        pref = np.sqrt((gamma**j) / fact) if gamma > 0.0 else 0.0
        kj = pref * (damp @ a_power)
        kraus.append(kj)
        completeness = completeness + kj.conj().T @ kj
    k0 = hermitian_sqrt(ident - completeness)
    return [k0] + kraus


def kraus_completeness_error(kraus: Sequence[np.ndarray]) -> float:
    dim = kraus[0].shape[0]
    acc = np.zeros((dim, dim), dtype=complex)
    for k in kraus:
        acc = acc + k.conj().T @ k
    return float(np.linalg.norm(acc - np.eye(dim)))


def lindblad_kraus(
    c_ops: Sequence[np.ndarray | qt.Qobj],
    tau: float,
    dim: int,
) -> list[np.ndarray]:
    """Kraus operators of exp(L τ) for H = 0 and the given collapse operators.

    The Liouvillian lives on the *local* space only (dim^2 × dim^2), e.g. 64×64
    for an 8-level cavity.
    """
    if abs(tau) == 0.0 or len(c_ops) == 0:
        return [np.eye(dim, dtype=complex)]
    q_ops = [c if isinstance(c, qt.Qobj) else qt.Qobj(c) for c in c_ops]
    liouv = qt.liouvillian(qt.qzero(dim), q_ops)
    superop = (liouv * float(tau)).expm()
    kraus_q = qt.to_kraus(superop)
    return [np.asarray(k.full(), dtype=complex) for k in kraus_q]


def _reshape_dm(rho: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    dims = tuple(int(d) for d in dims)
    return np.asarray(rho, dtype=complex).reshape(dims + dims)


def apply_kraus_local(
    rho: np.ndarray,
    kraus: Sequence[np.ndarray],
    subsystem: int,
    dims: Sequence[int] = (2, 8, 8),
) -> np.ndarray:
    """Apply a CPTP map on one tensor factor of a 3-partite density matrix."""
    dims = tuple(int(d) for d in dims)
    if subsystem not in (0, 1, 2):
        raise ValueError("subsystem must be 0 (qubit), 1 (cavity 1), or 2 (cavity 2).")
    rho_t = _reshape_dm(rho, dims)
    out = np.zeros_like(rho_t)
    if subsystem == 0:
        einsum = "aq,qnmQNM,AQ->anmANM"
    elif subsystem == 1:
        einsum = "an,qnmQNM,AN->qamQAM"
    else:
        einsum = "am,qnmQNM,AM->qnaQNA"
    for k in kraus:
        out += np.einsum(einsum, k, rho_t, k.conj(), optimize=True)
    return out.reshape(int(np.prod(dims)), int(np.prod(dims)))


def apply_local_unitary(
    rho: np.ndarray,
    unitary: np.ndarray,
    subsystem: int,
    dims: Sequence[int] = (2, 8, 8),
) -> np.ndarray:
    return apply_kraus_local(rho, [unitary], subsystem, dims)


def apply_full_unitary(rho: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    u = np.asarray(unitary, dtype=complex)
    r = np.asarray(rho, dtype=complex)
    return u @ r @ u.conj().T


def density_physicality(rho: np.ndarray, tol: float = 1e-8) -> dict[str, float]:
    r = np.asarray(rho, dtype=complex)
    herm = 0.5 * (r + r.conj().T)
    evals = np.linalg.eigvalsh(herm)
    trace = complex(np.trace(r))
    purity = complex(np.trace(r @ r))
    return {
        "trace_real": float(np.real(trace)),
        "trace_imag": float(np.imag(trace)),
        "hermiticity": float(np.linalg.norm(r - r.conj().T)),
        "min_eig": float(np.min(evals)),
        "purity": float(np.real(purity)),
        "trace_error": float(abs(trace - 1.0)),
        "is_physical": float(
            abs(trace - 1.0) < tol and np.linalg.norm(r - r.conj().T) < 1e-6 and np.min(evals) > -1e-8
        ),
    }


def choi_from_kraus(kraus: Sequence[np.ndarray]) -> np.ndarray:
    """Choi matrix (unnormalized convention: sum_i vec(K_i) vec(K_i)†)."""
    dim = kraus[0].shape[0]
    choi = np.zeros((dim * dim, dim * dim), dtype=complex)
    for k in kraus:
        vec = k.reshape(-1, order="F")  # column-stacking, matching QuTiP
        choi += np.outer(vec, vec.conj())
    return choi


def is_cptp_kraus(kraus: Sequence[np.ndarray], tol: float = 1e-7) -> bool:
    if kraus_completeness_error(kraus) > tol:
        return False
    choi = choi_from_kraus(kraus)
    evals = np.linalg.eigvalsh(0.5 * (choi + choi.conj().T))
    return bool(np.min(evals) > -tol)


def idle_kerr_unitary(kerr: float, tau: float, n_fock: int) -> np.ndarray:
    """U = exp(-i K τ n(n-1)/2)."""
    n = np.arange(n_fock, dtype=float)
    phase = np.exp(-1j * kerr * tau * n * (n - 1.0) / 2.0)
    return np.diag(phase.astype(complex))


def cross_kerr_phases(chi: float, tau: float, dims: Sequence[int]) -> np.ndarray:
    """Diagonal unitary exp(-i χ τ n1 n2) on the hybrid space."""
    l1, l2 = int(dims[1]), int(dims[2])
    phases = np.ones((2, l1, l2), dtype=complex)
    for n in range(l1):
        for m in range(l2):
            phases[:, n, m] = np.exp(-1j * chi * tau * n * m)
    return np.diag(phases.reshape(-1))


def dispersive_phases(chi: float, tau: float, dims: Sequence[int]) -> np.ndarray:
    """Residual dispersive coupling exp(-i χ τ/2 σ_z (n1 + n2))."""
    l1, l2 = int(dims[1]), int(dims[2])
    phases = np.empty((2, l1, l2), dtype=complex)
    for q, z in enumerate((1.0, -1.0)):
        for n in range(l1):
            for m in range(l2):
                phases[q, n, m] = np.exp(-1j * chi * tau * 0.5 * z * (n + m))
    return np.diag(phases.reshape(-1))


def vacuum_dm(dims: Sequence[int] = (2, 8, 8)) -> np.ndarray:
    dim = int(np.prod(dims))
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def ket_to_dm(ket: np.ndarray | qt.Qobj) -> np.ndarray:
    if isinstance(ket, qt.Qobj):
        vec = np.asarray(ket.full(), dtype=complex).reshape(-1)
    else:
        vec = np.asarray(ket, dtype=complex).reshape(-1)
    return np.outer(vec, vec.conj())
