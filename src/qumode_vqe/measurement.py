"""Readout, confusion matrices, and shot sampling.

Measurement errors act only on the final |q, n, m> histogram, never on the
physical density matrix. The diagonal QUBO energy reconstructed from a
histogram therefore differs from Tr(H ρ) whenever readout is imperfect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .hamiltonian import (
    TARGET_QNM,
    bitstring_from_bits,
    bits_from_qnm,
    hybrid_energy_tensor,
)


def joint_probabilities(rho: np.ndarray, dims: Sequence[int] = (2, 8, 8)) -> np.ndarray:
    """P(q, n, m) = <q,n,m| ρ |q,n,m> as an array of shape (2, L1, L2)."""
    dims = tuple(int(d) for d in dims)
    rho_t = np.asarray(rho, dtype=complex).reshape(dims + dims)
    probs = np.real(np.einsum("ijkijk->ijk", rho_t))
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total > 0:
        probs = probs / total
    return probs


def probabilities_from_ket(psi: np.ndarray, dims: Sequence[int] = (2, 8, 8)) -> np.ndarray:
    dims = tuple(int(d) for d in dims)
    vec = np.asarray(psi, dtype=complex).reshape(dims)
    return np.abs(vec) ** 2


def identity_confusion(dim: int) -> np.ndarray:
    return np.eye(dim, dtype=float)


def qubit_bitflip_confusion(p01: float, p10: float) -> np.ndarray:
    """Column-stochastic 2×2 matrix C[measured, true].

    p01 = P(measure 1 | true 0), p10 = P(measure 0 | true 1).
    """
    return np.array([[1.0 - p01, p10], [p01, 1.0 - p10]], dtype=float)


def nearest_neighbor_fock_confusion(n_fock: int, p_nn: float) -> np.ndarray:
    """QND photon-number misassignment to n±1, column-stochastic."""
    n_fock = int(n_fock)
    c = np.zeros((n_fock, n_fock), dtype=float)
    for true in range(n_fock):
        stay = 1.0 - p_nn
        left = p_nn / 2.0 if true > 0 else 0.0
        right = p_nn / 2.0 if true < n_fock - 1 else 0.0
        leftover = p_nn - left - right
        c[true, true] = stay + leftover
        if true > 0:
            c[true - 1, true] = left
        if true < n_fock - 1:
            c[true + 1, true] = right
    col = c.sum(axis=0, keepdims=True)
    col = np.where(col == 0.0, 1.0, col)
    return c / col


def is_column_stochastic(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    m = np.asarray(matrix, dtype=float)
    if np.any(m < -tol):
        return False
    return bool(np.allclose(m.sum(axis=0), 1.0, atol=tol))


def apply_confusion(
    probs: np.ndarray,
    qubit_c: np.ndarray | None = None,
    fock1_c: np.ndarray | None = None,
    fock2_c: np.ndarray | None = None,
) -> np.ndarray:
    """P_obs(q',n',m') = Σ Cq[q',q] Cn[n',n] Cm[m',m] P(q,n,m)."""
    p = np.asarray(probs, dtype=float)
    lq, l1, l2 = p.shape
    cq = np.eye(lq) if qubit_c is None else np.asarray(qubit_c, dtype=float)
    cn = np.eye(l1) if fock1_c is None else np.asarray(fock1_c, dtype=float)
    cm = np.eye(l2) if fock2_c is None else np.asarray(fock2_c, dtype=float)
    return np.einsum("aq,bn,cm,qnm->abc", cq, cn, cm, p, optimize=True)


def sample_shots(probs: np.ndarray, n_shots: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    flat = np.asarray(probs, dtype=float).reshape(-1)
    flat = np.clip(flat, 0.0, None)
    flat = flat / flat.sum()
    counts = rng.multinomial(int(n_shots), flat)
    return (counts / float(n_shots)).reshape(probs.shape)


def energy_from_histogram(
    probs: np.ndarray,
    energy_tensor: np.ndarray | None = None,
) -> float:
    coeffs = hybrid_energy_tensor(probs.shape[1:]) if energy_tensor is None else energy_tensor
    return float(np.sum(np.asarray(probs, dtype=float) * coeffs))


def most_likely_qnm(probs: np.ndarray) -> tuple[int, int, int]:
    p = np.asarray(probs)
    idx = int(np.argmax(p))
    q, n, m = np.unravel_index(idx, p.shape)
    return int(q), int(n), int(m)


def decode_bitstring(
    q: int,
    n: int,
    m: int,
    partition: tuple[int, int, int] = (1, 3, 3),
) -> str:
    return bitstring_from_bits(bits_from_qnm(q, n, m, partition))


@dataclass
class MeasurementConfig:
    qubit_c: np.ndarray | None = None
    fock1_c: np.ndarray | None = None
    fock2_c: np.ndarray | None = None
    n_shots: int | None = None
    seed: int | None = None

    def is_identity(self) -> bool:
        if self.n_shots is not None:
            return False
        return self.qubit_c is None and self.fock1_c is None and self.fock2_c is None


@dataclass
class MeasurementResult:
    physical_probs: np.ndarray
    observed_probs: np.ndarray
    energy_physical: float
    energy_observed: float
    most_likely: tuple[int, int, int]
    most_likely_bitstring: str
    target_prob_physical: float
    target_prob_observed: float
    n_shots: int | None

    def as_dict(self) -> dict:
        q, n, m = self.most_likely
        return {
            "energy_physical": self.energy_physical,
            "energy_observed": self.energy_observed,
            "most_likely": [q, n, m],
            "most_likely_bitstring": self.most_likely_bitstring,
            "target_prob_physical": self.target_prob_physical,
            "target_prob_observed": self.target_prob_observed,
            "n_shots": self.n_shots,
        }


def measure(
    physical_probs: np.ndarray,
    measurement: MeasurementConfig | None = None,
    energy_tensor: np.ndarray | None = None,
    target: tuple[int, int, int] | None = TARGET_QNM,
    partition: tuple[int, int, int] = (1, 3, 3),
) -> MeasurementResult:
    cfg = measurement or MeasurementConfig()
    coeffs = energy_tensor if energy_tensor is not None else hybrid_energy_tensor(
        (physical_probs.shape[1], physical_probs.shape[2])
    )
    observed = apply_confusion(physical_probs, cfg.qubit_c, cfg.fock1_c, cfg.fock2_c)
    if cfg.n_shots is not None:
        rng = np.random.default_rng(cfg.seed)
        observed = sample_shots(observed, cfg.n_shots, rng)
    ml = most_likely_qnm(observed)
    if target is None:
        tphys = float("nan")
        tobs = float("nan")
    else:
        tq, tn, tm = target
        tphys = float(physical_probs[tq, tn, tm])
        tobs = float(observed[tq, tn, tm])
    return MeasurementResult(
        physical_probs=physical_probs,
        observed_probs=observed,
        energy_physical=energy_from_histogram(physical_probs, coeffs),
        energy_observed=energy_from_histogram(observed, coeffs),
        most_likely=ml,
        most_likely_bitstring=decode_bitstring(*ml, partition=partition),
        target_prob_physical=tphys,
        target_prob_observed=tobs,
        n_shots=cfg.n_shots,
    )
