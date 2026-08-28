"""Echoed conditional displacement (ECD) rotation ansatz, Eqs. (20)–(22).

Gate order in one UER layer (paper Eq. 22a and the published notebooks)::

    U_ER = ECD_2(β2) R(θ2, φ2) ECD_1(β1) R(θ1, φ1)

so each ECD–rotation pair is ``ECD_k(βk) R(θk, φk)``. Depth ``N_d`` applies
``N_d`` such layers (2 N_d ECD gates).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import qutip as qt

from .params import ParamLayout, UnpackedParams, unpack

# Real parameters of Ry(θ)|0⟩ ⊗ |α₁⟩ ⊗ |α₂⟩: (θ, Re α₁, Im α₁, Re α₂, Im α₂).
N_PREP_PARAMS = 5


def qubit_rotation(theta: float, phi: float) -> qt.Qobj:
    """R(θ, φ) = exp[-i (θ/2) (X cos φ + Y sin φ)]."""
    gen = np.cos(phi) * qt.sigmax() + np.sin(phi) * qt.sigmay()
    return (-1j * (theta / 2.0) * gen).expm()


def _sigma_minus() -> qt.Qobj:
    """σ- = |1><0|."""
    return qt.Qobj(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex))


def _sigma_plus() -> qt.Qobj:
    """σ+ = |0><1|."""
    return qt.Qobj(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex))


def ecd_gate(beta: complex, cind: int, nfocks: Sequence[int]) -> qt.Qobj:
    """One-qubit one-qumode ECD, Eq. (22b–c).

    ECD(β) = σ- ⊗ D(β/2) + σ+ ⊗ D(-β/2) on the selected qumode.
    """
    if cind not in (0, 1):
        raise ValueError("cind must be 0 (first qumode) or 1 (second qumode).")
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    sm, sp = _sigma_minus(), _sigma_plus()
    if cind == 0:
        ecd = qt.tensor(sm, qt.displace(l1, beta / 2.0)) + qt.tensor(
            sp, qt.displace(l1, -beta / 2.0)
        )
        return qt.tensor(ecd, qt.qeye(l2))
    return qt.tensor(sm, qt.qeye(l1), qt.displace(l2, beta / 2.0)) + qt.tensor(
        sp, qt.qeye(l1), qt.displace(l2, -beta / 2.0)
    )


def rotation_on_hybrid(theta: float, phi: float, nfocks: Sequence[int]) -> qt.Qobj:
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    return qt.tensor(qubit_rotation(theta, phi), qt.qeye(l1), qt.qeye(l2))


def ecd_rotation_pair(
    beta: complex,
    theta: float,
    phi: float,
    cind: int,
    nfocks: Sequence[int],
) -> qt.Qobj:
    """Single ECD–rotation pair: ECD(β) R(θ, φ) on qumode ``cind``."""
    return ecd_gate(beta, cind, nfocks) * rotation_on_hybrid(theta, phi, nfocks)


def uer_layer(
    beta: Sequence[complex],
    theta: Sequence[float],
    phi: Sequence[float],
    nfocks: Sequence[int],
) -> qt.Qobj:
    """One UER block: pair on qumode 1 followed by pair on qumode 2."""
    pair0 = ecd_rotation_pair(beta[0], theta[0], phi[0], 0, nfocks)
    pair1 = ecd_rotation_pair(beta[1], theta[1], phi[1], 1, nfocks)
    return pair1 * pair0


def hybrid_identity(nfocks: Sequence[int]) -> qt.Qobj:
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    return qt.tensor(qt.qeye(2), qt.qeye(l1), qt.qeye(l2))


def ecd_ansatz_unitary(
    params: UnpackedParams,
    nfocks: Sequence[int],
    pair_mask: np.ndarray | None = None,
) -> qt.Qobj:
    """Full depth-N_d unitary U = U_ER(N_d) … U_ER(1).

    ``pair_mask`` is an optional (N_d, 2) boolean array. A False entry skips
    that ECD–rotation pair entirely (the AAS analog of dropping a QAOA edge).
    Zeroing the pair's parameters is *not* the same: ECD(0) is σx, not I.
    """
    ndepth = params.beta.shape[0]
    if pair_mask is None:
        mask = np.ones((ndepth, 2), dtype=bool)
    else:
        mask = np.asarray(pair_mask, dtype=bool).reshape(ndepth, 2)
    uni = None
    for i in range(ndepth):
        for cind in (0, 1):
            if not mask[i, cind]:
                continue
            gate = ecd_rotation_pair(
                params.beta[i, cind], params.theta[i, cind], params.phi[i, cind], cind, nfocks
            )
            uni = gate if uni is None else gate * uni
    return hybrid_identity(nfocks) if uni is None else uni


def vacuum(nfocks: Sequence[int]) -> qt.Qobj:
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    return qt.tensor(qt.basis(2, 0), qt.basis(l1, 0), qt.basis(l2, 0))


def coherent_radius_max(n_fock: int) -> float:
    """Largest |α| whose mean occupation |α|² still fits in a length-``n_fock`` grid."""
    return float(np.sqrt(max(int(n_fock) - 1, 0)))


def ry_qubit_ket(theta: float) -> qt.Qobj:
    """Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩.  θ = π/2 is Hadamard on |0⟩."""
    half = 0.5 * float(theta)
    vec = np.array([[np.cos(half)], [np.sin(half)]], dtype=complex)
    return qt.Qobj(vec, dims=[[2], [1]])


def truncated_coherent(n_fock: int, alpha: complex) -> qt.Qobj:
    ket = qt.coherent(int(n_fock), complex(alpha))
    nrm = float(ket.norm())
    if nrm == 0.0:
        raise ValueError("truncated coherent state has zero norm")
    return ket if abs(nrm - 1.0) <= 1e-12 else ket / nrm


def as_hybrid_ket(ket: qt.Qobj | np.ndarray, nfocks: Sequence[int]) -> qt.Qobj:
    """Validate, set hybrid dims, and normalize a qubit–qumode ket."""
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    expected = 2 * l1 * l2
    if isinstance(ket, qt.Qobj):
        psi = ket.copy()
    else:
        psi = qt.Qobj(np.asarray(ket, dtype=complex).reshape(-1, 1))
    if psi.shape[0] != expected:
        raise ValueError(f"Expected ket length {expected} for nfocks={nfocks}, got {psi.shape[0]}.")
    psi.dims = [[2, l1, l2], [1, 1, 1]]
    nrm = float(psi.norm())
    if nrm == 0.0:
        raise ValueError("initial state has zero norm")
    return psi if abs(nrm - 1.0) <= 1e-12 else psi / nrm


def ry_coherent_product(
    theta: float,
    alpha1: complex,
    alpha2: complex,
    nfocks: Sequence[int],
) -> qt.Qobj:
    """Normalized product state Ry(θ)|0⟩ ⊗ |α₁⟩ ⊗ |α₂⟩ with truncated coherents."""
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    psi = qt.tensor(
        ry_qubit_ket(theta),
        truncated_coherent(l1, alpha1),
        truncated_coherent(l2, alpha2),
    )
    return as_hybrid_ket(psi, nfocks)


def prep_params_to_ket(prep: np.ndarray | Sequence[float], nfocks: Sequence[int]) -> qt.Qobj:
    p = np.asarray(prep, dtype=float).reshape(-1)
    if p.size != N_PREP_PARAMS:
        raise ValueError(f"Expected {N_PREP_PARAMS} preparation parameters, got {p.size}.")
    return ry_coherent_product(p[0], p[1] + 1j * p[2], p[3] + 1j * p[4], nfocks)


def prep_bounds(nfocks: Sequence[int]) -> list[tuple[float, float]]:
    r1 = coherent_radius_max(nfocks[0])
    r2 = coherent_radius_max(nfocks[1])
    return [
        (0.0, np.pi),
        (-r1, r1),
        (-r1, r1),
        (-r2, r2),
        (-r2, r2),
    ]


def _project_disk(re: float, im: float, radius: float) -> tuple[float, float]:
    mag = float(np.hypot(re, im))
    if mag <= radius or mag == 0.0:
        return float(re), float(im)
    scale = radius / mag
    return float(re) * scale, float(im) * scale


def project_prep_params(prep: np.ndarray | Sequence[float], nfocks: Sequence[int]) -> np.ndarray:
    """Clip θ to [0, π] and each α onto the disk |α| ≤ √(N−1)."""
    p = np.asarray(prep, dtype=float).reshape(-1).copy()
    if p.size != N_PREP_PARAMS:
        raise ValueError(f"Expected {N_PREP_PARAMS} preparation parameters, got {p.size}.")
    p[0] = float(np.clip(p[0], 0.0, np.pi))
    p[1], p[2] = _project_disk(p[1], p[2], coherent_radius_max(nfocks[0]))
    p[3], p[4] = _project_disk(p[3], p[4], coherent_radius_max(nfocks[1]))
    return p


def prepare_state(
    xvec: np.ndarray,
    ndepth: int,
    nfocks: Sequence[int],
    layout: ParamLayout = ParamLayout.PAPER,
    initial_state: qt.Qobj | np.ndarray | None = None,
) -> qt.Qobj:
    params = unpack(xvec, ndepth, layout)
    ket0 = vacuum(nfocks) if initial_state is None else as_hybrid_ket(initial_state, nfocks)
    return ecd_ansatz_unitary(params, nfocks) * ket0


def apply_coherent_errors(
    beta: complex,
    theta: float,
    phi: float,
    amp_rel: float = 0.0,
    phase: float = 0.0,
    rot_rel: float = 0.0,
) -> tuple[complex, float, float]:
    """Static coherent control errors: β → |β|(1+ε_a) e^{i(arg β + δε)}, θ → θ(1+ε_r)."""
    mag = np.abs(beta) * (1.0 + amp_rel)
    arg = np.angle(beta) + phase
    return mag * np.exp(1j * arg), theta * (1.0 + rot_rel), phi
