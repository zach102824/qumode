"""Parameter packing for the ECD-rotation ansatz.

Paper parameterization (used by the published notebooks):
    X = [β_mag, β_arg, θ, φ]
each of shape (N_d, 2), so ``len(X) = 8 N_d`` (40 parameters for N_d = 5).

A Cartesian alternative stores Re(β), Im(β) instead of polar coordinates,
which avoids the |β| = 0 singularity and is used with L-BFGS-B.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from enum import Enum

import numpy as np


class ParamLayout(str, Enum):
    PAPER = "paper"
    CARTESIAN = "cartesian"


@dataclass(frozen=True)
class UnpackedParams:
    beta: np.ndarray  # complex, shape (ndepth, 2)
    theta: np.ndarray
    phi: np.ndarray


def n_parameters(ndepth: int) -> int:
    return 8 * int(ndepth)


def unpack(xvec: np.ndarray, ndepth: int, layout: ParamLayout = ParamLayout.PAPER) -> UnpackedParams:
    x = np.asarray(xvec, dtype=float).reshape(-1)
    size = int(ndepth) * 2
    expected = 4 * size
    if x.size != expected:
        raise ValueError(f"Expected {expected} parameters for ndepth={ndepth}, got {x.size}.")
    if layout is ParamLayout.PAPER:
        beta_mag = x[:size].reshape(ndepth, 2)
        beta_arg = x[size : 2 * size].reshape(ndepth, 2)
        beta = beta_mag * np.exp(1j * beta_arg)
    else:
        re_beta = x[:size].reshape(ndepth, 2)
        im_beta = x[size : 2 * size].reshape(ndepth, 2)
        beta = re_beta + 1j * im_beta
    theta = x[2 * size : 3 * size].reshape(ndepth, 2)
    phi = x[3 * size :].reshape(ndepth, 2)
    return UnpackedParams(beta=beta, theta=theta, phi=phi)


def pack(
    beta: np.ndarray | None = None,
    theta: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    *,
    beta_mag: np.ndarray | None = None,
    beta_arg: np.ndarray | None = None,
    layout: ParamLayout = ParamLayout.PAPER,
) -> np.ndarray:
    if beta is None:
        if beta_mag is None or beta_arg is None:
            raise ValueError("Provide either complex beta or polar (beta_mag, beta_arg).")
        beta = np.asarray(beta_mag) * np.exp(1j * np.asarray(beta_arg))
    beta = np.asarray(beta)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if layout is ParamLayout.PAPER:
        parts = [np.abs(beta).ravel(), np.angle(beta).ravel(), theta.ravel(), phi.ravel()]
    else:
        parts = [np.real(beta).ravel(), np.imag(beta).ravel(), theta.ravel(), phi.ravel()]
    return np.concatenate(parts)


def random_parameters(
    ndepth: int,
    rng: np.random.Generator | None = None,
    layout: ParamLayout = ParamLayout.PAPER,
) -> np.ndarray:
    """Match the published notebook initialization ranges."""
    rng = rng or np.random.default_rng()
    shape = (int(ndepth), 2)
    beta_mag = rng.uniform(0.0, 3.0, size=shape)
    beta_arg = rng.uniform(0.0, np.pi, size=shape)
    theta = rng.uniform(0.0, np.pi, size=shape)
    phi = rng.uniform(0.0, np.pi, size=shape)
    beta = beta_mag * np.exp(1j * beta_arg)
    return pack(beta=beta, theta=theta, phi=phi, layout=layout)


def slice_parameters(
    xvec: np.ndarray,
    ndepth_from: int,
    ndepth_to: int,
    layout: ParamLayout = ParamLayout.PAPER,
) -> np.ndarray:
    """Keep the first ``ndepth_to`` UER layers of a packed parameter vector."""
    d_from = int(ndepth_from)
    d_to = int(ndepth_to)
    if d_to < 1 or d_to > d_from:
        raise ValueError(f"ndepth_to must be in 1..{d_from}, got {d_to}.")
    p = unpack(xvec, d_from, layout)
    return pack(
        beta=p.beta[:d_to],
        theta=p.theta[:d_to],
        phi=p.phi[:d_to],
        layout=layout,
    )


def paper_bounds(ndepth: int) -> list[tuple[float, float]]:
    """Bounds supplied in the notebooks (ignored by SciPy BFGS, used by L-BFGS-B)."""
    size = int(ndepth) * 2
    return (
        [(0.0, 10.0)] * size
        + [(0.0, 2 * np.pi)] * size
        + [(0.0, np.pi)] * size
        + [(0.0, 2 * np.pi)] * size
    )


def cartesian_bounds(ndepth: int) -> list[tuple[float, float]]:
    size = int(ndepth) * 2
    return (
        [(-10.0, 10.0)] * size
        + [(-10.0, 10.0)] * size
        + [(0.0, np.pi)] * size
        + [(0.0, 2 * np.pi)] * size
    )


@dataclass(frozen=True)
class UnpackedSnapParams:
    """SNAP+displacement parameters with θ_0 fixed at 0 on each oscillator.

    ``alpha`` is complex with shape (ndepth, 2). ``phases`` has shape
    (ndepth, 2, max(L1, L2)); unused trailing slots for the smaller cutoff
    are zero, and ``phases[..., 0]`` is always the gauge-fixed zero.
    """

    alpha: np.ndarray
    phases: np.ndarray


def snap_phase_counts(nfocks: Sequence[int] | tuple[int, int] = (8, 8), *, gauge_fix: bool = True) -> tuple[int, int]:
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    shift = 1 if gauge_fix else 0
    return l1 - shift, l2 - shift


def n_snap_parameters(
    ndepth: int,
    nfocks: Sequence[int] | tuple[int, int] = (8, 8),
    *,
    gauge_fix: bool = True,
) -> int:
    """Real parameters of the two-mode displacement–SNAP ansatz.

    Each layer is ``SNAP D`` on qumode 1 then ``SNAP D`` on qumode 2. Each
    displacement contributes ``(|α|, arg α)`` and each SNAP contributes
    ``L−1`` phases when the Fock-0 phase is gauge-fixed to zero.
    """
    n0, n1 = snap_phase_counts(nfocks, gauge_fix=gauge_fix)
    return int(ndepth) * (4 + n0 + n1)


def unpack_snap(
    xvec: np.ndarray,
    ndepth: int,
    nfocks: Sequence[int] | tuple[int, int] = (8, 8),
    *,
    gauge_fix: bool = True,
) -> UnpackedSnapParams:
    x = np.asarray(xvec, dtype=float).reshape(-1)
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    expected = n_snap_parameters(ndepth, (l1, l2), gauge_fix=gauge_fix)
    if x.size != expected:
        raise ValueError(
            f"Expected {expected} SNAP parameters for ndepth={ndepth}, nfocks={(l1, l2)}, got {x.size}."
        )
    n0, n1 = snap_phase_counts((l1, l2), gauge_fix=gauge_fix)
    alpha = np.empty((int(ndepth), 2), dtype=complex)
    phases = np.zeros((int(ndepth), 2, max(l1, l2)), dtype=float)
    offset = 0
    for i in range(int(ndepth)):
        for cind, (n_phase, n_fock) in enumerate(((n0, l1), (n1, l2))):
            mag = float(x[offset])
            arg = float(x[offset + 1])
            alpha[i, cind] = mag * np.exp(1j * arg)
            offset += 2
            start = 1 if gauge_fix else 0
            phases[i, cind, start:n_fock] = x[offset : offset + n_phase]
            offset += n_phase
    return UnpackedSnapParams(alpha=alpha, phases=phases)


def pack_snap(
    alpha: np.ndarray,
    phases: np.ndarray,
    nfocks: Sequence[int] | tuple[int, int] = (8, 8),
    *,
    gauge_fix: bool = True,
) -> np.ndarray:
    alpha = np.asarray(alpha)
    phases = np.asarray(phases, dtype=float)
    ndepth = int(alpha.shape[0])
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    n0, n1 = snap_phase_counts((l1, l2), gauge_fix=gauge_fix)
    parts: list[np.ndarray] = []
    for i in range(ndepth):
        for cind, (n_phase, n_fock) in enumerate(((n0, l1), (n1, l2))):
            a = alpha[i, cind]
            parts.append(np.array([np.abs(a), np.angle(a)], dtype=float))
            start = 1 if gauge_fix else 0
            parts.append(np.asarray(phases[i, cind, start:n_fock], dtype=float).reshape(-1))
    return np.concatenate(parts)


def random_snap_parameters(
    ndepth: int,
    nfocks: Sequence[int] | tuple[int, int] = (8, 8),
    rng: np.random.Generator | None = None,
    *,
    gauge_fix: bool = True,
) -> np.ndarray:
    """Initialize SNAP+displacement with the same magnitude/angle ranges as ECD."""
    rng = rng or np.random.default_rng()
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    n0, n1 = snap_phase_counts((l1, l2), gauge_fix=gauge_fix)
    parts: list[np.ndarray] = []
    for _ in range(int(ndepth)):
        for n_phase in (n0, n1):
            mag = rng.uniform(0.0, 3.0)
            arg = rng.uniform(0.0, np.pi)
            phases = rng.uniform(0.0, np.pi, size=n_phase)
            parts.append(np.concatenate([[mag, arg], phases]))
    return np.concatenate(parts)


def snap_bounds(
    ndepth: int,
    nfocks: Sequence[int] | tuple[int, int] = (8, 8),
    *,
    gauge_fix: bool = True,
) -> list[tuple[float, float]]:
    n0, n1 = snap_phase_counts(nfocks, gauge_fix=gauge_fix)
    per_layer: list[tuple[float, float]] = []
    for n_phase in (n0, n1):
        per_layer.extend([(0.0, 10.0), (0.0, 2 * np.pi)])
        per_layer.extend([(0.0, 2 * np.pi)] * n_phase)
    return per_layer * int(ndepth)


def ansatz_inventory(
    ansatz: str,
    ndepth: int,
    nfocks: Sequence[int] | tuple[int, int] = (8, 8),
    n_prep_params: int = 5,
) -> dict:
    """Parameter and primitive-gate counts for one hybrid ansatz depth."""
    kind = str(ansatz).lower()
    nd = int(ndepth)
    if kind == "ecd":
        n_ansatz = n_parameters(nd)
        params_per_layer = n_parameters(1)
        primitive_gates = ("R", "ECD", "R", "ECD")
        description = "one layer = R, ECD on mode 1, then R, ECD on mode 2"
    elif kind == "snap":
        n_ansatz = n_snap_parameters(nd, nfocks)
        params_per_layer = n_snap_parameters(1, nfocks)
        primitive_gates = ("D", "SNAP", "D", "SNAP")
        description = "one layer = D, SNAP on mode 1, then D, SNAP on mode 2 (θ_0 = 0)"
    else:
        raise ValueError(f"unknown ansatz {ansatz!r}")
    n_gates = 4 * nd
    return {
        "ansatz": kind,
        "ndepth": nd,
        "nfocks": [int(nfocks[0]), int(nfocks[1])],
        "n_prep_params": int(n_prep_params),
        "n_ansatz_params": int(n_ansatz),
        "n_params": int(n_prep_params) + int(n_ansatz),
        "n_primitive_gates": int(n_gates),
        "n_scalar_controls": int(n_ansatz),
        "params_per_layer": int(params_per_layer),
        "gates_per_layer": 4,
        "primitive_gates": list(primitive_gates),
        "description": description,
    }
