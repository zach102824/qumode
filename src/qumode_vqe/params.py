"""Parameter packing for the ECD-rotation ansatz.

Paper parameterization (used by the published notebooks):
    X = [β_mag, β_arg, θ, φ]
each of shape (N_d, 2), so ``len(X) = 8 N_d`` (40 parameters for N_d = 5).

A Cartesian alternative stores Re(β), Im(β) instead of polar coordinates,
which avoids the |β| = 0 singularity and is used with L-BFGS-B.
"""

from __future__ import annotations

from dataclasses import dataclass
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
