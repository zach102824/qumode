"""Noiseless multi-mode ECD via local gates (no full 2048×2048 unitaries).

One pair is ``ECD(β) R(θ, φ)`` on a chosen (transmon, cavity), matching
production ``ecd_rotation_pair``. A layer applies every active transmon–cavity
pair, cavity-major then transmon (so n=7 reproduces the 1q+2cav UER order).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm

from .embedding import Embedding


def displace_matrix(n_fock: int, alpha: complex) -> np.ndarray:
    """Truncated D(α) = exp(α a† − α* a) on an ``n_fock``-level oscillator."""
    n = int(n_fock)
    a = np.zeros((n, n), dtype=complex)
    for j in range(1, n):
        a[j - 1, j] = np.sqrt(j)
    gen = complex(alpha) * a.conj().T - np.conjugate(alpha) * a
    return expm(gen)


def qubit_rotation(theta: float, phi: float) -> np.ndarray:
    """R(θ, φ) = exp[-i (θ/2) (X cos φ + Y sin φ)]."""
    half = 0.5 * float(theta)
    c = np.cos(half)
    s = np.sin(half)
    nx = np.cos(float(phi))
    ny = np.sin(float(phi))
    # n·σ = [[0, nx - i ny], [nx + i ny, 0]]
    off = nx - 1j * ny
    return np.array(
        [[c, -1j * s * off], [-1j * s * np.conjugate(off), c]],
        dtype=complex,
    )


def ry_ket(theta: float) -> np.ndarray:
    half = 0.5 * float(theta)
    return np.array([np.cos(half), np.sin(half)], dtype=complex)


def coherent_ket(n_fock: int, alpha: complex) -> np.ndarray:
    """Normalized truncated coherent, same recurrence as a Fock series."""
    n = int(n_fock)
    coeffs = np.zeros(n, dtype=complex)
    mag2 = float(np.abs(alpha) ** 2)
    coeffs[0] = np.exp(-0.5 * mag2)
    for k in range(1, n):
        coeffs[k] = coeffs[k - 1] * complex(alpha) / np.sqrt(k)
    nrm = float(np.linalg.norm(coeffs))
    if nrm == 0.0:
        raise ValueError("truncated coherent has zero norm")
    return coeffs / nrm


def _apply_matrix_on_axis(psi: np.ndarray, dims: tuple[int, ...], axis: int, mat: np.ndarray) -> np.ndarray:
    x = np.moveaxis(psi.reshape(dims), axis, -1)
    lead = x.shape[:-1]
    y = x.reshape(-1, dims[axis]) @ np.asarray(mat, dtype=complex).T
    return np.moveaxis(y.reshape(lead + (dims[axis],)), -1, axis).reshape(-1)


def apply_rotation(psi: np.ndarray, dims: tuple[int, ...], qubit_axis: int, theta: float, phi: float) -> np.ndarray:
    return _apply_matrix_on_axis(psi, dims, qubit_axis, qubit_rotation(theta, phi))


def apply_ecd(
    psi: np.ndarray,
    dims: tuple[int, ...],
    qubit_axis: int,
    cavity_axis: int,
    beta: complex,
) -> np.ndarray:
    """ECD(β) = |1⟩⟨0| ⊗ D(β/2) + |0⟩⟨1| ⊗ D(−β/2) on the selected axes."""
    L = int(dims[cavity_axis])
    x = np.moveaxis(psi.reshape(dims), (qubit_axis, cavity_axis), (-2, -1))
    lead = x.shape[:-2]
    x = x.reshape(-1, 2, L)
    d_plus = displace_matrix(L, 0.5 * complex(beta))
    # Truncated generator is anti-Hermitian, so D(-α) = D(α)†.
    d_minus = d_plus.conj().T
    y = np.empty_like(x)
    y[:, 1, :] = x[:, 0, :] @ d_plus.T
    y[:, 0, :] = x[:, 1, :] @ d_minus.T
    y = y.reshape(lead + (2, L))
    return np.moveaxis(y, (-2, -1), (qubit_axis, cavity_axis)).reshape(-1)


def coherent_radius_max(n_fock: int) -> float:
    return float(np.sqrt(max(int(n_fock) - 1, 0)))


def vacuum_prep(emb: Embedding) -> np.ndarray:
    return np.zeros(emb.n_prep_params, dtype=float)


def unpack_prep(prep: np.ndarray, emb: Embedding) -> tuple[list[float], list[complex]]:
    p = np.asarray(prep, dtype=float).reshape(-1)
    if p.size != emb.n_prep_params:
        raise ValueError(f"expected {emb.n_prep_params} prep params, got {p.size}")
    thetas = [float(p[i]) for i in range(emb.n_transmons)]
    alphas: list[complex] = []
    off = emb.n_transmons
    for k in range(emb.n_cavities):
        alphas.append(complex(p[off + 2 * k], p[off + 2 * k + 1]))
    return thetas, alphas


def project_prep(prep: np.ndarray, emb: Embedding) -> np.ndarray:
    p = np.asarray(prep, dtype=float).reshape(-1).copy()
    if p.size != emb.n_prep_params:
        raise ValueError(f"expected {emb.n_prep_params} prep params, got {p.size}")
    for i in range(emb.n_transmons):
        p[i] = float(np.clip(p[i], 0.0, np.pi))
    off = emb.n_transmons
    for mode in emb.modes:
        if mode.kind != "cavity":
            continue
        radius = coherent_radius_max(mode.dim)
        re, im = float(p[off]), float(p[off + 1])
        mag = float(np.hypot(re, im))
        if mag > radius and mag > 0.0:
            scale = radius / mag
            p[off] = re * scale
            p[off + 1] = im * scale
        off += 2
    return p


def prep_bounds(emb: Embedding) -> list[tuple[float, float]]:
    bounds = [(0.0, float(np.pi))] * emb.n_transmons
    for mode in emb.modes:
        if mode.kind != "cavity":
            continue
        r = coherent_radius_max(mode.dim)
        bounds.extend([(-r, r), (-r, r)])
    return bounds


def prep_to_ket(prep: np.ndarray, emb: Embedding) -> np.ndarray:
    thetas, alphas = unpack_prep(prep, emb)
    t_i = 0
    c_i = 0
    kets: list[np.ndarray] = []
    for mode in emb.modes:
        if mode.kind == "transmon":
            kets.append(ry_ket(thetas[t_i]))
            t_i += 1
        else:
            kets.append(coherent_ket(mode.dim, alphas[c_i]))
            c_i += 1
    psi = kets[0]
    for ket in kets[1:]:
        psi = np.kron(psi, ket)
    nrm = float(np.linalg.norm(psi))
    if nrm == 0.0:
        raise ValueError("product state has zero norm")
    return psi if abs(nrm - 1.0) <= 1e-12 else psi / nrm


@dataclass(frozen=True)
class UnpackedECD:
    beta: np.ndarray  # complex (ndepth, n_pairs)
    theta: np.ndarray
    phi: np.ndarray


def n_ecd_parameters(ndepth: int, n_pairs: int) -> int:
    return 4 * int(ndepth) * int(n_pairs)


def unpack_ecd(xvec: np.ndarray, ndepth: int, n_pairs: int) -> UnpackedECD:
    x = np.asarray(xvec, dtype=float).reshape(-1)
    size = int(ndepth) * int(n_pairs)
    expected = 4 * size
    if x.size != expected:
        raise ValueError(f"expected {expected} ECD params for L={ndepth} pairs={n_pairs}, got {x.size}")
    beta_mag = x[:size].reshape(ndepth, n_pairs)
    beta_arg = x[size : 2 * size].reshape(ndepth, n_pairs)
    theta = x[2 * size : 3 * size].reshape(ndepth, n_pairs)
    phi = x[3 * size :].reshape(ndepth, n_pairs)
    return UnpackedECD(beta=beta_mag * np.exp(1j * beta_arg), theta=theta, phi=phi)


def random_ecd_parameters(ndepth: int, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Notebook ranges: |β|~U(0,3), arg/θ/φ ~ U(0,π)."""
    shape = (int(ndepth), int(n_pairs))
    mag = rng.uniform(0.0, 3.0, size=shape)
    arg = rng.uniform(0.0, np.pi, size=shape)
    theta = rng.uniform(0.0, np.pi, size=shape)
    phi = rng.uniform(0.0, np.pi, size=shape)
    return np.concatenate([mag.ravel(), arg.ravel(), theta.ravel(), phi.ravel()])


def ecd_bounds(ndepth: int, n_pairs: int) -> list[tuple[float, float]]:
    size = int(ndepth) * int(n_pairs)
    return (
        [(0.0, 10.0)] * size
        + [(0.0, 2 * np.pi)] * size
        + [(0.0, np.pi)] * size
        + [(0.0, 2 * np.pi)] * size
    )


def apply_ecd_ansatz(psi: np.ndarray, xvec: np.ndarray, emb: Embedding, ndepth: int) -> np.ndarray:
    """Apply L layers of ECD–rotation pairs to a hybrid statevector."""
    dims = emb.dims
    pairs = emb.ecd_pairs()
    params = unpack_ecd(xvec, ndepth, len(pairs))
    out = np.asarray(psi, dtype=complex).reshape(-1).copy()
    for layer in range(int(ndepth)):
        for p, (t_ax, c_ax) in enumerate(pairs):
            out = apply_rotation(out, dims, t_ax, float(params.theta[layer, p]), float(params.phi[layer, p]))
            out = apply_ecd(out, dims, t_ax, c_ax, complex(params.beta[layer, p]))
    nrm = float(np.linalg.norm(out))
    if nrm == 0.0:
        raise ValueError("ECD state has zero norm")
    return out if abs(nrm - 1.0) <= 1e-12 else out / nrm


def hybrid_energy_tensor(
    emb: Embedding,
    logical_energies: np.ndarray,
) -> np.ndarray:
    """Diagonal hybrid energies: each Fock occupation decodes to an n-bit energy."""
    logical = np.asarray(logical_energies, dtype=float).reshape(1 << emb.n_qubits)
    tensor = np.empty(emb.dims, dtype=float)
    for idx in np.ndindex(emb.dims):
        bits = emb.decode_occupations(idx)
        acc = 0
        for b in bits:
            acc = (acc << 1) | int(b)
        tensor[idx] = logical[acc]
    return tensor
