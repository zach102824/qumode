"""Single-mode ECD state preparation: cats, Fock states, and fidelities.

This module is for noiseless *state-prep* experiments (overlap / fidelity),
not Gibbs VQE. The two-cavity ECD–rotation ansatz in ``circuit.py`` is left
untouched.

Single-mode layer used here (one transmon + one oscillator)::

    R(θ, φ) then ECD(β)

``N_d`` layers give ``4 N_d`` real parameters (Re β, Im β, θ, φ). A terminal
qubit rotation may be appended (+2 parameters) when the target is
``|g⟩ ⊗ |target⟩``.

ECD convention matches Eickbusch and ``circuit.ecd_gate``:

    ECD(β) = D(β/2)|e⟩⟨g| + D(−β/2)|g⟩⟨e|

so ``ECD(β)|g, 0⟩ = |e⟩|β/2⟩`` with ``|g⟩ ≡ |0⟩``, ``|e⟩ ≡ |1⟩``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt
from scipy.linalg import expm
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.special import gammaln

from .circuit import qubit_rotation
from .qaoa import n_hea_params

# |g⟩ = |0⟩, |e⟩ = |1⟩, matching circuit.py and QuTiP tensor order.
G = 0
E = 1

INFINITE_FOCK = 256
DEFAULT_TRUNC_INFIDELITY = 1e-4
DEFAULT_L_CAP = 64
N_PARAMS_PER_LAYER = 4


def n_ecd_params(n_layers: int, terminal_rotation: bool = False) -> int:
    """Real parameter count: 4 N_d, plus 2 if a terminal R(θ, φ) is included."""
    return N_PARAMS_PER_LAYER * int(n_layers) + (2 if terminal_rotation else 0)


def n_qubits_for_cutoff(n_fock: int) -> int:
    """Qubits needed to binary-encode Fock indices ``0 .. L-1`` (MSB-first)."""
    l = int(n_fock)
    if l < 2:
        raise ValueError(f"n_fock must be >= 2, got {n_fock}")
    return int(np.ceil(np.log2(l)))


def matched_hea_layers(n_qubits: int, n_ecd_params: int) -> int:
    """``n_layers`` minimizing ``|n_qubits (n_layers+1) - n_ecd_params|``.

    HEA parameter count is ``n_qubits * (n_layers + 1)`` (see ``qaoa.n_hea_params``).
    Ties keep the smaller layer count.
    """
    nq = int(n_qubits)
    budget = int(n_ecd_params)
    if nq < 1:
        raise ValueError("n_qubits must be >= 1")
    best_l = 0
    best_err = abs(nq - budget)
    # Search a finite range; n_layers=0 is the product-Ry circuit.
    for n_layers in range(0, max(budget, 1) + 2):
        err = abs(nq * (n_layers + 1) - budget)
        if err < best_err:
            best_l = n_layers
            best_err = err
    return best_l


def cutoff_formula(alpha: complex | float) -> int:
    """Heuristic ``L >= |α|² + 8|α| + 16`` (ceil)."""
    a = abs(complex(alpha))
    return int(np.ceil(a * a + 8.0 * a + 16.0))


def coherent_amplitude(n: int, alpha: complex) -> complex:
    """⟨n|α⟩ = e^{−|α|²/2} α^n / √n! (infinite-dimensional oscillator)."""
    a = complex(alpha)
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    if a == 0.0:
        return 1.0 + 0.0j if n == 0 else 0.0j
    log_mag = -0.5 * abs(a) ** 2 + n * np.log(abs(a)) - 0.5 * gammaln(n + 1.0)
    return np.exp(log_mag + 1j * n * np.angle(a))


def _coherent_amplitudes(alpha: complex, n_max: int) -> np.ndarray:
    return np.array([coherent_amplitude(n, alpha) for n in range(int(n_max))], dtype=complex)


def _normalize(vec: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=complex).reshape(-1).copy()
    nrm = float(np.linalg.norm(out))
    if nrm == 0.0:
        raise ValueError("attempted to normalize a zero vector")
    return out / nrm


def even_cat_amplitudes_infinite(alpha: complex, n_max: int = INFINITE_FOCK) -> np.ndarray:
    """Normalized even two-legged cat ``|C_α⟩ ∝ |α⟩ + |−α⟩`` on ``0 .. n_max-1``.

    The discarded tail past ``n_max`` is required to be negligible (the
    coefficients are the exact infinite-dim ones, then renormalized on the
    finite window).
    """
    a = complex(alpha)
    raw = _coherent_amplitudes(a, n_max) + _coherent_amplitudes(-a, n_max)
    return _normalize(raw)


def compass_cat_amplitudes_infinite(alpha: complex, n_max: int = INFINITE_FOCK) -> np.ndarray:
    """Normalized 4-legged compass cat ``∝ |α⟩ + |iα⟩ + |−α⟩ + |−iα⟩``."""
    a = complex(alpha)
    raw = (
        _coherent_amplitudes(a, n_max)
        + _coherent_amplitudes(1j * a, n_max)
        + _coherent_amplitudes(-a, n_max)
        + _coherent_amplitudes(-1j * a, n_max)
    )
    return _normalize(raw)


def truncation_fidelity(infinite_amps: np.ndarray, n_fock: int) -> float:
    """``‖P_L |ψ_∞⟩‖²`` for an already-normalized infinite-window ket."""
    psi = np.asarray(infinite_amps, dtype=complex).reshape(-1)
    l = int(n_fock)
    if l <= 0:
        return 0.0
    return float(np.vdot(psi[:l], psi[:l]).real)


def truncate_normalize(infinite_amps: np.ndarray, n_fock: int) -> np.ndarray:
    psi = np.asarray(infinite_amps, dtype=complex).reshape(-1)
    l = int(n_fock)
    if psi.size < l:
        padded = np.zeros(l, dtype=complex)
        padded[: psi.size] = psi
        return _normalize(padded)
    return _normalize(psi[:l])


def even_cat_amplitudes(alpha: complex, n_fock: int, n_max: int = INFINITE_FOCK) -> np.ndarray:
    """Even cat, truncated to ``L`` and renormalized."""
    return truncate_normalize(even_cat_amplitudes_infinite(alpha, n_max), n_fock)


def compass_cat_amplitudes(alpha: complex, n_fock: int, n_max: int = INFINITE_FOCK) -> np.ndarray:
    """4-legged compass cat, truncated to ``L`` and renormalized."""
    return truncate_normalize(compass_cat_amplitudes_infinite(alpha, n_max), n_fock)


def fock_amplitudes(n: int, n_fock: int) -> np.ndarray:
    l = int(n_fock)
    k = int(n)
    if not (0 <= k < l):
        raise ValueError(f"Fock index n={k} is outside [0, {l})")
    psi = np.zeros(l, dtype=complex)
    psi[k] = 1.0
    return psi


def fock_index_for_alpha(alpha: complex, n_fock: int) -> int:
    """Negative-control photon number: ``round(|α|²)`` clipped to ``[1, L-2]``."""
    l = int(n_fock)
    if l < 3:
        raise ValueError(f"n_fock must be >= 3 for the Fock control, got {n_fock}")
    n = int(np.round(abs(complex(alpha)) ** 2))
    return int(np.clip(n, 1, l - 2))


def even_parity_weight(amplitudes: np.ndarray) -> float:
    """Probability on even Fock states."""
    psi = np.asarray(amplitudes, dtype=complex).reshape(-1)
    even = psi[0::2]
    return float(np.vdot(even, even).real)


def choose_cutoff(
    alpha: complex | float,
    *,
    max_infidelity: float = DEFAULT_TRUNC_INFIDELITY,
    cap: int = DEFAULT_L_CAP,
    n_max: int = INFINITE_FOCK,
    kind: str = "even_cat",
) -> tuple[int, float]:
    """Pick ``L`` so the cat's truncated-norm infidelity is below ``max_infidelity``.

    Starts from ``L = |α|² + 8|α| + 16`` and grows (up to ``cap``) if needed.
    Returns ``(L, truncation_fidelity)``. Raises if the cap still misses the
    infidelity target.
    """
    if kind == "even_cat":
        inf = even_cat_amplitudes_infinite(alpha, n_max)
    elif kind == "compass":
        inf = compass_cat_amplitudes_infinite(alpha, n_max)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    l = max(cutoff_formula(alpha), 4)
    l = min(int(l), int(cap))
    f_trunc = truncation_fidelity(inf, l)
    while (1.0 - f_trunc) >= float(max_infidelity) and l < int(cap):
        l += 1
        f_trunc = truncation_fidelity(inf, l)
    if (1.0 - f_trunc) >= float(max_infidelity):
        raise ValueError(
            f"cutoff cap L={cap} still has truncation infidelity "
            f"{1.0 - f_trunc:.3e} >= {max_infidelity} for α={alpha} ({kind})"
        )
    return l, f_trunc


def embed_fock_in_qubits(amplitudes: np.ndarray, n_qubits: int | None = None) -> np.ndarray:
    """Pad a length-``L`` Fock ket into a ``2^{n_q}`` MSB-first computational register.

    Unused levels ``L .. 2^{n_q}-1`` are left at zero amplitude. ``n_qubits``
    defaults to ``ceil(log2 L)``.
    """
    psi = np.asarray(amplitudes, dtype=complex).reshape(-1)
    nq = int(n_qubits) if n_qubits is not None else n_qubits_for_cutoff(psi.size)
    dim = 1 << nq
    if psi.size > dim:
        raise ValueError(f"Fock length {psi.size} does not fit in {nq} qubits (dim={dim})")
    out = np.zeros(dim, dtype=complex)
    out[: psi.size] = psi
    return out


def state_fidelity(psi: np.ndarray, target: np.ndarray) -> float:
    """Pure-state fidelity ``|⟨target|ψ⟩|²``. Both kets are renormalized."""
    a = _normalize(psi)
    b = _normalize(target)
    if a.size != b.size:
        raise ValueError(f"state size {a.size} != target size {b.size}")
    return float(abs(np.vdot(b, a)) ** 2)


def hybrid_to_components(psi: np.ndarray, n_fock: int) -> np.ndarray:
    """Reshape a length-``2L`` ket to ``(2, L)`` with axis 0 = ``(g, e)``."""
    vec = np.asarray(psi, dtype=complex).reshape(-1)
    l = int(n_fock)
    if vec.size != 2 * l:
        raise ValueError(f"expected length {2 * l}, got {vec.size}")
    return vec.reshape(2, l)


def cavity_reduced_fidelity(psi_hybrid: np.ndarray, cavity_target: np.ndarray) -> float:
    """``⟨target|ρ_cav|target⟩`` after tracing out the transmon.

    For components ``|ψ⟩ = |g⟩|ψ_g⟩ + |e⟩|ψ_e⟩`` this is
    ``|⟨t|ψ_g⟩|² + |⟨t|ψ_e⟩|²``.
    """
    target = _normalize(cavity_target)
    comps = hybrid_to_components(psi_hybrid, target.size)
    overlap = 0.0
    for q in (G, E):
        overlap += abs(np.vdot(target, comps[q])) ** 2
    return float(overlap)


def joint_fidelity_ground_cavity(psi_hybrid: np.ndarray, cavity_target: np.ndarray) -> float:
    """Fidelity to ``|g⟩ ⊗ |target⟩``."""
    target = _normalize(cavity_target)
    comps = hybrid_to_components(psi_hybrid, target.size)
    return float(abs(np.vdot(target, comps[G])) ** 2)


def qubit_ground_prob(psi_hybrid: np.ndarray, n_fock: int) -> float:
    comps = hybrid_to_components(psi_hybrid, n_fock)
    return float(np.vdot(comps[G], comps[G]).real)


def plus_postselect_cavity(psi_hybrid: np.ndarray, n_fock: int) -> tuple[np.ndarray, float]:
    """X-basis post-selection: unnormalized cavity is ``(ψ_g + ψ_e)/√2``.

    Returns ``(normalized_cavity, success_probability)``.
    """
    comps = hybrid_to_components(psi_hybrid, n_fock)
    unnorm = (comps[G] + comps[E]) / np.sqrt(2.0)
    p = float(np.vdot(unnorm, unnorm).real)
    if p <= 0.0:
        raise ValueError("post-selection probability is zero")
    return unnorm / np.sqrt(p), p


# ---------------------------------------------------------------------------
# Single-mode ECD / rotation apply (numpy, Fock truncation L)
# ---------------------------------------------------------------------------

_ladder_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_sqrtn_cache: dict[int, np.ndarray] = {}
_rot_cache: dict[tuple[float, float], np.ndarray] = {}


def _ladders(n_fock: int) -> tuple[np.ndarray, np.ndarray]:
    l = int(n_fock)
    cached = _ladder_cache.get(l)
    if cached is not None:
        return cached
    a = np.zeros((l, l), dtype=complex)
    for n in range(1, l):
        a[n - 1, n] = np.sqrt(n)
    adag = a.T.conj()
    _ladder_cache[l] = (a, adag)
    return a, adag


def _sqrtn(n_fock: int) -> np.ndarray:
    l = int(n_fock)
    cached = _sqrtn_cache.get(l)
    if cached is None:
        cached = np.sqrt(np.arange(1, l, dtype=float))
        _sqrtn_cache[l] = cached
    return cached


def displace_matrix(n_fock: int, alpha: complex) -> np.ndarray:
    """Truncated ``D(α) = exp(α a† − α* a)`` on ``span{|0⟩,…,|L−1⟩}``."""
    a, adag = _ladders(n_fock)
    z = complex(alpha)
    return expm(z * adag - np.conjugate(z) * a)


def apply_displace(vec: np.ndarray, alpha: complex, n_fock: int) -> np.ndarray:
    """Apply truncated ``D(α)`` to a Fock vector via sparse ``expm_multiply``."""
    l = int(n_fock)
    z = complex(alpha)
    if z == 0.0:
        return np.asarray(vec, dtype=complex).reshape(l).copy()
    n = _sqrtn(l)
    # Generator G = α a† − α* a is tridiagonal in the Fock basis.
    gen = diags([z * n, -np.conjugate(z) * n], [-1, 1], shape=(l, l), dtype=complex)
    return np.asarray(expm_multiply(gen, np.asarray(vec, dtype=complex).reshape(l)), dtype=complex)


def qubit_rotation_matrix(theta: float, phi: float) -> np.ndarray:
    """``R(θ, φ) = exp[−i (θ/2) (X cos φ + Y sin φ)]``, same as ``circuit``."""
    key = (float(theta), float(phi))
    cached = _rot_cache.get(key)
    if cached is not None:
        return cached
    mat = np.asarray(qubit_rotation(theta, phi).full(), dtype=complex)
    _rot_cache[key] = mat
    return mat


def vacuum_hybrid(n_fock: int) -> np.ndarray:
    """``|g, 0⟩`` as a length-``2L`` ket."""
    psi = np.zeros(2 * int(n_fock), dtype=complex)
    psi[0] = 1.0
    return psi


def apply_qubit_rotation(psi_hybrid: np.ndarray, theta: float, phi: float, n_fock: int) -> np.ndarray:
    comps = hybrid_to_components(psi_hybrid, n_fock)
    rot = qubit_rotation_matrix(theta, phi)
    out = rot @ comps
    return out.reshape(-1)


def apply_ecd(psi_hybrid: np.ndarray, beta: complex, n_fock: int) -> np.ndarray:
    """Apply ``ECD(β)``: ``|g, ψ⟩ → |e, D(β/2) ψ⟩``, ``|e, ψ⟩ → |g, D(−β/2) ψ⟩``."""
    comps = hybrid_to_components(psi_hybrid, n_fock)
    out = np.zeros((2, int(n_fock)), dtype=complex)
    out[E] = apply_displace(comps[G], 0.5 * complex(beta), n_fock)
    out[G] = apply_displace(comps[E], -0.5 * complex(beta), n_fock)
    return out.reshape(-1)


def unpack_ecd_params(
    xvec: np.ndarray,
    n_layers: int,
    terminal_rotation: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float] | None]:
    """Return ``(beta, theta, phi, terminal_or_None)`` from a packed real vector."""
    x = np.asarray(xvec, dtype=float).reshape(-1)
    nd = int(n_layers)
    expected = n_ecd_params(nd, terminal_rotation)
    if x.size != expected:
        raise ValueError(f"expected {expected} ECD parameters for N_d={nd}, got {x.size}")
    beta = np.empty(nd, dtype=complex)
    theta = np.empty(nd, dtype=float)
    phi = np.empty(nd, dtype=float)
    for i in range(nd):
        beta[i] = complex(x[4 * i], x[4 * i + 1])
        theta[i] = float(x[4 * i + 2])
        phi[i] = float(x[4 * i + 3])
    terminal = None
    if terminal_rotation:
        terminal = (float(x[-2]), float(x[-1]))
    return beta, theta, phi, terminal


def ecd_statevector(
    xvec: np.ndarray,
    n_fock: int,
    n_layers: int,
    terminal_rotation: bool = False,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ``N_d`` layers of ``R(θ, φ)`` then ``ECD(β)``, optional terminal ``R``."""
    beta, theta, phi, terminal = unpack_ecd_params(xvec, n_layers, terminal_rotation)
    psi = vacuum_hybrid(n_fock) if initial is None else np.asarray(initial, dtype=complex).reshape(-1)
    if psi.size != 2 * int(n_fock):
        raise ValueError(f"initial state size {psi.size} != {2 * int(n_fock)}")
    for i in range(int(n_layers)):
        psi = apply_qubit_rotation(psi, theta[i], phi[i], n_fock)
        psi = apply_ecd(psi, beta[i], n_fock)
    if terminal is not None:
        psi = apply_qubit_rotation(psi, terminal[0], terminal[1], n_fock)
    nrm = float(np.linalg.norm(psi))
    if nrm == 0.0:
        raise ValueError("ECD circuit produced a zero state")
    return psi / nrm


def random_ecd_params(
    n_layers: int,
    rng: np.random.Generator,
    *,
    terminal_rotation: bool = False,
    alpha: complex | float = 1.0,
) -> np.ndarray:
    """Random starts: β in a disk of radius ``max(3, 2|α|+1)``, angles in ``[0, 2π)``."""
    nd = int(n_layers)
    radius = max(3.0, 2.0 * abs(complex(alpha)) + 1.0)
    x = np.empty(n_ecd_params(nd, terminal_rotation), dtype=float)
    for i in range(nd):
        mag = float(rng.uniform(0.0, radius))
        arg = float(rng.uniform(0.0, 2.0 * np.pi))
        x[4 * i] = mag * np.cos(arg)
        x[4 * i + 1] = mag * np.sin(arg)
        x[4 * i + 2] = float(rng.uniform(0.0, 2.0 * np.pi))
        x[4 * i + 3] = float(rng.uniform(0.0, 2.0 * np.pi))
    if terminal_rotation:
        x[-2] = float(rng.uniform(0.0, 2.0 * np.pi))
        x[-1] = float(rng.uniform(0.0, 2.0 * np.pi))
    return x


def ecd_bounds(
    n_layers: int,
    *,
    terminal_rotation: bool = False,
    alpha: complex | float = 1.0,
) -> list[tuple[float, float]]:
    """L-BFGS-B bounds. |β| is capped so truncated displacements stay meaningful."""
    radius = max(15.0, 5.0 * abs(complex(alpha)) + 5.0)
    bounds: list[tuple[float, float]] = []
    for _ in range(int(n_layers)):
        bounds.extend([(-radius, radius), (-radius, radius), (-4.0 * np.pi, 4.0 * np.pi), (-4.0 * np.pi, 4.0 * np.pi)])
    if terminal_rotation:
        bounds.extend([(-4.0 * np.pi, 4.0 * np.pi), (-4.0 * np.pi, 4.0 * np.pi)])
    return bounds


def ecd_gate_single(beta: complex, n_fock: int) -> qt.Qobj:
    """QuTiP ECD on one qubit + one oscillator (same convention as ``circuit.ecd_gate``)."""
    sm = qt.Qobj(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex))  # |e⟩⟨g|
    sp = qt.Qobj(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex))  # |g⟩⟨e|
    return qt.tensor(sm, qt.displace(int(n_fock), complex(beta) / 2.0)) + qt.tensor(
        sp, qt.displace(int(n_fock), -complex(beta) / 2.0)
    )


@dataclass(frozen=True)
class ConstructiveCat:
    """O(1) even-cat circuit: ``R_y(π/2)`` then ``ECD(2α)``, X-basis post-select."""

    alpha: complex
    n_fock: int
    cavity: np.ndarray
    fidelity: float
    success_probability: float
    f_reduced: float
    f_joint: float
    f_truncation: float
    hybrid: np.ndarray


def constructive_even_cat(alpha: complex, n_fock: int) -> ConstructiveCat:
    """Existence proof: one rotation + one ECD, post-select the even cat.

    Start ``|g, 0⟩``. ``R_y(π/2)`` (``R(π/2, π/2)``) then ``ECD(β=2α)`` yields
    ``(|e, α⟩ + |g, −α⟩)/√2`` (truncated displacements). Projecting the qubit
    onto ``|+⟩ = (|g⟩+|e⟩)/√2`` leaves the even cat on the cavity.
    """
    a = complex(alpha)
    l = int(n_fock)
    # R_y(π/2)|g⟩ = (|g⟩ + |e⟩)/√2.  R(θ, φ=π/2) = R_y(θ).
    x = np.array([2.0 * a.real, 2.0 * a.imag, 0.5 * np.pi, 0.5 * np.pi], dtype=float)
    hybrid = ecd_statevector(x, l, n_layers=1, terminal_rotation=False)
    target = even_cat_amplitudes(a, l)
    cavity, p_ok = plus_postselect_cavity(hybrid, l)
    inf = even_cat_amplitudes_infinite(a)
    return ConstructiveCat(
        alpha=a,
        n_fock=l,
        cavity=cavity,
        fidelity=state_fidelity(cavity, target),
        success_probability=p_ok,
        f_reduced=cavity_reduced_fidelity(hybrid, target),
        f_joint=joint_fidelity_ground_cavity(hybrid, target),
        f_truncation=truncation_fidelity(inf, l),
        hybrid=hybrid,
    )


def evaluate_ecd_fidelities(
    xvec: np.ndarray,
    cavity_target: np.ndarray,
    n_layers: int,
    terminal_rotation: bool = False,
) -> dict[str, float]:
    """Joint, reduced, and |+⟩-post-selected fidelities for a packed ECD vector."""
    l = int(np.asarray(cavity_target).size)
    psi = ecd_statevector(xvec, l, n_layers, terminal_rotation=terminal_rotation)
    cav, p_ok = plus_postselect_cavity(psi, l)
    return {
        "F_joint": joint_fidelity_ground_cavity(psi, cavity_target),
        "F_reduced": cavity_reduced_fidelity(psi, cavity_target),
        "F_postselect": state_fidelity(cav, cavity_target),
        "P_plus": p_ok,
        "P_g": qubit_ground_prob(psi, l),
    }


__all__ = [
    "ConstructiveCat",
    "apply_ecd",
    "apply_qubit_rotation",
    "cavity_reduced_fidelity",
    "choose_cutoff",
    "compass_cat_amplitudes",
    "constructive_even_cat",
    "cutoff_formula",
    "ecd_bounds",
    "ecd_gate_single",
    "ecd_statevector",
    "embed_fock_in_qubits",
    "even_cat_amplitudes",
    "even_parity_weight",
    "evaluate_ecd_fidelities",
    "fock_amplitudes",
    "fock_index_for_alpha",
    "joint_fidelity_ground_cavity",
    "matched_hea_layers",
    "n_ecd_params",
    "n_hea_params",
    "n_qubits_for_cutoff",
    "random_ecd_params",
    "state_fidelity",
    "truncation_fidelity",
    "vacuum_hybrid",
]
