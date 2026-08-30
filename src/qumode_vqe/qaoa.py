"""Noiseless qubit QAOA and a hardware-efficient ansatz on the 7-bit embedding.

The hybrid ECD simulator lives in |q, n, m⟩. These qubit circuits act on the
same 128-dimensional space via the existing MSB-first binary map
(``bits_from_qnm`` / ``qubit_index_from_bits``, partition (1, 3, 3)).

QAOA uses mixer H_M = Σ_i X_i and diagonal cost H_C = diag(E[z]), native
initial state |+⟩^{⊗ n}. The hardware-efficient ansatz (HEA) is Ry on each
qubit plus nearest-neighbour CZ, starting from |0⟩^{⊗ n}.

Gibbs η is always ``sampled_tail`` (this repo's heuristic, not Dutta/Li).
The Gibbs cost f = −ln ⟨e^{−η E}⟩ is Li et al., Phys. Rev. Research 2,
023074 (2020). SPSA is ``run_spsa`` with the mixed-suite gains.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .eta import SampledTailEta
from .hamiltonian import (
    DEFAULT_NFOCKS,
    N_QUBITS,
    bits_from_qubit_index,
    bitstring_from_bits,
    qnm_from_bits,
    qubit_index_from_bits,
)
from .vqe import OptimizeResult, gibbs_objective, run_spsa

GROUND_ATOL = 1e-8
DEFAULT_QAOA_P = 20
DEFAULT_HEA_LAYERS = 5  # n_qubits * (L + 1) = 7 * 6 = 42
DEFAULT_SPSA = dict(a=0.2, c=0.15, A=10.0, alpha=0.602, gamma=0.101)


def n_qaoa_params(p: int) -> int:
    return 2 * int(p)


def n_hea_params(n_qubits: int, n_layers: int) -> int:
    return int(n_qubits) * (int(n_layers) + 1)


def bitstring_to_index(bitstring: str) -> int:
    text = str(bitstring).strip()
    if not text or any(ch not in "01" for ch in text):
        raise ValueError(f"expected a computational bitstring, got {bitstring!r}")
    return int(text, 2)


def energy_vector_from_tensor(
    energy_tensor: np.ndarray,
    partition: tuple[int, int, int] = (1, 3, 3),
    *,
    n_qubits: int | None = None,
    ground_bitstring: str | None = None,
    atol: float = GROUND_ATOL,
) -> np.ndarray:
    """Flatten a (2, L1, L2) hybrid energy tensor onto the 7-qubit basis.

    Index ``z`` uses qubit 0 as MSB (QuTiP / ``bits_from_qubit_index``).
    Raises ``RuntimeError`` if the decoded ground bitstring is not an
    exact minimizer of the tensor (within ``atol``).
    """
    tensor = np.asarray(energy_tensor, dtype=float)
    if tensor.ndim != 3 or tensor.shape[0] != 2:
        raise ValueError(f"expected energy tensor of shape (2, L1, L2), got {tensor.shape}")
    n_nbits, m_nbits = int(partition[1]), int(partition[2])
    nq = int(n_qubits) if n_qubits is not None else 1 + n_nbits + m_nbits
    if nq != 1 + n_nbits + m_nbits:
        raise ValueError(f"n_qubits={nq} does not match partition {partition}")
    l1, l2 = int(tensor.shape[1]), int(tensor.shape[2])
    dim = 1 << nq
    energies = np.empty(dim, dtype=float)
    for idx in range(dim):
        bits = bits_from_qubit_index(idx, nq)
        q, n, m = qnm_from_bits(bits, partition)
        if not (0 <= n < l1 and 0 <= m < l2):
            raise RuntimeError(f"decoded Fock ({n}, {m}) is outside tensor {(l1, l2)}")
        energies[idx] = float(tensor[q, n, m])
    verify_energy_vector(energies, tensor, partition, ground_bitstring=ground_bitstring, atol=atol)
    return energies


def verify_energy_vector(
    energies: np.ndarray,
    energy_tensor: np.ndarray,
    partition: tuple[int, int, int] = (1, 3, 3),
    *,
    ground_bitstring: str | None = None,
    atol: float = GROUND_ATOL,
) -> None:
    """Fail if E[z] disagrees with the tensor or the labeled ground is not a min."""
    e = np.asarray(energies, dtype=float).reshape(-1)
    tensor = np.asarray(energy_tensor, dtype=float)
    nq = int(round(np.log2(e.size)))
    if (1 << nq) != e.size:
        raise RuntimeError(f"energy vector length {e.size} is not a power of two")
    for idx in range(e.size):
        bits = bits_from_qubit_index(idx, nq)
        q, n, m = qnm_from_bits(bits, partition)
        got = float(e[idx])
        want = float(tensor[q, n, m])
        if not np.isclose(got, want, atol=atol, rtol=0.0):
            raise RuntimeError(
                f"encoding mismatch at z={idx} bits={bitstring_from_bits(bits)}: "
                f"E[z]={got} vs tensor[{q},{n},{m}]={want}"
            )
        if qubit_index_from_bits(bits) != idx:
            raise RuntimeError(f"qubit_index_from_bits is not inverse of bits_from_qubit_index at {idx}")
    emin_t = float(np.min(tensor))
    emin_e = float(np.min(e))
    if not np.isclose(emin_e, emin_t, atol=atol, rtol=0.0):
        raise RuntimeError(f"min(E)={emin_e} != min(tensor)={emin_t}")
    if ground_bitstring is not None:
        gidx = bitstring_to_index(ground_bitstring)
        if gidx < 0 or gidx >= e.size:
            raise RuntimeError(f"ground bitstring {ground_bitstring!r} is out of range for dim {e.size}")
        if not np.isclose(float(e[gidx]), emin_e, atol=atol, rtol=0.0):
            raise RuntimeError(
                f"E[{ground_bitstring}]={float(e[gidx])} != energy_min={emin_e}"
            )


def plus_state(n_qubits: int) -> np.ndarray:
    dim = 1 << int(n_qubits)
    return np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)


def zero_state(n_qubits: int) -> np.ndarray:
    psi = np.zeros(1 << int(n_qubits), dtype=complex)
    psi[0] = 1.0
    return psi


def apply_diag_phase(psi: np.ndarray, gamma: float, energies: np.ndarray) -> np.ndarray:
    """In-place multiply by e^{-i γ E[z]}."""
    psi *= np.exp(-1j * float(gamma) * np.asarray(energies, dtype=float))
    return psi


def apply_x_rotation(psi: np.ndarray, beta: float, qubit: int, n_qubits: int) -> np.ndarray:
    """In-place exp(-i β X) on one qubit (qubit 0 = MSB)."""
    nq = int(n_qubits)
    q = int(qubit)
    hi = 1 << q
    lo = 1 << (nq - 1 - q)
    view = psi.reshape(hi, 2, lo)
    c = np.cos(float(beta))
    s = -1j * np.sin(float(beta))
    a = view[:, 0, :].copy()
    b = view[:, 1, :].copy()
    view[:, 0, :] = c * a + s * b
    view[:, 1, :] = s * a + c * b
    return psi


def apply_mixer(psi: np.ndarray, beta: float, n_qubits: int) -> np.ndarray:
    """In-place exp(-i β Σ_i X_i)."""
    for q in range(int(n_qubits)):
        apply_x_rotation(psi, beta, q, n_qubits)
    return psi


def apply_ry(psi: np.ndarray, theta: float, qubit: int, n_qubits: int) -> np.ndarray:
    """In-place R_y(θ) on one qubit (qubit 0 = MSB)."""
    nq = int(n_qubits)
    q = int(qubit)
    hi = 1 << q
    lo = 1 << (nq - 1 - q)
    view = psi.reshape(hi, 2, lo)
    half = 0.5 * float(theta)
    c = np.cos(half)
    s = np.sin(half)
    a = view[:, 0, :].copy()
    b = view[:, 1, :].copy()
    view[:, 0, :] = c * a - s * b
    view[:, 1, :] = s * a + c * b
    return psi


def apply_cz(psi: np.ndarray, q1: int, q2: int, n_qubits: int) -> np.ndarray:
    """In-place CZ between two qubits (qubit 0 = MSB)."""
    nq = int(n_qubits)
    bit1 = nq - 1 - int(q1)
    bit2 = nq - 1 - int(q2)
    idx = np.arange(psi.size)
    both = ((idx >> bit1) & 1) & ((idx >> bit2) & 1)
    psi[both.astype(bool)] *= -1.0
    return psi


def nearest_cz_even_odd_pairs(n_qubits: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Even then odd nearest-neighbour CZ pairs (IBM-style depth-2 packing).

    Even (parallel): CZ(0,1), CZ(2,3), … ; the last qubit is idle when n is odd.
    Odd (parallel): CZ(1,2), CZ(3,4), … ; qubit 0 is idle.
    All CZs commute, so this is the same unitary as sequential CZ(i, i+1).
    """
    nq = int(n_qubits)
    even = [(q, q + 1) for q in range(0, nq - 1, 2)]
    odd = [(q, q + 1) for q in range(1, nq - 1, 2)]
    return even, odd


def apply_nearest_cz_sequential(psi: np.ndarray, n_qubits: int) -> np.ndarray:
    """In-place sequential CZ(0,1) CZ(1,2) … CZ(n-2, n-1). Same unitary as even/odd."""
    nq = int(n_qubits)
    for q in range(nq - 1):
        apply_cz(psi, q, q + 1, nq)
    return psi


def apply_nearest_cz(psi: np.ndarray, n_qubits: int) -> np.ndarray:
    """In-place nearest-neighbour CZ: even commuting group, then odd.

    On 7 qubits: CZ(0,1), CZ(2,3), CZ(4,5) (q6 idle), then CZ(1,2), CZ(3,4),
    CZ(5,6) (q0 idle). Qubit 0 is MSB. Same unitary as the sequential staircase.
    """
    nq = int(n_qubits)
    even, odd = nearest_cz_even_odd_pairs(nq)
    for q1, q2 in even:
        apply_cz(psi, q1, q2, nq)
    for q1, q2 in odd:
        apply_cz(psi, q1, q2, nq)
    return psi


def unpack_qaoa(xvec: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Stacked layout: [γ_0 … γ_{p-1}, β_0 … β_{p-1}]."""
    x = np.asarray(xvec, dtype=float).reshape(-1)
    p = int(p)
    if x.size != 2 * p:
        raise ValueError(f"expected {2 * p} QAOA parameters for p={p}, got {x.size}")
    return x[:p].copy(), x[p:].copy()


def pack_qaoa(gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    g = np.asarray(gammas, dtype=float).reshape(-1)
    b = np.asarray(betas, dtype=float).reshape(-1)
    if g.size != b.size:
        raise ValueError("gammas and betas must have the same length")
    return np.concatenate([g, b])


def unpack_hea(xvec: np.ndarray, n_qubits: int, n_layers: int) -> np.ndarray:
    """θ[layer, qubit], layer-major, including the final Ry layer."""
    nq = int(n_qubits)
    nl = int(n_layers)
    x = np.asarray(xvec, dtype=float).reshape(-1)
    expected = n_hea_params(nq, nl)
    if x.size != expected:
        raise ValueError(f"expected {expected} HEA parameters for n={nq} L={nl}, got {x.size}")
    return x.reshape(nl + 1, nq)


def random_qaoa_params(p: int, rng: np.random.Generator) -> np.ndarray:
    """γ ~ U(0, 2π), β ~ U(0, π)."""
    p = int(p)
    gammas = rng.uniform(0.0, 2.0 * np.pi, size=p)
    betas = rng.uniform(0.0, np.pi, size=p)
    return pack_qaoa(gammas, betas)


def random_hea_params(n_qubits: int, n_layers: int, rng: np.random.Generator) -> np.ndarray:
    """Each R_y angle ~ U(0, 2π)."""
    return rng.uniform(0.0, 2.0 * np.pi, size=n_hea_params(n_qubits, n_layers))


def qaoa_statevector(
    gammas: Sequence[float],
    betas: Sequence[float],
    energies: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Noiseless QAOA from |+⟩^{⊗ n}. Do not start from |0⟩."""
    g = np.asarray(gammas, dtype=float).reshape(-1)
    b = np.asarray(betas, dtype=float).reshape(-1)
    if g.size != b.size:
        raise ValueError("gammas and betas must have the same length")
    e = np.asarray(energies, dtype=float).reshape(-1)
    dim = 1 << int(n_qubits)
    if e.size != dim:
        raise ValueError(f"energies length {e.size} does not match 2^{n_qubits}={dim}")
    psi = plus_state(n_qubits)
    for gamma, beta in zip(g, b):
        apply_diag_phase(psi, gamma, e)
        apply_mixer(psi, beta, n_qubits)
    return psi


def hea_statevector(
    thetas: np.ndarray,
    n_qubits: int,
    n_layers: int | None = None,
) -> np.ndarray:
    """Noiseless HEA from |0⟩^{⊗ n}: L × (Ry + even/odd NN-CZ) plus a final Ry layer."""
    nq = int(n_qubits)
    th = np.asarray(thetas, dtype=float)
    if th.ndim == 1:
        if n_layers is None:
            raise ValueError("n_layers is required when thetas is flat")
        th = unpack_hea(th, nq, n_layers)
    nl = int(th.shape[0] - 1)
    if n_layers is not None and int(n_layers) != nl:
        raise ValueError(f"thetas imply L={nl}, expected {n_layers}")
    if th.shape != (nl + 1, nq):
        raise ValueError(f"thetas shape {th.shape} does not match (L+1, n)=({nl + 1}, {nq})")
    psi = zero_state(nq)
    for layer in range(nl):
        for q in range(nq):
            apply_ry(psi, th[layer, q], q, nq)
        apply_nearest_cz(psi, nq)
    for q in range(nq):
        apply_ry(psi, th[nl, q], q, nq)
    return psi


def qaoa_unitary_matrix(
    gammas: Sequence[float],
    betas: Sequence[float],
    energies: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Full QAOA unitary on the computational basis (small n only)."""
    dim = 1 << int(n_qubits)
    cols = []
    e = np.asarray(energies, dtype=float).reshape(-1)
    g = np.asarray(gammas, dtype=float).reshape(-1)
    b = np.asarray(betas, dtype=float).reshape(-1)
    for idx in range(dim):
        psi = np.zeros(dim, dtype=complex)
        psi[idx] = 1.0
        for gamma, beta in zip(g, b):
            apply_diag_phase(psi, gamma, e)
            apply_mixer(psi, beta, n_qubits)
        cols.append(psi)
    return np.column_stack(cols)


def probabilities(psi: np.ndarray) -> np.ndarray:
    p = np.abs(np.asarray(psi, dtype=complex).reshape(-1)) ** 2
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total > 0.0:
        p = p / total
    return p


def ground_mask(energies: np.ndarray, atol: float = GROUND_ATOL) -> np.ndarray:
    e = np.asarray(energies, dtype=float).reshape(-1)
    return np.isclose(e, float(np.min(e)), atol=atol, rtol=0.0)


@dataclass
class HistogramEval:
    probs: np.ndarray
    most_likely_index: int
    most_likely_bitstring: str
    most_likely_prob: float
    energy_diag: float
    energy_expect: float
    p_ground: float
    gap_to_min: float
    rel_gap: float
    success: bool
    energy_min: float
    energy_max: float
    ground_bitstring: str
    cost: float
    eta: float | None = None


def evaluate_histogram(
    probs: np.ndarray,
    energies: np.ndarray,
    *,
    eta: float | None = None,
    atol: float = GROUND_ATOL,
) -> HistogramEval:
    raw = np.clip(np.asarray(probs, dtype=float).reshape(-1), 0.0, None)
    total = float(raw.sum())
    p = raw / total if total > 0.0 else raw
    e = np.asarray(energies, dtype=float).reshape(-1)
    if p.size != e.size:
        raise ValueError(f"probs length {p.size} != energies length {e.size}")
    nq = int(round(np.log2(e.size)))
    idx = int(np.argmax(p))
    bits = bits_from_qubit_index(idx, nq)
    emin = float(np.min(e))
    emax = float(np.max(e))
    spread = max(emax - emin, 1e-12)
    found = float(e[idx])
    mask = ground_mask(e, atol=atol)
    gs_idx = int(np.argmin(e))
    cost = (
        gibbs_objective(p, e, float(eta))
        if eta is not None
        else float(np.dot(p, e))
    )
    return HistogramEval(
        probs=p,
        most_likely_index=idx,
        most_likely_bitstring=bitstring_from_bits(bits),
        most_likely_prob=float(p[idx]),
        energy_diag=found,
        energy_expect=float(np.dot(p, e)),
        p_ground=float(np.dot(p, mask.astype(float))),
        gap_to_min=found - emin,
        rel_gap=(found - emin) / spread,
        success=bool(np.isclose(found, emin, atol=atol, rtol=0.0)),
        energy_min=emin,
        energy_max=emax,
        ground_bitstring=bitstring_from_bits(bits_from_qubit_index(gs_idx, nq)),
        cost=cost,
        eta=None if eta is None else float(eta),
    )


def qaoa_probs(xvec: np.ndarray, energies: np.ndarray, p: int, n_qubits: int) -> np.ndarray:
    gammas, betas = unpack_qaoa(xvec, p)
    return probabilities(qaoa_statevector(gammas, betas, energies, n_qubits))


def hea_probs(xvec: np.ndarray, n_qubits: int, n_layers: int) -> np.ndarray:
    return probabilities(hea_statevector(xvec, n_qubits, n_layers))


@dataclass
class QubitVQEResult:
    fun: float
    x: np.ndarray
    x0: np.ndarray
    nfev: int
    nit: int
    eval: HistogramEval
    eta_policy: str | None = None
    eta0: float | None = None
    eta: float | None = None
    eta_history: list[dict] = field(default_factory=list)
    n_eta_clamps: int = 0
    n_eta_fallbacks: int = 0
    message: str = ""
    success_opt: bool = True

    def as_score_dict(self) -> dict:
        ev = self.eval
        return {
            "cost": float(self.fun),
            "energy_diag": ev.energy_diag,
            "energy_physical": ev.energy_expect,
            "energy_expect": ev.energy_expect,
            "p_ground": ev.p_ground,
            "p_most_likely": ev.most_likely_prob,
            "most_likely_bitstring": ev.most_likely_bitstring,
            "ground_bitstring": ev.ground_bitstring,
            "energy_min": ev.energy_min,
            "energy_max": ev.energy_max,
            "gap_to_min": ev.gap_to_min,
            "rel_gap": ev.rel_gap,
            "success": ev.success,
            "nfev": int(self.nfev),
            "nit": int(self.nit),
            "eta": None if self.eta is None else float(self.eta),
            "eta0": None if self.eta0 is None else float(self.eta0),
        }


def _probs_factory(
    ansatz: str,
    *,
    p: int,
    n_qubits: int,
    n_layers: int,
    energies: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    kind = str(ansatz).lower()
    if kind == "qaoa":
        def qaoa_fn(x: np.ndarray) -> np.ndarray:
            return qaoa_probs(x, energies, p, n_qubits)

        return qaoa_fn
    if kind == "hea":
        def hea_fn(x: np.ndarray) -> np.ndarray:
            return hea_probs(x, n_qubits, n_layers)

        return hea_fn
    raise ValueError(f"unknown ansatz {ansatz!r}")


def optimize_qubit_ansatz(
    x0: np.ndarray,
    energies: np.ndarray,
    *,
    ansatz: str,
    objective: str = "gibbs",
    p: int = DEFAULT_QAOA_P,
    n_qubits: int = N_QUBITS,
    n_layers: int = DEFAULT_HEA_LAYERS,
    maxiter: int = 70,
    rng: np.random.Generator,
    a: float = 0.2,
    c: float = 0.15,
    A: float = 10.0,
    alpha: float = 0.602,
    gamma: float = 0.101,
    bounds: Sequence[tuple[float, float]] | None = None,
) -> QubitVQEResult:
    """SPSA on QAOA or HEA. Gibbs uses sampled_tail η, held fixed inside each pair."""
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    e = np.asarray(energies, dtype=float).reshape(-1)
    kind = str(objective).lower()
    if kind not in ("gibbs", "energy"):
        raise ValueError(f"unknown objective {objective!r}")
    probs_at = _probs_factory(ansatz, p=p, n_qubits=n_qubits, n_layers=n_layers, energies=e)
    policy: SampledTailEta | None = None
    eta0: float | None = None
    if kind == "gibbs":
        policy = SampledTailEta()
        st0 = policy.initialize(e, probs_at(x0))
        eta0 = float(st0.eta)

        def fun(x: np.ndarray) -> float:
            return gibbs_objective(probs_at(x), e, float(policy.eta))

        def before(k: int, x: np.ndarray) -> None:
            policy.maybe_update(k, int(maxiter), e, probs_at(x))

    else:
        def fun(x: np.ndarray) -> float:
            return float(np.dot(probs_at(x), e))

        def before(k: int, x: np.ndarray) -> None:
            del k, x

    opt: OptimizeResult = run_spsa(
        fun,
        x0,
        maxiter=int(maxiter),
        rng=rng,
        bounds=bounds,
        a=a,
        c=c,
        A=A,
        alpha=alpha,
        gamma=gamma,
        on_before_step=before if kind == "gibbs" else None,
    )
    eta_final = None if policy is None else float(policy.eta)
    ev = evaluate_histogram(probs_at(opt.x), e, eta=eta_final if kind == "gibbs" else None)
    snap = policy.snapshot() if policy is not None else None
    return QubitVQEResult(
        fun=float(opt.fun),
        x=np.asarray(opt.x, dtype=float),
        x0=x0,
        nfev=int(opt.nfev),
        nit=int(opt.nit),
        eval=ev,
        eta_policy=None if snap is None else str(snap["name"]),
        eta0=eta0,
        eta=eta_final,
        eta_history=[] if snap is None else list(snap["history"]),
        n_eta_clamps=0 if snap is None else int(snap["n_clamps"]),
        n_eta_fallbacks=0 if snap is None else int(snap["n_fallbacks"]),
        message=str(opt.message),
        success_opt=bool(opt.success),
    )


def default_nfocks() -> tuple[int, int]:
    return (int(DEFAULT_NFOCKS[0]), int(DEFAULT_NFOCKS[1]))
