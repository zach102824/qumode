"""Binary knapsack QUBO Hamiltonian and hybrid qubit-qumode mapping.

Implements Dutta et al. Eqs. (23)–(26): the 7-qubit diagonal Ising
Hamiltonian and its embedding into one qubit plus two qumodes with
Fock cutoffs L1 = L2 = 8.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import qutip as qt

N_PRIMARY = 4
N_AUX = 3
N_QUBITS = N_PRIMARY + N_AUX

DEFAULT_VALUES = np.array([2.0, 5.0, 7.0, 3.0], dtype=float)
DEFAULT_WEIGHTS = np.array([2.5, 3.0, 4.0, 3.5], dtype=float)
DEFAULT_CAPACITY = 7.0
DEFAULT_LAMBDA = 2.0
DEFAULT_NFOCKS = (8, 8)

TARGET_QNM = (0, 6, 0)
TARGET_BITSTRING = "0110000"
EXACT_GROUND_ENERGY = -12.0

# Eq. (25): identity, single-Z, and ZZ coefficients.
EQ25_IDENTITY = 41.75
EQ25_Z = {
    0: -14.0,
    1: -15.5,
    2: -20.5,
    3: -19.5,
    4: -6.0,
    5: -12.0,
    6: -24.0,
}
EQ25_ZZ = {
    (0, 1): 7.5,
    (0, 2): 10.0,
    (0, 3): 8.75,
    (0, 4): 2.5,
    (0, 5): 5.0,
    (0, 6): 10.0,
    (1, 2): 12.0,
    (1, 3): 10.5,
    (1, 4): 3.0,
    (1, 5): 6.0,
    (1, 6): 12.0,
    (2, 3): 14.0,
    (2, 4): 4.0,
    (2, 5): 8.0,
    (2, 6): 16.0,
    (3, 4): 3.5,
    (3, 5): 7.0,
    (3, 6): 14.0,
    (4, 5): 2.0,
    (4, 6): 4.0,
    (5, 6): 8.0,
}


class BKPInstance:
    def __init__(
        self,
        values: Sequence[float] | None = None,
        weights: Sequence[float] | None = None,
        capacity: float = DEFAULT_CAPACITY,
        penalty: float = DEFAULT_LAMBDA,
    ) -> None:
        self.values = np.asarray(DEFAULT_VALUES if values is None else values, dtype=float)
        self.weights = np.asarray(DEFAULT_WEIGHTS if weights is None else weights, dtype=float)
        self.capacity = float(capacity)
        self.penalty = float(penalty)
        if self.values.shape != (N_PRIMARY,) or self.weights.shape != (N_PRIMARY,):
            raise ValueError("This module implements the N0=4 paper instance.")


def default_instance() -> BKPInstance:
    return BKPInstance()


def bkp_instance_as_dict(instance: BKPInstance) -> dict:
    return {
        "values": [float(v) for v in instance.values],
        "weights": [float(v) for v in instance.weights],
        "capacity": float(instance.capacity),
        "penalty": float(instance.penalty),
    }


def _choice_integers(rng: np.random.Generator, lo: int, hi: int, n: int) -> np.ndarray:
    if int(hi) < int(lo):
        raise ValueError(f"empty integer grid [{lo}, {hi}]")
    return rng.integers(int(lo), int(hi) + 1, size=int(n)).astype(float)


def sample_bkp_instance(
    rng: np.random.Generator,
    *,
    penalty: float = DEFAULT_LAMBDA,
    capacity_frac: float = 0.55,
    value_lo: float = 1.0,
    value_hi: float = 10.0,
    weight_lo: float = 1.0,
    weight_hi: float = 5.0,
    anti_correlated: bool = False,
) -> BKPInstance:
    """Random 4-item / 3-slack knapsack in the paper encoding.

    Integer values/weights/capacity so the slack bits can cancel leftover
    exactly. ``penalty`` is raised if needed so the QUBO ground state is a
    feasible packing.
    """
    for _ in range(32):
        values = _choice_integers(rng, int(round(value_lo)), int(round(value_hi)), N_PRIMARY)
        weights = _choice_integers(rng, int(round(weight_lo)), int(round(weight_hi)), N_PRIMARY)
        if anti_correlated:
            values = np.sort(values)[::-1]
            weights = np.sort(weights)
        wsum = float(np.sum(weights))
        wmax = float(np.max(weights))
        wmin = float(np.min(weights))
        c_lo = wmax
        c_hi = min(7.0, max(wsum - wmin, wmax))
        if c_hi < c_lo:
            c_hi = min(7.0, wsum)
            c_lo = min(c_lo, c_hi)
        cap = c_lo + float(capacity_frac) * (c_hi - c_lo)
        cap = float(min(max(round(cap), 1.0), 7.0))
        inst = BKPInstance(values, weights, cap, float(penalty))
        need = min_faithful_penalty(inst)
        if math.isfinite(need):
            return with_faithful_penalty(inst, floor=float(penalty))
    raise RuntimeError("failed to sample a slack-faithful knapsack instance")


def knapsack_packing_stats(
    bits: Sequence[int] | np.ndarray,
    instance: BKPInstance,
) -> dict:
    x = np.asarray(bits, dtype=float).reshape(N_QUBITS)
    value = float(np.dot(instance.values, x[:N_PRIMARY]))
    weight = float(np.dot(instance.weights, x[:N_PRIMARY]))
    slack = float(x[4] + 2.0 * x[5] + 4.0 * x[6])
    residual = float(instance.capacity - weight - slack)
    return {
        "value": value,
        "weight": weight,
        "slack": slack,
        "residual": residual,
        "feasible": bool(weight <= instance.capacity + 1e-8),
    }


def _primary_subset(mask: int) -> np.ndarray:
    bits = np.zeros(N_PRIMARY, dtype=int)
    for i in range(N_PRIMARY):
        bits[i] = (int(mask) >> i) & 1
    return bits


def best_feasible_subset(instance: BKPInstance) -> tuple[np.ndarray, float, float]:
    """Highest-value 4-bit packing with weight ≤ capacity (empty allowed)."""
    best_bits = np.zeros(N_PRIMARY, dtype=int)
    best_val = 0.0
    best_w = 0.0
    for mask in range(1 << N_PRIMARY):
        bits = _primary_subset(mask)
        weight = float(np.dot(instance.weights, bits))
        value = float(np.dot(instance.values, bits))
        if weight > instance.capacity + 1e-8:
            continue
        better = value > best_val + 1e-12
        tie_lighter = abs(value - best_val) <= 1e-12 and weight < best_w - 1e-12
        if better or tie_lighter:
            best_bits, best_val, best_w = bits, value, weight
    return best_bits, best_val, best_w


def _best_slack_sqerr(leftover: float) -> float:
    return min((float(leftover) - s) ** 2 for s in range(8))


def min_faithful_penalty(instance: BKPInstance) -> float:
    """Smallest λ so some feasible packing beats every overweight QUBO energy.

    Compares actual slack-minimized residuals, not just (weight − C)². Returns
    +inf if slack quantization makes the encodings inseparable.
    """
    feas: list[tuple[float, float]] = []
    infeas: list[tuple[float, float]] = []
    for mask in range(1 << N_PRIMARY):
        bits = _primary_subset(mask)
        weight = float(np.dot(instance.weights, bits))
        value = float(np.dot(instance.values, bits))
        err = _best_slack_sqerr(instance.capacity - weight)
        if weight <= instance.capacity + 1e-8:
            feas.append((value, err))
        else:
            infeas.append((value, err))
    if not feas:
        return 0.0
    feas.sort(key=lambda rec: (rec[1], -rec[0]))
    v_f, err_f = feas[0]
    need = 0.0
    for v_i, err_i in infeas:
        denom = err_i - err_f
        dv = v_i - v_f
        if denom <= 1e-12:
            if dv > 1e-12:
                return float("inf")
            continue
        need = max(need, dv / denom)
    return float(need)


def with_faithful_penalty(
    instance: BKPInstance,
    *,
    floor: float | None = None,
    margin: float = 1.05,
) -> BKPInstance:
    """Raise λ if needed so the QUBO ground state is a feasible knapsack packing."""
    lam_floor = DEFAULT_LAMBDA if floor is None else float(floor)
    need = min_faithful_penalty(instance)
    if not math.isfinite(need):
        raise ValueError("no finite penalty separates feasible and overweight packings")
    lam = max(lam_floor, float(margin) * need)
    return BKPInstance(instance.values, instance.weights, instance.capacity, lam)


def knapsack_sweep_instance(index: int, ham_seed: int = 7000) -> tuple[str, BKPInstance]:
    """Ten knapsack-like instances: paper BKP plus penalty/capacity/item variants."""
    paper = default_instance()
    idx = int(index) % 10
    if idx == 0:
        return "paper", paper
    if idx == 1:
        return "paper_lambda4", BKPInstance(paper.values, paper.weights, paper.capacity, 4.0)
    if idx == 2:
        return "paper_lambda1", BKPInstance(paper.values, paper.weights, paper.capacity, 1.0)
    if idx == 3:
        tight = BKPInstance(paper.values, paper.weights, 5.5, paper.penalty)
        return "paper_tight_C55", with_faithful_penalty(tight)
    if idx == 4:
        mid = BKPInstance(paper.values, paper.weights, 6.5, paper.penalty)
        return "paper_C65", with_faithful_penalty(mid)
    rng = np.random.default_rng(int(ham_seed) + 1000 + idx)
    if idx == 5:
        return "random", sample_bkp_instance(rng)
    if idx == 6:
        return "random_lambda3", sample_bkp_instance(rng, penalty=3.0)
    if idx == 7:
        return "random_tight", sample_bkp_instance(rng, capacity_frac=0.35)
    if idx == 8:
        return "high_values", sample_bkp_instance(
            rng, value_lo=4.0, value_hi=12.0, weight_lo=1.0, weight_hi=4.5
        )
    return "anti_correlated", sample_bkp_instance(rng, anti_correlated=True)


def qubo_energy(
    bits: Sequence[int] | np.ndarray,
    instance: BKPInstance | None = None,
) -> float:
    """Evaluate Eq. (24) on a length-7 binary vector."""
    inst = instance or default_instance()
    x = np.asarray(bits, dtype=float).reshape(N_QUBITS)
    value = float(np.dot(inst.values, x[:N_PRIMARY]))
    weight = float(np.dot(inst.weights, x[:N_PRIMARY]))
    slack = float(x[4] + 2.0 * x[5] + 4.0 * x[6])
    return -value + inst.penalty * (inst.capacity - weight - slack) ** 2


def bits_from_qnm_partitioned(
    q: int,
    n: int,
    m: int,
    n_nbits: int,
    m_nbits: int,
) -> np.ndarray:
    """Decode |q, n, m> with MSB-first binary on each Fock register."""
    bits = [int(q)]
    for k in range(int(n_nbits) - 1, -1, -1):
        bits.append((int(n) >> k) & 1)
    for k in range(int(m_nbits) - 1, -1, -1):
        bits.append((int(m) >> k) & 1)
    return np.asarray(bits, dtype=int)


def bits_from_qnm(q: int, n: int, m: int, partition: tuple[int, int, int] = (1, 3, 3)) -> np.ndarray:
    """Decode |q, n, m> using MSB-first binary on each Fock register.

    Default partition (1, 3, 3) matches Eq. (26): |0,1,1,0> ⊗ |0,0,0> ↔ |0, 6, 0>.
    The multi-constraint instance uses (1, 2, 3): |1,0,0> ⊗ |1,0,0> ↔ |1, 0, 4>.
    """
    _, n_nbits, m_nbits = partition
    return bits_from_qnm_partitioned(q, n, m, n_nbits, m_nbits)


def qnm_from_bits(
    bits: Sequence[int],
    partition: tuple[int, int, int] = (1, 3, 3),
) -> tuple[int, int, int]:
    x = np.asarray(bits, dtype=int).reshape(-1)
    _, n_nbits, m_nbits = partition
    expected = 1 + n_nbits + m_nbits
    if x.size != expected:
        raise ValueError(f"Expected {expected} bits for partition {partition}, got {x.size}.")
    q = int(x[0])
    n = 0
    for b in x[1 : 1 + n_nbits]:
        n = (n << 1) | int(b)
    m = 0
    for b in x[1 + n_nbits :]:
        m = (m << 1) | int(b)
    return q, n, m


def bitstring_from_bits(bits: Sequence[int]) -> str:
    return "".join(str(int(b)) for b in np.asarray(bits).reshape(-1))


def bits_from_qubit_index(index: int, n_qubits: int = N_QUBITS) -> np.ndarray:
    """QuTiP tensor order: qubit 0 is the slowest index (MSB)."""
    x = np.zeros(n_qubits, dtype=int)
    for k in range(n_qubits):
        x[k] = (index >> (n_qubits - 1 - k)) & 1
    return x


def qubit_index_from_bits(bits: Sequence[int]) -> int:
    x = np.asarray(bits, dtype=int).reshape(N_QUBITS)
    idx = 0
    for b in x:
        idx = (idx << 1) | int(b)
    return idx


def hybrid_index(q: int, n: int, m: int, nfocks: Sequence[int] = DEFAULT_NFOCKS) -> int:
    _, l2 = nfocks
    return int(q) * (nfocks[0] * l2) + int(n) * l2 + int(m)


def qnm_from_hybrid_index(index: int, nfocks: Sequence[int] = DEFAULT_NFOCKS) -> tuple[int, int, int]:
    l1, l2 = nfocks
    q, rem = divmod(int(index), l1 * l2)
    n, m = divmod(rem, l2)
    return q, n, m


def _pauli_z(n_qubits: int, *sites: int) -> qt.Qobj:
    ops = [qt.sigmaz() if i in sites else qt.qeye(2) for i in range(n_qubits)]
    return qt.tensor(*ops)


def qubit_hamiltonian_eq25() -> qt.Qobj:
    """Construct HQ exactly from the published Pauli coefficients in Eq. (25)."""
    ident = qt.tensor(*[qt.qeye(2)] * N_QUBITS)
    h = EQ25_IDENTITY * ident
    for site, coeff in EQ25_Z.items():
        h = h + coeff * _pauli_z(N_QUBITS, site)
    for (i, j), coeff in EQ25_ZZ.items():
        h = h + coeff * _pauli_z(N_QUBITS, i, j)
    return h


def qubit_hamiltonian_from_qubo(instance: BKPInstance | None = None) -> qt.Qobj:
    """Diagonal 7-qubit Hamiltonian obtained by evaluating Eq. (24) on each bitstring."""
    inst = instance or default_instance()
    dim = 2**N_QUBITS
    diag = np.empty(dim, dtype=float)
    for idx in range(dim):
        diag[idx] = qubo_energy(bits_from_qubit_index(idx), inst)
    return qt.Qobj(np.diag(diag), dims=[[2] * N_QUBITS, [2] * N_QUBITS])


def hybrid_energy_tensor(
    nfocks: Sequence[int] = DEFAULT_NFOCKS,
    instance: BKPInstance | None = None,
) -> np.ndarray:
    """C[q, n, m] = QUBO energy of the decoded bitstring, C-ordered with m fastest."""
    inst = instance or default_instance()
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    coeffs = np.empty((2, l1, l2), dtype=float)
    for q in range(2):
        for n in range(l1):
            for m in range(l2):
                coeffs[q, n, m] = qubo_energy(bits_from_qnm(q, n, m), inst)
    return coeffs


def hybrid_hamiltonian(
    nfocks: Sequence[int] = DEFAULT_NFOCKS,
    instance: BKPInstance | None = None,
) -> qt.Qobj:
    """Diagonal operator on |q> ⊗ |n> ⊗ |m> with dims [[2, L1, L2], [2, L1, L2]]."""
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    diag = hybrid_energy_tensor((l1, l2), instance).reshape(-1)
    return qt.Qobj(np.diag(diag), dims=[[2, l1, l2], [2, l1, l2]])


def diagonal_hybrid_hamiltonian(energy_tensor: np.ndarray) -> qt.Qobj:
    """Build a diagonal hybrid Hamiltonian from a (2, L1, L2) energy tensor."""
    e = np.asarray(energy_tensor, dtype=float)
    if e.ndim != 3 or e.shape[0] != 2:
        raise ValueError(f"Expected energy tensor of shape (2, L1, L2), got {e.shape}.")
    l1, l2 = int(e.shape[1]), int(e.shape[2])
    return qt.Qobj(np.diag(e.reshape(-1)), dims=[[2, l1, l2], [2, l1, l2]])


def _z_string_eigenvalue(bits: np.ndarray, sites: Sequence[int]) -> float:
    """Eigenvalue of ∏_{i∈S} Z_i on a computational bitstring."""
    val = 1.0
    for i in sites:
        val *= 1.0 - 2.0 * float(bits[int(i)])
    return val


def pauli_z_label(sites: Sequence[int]) -> str:
    if not sites:
        return "I"
    return "".join(f"Z{int(i)}" for i in sites)


def energy_from_z_terms(
    bits: Sequence[int] | np.ndarray,
    terms: Sequence[tuple[tuple[int, ...], float]],
    identity: float = 0.0,
) -> float:
    x = np.asarray(bits, dtype=int).reshape(-1)
    energy = float(identity)
    for sites, coeff in terms:
        energy += float(coeff) * _z_string_eigenvalue(x, sites)
    return energy


def random_diag_ising_terms(
    n_qubits: int = N_QUBITS,
    rng: np.random.Generator | None = None,
    *,
    n_body: dict[int, int] | None = None,
    scale: float = 1.0,
) -> list[tuple[tuple[int, ...], float]]:
    """Sparse random diagonal Ising/PUBO: H = Σ c_S ∏_{i∈S} Z_i.

    ``n_body[k]`` is how many distinct weight-k Z-strings to sample.
    k=1,2 is QUBO/Ising; k≥3 is PUBO (e.g. Z1 Z5 Z6).
    """
    from itertools import combinations

    rng = rng or np.random.default_rng()
    counts = n_body if n_body is not None else {1: n_qubits, 2: min(12, n_qubits * (n_qubits - 1) // 2)}
    terms: list[tuple[tuple[int, ...], float]] = []
    seen: set[tuple[int, ...]] = set()
    for k, count in sorted((int(kk), int(vv)) for kk, vv in counts.items()):
        if k < 1 or count <= 0:
            continue
        if k > n_qubits:
            raise ValueError(f"body order {k} exceeds n_qubits={n_qubits}")
        pool = list(combinations(range(int(n_qubits)), k))
        take = min(int(count), len(pool))
        chosen = rng.choice(len(pool), size=take, replace=False)
        for idx in np.atleast_1d(chosen):
            sites = tuple(int(i) for i in pool[int(idx)])
            if sites in seen:
                continue
            seen.add(sites)
            terms.append((sites, float(rng.normal(0.0, scale))))
    if not terms:
        raise ValueError("n_body produced an empty Hamiltonian")
    return terms


def z_terms_as_list(terms: Sequence[tuple[tuple[int, ...], float]]) -> list[dict]:
    return [{"sites": [int(i) for i in sites], "coeff": float(coeff)} for sites, coeff in terms]


def rms_normalize_z_terms(
    terms: Sequence[tuple[tuple[int, ...], float]],
) -> list[tuple[tuple[int, ...], float]]:
    coeffs = np.asarray([float(c) for _, c in terms], dtype=float)
    if coeffs.size == 0:
        return []
    rms = float(np.sqrt(np.mean(coeffs * coeffs)))
    if rms < 1e-12:
        return [(tuple(s), float(c)) for s, c in terms]
    return [(tuple(s), float(c) / rms) for s, c in terms]


def knapsack_benchmark_instance(index: int, ham_seed: int = 8000) -> BKPInstance:
    """Paper encoding (4 items + 3 slack), random values / weights / capacity."""
    rng = np.random.default_rng(int(ham_seed) + 10_000 + int(index))
    return sample_bkp_instance(rng)


def ising_benchmark_terms(index: int, ham_seed: int = 8000) -> list[tuple[tuple[int, ...], float]]:
    """7-spin diagonal Ising: all local Z fields plus 12 random ZZ couplings, RMS-normalized."""
    rng = np.random.default_rng(int(ham_seed) + 20_000 + int(index))
    terms = random_diag_ising_terms(N_QUBITS, rng, n_body={1: N_QUBITS, 2: 12}, scale=1.0)
    return rms_normalize_z_terms(terms)


def energy_tensor_from_z_terms(
    terms: Sequence[tuple[tuple[int, ...], float]],
    nfocks: Sequence[int] = DEFAULT_NFOCKS,
    partition: tuple[int, int, int] = (1, 3, 3),
    identity: float = 0.0,
) -> np.ndarray:
    """Hybrid |q,n,m> diagonal from a 7-bit Z-string expansion."""
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    coeffs = np.empty((2, l1, l2), dtype=float)
    for q in range(2):
        for n in range(l1):
            for m in range(l2):
                bits = bits_from_qnm(q, n, m, partition)
                coeffs[q, n, m] = energy_from_z_terms(bits, terms, identity)
    tilt = 1e-10 * np.arange(coeffs.size, dtype=float).reshape(coeffs.shape)
    return coeffs + tilt


def max_pauli_weight(terms: Sequence[tuple[tuple[int, ...], float]]) -> int:
    return max((len(sites) for sites, _ in terms), default=0)


def ground_qnm_from_tensor(energy_tensor: np.ndarray) -> tuple[int, int, int]:
    e = np.asarray(energy_tensor, dtype=float)
    return tuple(int(v) for v in np.unravel_index(int(np.argmin(e)), e.shape))


def energy_spectrum_stats(energy_tensor: np.ndarray) -> dict:
    """Gap, range, and ground-state degeneracy of a diagonal hybrid Hamiltonian."""
    e = np.asarray(energy_tensor, dtype=float)
    flat = e.reshape(-1)
    emin = float(np.min(flat))
    emax = float(np.max(flat))
    spread = float(emax - emin)
    uniq = np.unique(np.round(flat, decimals=10))
    gap = float(uniq[1] - uniq[0]) if uniq.size > 1 else 0.0
    n_ground = int(np.sum(np.isclose(flat, emin, atol=1e-8, rtol=0.0)))
    gs = ground_qnm_from_tensor(e)
    return {
        "energy_min": emin,
        "energy_max": emax,
        "spread": spread,
        "gap": gap,
        "n_ground": n_ground,
        "ground_qnm": [int(v) for v in gs],
    }


def tabu_penalized_energy_tensor(
    forbidden_qnm: Sequence[Sequence[int]],
    penalty: float = 50.0,
    nfocks: Sequence[int] = DEFAULT_NFOCKS,
    instance: BKPInstance | None = None,
) -> np.ndarray:
    """Raise the diagonal energy of previously found |q,n,m> attractors."""
    tensor = hybrid_energy_tensor(nfocks, instance).copy()
    lq, l1, l2 = tensor.shape
    for qnm in forbidden_qnm:
        q, n, m = (int(v) for v in qnm)
        if not (0 <= q < lq and 0 <= n < l1 and 0 <= m < l2):
            raise ValueError(f"forbidden qnm {(q, n, m)} is outside {(lq, l1, l2)}.")
        tensor[q, n, m] += float(penalty)
    return tensor


def tabu_penalized_hamiltonian(
    forbidden_qnm: Sequence[Sequence[int]],
    penalty: float = 50.0,
    nfocks: Sequence[int] = DEFAULT_NFOCKS,
    instance: BKPInstance | None = None,
) -> qt.Qobj:
    """Diagonal hybrid Hamiltonian with tabu penalties on listed basis states."""
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    diag = tabu_penalized_energy_tensor(forbidden_qnm, penalty, (l1, l2), instance).reshape(-1)
    return qt.Qobj(np.diag(diag), dims=[[2, l1, l2], [2, l1, l2]])


def extract_ising_coefficients(h: qt.Qobj, n_qubits: int = N_QUBITS) -> dict:
    """Return identity, Z, and ZZ coefficients via Hilbert-Schmidt inner products."""
    dim = 2**n_qubits
    ident = qt.tensor(*[qt.qeye(2)] * n_qubits)
    out: dict = {"I": float(np.real((h.dag() * ident).tr() / dim))}
    z_coeff = {}
    for i in range(n_qubits):
        zi = _pauli_z(n_qubits, i)
        z_coeff[i] = float(np.real((h.dag() * zi).tr() / dim))
    zz_coeff = {}
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            zij = _pauli_z(n_qubits, i, j)
            zz_coeff[(i, j)] = float(np.real((h.dag() * zij).tr() / dim))
    out["Z"] = z_coeff
    out["ZZ"] = zz_coeff
    return out


def exact_ground(h: qt.Qobj) -> tuple[float, qt.Qobj]:
    evals, evecs = h.eigenstates()
    return float(np.real(evals[0])), evecs[0]


def computational_label(state: qt.Qobj, n_qubits: int = N_QUBITS) -> str | None:
    """Return the bitstring if `state` is a computational-basis vector."""
    vec = np.asarray(state.full()).reshape(-1)
    if not np.isclose(np.linalg.norm(vec), 1.0):
        return None
    idx = int(np.argmax(np.abs(vec)))
    if not np.isclose(np.abs(vec[idx]), 1.0):
        return None
    return bitstring_from_bits(bits_from_qubit_index(idx, n_qubits))


# ---------------------------------------------------------------------------
# Multiple-constraint instance, Eqs. (29)–(32)
# ---------------------------------------------------------------------------

MC_N_QUBITS = 6
MC_NFOCKS = (4, 8)
MC_PARTITION = (1, 2, 3)
MC_NDEPTH = 10
MC_TARGET_QNM = (1, 0, 4)
MC_TARGET_BITSTRING = "100100"
MC_EXACT_GROUND_ENERGY = 1.0
MC_LAMBDAS = (5.0, 5.0, 5.0)

EQ31_IDENTITY = 32.0
EQ31_Z = {
    0: -10.5,
    1: -11.0,
    2: -5.5,
    3: -5.0,
    4: -10.0,
}
EQ31_ZZ = {
    (0, 1): 15.0,
    (0, 2): 7.5,
    (0, 3): 5.0,
    (0, 4): 10.0,
    (0, 5): -2.5,
    (1, 2): 7.5,
    (1, 3): 5.0,
    (1, 4): 10.0,
    (1, 5): -2.5,
    (2, 3): 2.5,
    (2, 4): 5.0,
    (2, 5): -2.5,
    (3, 4): 5.0,
}


def mc_qubo_energy(bits: Sequence[int] | np.ndarray, lambdas: Sequence[float] = MC_LAMBDAS) -> float:
    """Eq. (29)–(30) with the notebook's linear term x0 + 2 x1 + x2."""
    x = np.asarray(bits, dtype=float).reshape(MC_N_QUBITS)
    x0, x1, x2, x3, x4, x5 = x
    lam1, lam2, lam3 = (float(v) for v in lambdas)
    objective = x0 + 2.0 * x1 + x2
    p1 = lam1 * (1.0 - x0 - x1) ** 2
    p2 = lam2 * (3.0 - (2.0 * x0 + 2.0 * x1 + x2) - (x3 + 2.0 * x4)) ** 2
    p3 = lam3 * ((x0 + x1 + x2) - x5 - 1.0) ** 2
    return float(objective + p1 + p2 + p3)


def qubit_hamiltonian_eq31() -> qt.Qobj:
    ident = qt.tensor(*[qt.qeye(2)] * MC_N_QUBITS)
    h = EQ31_IDENTITY * ident
    for site, coeff in EQ31_Z.items():
        h = h + coeff * _pauli_z(MC_N_QUBITS, site)
    for (i, j), coeff in EQ31_ZZ.items():
        h = h + coeff * _pauli_z(MC_N_QUBITS, i, j)
    return h


def qubit_hamiltonian_from_mc_qubo(lambdas: Sequence[float] = MC_LAMBDAS) -> qt.Qobj:
    dim = 2**MC_N_QUBITS
    diag = np.empty(dim, dtype=float)
    for idx in range(dim):
        diag[idx] = mc_qubo_energy(bits_from_qubit_index(idx, MC_N_QUBITS), lambdas)
    return qt.Qobj(np.diag(diag), dims=[[2] * MC_N_QUBITS, [2] * MC_N_QUBITS])


def mc_hybrid_energy_tensor(nfocks: Sequence[int] = MC_NFOCKS) -> np.ndarray:
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    coeffs = np.empty((2, l1, l2), dtype=float)
    for q in range(2):
        for n in range(l1):
            for m in range(l2):
                coeffs[q, n, m] = mc_qubo_energy(bits_from_qnm(q, n, m, MC_PARTITION))
    return coeffs


def mc_hybrid_hamiltonian(nfocks: Sequence[int] = MC_NFOCKS) -> qt.Qobj:
    l1, l2 = int(nfocks[0]), int(nfocks[1])
    diag = mc_hybrid_energy_tensor((l1, l2)).reshape(-1)
    return qt.Qobj(np.diag(diag), dims=[[2, l1, l2], [2, l1, l2]])
