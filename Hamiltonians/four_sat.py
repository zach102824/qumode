"""Generate a 7-qubit 4-SAT Hamiltonian with a nontrivial unique ground state.

Each clause C contributes the diagonal projector onto the unique assignment
that falsifies C,

    H_C = prod_{positive x_i} (I + Z_i)/2  *  prod_{negative ~x_j} (I - Z_j)/2,

and H = sum_C H_C counts unsatisfied clauses. Computational bits use
x = 0, 1 with Z|0> = +1 and Z|1> = -1.

Random 4-CNF at 12–20 clauses on 7 variables is typically highly degenerate.
This generator plants a locally rigid satisfying assignment, then greedily
adds compatible 4-clauses until that assignment is the unique solution, so
the ground-state bitstring is not a trivial pattern and is not found by
greedy single-bit descent from most start states.

Run this file directly after editing the hyperparameters below. The instance
is written as a compressed ``.npz`` file in the ``four_sat/`` subdirectory.
The Pauli arrays ``sites``, ``orders``, ``coefficients`` match the mixed
p-spin layout (plus a scalar ``identity`` shift).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_HAMILTONIANS = 20
NUM_SPINS = 7
CLAUSE_WIDTH = 4
MIN_CLAUSES = 12
MAX_CLAUSES = 20
TARGET_CLAUSES = 16
SEARCH_TRIALS = 4000
RANDOM_SEED = 11000
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "four_sat"
FILE_PREFIX = "four_sat"

_TRIVIAL_BITSTRINGS = {
    "0000000",
    "1111111",
    "0101010",
    "1010101",
    "0001111",
    "1110000",
    "0011100",
    "1100011",
}


def _validate_hyperparameters() -> None:
    if NUM_HAMILTONIANS < 1 or NUM_SPINS < 1:
        raise ValueError("NUM_HAMILTONIANS and NUM_SPINS must be positive.")
    if not 1 <= CLAUSE_WIDTH <= NUM_SPINS:
        raise ValueError("CLAUSE_WIDTH must be between 1 and NUM_SPINS.")
    if not 1 <= MIN_CLAUSES <= TARGET_CLAUSES <= MAX_CLAUSES:
        raise ValueError("Clause counts must satisfy 1 <= MIN <= TARGET <= MAX.")
    if SEARCH_TRIALS < 1:
        raise ValueError("SEARCH_TRIALS must be positive.")


def _bits_from_index(index: int) -> np.ndarray:
    bits = np.zeros(NUM_SPINS, dtype=np.int64)
    for i in range(NUM_SPINS):
        bits[i] = (int(index) >> (NUM_SPINS - 1 - i)) & 1
    return bits


def _index_from_bits(bits: np.ndarray) -> int:
    idx = 0
    for bit in np.asarray(bits, dtype=int).reshape(NUM_SPINS):
        idx = (idx << 1) | int(bit)
    return int(idx)


def _bitstring(bits: np.ndarray) -> str:
    return "".join(str(int(b)) for b in np.asarray(bits).reshape(NUM_SPINS))


def clause_violated(
    bits: np.ndarray,
    variables: np.ndarray,
    polarities: np.ndarray,
) -> bool:
    pattern = np.where(np.asarray(polarities, dtype=int) > 0, 0, 1)
    return bool(np.all(np.asarray(bits, dtype=int)[np.asarray(variables, dtype=int)] == pattern))


def unsat_counts(clauses: np.ndarray, polarities: np.ndarray) -> np.ndarray:
    """Number of violated clauses for every computational basis state."""
    n_states = 1 << NUM_SPINS
    energies = np.zeros(n_states, dtype=np.int64)
    all_bits = np.array([_bits_from_index(idx) for idx in range(n_states)], dtype=np.int64)
    for variables, poles in zip(clauses, polarities, strict=True):
        pattern = np.where(np.asarray(poles, dtype=int) > 0, 0, 1)
        energies += np.all(all_bits[:, np.asarray(variables, dtype=int)] == pattern, axis=1)
    return energies


def _clause_key(variables: np.ndarray, polarities: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (tuple(int(v) for v in variables), tuple(int(p) for p in polarities))


def _polarity_for_literal(bit: int, *, want_true: bool) -> int:
    """Return ±1 so the literal is true (or false) on this bit value."""
    if want_true:
        return 1 if int(bit) == 1 else -1
    return 1 if int(bit) == 0 else -1


def _rigidity_clauses(planted: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One minimally satisfied clause per variable so each single bit-flip costs energy 1."""
    clauses: list[np.ndarray] = []
    polarities: list[np.ndarray] = []
    others = np.arange(NUM_SPINS)
    for site in range(NUM_SPINS):
        pool = others[others != site]
        companions = rng.choice(pool, size=CLAUSE_WIDTH - 1, replace=False)
        variables = np.sort(np.concatenate(([site], companions))).astype(np.int64)
        poles = np.empty(CLAUSE_WIDTH, dtype=np.int64)
        for j, var in enumerate(variables):
            poles[j] = _polarity_for_literal(int(planted[var]), want_true=(int(var) == site))
        clauses.append(variables)
        polarities.append(poles)
    return np.stack(clauses), np.stack(polarities)


def _violating_mask(variables: np.ndarray, polarities: np.ndarray) -> int:
    pattern = np.where(np.asarray(polarities, dtype=int) > 0, 0, 1)
    mask = 0
    for idx in range(1 << NUM_SPINS):
        bits = _bits_from_index(idx)
        if np.all(bits[variables] == pattern):
            mask |= 1 << idx
    return int(mask)


def _full_clause_catalog() -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Every 4-clause together with the bit-mask of assignments it forbids."""
    out: list[tuple[np.ndarray, np.ndarray, int]] = []
    for variables in combinations(range(NUM_SPINS), CLAUSE_WIDTH):
        vars_arr = np.asarray(variables, dtype=np.int64)
        for mask in range(1 << CLAUSE_WIDTH):
            poles = np.array(
                [1 if ((mask >> j) & 1) == 0 else -1 for j in range(CLAUSE_WIDTH)],
                dtype=np.int64,
            )
            out.append((vars_arr, poles, _violating_mask(vars_arr, poles)))
    return out


def _compatible_catalog(
    planted_idx: int,
    catalog: list[tuple[np.ndarray, np.ndarray, int]],
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    planted_bit = 1 << int(planted_idx)
    return [(v, p, m) for v, p, m in catalog if (m & planted_bit) == 0]


def _greedy_unique_cover(
    planted: np.ndarray,
    clauses: np.ndarray,
    polarities: np.ndarray,
    catalog: list[tuple[np.ndarray, np.ndarray, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Add compatible clauses until the planted assignment is the unique SAT solution."""
    planted_idx = _index_from_bits(planted)
    used = {_clause_key(v, p) for v, p in zip(clauses, polarities, strict=True)}
    remaining = 0
    energies = unsat_counts(clauses, polarities)
    for idx, energy in enumerate(energies):
        if energy == 0 and idx != planted_idx:
            remaining |= 1 << idx
    if remaining == 0:
        return clauses, polarities

    cur_clauses = [np.asarray(row).copy() for row in clauses]
    cur_poles = [np.asarray(row).copy() for row in polarities]

    while remaining and len(cur_clauses) < MAX_CLAUSES:
        best_i = -1
        best_cover = 0
        for i, (variables, poles, vmask) in enumerate(catalog):
            key = _clause_key(variables, poles)
            if key in used:
                continue
            cover = int(vmask & remaining).bit_count()
            if cover > best_cover:
                best_cover = cover
                best_i = i
        if best_i < 0 or best_cover == 0:
            break
        variables, poles, vmask = catalog[best_i]
        used.add(_clause_key(variables, poles))
        cur_clauses.append(variables)
        cur_poles.append(poles)
        remaining &= ~vmask & ((1 << (1 << NUM_SPINS)) - 1)

    if remaining:
        return None
    return np.stack(cur_clauses), np.stack(cur_poles)


def _pad_to_target(
    planted: np.ndarray,
    clauses: np.ndarray,
    polarities: np.ndarray,
    catalog: list[tuple[np.ndarray, np.ndarray, int]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Add extra GS-compatible clauses up to TARGET_CLAUSES without destroying uniqueness."""
    if len(clauses) >= TARGET_CLAUSES:
        return clauses, polarities
    used = {_clause_key(v, p) for v, p in zip(clauses, polarities, strict=True)}
    unused = [(v, p) for v, p, _m in catalog if _clause_key(v, p) not in used]
    unused = [unused[i] for i in rng.permutation(len(unused))]
    cur_c = [np.asarray(row).copy() for row in clauses]
    cur_p = [np.asarray(row).copy() for row in polarities]
    planted_idx = _index_from_bits(planted)
    for variables, poles in unused:
        if len(cur_c) >= TARGET_CLAUSES:
            break
        trial_c = np.stack(cur_c + [variables])
        trial_p = np.stack(cur_p + [poles])
        energies = unsat_counts(trial_c, trial_p)
        if int(np.sum(energies == 0)) != 1 or int(np.argmin(energies)) != planted_idx:
            continue
        cur_c.append(variables)
        cur_p.append(poles)
    return np.stack(cur_c), np.stack(cur_p)


def _greedy_basin(energies: np.ndarray) -> tuple[int, int]:
    """Deterministic steepest single-bit descent: basin size of the unique GS, plus local-min count."""
    n_states = energies.size
    gs = int(np.argmin(energies))
    dest = np.empty(n_states, dtype=np.int64)
    is_local = np.ones(n_states, dtype=bool)
    for idx in range(n_states):
        best = idx
        best_e = int(energies[idx])
        for site in range(NUM_SPINS):
            nbr = idx ^ (1 << (NUM_SPINS - 1 - site))
            e_nbr = int(energies[nbr])
            if e_nbr < int(energies[idx]):
                is_local[idx] = False
            if e_nbr < best_e or (e_nbr == best_e and nbr < best and e_nbr < int(energies[idx])):
                best_e = e_nbr
                best = nbr
        dest[idx] = best if best_e < int(energies[idx]) else idx

    attractor = np.arange(n_states, dtype=np.int64)
    for _ in range(NUM_SPINS + 2):
        attractor = dest[attractor]
    basin = int(np.sum(attractor == gs))
    n_local = int(np.sum(is_local))
    return basin, n_local


def clause_to_z_terms(
    variables: np.ndarray,
    polarities: np.ndarray,
) -> dict[tuple[int, ...], float]:
    """Expand one clause projector into Z-string coefficients, including identity."""
    terms: dict[tuple[int, ...], float] = {}
    width = int(len(variables))
    scale = 2.0 ** (-width)
    for mask in range(1 << width):
        sites: list[int] = []
        sign = 1.0
        for j in range(width):
            if mask & (1 << j):
                sites.append(int(variables[j]))
                sign *= float(polarities[j])
        key = tuple(sorted(sites))
        terms[key] = terms.get(key, 0.0) + scale * sign
    return terms


def combine_z_terms(
    clauses: np.ndarray,
    polarities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return padded sites, orders, non-identity coefficients, and identity shift."""
    merged: dict[tuple[int, ...], float] = {}
    for variables, poles in zip(clauses, polarities, strict=True):
        for key, coeff in clause_to_z_terms(variables, poles).items():
            merged[key] = merged.get(key, 0.0) + coeff
    identity = float(merged.pop((), 0.0))
    items = [(key, coeff) for key, coeff in merged.items() if abs(coeff) > 1e-14]
    items.sort(key=lambda rec: (len(rec[0]), rec[0]))
    n_terms = len(items)
    max_order = max((len(key) for key, _ in items), default=1)
    sites = np.full((n_terms, max_order), -1, dtype=np.int64)
    orders = np.empty(n_terms, dtype=np.int64)
    coefficients = np.empty(n_terms, dtype=float)
    for row, (key, coeff) in enumerate(items):
        orders[row] = len(key)
        sites[row, : len(key)] = key
        coefficients[row] = coeff
    return sites, orders, coefficients, identity


def energy(
    spins_or_bits: np.ndarray,
    sites: np.ndarray,
    orders: np.ndarray,
    coefficients: np.ndarray,
    identity: float = 0.0,
    *,
    bits: bool = True,
) -> float:
    """Evaluate the diagonal Hamiltonian. ``bits=True`` uses {0,1}; otherwise {-1,+1}."""
    state = np.asarray(spins_or_bits, dtype=float).reshape(NUM_SPINS)
    if bits:
        if not np.all(np.isin(state, (0.0, 1.0))):
            raise ValueError(f"bits must have shape ({NUM_SPINS},) with values 0 or 1.")
        z_eig = 1.0 - 2.0 * state
    else:
        if not np.all(np.isin(state, (-1.0, 1.0))):
            raise ValueError(f"spins must have shape ({NUM_SPINS},) with values -1 or +1.")
        z_eig = state
    total = float(identity)
    for row, order in enumerate(orders):
        total += float(coefficients[row]) * float(np.prod(z_eig[sites[row, : int(order)]]))
    return total


def _score_instance(energies: np.ndarray, bits: np.ndarray) -> tuple[int, int, int, int] | None:
    if int(np.min(energies)) != 0:
        return None
    if int(np.sum(energies == 0)) != 1:
        return None
    gs_bits = _bits_from_index(int(np.argmin(energies)))
    if not np.array_equal(gs_bits, bits):
        return None
    label = _bitstring(gs_bits)
    if label in _TRIVIAL_BITSTRINGS:
        return None
    weight = int(gs_bits.sum())
    if weight < 2 or weight > NUM_SPINS - 2:
        return None
    basin, n_local = _greedy_basin(energies)
    # Prefer a small descent basin, then many competing local minima, then a mid-weight GS.
    return (-basin, n_local, -abs(weight - NUM_SPINS // 2), weight)


def sample_instance(
    rng: np.random.Generator,
    catalog: list[tuple[np.ndarray, np.ndarray, int]],
) -> dict[str, object] | None:
    """Try one planted rigid unique 4-SAT instance in the target clause window."""
    weight = int(rng.integers(2, NUM_SPINS - 1))
    ones = rng.choice(NUM_SPINS, size=weight, replace=False)
    planted = np.zeros(NUM_SPINS, dtype=np.int64)
    planted[ones] = 1
    if _bitstring(planted) in _TRIVIAL_BITSTRINGS:
        return None

    clauses, polarities = _rigidity_clauses(planted, rng)
    planted_idx = _index_from_bits(planted)
    if int(unsat_counts(clauses, polarities)[planted_idx]) != 0:
        return None
    compatible = _compatible_catalog(planted_idx, catalog)
    covered = _greedy_unique_cover(planted, clauses, polarities, compatible)
    if covered is None:
        return None
    clauses, polarities = covered
    clauses, polarities = _pad_to_target(planted, clauses, polarities, compatible, rng)
    n_clauses = int(len(clauses))
    if not MIN_CLAUSES <= n_clauses <= MAX_CLAUSES:
        return None

    energies = unsat_counts(clauses, polarities)
    score = _score_instance(energies, planted)
    if score is None:
        return None
    sites, orders, coefficients, identity = combine_z_terms(clauses, polarities)
    gs_idx = int(np.argmin(energies))
    uniq = np.unique(energies)
    gap = int(uniq[1] - uniq[0]) if uniq.size > 1 else 0
    basin, n_local = _greedy_basin(energies)
    return {
        "clauses": clauses,
        "polarities": polarities,
        "sites": sites,
        "orders": orders,
        "coefficients": coefficients,
        "identity": identity,
        "ground_bitstring": _bitstring(planted),
        "ground_index": gs_idx,
        "energy_min": int(energies[gs_idx]),
        "energy_max": int(np.max(energies)),
        "gap": gap,
        "n_ground": 1,
        "n_clauses": n_clauses,
        "n_pauli_terms": int(len(coefficients)),
        "greedy_basin": basin,
        "n_local_minima": n_local,
        "score": score,
    }


def generate_instance(
    rng: np.random.Generator,
    catalog: list[tuple[np.ndarray, np.ndarray, int]] | None = None,
) -> dict[str, object]:
    """Search random planted assignments for a unique, locally rigid 4-SAT Hamiltonian."""
    if catalog is None:
        catalog = _full_clause_catalog()
    best: dict[str, object] | None = None
    for _ in range(SEARCH_TRIALS):
        candidate = sample_instance(rng, catalog)
        if candidate is None:
            continue
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    if best is None:
        raise RuntimeError(
            f"no unique 4-SAT instance with {MIN_CLAUSES}-{MAX_CLAUSES} clauses "
            f"in {SEARCH_TRIALS} trials."
        )
    return best


def _verify_instance(instance: dict[str, object]) -> None:
    clauses = np.asarray(instance["clauses"])
    polarities = np.asarray(instance["polarities"])
    sites = np.asarray(instance["sites"])
    orders = np.asarray(instance["orders"])
    coefficients = np.asarray(instance["coefficients"])
    identity = float(instance["identity"])
    energies = unsat_counts(clauses, polarities)
    if int(np.sum(energies == 0)) != 1:
        raise ValueError("generated 4-SAT instance is not unique-SAT.")
    gs_bits = _bits_from_index(int(np.argmin(energies)))
    if _bitstring(gs_bits) != str(instance["ground_bitstring"]):
        raise ValueError("stored ground bitstring does not match the enumerated spectrum.")
    for idx in range(1 << NUM_SPINS):
        bits = _bits_from_index(idx)
        pauli_e = energy(bits, sites, orders, coefficients, identity)
        if abs(pauli_e - float(energies[idx])) > 1e-10:
            raise ValueError("Pauli expansion does not match clause-count energies.")


def generate_dataset() -> list[dict[str, object]]:
    """Generate and save unique 4-SAT Hamiltonians."""
    _validate_hyperparameters()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    catalog = _full_clause_catalog()
    manifest: list[dict[str, object]] = []

    for index in range(NUM_HAMILTONIANS):
        instance = generate_instance(rng, catalog)
        _verify_instance(instance)
        path = OUTPUT_DIRECTORY / f"{FILE_PREFIX}_{index:03d}.npz"
        np.savez_compressed(
            path,
            sites=instance["sites"],
            orders=instance["orders"],
            coefficients=instance["coefficients"],
            identity=instance["identity"],
            clauses=instance["clauses"],
            polarities=instance["polarities"],
            num_spins=NUM_SPINS,
            clause_width=CLAUSE_WIDTH,
            num_clauses=instance["n_clauses"],
            min_body_order=int(np.min(instance["orders"])),
            max_body_order=int(np.max(instance["orders"])),
        )
        manifest.append(
            {
                "file": path.name,
                "num_spins": NUM_SPINS,
                "clause_width": CLAUSE_WIDTH,
                "num_clauses": int(instance["n_clauses"]),
                "n_pauli_terms": int(instance["n_pauli_terms"]),
                "identity": float(instance["identity"]),
                "energy_min": int(instance["energy_min"]),
                "energy_max": int(instance["energy_max"]),
                "gap": int(instance["gap"]),
                "n_ground": int(instance["n_ground"]),
                "ground_bitstring": str(instance["ground_bitstring"]),
                "greedy_basin": int(instance["greedy_basin"]),
                "n_local_minima": int(instance["n_local_minima"]),
                "clauses": np.asarray(instance["clauses"]).tolist(),
                "polarities": np.asarray(instance["polarities"]).tolist(),
            }
        )
        print(
            f"  [{index + 1}/{NUM_HAMILTONIANS}] {path.name}: "
            f"{instance['n_clauses']} clauses, GS={instance['ground_bitstring']}, "
            f"greedy_basin={instance['greedy_basin']}/{1 << NUM_SPINS}",
            flush=True,
        )

    manifest_path = OUTPUT_DIRECTORY / f"{FILE_PREFIX}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    generated = generate_dataset()
    print(f"Saved {len(generated)} 4-SAT Hamiltonians to {OUTPUT_DIRECTORY}")
    for rec in generated:
        print(
            f"  {rec['file']}: {rec['num_clauses']} clauses, "
            f"GS={rec['ground_bitstring']}, gap={rec['gap']}, "
            f"greedy_basin={rec['greedy_basin']}/{1 << NUM_SPINS}, "
            f"local_minima={rec['n_local_minima']}"
        )
