"""Unique-planted 4-SAT → diagonal Ising, generalized to n=7…11.

Same family as ``Hamiltonians/four_sat.py``: plant a locally rigid satisfying
assignment, greedily add compatible 4-clauses until that assignment is the
unique solution, then pad toward the density target. Clause density is fixed
at 18/7; the uniqueness loop is allowed a band around the target (not the
old 12–20 lock).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import CLAUSE_WIDTH, N_HAMILTONIANS, SEARCH_TRIALS, clause_window, ham_dir, ham_seed


def trivial_bitstrings(n: int) -> set[str]:
    n = int(n)
    half = n // 2
    return {
        "0" * n,
        "1" * n,
        ("01" * ((n + 1) // 2))[:n],
        ("10" * ((n + 1) // 2))[:n],
        "0" * half + "1" * (n - half),
        "1" * half + "0" * (n - half),
    }


def bits_from_index(index: int, n: int) -> np.ndarray:
    bits = np.zeros(n, dtype=np.int64)
    for i in range(n):
        bits[i] = (int(index) >> (n - 1 - i)) & 1
    return bits


def index_from_bits(bits: np.ndarray) -> int:
    bits = np.asarray(bits, dtype=int).reshape(-1)
    idx = 0
    for bit in bits:
        idx = (idx << 1) | int(bit)
    return int(idx)


def bitstring(bits: np.ndarray) -> str:
    return "".join(str(int(b)) for b in np.asarray(bits).reshape(-1))


def all_bit_table(n: int) -> np.ndarray:
    n = int(n)
    idx = np.arange(1 << n, dtype=np.int64)
    bits = np.empty((1 << n, n), dtype=np.int64)
    for i in range(n):
        bits[:, i] = (idx >> (n - 1 - i)) & 1
    return bits


def violating_mask(all_bits: np.ndarray, variables: np.ndarray, polarities: np.ndarray) -> np.ndarray:
    pattern = np.where(np.asarray(polarities, dtype=int) > 0, 0, 1)
    return np.all(all_bits[:, np.asarray(variables, dtype=int)] == pattern, axis=1)


def unsat_counts(all_bits: np.ndarray, clauses: np.ndarray, polarities: np.ndarray) -> np.ndarray:
    energies = np.zeros(all_bits.shape[0], dtype=np.int64)
    for variables, poles in zip(clauses, polarities, strict=True):
        energies += violating_mask(all_bits, variables, poles)
    return energies


def _clause_key(variables: np.ndarray, polarities: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (tuple(int(v) for v in variables), tuple(int(p) for p in polarities))


def _polarity_for_literal(bit: int, *, want_true: bool) -> int:
    if want_true:
        return 1 if int(bit) == 1 else -1
    return 1 if int(bit) == 0 else -1


def rigidity_clauses(planted: np.ndarray, rng: np.random.Generator, n: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    clauses: list[np.ndarray] = []
    polarities: list[np.ndarray] = []
    others = np.arange(n)
    for site in range(n):
        pool = others[others != site]
        companions = rng.choice(pool, size=width - 1, replace=False)
        variables = np.sort(np.concatenate(([site], companions))).astype(np.int64)
        poles = np.empty(width, dtype=np.int64)
        for j, var in enumerate(variables):
            poles[j] = _polarity_for_literal(int(planted[var]), want_true=(int(var) == site))
        clauses.append(variables)
        polarities.append(poles)
    return np.stack(clauses), np.stack(polarities)


def full_clause_catalog(all_bits: np.ndarray, n: int, width: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for variables in combinations(range(n), width):
        vars_arr = np.asarray(variables, dtype=np.int64)
        for mask in range(1 << width):
            poles = np.array(
                [1 if ((mask >> j) & 1) == 0 else -1 for j in range(width)],
                dtype=np.int64,
            )
            out.append((vars_arr, poles, violating_mask(all_bits, vars_arr, poles)))
    return out


def compatible_catalog(
    planted_idx: int,
    catalog: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [(v, p, m) for v, p, m in catalog if not bool(m[int(planted_idx)])]


def greedy_unique_cover(
    planted: np.ndarray,
    clauses: np.ndarray,
    polarities: np.ndarray,
    catalog: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    all_bits: np.ndarray,
    max_clauses: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    planted_idx = index_from_bits(planted)
    used = {_clause_key(v, p) for v, p in zip(clauses, polarities, strict=True)}
    energies = unsat_counts(all_bits, clauses, polarities)
    remaining = energies == 0
    remaining[planted_idx] = False
    if not bool(remaining.any()):
        return clauses, polarities

    cur_c = [np.asarray(row).copy() for row in clauses]
    cur_p = [np.asarray(row).copy() for row in polarities]
    stack = np.stack([m for _v, _p, m in catalog], axis=0)
    taken = np.array([_clause_key(v, p) in used for v, p, _m in catalog], dtype=bool)
    while bool(remaining.any()) and len(cur_c) < int(max_clauses):
        live = ~taken
        if not bool(live.any()):
            break
        covers = np.zeros(len(catalog), dtype=np.int64)
        covers[live] = np.count_nonzero(stack[live] & remaining, axis=1)
        best_i = int(np.argmax(covers))
        if int(covers[best_i]) <= 0:
            break
        variables, poles, vmask = catalog[best_i]
        used.add(_clause_key(variables, poles))
        taken[best_i] = True
        cur_c.append(variables)
        cur_p.append(poles)
        remaining = remaining & ~vmask
    if bool(remaining.any()):
        return None
    return np.stack(cur_c), np.stack(cur_p)


def pad_to_target(
    planted: np.ndarray,
    clauses: np.ndarray,
    polarities: np.ndarray,
    catalog: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    all_bits: np.ndarray,
    target: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if len(clauses) >= int(target):
        return clauses, polarities
    used = {_clause_key(v, p) for v, p in zip(clauses, polarities, strict=True)}
    unused = [(v, p) for v, p, _m in catalog if _clause_key(v, p) not in used]
    unused = [unused[i] for i in rng.permutation(len(unused))]
    cur_c = [np.asarray(row).copy() for row in clauses]
    cur_p = [np.asarray(row).copy() for row in polarities]
    planted_idx = index_from_bits(planted)
    for variables, poles in unused:
        if len(cur_c) >= int(target):
            break
        trial_c = np.stack(cur_c + [variables])
        trial_p = np.stack(cur_p + [poles])
        energies = unsat_counts(all_bits, trial_c, trial_p)
        if int(np.sum(energies == 0)) != 1 or int(np.argmin(energies)) != planted_idx:
            continue
        cur_c.append(variables)
        cur_p.append(poles)
    return np.stack(cur_c), np.stack(cur_p)


def greedy_basin(energies: np.ndarray, n: int) -> tuple[int, int]:
    n_states = energies.size
    gs = int(np.argmin(energies))
    dest = np.empty(n_states, dtype=np.int64)
    is_local = np.ones(n_states, dtype=bool)
    for idx in range(n_states):
        best = idx
        best_e = int(energies[idx])
        for site in range(n):
            nbr = idx ^ (1 << (n - 1 - site))
            e_nbr = int(energies[nbr])
            if e_nbr < int(energies[idx]):
                is_local[idx] = False
            if e_nbr < best_e or (e_nbr == best_e and nbr < best and e_nbr < int(energies[idx])):
                best_e = e_nbr
                best = nbr
        dest[idx] = best if best_e < int(energies[idx]) else idx
    attractor = np.arange(n_states, dtype=np.int64)
    for _ in range(n + 2):
        attractor = dest[attractor]
    return int(np.sum(attractor == gs)), int(np.sum(is_local))


def clause_to_z_terms(variables: np.ndarray, polarities: np.ndarray) -> dict[tuple[int, ...], float]:
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


def energy_from_z_terms(
    bits: Sequence[int] | np.ndarray,
    sites: np.ndarray,
    orders: np.ndarray,
    coefficients: np.ndarray,
    identity: float = 0.0,
) -> float:
    z_eig = 1.0 - 2.0 * np.asarray(bits, dtype=float).reshape(-1)
    total = float(identity)
    for row, order in enumerate(orders):
        total += float(coefficients[row]) * float(np.prod(z_eig[sites[row, : int(order)]]))
    return total


def z_terms_list(
    sites: np.ndarray,
    orders: np.ndarray,
    coefficients: np.ndarray,
) -> list[tuple[tuple[int, ...], float]]:
    out: list[tuple[tuple[int, ...], float]] = []
    for row, coeff in enumerate(coefficients):
        order = int(orders[row])
        out.append((tuple(int(v) for v in sites[row, :order]), float(coeff)))
    return out


def _score_instance(energies: np.ndarray, planted: np.ndarray, n: int) -> tuple[int, int, int, int] | None:
    if int(np.min(energies)) != 0:
        return None
    if int(np.sum(energies == 0)) != 1:
        return None
    gs_bits = bits_from_index(int(np.argmin(energies)), n)
    if not np.array_equal(gs_bits, planted):
        return None
    label = bitstring(gs_bits)
    if label in trivial_bitstrings(n):
        return None
    weight = int(gs_bits.sum())
    if weight < 2 or weight > n - 2:
        return None
    basin, n_local = greedy_basin(energies, n)
    return (-basin, n_local, -abs(weight - n // 2), weight)


def sample_instance(
    rng: np.random.Generator,
    catalog: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    all_bits: np.ndarray,
    n: int,
    min_clauses: int,
    target: int,
    max_clauses: int,
    width: int = CLAUSE_WIDTH,
) -> dict[str, object] | None:
    weight = int(rng.integers(2, n - 1))
    ones = rng.choice(n, size=weight, replace=False)
    planted = np.zeros(n, dtype=np.int64)
    planted[ones] = 1
    if bitstring(planted) in trivial_bitstrings(n):
        return None
    clauses, polarities = rigidity_clauses(planted, rng, n, width)
    planted_idx = index_from_bits(planted)
    if int(unsat_counts(all_bits, clauses, polarities)[planted_idx]) != 0:
        return None
    compatible = compatible_catalog(planted_idx, catalog)
    covered = greedy_unique_cover(planted, clauses, polarities, compatible, all_bits, max_clauses)
    if covered is None:
        return None
    clauses, polarities = covered
    clauses, polarities = pad_to_target(planted, clauses, polarities, compatible, all_bits, target, rng)
    n_clauses = int(len(clauses))
    if not min_clauses <= n_clauses <= max_clauses:
        return None
    energies = unsat_counts(all_bits, clauses, polarities)
    score = _score_instance(energies, planted, n)
    if score is None:
        return None
    sites, orders, coefficients, identity = combine_z_terms(clauses, polarities)
    gs_idx = int(np.argmin(energies))
    uniq = np.unique(energies)
    gap = int(uniq[1] - uniq[0]) if uniq.size > 1 else 0
    basin, n_local = greedy_basin(energies, n)
    return {
        "clauses": clauses,
        "polarities": polarities,
        "sites": sites,
        "orders": orders,
        "coefficients": coefficients,
        "identity": identity,
        "ground_bitstring": bitstring(planted),
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
        "logical_energies": energies,
    }


def generate_instance(
    rng: np.random.Generator,
    n: int,
    *,
    catalog: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    all_bits: np.ndarray | None = None,
    search_trials: int = SEARCH_TRIALS,
    max_accepts: int = 40,
) -> dict[str, object]:
    min_c, target, max_c = clause_window(n)
    if all_bits is None:
        all_bits = all_bit_table(n)
    if catalog is None:
        catalog = full_clause_catalog(all_bits, n, CLAUSE_WIDTH)
    best: dict[str, object] | None = None
    n_accept = 0
    for _ in range(int(search_trials)):
        candidate = sample_instance(rng, catalog, all_bits, n, min_c, target, max_c)
        if candidate is None:
            continue
        n_accept += 1
        if best is None or candidate["score"] > best["score"]:
            best = candidate
        if n_accept >= int(max_accepts):
            break
    if best is None:
        raise RuntimeError(
            f"no unique 4-SAT instance with {min_c}-{max_c} clauses "
            f"(target {target}) on n={n} in {search_trials} trials."
        )
    return best


def verify_instance(instance: dict[str, object], n: int, all_bits: np.ndarray | None = None) -> None:
    if all_bits is None:
        all_bits = all_bit_table(n)
    clauses = np.asarray(instance["clauses"])
    polarities = np.asarray(instance["polarities"])
    sites = np.asarray(instance["sites"])
    orders = np.asarray(instance["orders"])
    coefficients = np.asarray(instance["coefficients"])
    identity = float(instance["identity"])
    energies = unsat_counts(all_bits, clauses, polarities)
    if int(np.sum(energies == 0)) != 1:
        raise ValueError("generated 4-SAT instance is not unique-SAT.")
    gs_bits = bits_from_index(int(np.argmin(energies)), n)
    if bitstring(gs_bits) != str(instance["ground_bitstring"]):
        raise ValueError("stored ground bitstring does not match the enumerated spectrum.")
    for idx in range(1 << n):
        bits = bits_from_index(idx, n)
        pauli_e = energy_from_z_terms(bits, sites, orders, coefficients, identity)
        if abs(pauli_e - float(energies[idx])) > 1e-10:
            raise ValueError("Pauli expansion does not match clause-count energies.")


def generate_dataset(
    n: int,
    *,
    n_hamiltonians: int = N_HAMILTONIANS,
    search_trials: int = SEARCH_TRIALS,
    seed: int | None = None,
    outdir: Path | None = None,
) -> list[dict[str, object]]:
    n = int(n)
    outdir = Path(outdir) if outdir is not None else ham_dir(n)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(ham_seed(n) if seed is None else int(seed))
    all_bits = all_bit_table(n)
    print(f"Building 4-clause catalog for n={n} ({1 << n} assignments)...", flush=True)
    catalog = full_clause_catalog(all_bits, n, CLAUSE_WIDTH)
    min_c, target, max_c = clause_window(n)
    print(
        f"Generating {n_hamiltonians} unique 4-SAT instances  n={n}  "
        f"clause window {min_c}–{max_c} (target {target}, density {18/7:.3f})",
        flush=True,
    )
    manifest: list[dict[str, object]] = []
    for index in range(int(n_hamiltonians)):
        instance = generate_instance(rng, n, catalog=catalog, all_bits=all_bits, search_trials=search_trials)
        verify_instance(instance, n, all_bits)
        path = outdir / f"four_sat_{index:03d}.npz"
        np.savez_compressed(
            path,
            sites=instance["sites"],
            orders=instance["orders"],
            coefficients=instance["coefficients"],
            identity=instance["identity"],
            clauses=instance["clauses"],
            polarities=instance["polarities"],
            logical_energies=instance["logical_energies"],
            num_spins=n,
            clause_width=CLAUSE_WIDTH,
            num_clauses=instance["n_clauses"],
            min_body_order=int(np.min(instance["orders"])),
            max_body_order=int(np.max(instance["orders"])),
            ground_index=instance["ground_index"],
        )
        rec = {
            "file": path.name,
            "num_spins": n,
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
            "clause_window": [min_c, target, max_c],
        }
        manifest.append(rec)
        print(
            f"  [{index + 1}/{n_hamiltonians}] {path.name}: "
            f"{instance['n_clauses']} clauses, GS={instance['ground_bitstring']}, "
            f"greedy_basin={instance['greedy_basin']}/{1 << n}",
            flush=True,
        )
    manifest_path = outdir / "four_sat_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_instance(path: Path | str) -> dict:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    n = int(np.asarray(data["num_spins"]).reshape(-1)[0])
    identity = float(np.asarray(data["identity"]).reshape(-1)[0])
    sites = np.asarray(data["sites"])
    orders = np.asarray(data["orders"])
    coefficients = np.asarray(data["coefficients"])
    if "logical_energies" in data.files:
        logical = np.asarray(data["logical_energies"], dtype=float)
    else:
        all_bits = all_bit_table(n)
        logical = unsat_counts(all_bits, data["clauses"], data["polarities"]).astype(float)
    hid = int(path.stem.rsplit("_", 1)[-1])
    gs = bits_from_index(int(np.argmin(logical)), n)
    return {
        "path": str(path),
        "file": path.name,
        "hamiltonian_id": hid,
        "num_spins": n,
        "sites": sites,
        "orders": orders,
        "coefficients": coefficients,
        "identity": identity,
        "clauses": np.asarray(data["clauses"]),
        "polarities": np.asarray(data["polarities"]),
        "num_clauses": int(np.asarray(data["num_clauses"]).reshape(-1)[0]),
        "logical_energies": logical,
        "ground_bitstring": bitstring(gs),
        "ground_index": int(np.argmin(logical)),
        "energy_min": float(np.min(logical)),
        "energy_max": float(np.max(logical)),
        "terms": z_terms_list(sites, orders, coefficients),
    }
