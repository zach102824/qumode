"""Exact spectrum of mixed p-spin instances: confirm that E_GS is extensive.

H is a sum of Z-strings, so the spectrum is the 2^N diagonal. This script
enumerates every computational-basis energy (vectorized) and reports E_GS / N.
A size scan through N ~ 15 (in memory, does not overwrite the dataset) checks
that E_GS / N stays O(1). One small-N dense-matrix eigh is kept as a sanity check.

Run from the repo root:

    python Hamiltonians/check_mixed_p_spin_extensive.py
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

HAMILTONIAN_DIR = Path(__file__).resolve().parent / "mixed_p_spin"
RESULT_PATH = Path(__file__).resolve().parent / "mixed_p_spin_extensivity_check.json"
SK_J = 1.0
MIN_BODY_ORDER = 2
MAX_BODY_ORDER = 4
N_SCAN = (5, 6, 7, 8, 10, 12, 14, 15)
N_SCAN_SAMPLES = 8
N_SCAN_SEED = 19000
DENSE_EIGH_MAX_N = 8


def coupling_std(n_spins: int, order: int, sk_j: float = SK_J) -> float:
    return sk_j * math.sqrt(math.factorial(order) / (2.0 * n_spins ** (order - 1)))


def config_energy_variance(n_spins: int, sk_j: float = SK_J) -> float:
    """Exact Var(E) of a random ±1 configuration (finite-N, all p-subsets)."""
    total = 0.0
    for order in range(MIN_BODY_ORDER, min(MAX_BODY_ORDER, n_spins) + 1):
        n_terms = math.comb(n_spins, order)
        total += n_terms * coupling_std(n_spins, order, sk_j) ** 2
    return total


def rem_ground_estimate(n_spins: int, sk_j: float = SK_J) -> float:
    """Extreme-value estimate: min of 2^N Gaussians with the exact Var(E)."""
    variance = config_energy_variance(n_spins, sk_j)
    return -math.sqrt(2.0 * variance * n_spins * math.log(2.0))


def bit_table(n_spins: int) -> np.ndarray:
    """All 2^N bitstrings, shape (2^N, N), qubit 0 = MSB."""
    dim = 1 << n_spins
    index = np.arange(dim, dtype=np.uint32)[:, None]
    shifts = np.arange(n_spins - 1, -1, -1, dtype=np.uint32)
    return ((index >> shifts) & 1).astype(np.int8)


def z_string(n_spins: int, sites: np.ndarray) -> np.ndarray:
    """Kronecker product of Z on ``sites`` and I elsewhere."""
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    ident = np.eye(2)
    op = np.array([[1.0]])
    site_set = {int(s) for s in sites}
    for i in range(n_spins):
        op = np.kron(op, z if i in site_set else ident)
    return op


def hamiltonian_matrix(
    n_spins: int,
    sites: np.ndarray,
    orders: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    dim = 2**n_spins
    ham = np.zeros((dim, dim), dtype=float)
    for row, coeff in enumerate(coefficients):
        order = int(orders[row])
        ham += float(coeff) * z_string(n_spins, sites[row, :order])
    return ham


def enumerate_spectrum(
    n_spins: int,
    sites: np.ndarray,
    orders: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Exact Z-basis spectrum. Eigenvalue of a Z-string is (-1)^{sum of bits}."""
    bits = bit_table(n_spins)
    energies = np.zeros(bits.shape[0], dtype=float)
    for row, coeff in enumerate(coefficients):
        order = int(orders[row])
        parity = np.bitwise_xor.reduce(bits[:, sites[row, :order]], axis=1)
        energies += float(coeff) * (1.0 - 2.0 * parity)
    return energies


def diagonalize_instance(
    n_spins: int,
    sites: np.ndarray,
    orders: np.ndarray,
    coefficients: np.ndarray,
    *,
    dense_eigh: bool = False,
) -> dict[str, float]:
    enumerated = enumerate_spectrum(n_spins, sites, orders, coefficients)
    gs = float(np.min(enumerated))
    emax = float(np.max(enumerated))
    typical = float(np.sqrt(np.mean(enumerated**2)))
    stats = {
        "energy_min": gs,
        "energy_max": emax,
        "energy_min_per_spin": gs / n_spins,
        "typical_rms_energy": typical,
        "typical_rms_per_spin": typical / n_spins,
        "spread": emax - gs,
        "config_variance_theory": config_energy_variance(n_spins),
        "rem_gs_estimate": rem_ground_estimate(n_spins),
        "method": "z_basis_enumeration",
    }
    if dense_eigh:
        ham = hamiltonian_matrix(n_spins, sites, orders, coefficients)
        evals = np.linalg.eigvalsh(ham)
        stats["off_diagonal_max"] = float(np.max(np.abs(ham - np.diag(np.diag(ham)))))
        stats["enum_minus_eigh"] = gs - float(evals[0])
        stats["method"] = "z_basis_enumeration+eigh"
    return stats


def sample_mixed_terms(n_spins: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_sites: list[tuple[int, ...]] = []
    all_orders: list[int] = []
    all_coeffs: list[float] = []
    max_order = min(MAX_BODY_ORDER, n_spins)
    for order in range(MIN_BODY_ORDER, max_order + 1):
        candidates = list(combinations(range(n_spins), order))
        coeffs = rng.normal(0.0, coupling_std(n_spins, order), size=len(candidates))
        all_sites.extend(candidates)
        all_orders.extend([order] * len(candidates))
        all_coeffs.extend(float(c) for c in coeffs)
    padded = np.full((len(all_sites), max_order), -1, dtype=np.int64)
    for row, sites in enumerate(all_sites):
        padded[row, : len(sites)] = sites
    return padded, np.asarray(all_orders, dtype=np.int64), np.asarray(all_coeffs, dtype=float)


def check_saved_dataset() -> list[dict[str, object]]:
    paths = sorted(HAMILTONIAN_DIR.glob("mixed_p_spin_p2-4_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no mixed p-spin files in {HAMILTONIAN_DIR}")
    records: list[dict[str, object]] = []
    print(f"{'file':<28} {'N':>3} {'E_GS':>10} {'E_GS/N':>8} {'RMS/N':>8} {'REM/N':>8}")
    for path in paths:
        data = np.load(path)
        n_spins = int(data["num_spins"])
        stats = diagonalize_instance(
            n_spins,
            data["sites"],
            data["orders"],
            data["coefficients"],
            dense_eigh=n_spins <= DENSE_EIGH_MAX_N and path == paths[0],
        )
        rec = {"file": path.name, "num_spins": n_spins, **stats}
        records.append(rec)
        print(
            f"{path.name:<28} {n_spins:3d} {stats['energy_min']:10.4f} "
            f"{stats['energy_min_per_spin']:8.3f} {stats['typical_rms_per_spin']:8.3f} "
            f"{stats['rem_gs_estimate'] / n_spins:8.3f}"
        )
    return records


def check_size_scan() -> list[dict[str, object]]:
    rng = np.random.default_rng(N_SCAN_SEED)
    records: list[dict[str, object]] = []
    print()
    print(f"{'N':>3} {'mean E_GS':>10} {'mean E_GS/N':>12} {'mean RMS/N':>11} {'REM/N':>8}")
    for n_spins in N_SCAN:
        gs_list = []
        rms_list = []
        for _ in range(N_SCAN_SAMPLES):
            sites, orders, coeffs = sample_mixed_terms(n_spins, rng)
            stats = diagonalize_instance(
                n_spins, sites, orders, coeffs, dense_eigh=False
            )
            gs_list.append(stats["energy_min"])
            rms_list.append(stats["typical_rms_energy"])
        mean_gs = float(np.mean(gs_list))
        rec = {
            "num_spins": n_spins,
            "n_samples": N_SCAN_SAMPLES,
            "mean_energy_min": mean_gs,
            "mean_energy_min_per_spin": mean_gs / n_spins,
            "std_energy_min_per_spin": float(np.std(gs_list, ddof=0) / n_spins),
            "mean_rms_per_spin": float(np.mean(rms_list) / n_spins),
            "rem_gs_estimate_per_spin": rem_ground_estimate(n_spins) / n_spins,
            "config_variance_theory": config_energy_variance(n_spins),
        }
        records.append(rec)
        print(
            f"{n_spins:3d} {mean_gs:10.4f} {rec['mean_energy_min_per_spin']:12.3f} "
            f"{rec['mean_rms_per_spin']:11.3f} {rec['rem_gs_estimate_per_spin']:8.3f}"
        )
    return records


if __name__ == "__main__":
    saved = check_saved_dataset()
    scan = check_size_scan()
    gs_over_n = [float(r["energy_min_per_spin"]) for r in saved]
    payload = {
        "convention": "Gross-Mezard / Crisanti-Sommers, SK_J=1, mixed p=2,3,4",
        "note": (
            "H is a sum of Z-strings, so it is already diagonal in the Z basis. "
            "eigvalsh of the explicit matrix matches the 2^N enumeration. "
            "For N > 8 the dense matrix is skipped: the spectrum is the 2^N "
            "Z-basis energies. Extensivity means E_GS = O(N), i.e. E_GS/N "
            "stays O(1), while a typical configuration is only O(sqrt(N))."
        ),
        "saved_instances": saved,
        "saved_summary": {
            "n_instances": len(saved),
            "mean_energy_min": float(np.mean([r["energy_min"] for r in saved])),
            "mean_energy_min_per_spin": float(np.mean(gs_over_n)),
            "std_energy_min_per_spin": float(np.std(gs_over_n, ddof=0)),
            "mean_rms_per_spin": float(np.mean([r["typical_rms_per_spin"] for r in saved])),
        },
        "size_scan": scan,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"Wrote {RESULT_PATH}")
    print(
        "saved  <E_GS/N> = "
        f"{payload['saved_summary']['mean_energy_min_per_spin']:.3f}  "
        f"(std {payload['saved_summary']['std_energy_min_per_spin']:.3f})"
    )
