"""Generate random mixed higher-order Ising Hamiltonians.

With ``MAX_BODY_ORDER = 4``, each Hamiltonian contains random 2-, 3-, and
4-body terms:

    H = sum_{p=2}^{4} sum_{|S|=p} J[p,S] product_{i in S} Z[i].

Couplings follow Gross–Mézard / Crisanti–Sommers with energy unit ``SK_J``:

    J[p,S] ~ Normal(0, SK_J^2 * p! / (2 N^{p-1})).

That choice makes each order extensive and recovers SK at p = 2 when SK_J = 1.
Edit the hyperparameters below and run this file directly.
Instances are written to the ``mixed_p_spin/`` subdirectory.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_HAMILTONIANS = 20  # Common choices are 20 or 40.
NUM_SPINS = 7
MIN_BODY_ORDER = 2
MAX_BODY_ORDER = 4
TERMS_PER_ORDER: int | None = None  # None uses every term at each order.
SK_J = 1.0  # Gross–Mézard / Crisanti–Sommers energy unit.
COUPLING_SCALE_BY_ORDER = {2: 1.0, 3: 1.0, 4: 1.0}  # Extra per-order weights.
RANDOM_SEED = 9000
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "mixed_p_spin"
FILE_PREFIX = "mixed_p_spin"


def _validate_hyperparameters() -> None:
    if NUM_HAMILTONIANS < 1 or NUM_SPINS < 1:
        raise ValueError("NUM_HAMILTONIANS and NUM_SPINS must be positive.")
    if not 1 <= MIN_BODY_ORDER <= MAX_BODY_ORDER <= NUM_SPINS:
        raise ValueError("Body orders must satisfy 1 <= MIN <= MAX <= NUM_SPINS.")
    if TERMS_PER_ORDER is not None and TERMS_PER_ORDER < 1:
        raise ValueError("TERMS_PER_ORDER must be positive or None.")
    missing = set(range(MIN_BODY_ORDER, MAX_BODY_ORDER + 1)) - set(
        COUPLING_SCALE_BY_ORDER
    )
    if missing:
        raise ValueError(f"COUPLING_SCALE_BY_ORDER is missing orders: {sorted(missing)}")
    if SK_J <= 0.0:
        raise ValueError("SK_J must be positive.")
    if any(COUPLING_SCALE_BY_ORDER[p] <= 0.0 for p in COUPLING_SCALE_BY_ORDER):
        raise ValueError("All coupling scales must be positive.")


def coupling_std(order: int) -> float:
    """Gross–Mézard / Crisanti–Sommers width for a p-body coupling."""
    variance = (SK_J**2) * math.factorial(order) / (2.0 * NUM_SPINS ** (order - 1))
    return float(COUPLING_SCALE_BY_ORDER[order]) * math.sqrt(variance)


def generate_terms(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return padded sites, body orders, and coefficients for one instance."""
    all_sites: list[tuple[int, ...]] = []
    all_orders: list[int] = []
    all_coefficients: list[float] = []

    for order in range(MIN_BODY_ORDER, MAX_BODY_ORDER + 1):
        candidates = list(combinations(range(NUM_SPINS), order))
        count = len(candidates)
        if TERMS_PER_ORDER is not None:
            count = min(TERMS_PER_ORDER, count)
            selected = rng.choice(len(candidates), size=count, replace=False)
            candidates = [candidates[int(i)] for i in selected]

        coefficients = rng.normal(0.0, coupling_std(order), size=count)
        if count < 1:
            raise ValueError(f"order {order} produced no terms; every order in the range must appear.")
        all_sites.extend(candidates)
        all_orders.extend([order] * count)
        all_coefficients.extend(float(value) for value in coefficients)

    # Rows use -1 padding so all interaction orders fit one rectangular array.
    padded_sites = np.full((len(all_sites), MAX_BODY_ORDER), -1, dtype=np.int64)
    for row, sites in enumerate(all_sites):
        padded_sites[row, : len(sites)] = sites
    return (
        padded_sites,
        np.asarray(all_orders, dtype=np.int64),
        np.asarray(all_coefficients, dtype=float),
    )


def energy(
    spins: np.ndarray,
    sites: np.ndarray,
    orders: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    """Evaluate the mixed Hamiltonian for spins in {-1, +1}."""
    state = np.asarray(spins, dtype=float)
    if state.shape != (NUM_SPINS,) or not np.all(np.isin(state, (-1.0, 1.0))):
        raise ValueError(f"spins must have shape ({NUM_SPINS},) with values -1 or +1.")
    products = [
        np.prod(state[sites[row, : int(order)]])
        for row, order in enumerate(orders)
    ]
    return float(np.dot(coefficients, products))


def generate_dataset() -> list[dict[str, object]]:
    """Generate and save all configured mixed p-spin Hamiltonians."""
    _validate_hyperparameters()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    manifest: list[dict[str, object]] = []

    for index in range(NUM_HAMILTONIANS):
        sites, orders, coefficients = generate_terms(rng)
        path = OUTPUT_DIRECTORY / (
            f"{FILE_PREFIX}_p{MIN_BODY_ORDER}-{MAX_BODY_ORDER}_{index:03d}.npz"
        )
        np.savez_compressed(
            path,
            sites=sites,
            orders=orders,
            coefficients=coefficients,
            num_spins=NUM_SPINS,
            min_body_order=MIN_BODY_ORDER,
            max_body_order=MAX_BODY_ORDER,
            sk_j=SK_J,
        )
        body_orders = list(range(MIN_BODY_ORDER, MAX_BODY_ORDER + 1))
        counts = {
            str(order): int(np.count_nonzero(orders == order))
            for order in body_orders
        }
        if any(counts[str(order)] < 1 for order in body_orders):
            raise ValueError(f"instance {index} is missing a body order: {counts}")
        manifest.append(
            {
                "file": path.name,
                "num_spins": NUM_SPINS,
                "body_orders": body_orders,
                "terms_by_order": counts,
                "sk_j": SK_J,
                "coupling_std_by_order": {
                    str(order): coupling_std(order) for order in body_orders
                },
            }
        )

    manifest_path = OUTPUT_DIRECTORY / (
        f"{FILE_PREFIX}_p{MIN_BODY_ORDER}-{MAX_BODY_ORDER}_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    generated = generate_dataset()
    print(
        f"Saved {len(generated)} mixed {MIN_BODY_ORDER}- through "
        f"{MAX_BODY_ORDER}-spin Hamiltonians to {OUTPUT_DIRECTORY}"
    )
