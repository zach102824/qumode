"""Generate pure random p-spin Ising Hamiltonians.

A pure p-spin instance contains only terms of exactly one body order:

    H = sum_{i1 < ... < ip} J[i1,...,ip] Z[i1] ... Z[ip].

Edit the hyperparameters below and run this file directly. Setting
``BODY_ORDER`` to 2, 3, 4, ... selects two-, three-, four-body interactions.
Instances are written to the ``pure_p_spin/`` subdirectory.
"""

from __future__ import annotations

import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_HAMILTONIANS = 20  # Common choices are 20 or 40.
NUM_SPINS = 7
BODY_ORDER = 3
TERMS_PER_HAMILTONIAN: int | None = None  # None uses every p-spin term.
COUPLING_SCALE = 1.0
NORMALIZE_BY_SYSTEM_SIZE = True
RANDOM_SEED = 8000
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "pure_p_spin"
FILE_PREFIX = "pure_p_spin"


def _validate_hyperparameters() -> None:
    if NUM_HAMILTONIANS < 1 or NUM_SPINS < 1:
        raise ValueError("NUM_HAMILTONIANS and NUM_SPINS must be positive.")
    if not 1 <= BODY_ORDER <= NUM_SPINS:
        raise ValueError("BODY_ORDER must be between 1 and NUM_SPINS.")
    if TERMS_PER_HAMILTONIAN is not None and TERMS_PER_HAMILTONIAN < 1:
        raise ValueError("TERMS_PER_HAMILTONIAN must be positive or None.")
    if COUPLING_SCALE <= 0.0:
        raise ValueError("COUPLING_SCALE must be positive.")


def generate_terms(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return interaction sites and Gaussian couplings for one instance."""
    available = np.asarray(list(combinations(range(NUM_SPINS), BODY_ORDER)), dtype=np.int64)
    count = len(available)
    if TERMS_PER_HAMILTONIAN is not None:
        count = min(TERMS_PER_HAMILTONIAN, count)
        chosen = rng.choice(len(available), size=count, replace=False)
        available = available[chosen]

    scale = COUPLING_SCALE
    if NORMALIZE_BY_SYSTEM_SIZE:
        scale /= NUM_SPINS ** ((BODY_ORDER - 1) / 2)
    coefficients = rng.normal(0.0, scale, size=count)
    return available, coefficients


def energy(spins: np.ndarray, sites: np.ndarray, coefficients: np.ndarray) -> float:
    """Evaluate the Hamiltonian for spins taking values in {-1, +1}."""
    state = np.asarray(spins, dtype=float)
    if state.shape != (NUM_SPINS,) or not np.all(np.isin(state, (-1.0, 1.0))):
        raise ValueError(f"spins must have shape ({NUM_SPINS},) with values -1 or +1.")
    return float(np.sum(coefficients * np.prod(state[sites], axis=1)))


def generate_dataset() -> list[dict[str, object]]:
    """Generate and save all configured pure p-spin Hamiltonians."""
    _validate_hyperparameters()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    manifest: list[dict[str, object]] = []

    for index in range(NUM_HAMILTONIANS):
        sites, coefficients = generate_terms(rng)
        path = OUTPUT_DIRECTORY / f"{FILE_PREFIX}_p{BODY_ORDER}_{index:03d}.npz"
        np.savez_compressed(
            path,
            sites=sites,
            coefficients=coefficients,
            num_spins=NUM_SPINS,
            body_order=BODY_ORDER,
        )
        manifest.append(
            {
                "file": path.name,
                "num_spins": NUM_SPINS,
                "body_order": BODY_ORDER,
                "num_terms": len(coefficients),
            }
        )

    manifest_path = OUTPUT_DIRECTORY / f"{FILE_PREFIX}_p{BODY_ORDER}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    generated = generate_dataset()
    total_available = comb(NUM_SPINS, BODY_ORDER)
    print(
        f"Saved {len(generated)} pure {BODY_ORDER}-spin Hamiltonians "
        f"(up to {total_available} terms each) to {OUTPUT_DIRECTORY}"
    )
