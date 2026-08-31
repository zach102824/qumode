"""Generate random binary-knapsack QUBO Hamiltonians.

The construction follows the paper's slack-variable encoding,

    H(x, y) = -sum_i(value_i x_i)
              + penalty * (capacity - sum_i(weight_i x_i)
                           - sum_k(2**k y_k))**2.

Run this file directly after editing the hyperparameters below. Each instance
is written as a compressed ``.npz`` file in the ``binary_knapsack/`` subdirectory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_HAMILTONIANS = 20  # Common choices are 20 or 40.
NUM_ITEMS = 4
VALUE_RANGE = (1, 10)  # Inclusive integer range.
WEIGHT_RANGE = (1, 5)  # Inclusive integer range.
CAPACITY_FRACTION_RANGE = (0.40, 0.70)
PENALTY_MULTIPLIER = 1.10
RANDOM_SEED = 7000
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "binary_knapsack"
FILE_PREFIX = "binary_knapsack"


def _validate_hyperparameters() -> None:
    if NUM_HAMILTONIANS < 1 or NUM_ITEMS < 1:
        raise ValueError("NUM_HAMILTONIANS and NUM_ITEMS must be positive.")
    if VALUE_RANGE[0] < 1 or VALUE_RANGE[1] < VALUE_RANGE[0]:
        raise ValueError("VALUE_RANGE must be a positive increasing range.")
    if WEIGHT_RANGE[0] < 1 or WEIGHT_RANGE[1] < WEIGHT_RANGE[0]:
        raise ValueError("WEIGHT_RANGE must be a positive increasing range.")
    lo, hi = CAPACITY_FRACTION_RANGE
    if not 0.0 < lo <= hi < 1.0:
        raise ValueError("CAPACITY_FRACTION_RANGE must lie strictly inside (0, 1).")
    if PENALTY_MULTIPLIER <= 1.0:
        raise ValueError("PENALTY_MULTIPLIER must exceed 1.")


def build_qubo(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: int,
    penalty: float,
) -> tuple[np.ndarray, float, int]:
    """Return upper-triangular Q, offset, and number of slack bits.

    Energies use ``E(z) = offset + sum_{i<=j} Q[i,j] z[i] z[j]``.
    Item variables precede slack variables in ``z``.
    """
    n_items = len(values)
    n_slack = max(1, math.ceil(math.log2(capacity + 1)))
    slack_weights = 2 ** np.arange(n_slack, dtype=np.int64)
    constraint = np.concatenate((weights, slack_weights)).astype(float)
    linear_objective = np.concatenate((-values, np.zeros(n_slack)))

    n_variables = n_items + n_slack
    qubo = np.zeros((n_variables, n_variables), dtype=float)
    qubo[np.diag_indices(n_variables)] = (
        linear_objective
        + penalty * constraint**2
        - 2.0 * penalty * capacity * constraint
    )
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            qubo[i, j] = 2.0 * penalty * constraint[i] * constraint[j]
    return qubo, float(penalty * capacity**2), n_slack


def generate_instance(rng: np.random.Generator) -> dict[str, object]:
    """Sample one nontrivial integer knapsack and construct its QUBO."""
    values = rng.integers(VALUE_RANGE[0], VALUE_RANGE[1] + 1, size=NUM_ITEMS)
    weights = rng.integers(WEIGHT_RANGE[0], WEIGHT_RANGE[1] + 1, size=NUM_ITEMS)

    total_weight = int(weights.sum())
    fraction = rng.uniform(*CAPACITY_FRACTION_RANGE)
    capacity = int(np.clip(round(fraction * total_weight), weights.min(), total_weight - 1))

    # Integer weights imply every violated equality has squared residual >= 1.
    # This bound makes every infeasible state worse than the empty packing.
    penalty = float(PENALTY_MULTIPLIER * values.sum())
    qubo, offset, n_slack = build_qubo(values, weights, capacity, penalty)
    return {
        "values": values,
        "weights": weights,
        "capacity": capacity,
        "penalty": penalty,
        "num_slack_bits": n_slack,
        "qubo": qubo,
        "offset": offset,
    }


def generate_dataset() -> list[dict[str, object]]:
    """Generate and save all configured Hamiltonians."""
    _validate_hyperparameters()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    manifest: list[dict[str, object]] = []

    for index in range(NUM_HAMILTONIANS):
        instance = generate_instance(rng)
        path = OUTPUT_DIRECTORY / f"{FILE_PREFIX}_{index:03d}.npz"
        np.savez_compressed(
            path,
            qubo=instance["qubo"],
            offset=instance["offset"],
            values=instance["values"],
            weights=instance["weights"],
            capacity=instance["capacity"],
            penalty=instance["penalty"],
            num_items=NUM_ITEMS,
            num_slack_bits=instance["num_slack_bits"],
        )
        manifest.append(
            {
                "file": path.name,
                "capacity": int(instance["capacity"]),
                "penalty": float(instance["penalty"]),
                "num_items": NUM_ITEMS,
                "num_slack_bits": int(instance["num_slack_bits"]),
                "values": np.asarray(instance["values"]).tolist(),
                "weights": np.asarray(instance["weights"]).tolist(),
            }
        )

    manifest_path = OUTPUT_DIRECTORY / f"{FILE_PREFIX}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    generated = generate_dataset()
    print(f"Saved {len(generated)} binary-knapsack Hamiltonians to {OUTPUT_DIRECTORY}")
