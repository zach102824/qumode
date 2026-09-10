"""7-qubit 4-SAT Hamiltonian: unique SAT ground state and Pauli expansion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qumode_vqe.hamiltonian import (
    DEFAULT_NFOCKS,
    N_QUBITS,
    bits_from_qubit_index,
    energy_from_z_terms,
    energy_tensor_from_z_terms,
    load_four_sat_instances,
    z_terms_from_four_sat_npz,
)

ROOT = Path(__file__).resolve().parents[1]
HAM_DIR = ROOT / "Hamiltonians" / "four_sat"


def _first_npz() -> Path:
    paths = sorted(HAM_DIR.glob("four_sat_[0-9][0-9][0-9].npz"))
    if not paths:
        pytest.skip(f"no 4-SAT NPZ files in {HAM_DIR}")
    return paths[0]


def _clause_energy(bits: np.ndarray, clauses: np.ndarray, polarities: np.ndarray) -> int:
    energy = 0
    for variables, poles in zip(clauses, polarities, strict=True):
        pattern = np.where(np.asarray(poles, dtype=int) > 0, 0, 1)
        energy += int(np.all(bits[np.asarray(variables, dtype=int)] == pattern))
    return energy


def test_four_sat_instance_is_unique_and_matches_clause_counts():
    path = _first_npz()
    data = np.load(path)
    terms, meta = z_terms_from_four_sat_npz(path)
    assert meta["num_spins"] == N_QUBITS
    assert 12 <= int(meta["num_clauses"]) <= 20
    assert int(meta["clause_width"]) == 4
    assert meta["max_body_order"] == 4

    clauses = np.asarray(data["clauses"])
    polarities = np.asarray(data["polarities"])
    identity = float(meta["identity"])
    clause_energies = np.array(
        [
            _clause_energy(bits_from_qubit_index(idx), clauses, polarities)
            for idx in range(1 << N_QUBITS)
        ]
    )
    assert int(np.sum(clause_energies == 0)) == 1
    gs_idx = int(np.argmin(clause_energies))
    gs_bits = bits_from_qubit_index(gs_idx)
    assert gs_bits.sum() not in (0, N_QUBITS)

    for idx in range(1 << N_QUBITS):
        bits = bits_from_qubit_index(idx)
        pauli_e = energy_from_z_terms(bits, terms, identity)
        assert pauli_e == pytest.approx(float(clause_energies[idx]), abs=1e-12)

    for site in range(N_QUBITS):
        flipped = gs_bits.copy()
        flipped[site] = 1 - flipped[site]
        assert _clause_energy(flipped, clauses, polarities) >= 1


def test_load_four_sat_instances_hybrid_ground_state():
    instances = load_four_sat_instances(HAM_DIR)
    assert len(instances) >= 1
    inst = instances[0]
    tensor = inst["energy_tensor"]
    assert tensor.shape == (2, DEFAULT_NFOCKS[0], DEFAULT_NFOCKS[1])
    assert inst["family"] == "four_sat"
    assert inst["n_ground"] == 1
    assert inst["max_pauli_weight"] == 4
    assert inst["energy_min"] == pytest.approx(0.0, abs=1e-7)
    terms, meta = z_terms_from_four_sat_npz(HAM_DIR / inst["file"])
    untilted = energy_tensor_from_z_terms(
        terms, DEFAULT_NFOCKS, identity=float(meta["identity"]), tilt=0.0
    )
    assert untilted[tuple(inst["ground_qnm"])] == pytest.approx(0.0, abs=1e-12)
    assert inst["ground_bitstring"] not in {"0000000", "1111111"}
