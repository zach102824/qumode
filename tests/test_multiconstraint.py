"""Multiple-constraint Hamiltonian Eq. (31) and hybrid mapping Eq. (32)."""

from __future__ import annotations

import numpy as np
import pytest

from qumode_vqe.hamiltonian import (
    EQ31_IDENTITY,
    EQ31_Z,
    EQ31_ZZ,
    MC_EXACT_GROUND_ENERGY,
    MC_N_QUBITS,
    MC_PARTITION,
    MC_TARGET_BITSTRING,
    MC_TARGET_QNM,
    bits_from_qnm,
    bitstring_from_bits,
    computational_label,
    exact_ground,
    extract_ising_coefficients,
    hybrid_index,
    mc_hybrid_energy_tensor,
    mc_hybrid_hamiltonian,
    mc_qubo_energy,
    qnm_from_bits,
    qubit_hamiltonian_eq31,
    qubit_hamiltonian_from_mc_qubo,
)


def test_target_maps_to_104():
    bits = bits_from_qnm(*MC_TARGET_QNM, MC_PARTITION)
    assert bitstring_from_bits(bits) == MC_TARGET_BITSTRING
    assert qnm_from_bits(bits, MC_PARTITION) == MC_TARGET_QNM
    assert mc_qubo_energy(bits) == pytest.approx(MC_EXACT_GROUND_ENERGY)


def test_eq31_coefficients_from_qubo():
    coeffs = extract_ising_coefficients(qubit_hamiltonian_from_mc_qubo(), n_qubits=MC_N_QUBITS)
    assert coeffs["I"] == pytest.approx(EQ31_IDENTITY, abs=1e-8)
    for i, val in EQ31_Z.items():
        assert coeffs["Z"][i] == pytest.approx(val, abs=1e-8)
    for pair, val in EQ31_ZZ.items():
        assert coeffs["ZZ"][pair] == pytest.approx(val, abs=1e-8)
    assert coeffs["Z"].get(5, 0.0) == pytest.approx(0.0, abs=1e-8)


def test_eq31_operator_matches_qubo():
    err = (qubit_hamiltonian_eq31() - qubit_hamiltonian_from_mc_qubo()).norm()
    assert err < 1e-8


def test_six_qubit_ground_state():
    energy, vec = exact_ground(qubit_hamiltonian_from_mc_qubo())
    assert energy == pytest.approx(MC_EXACT_GROUND_ENERGY, abs=1e-10)
    assert computational_label(vec, n_qubits=MC_N_QUBITS) == MC_TARGET_BITSTRING


def test_hybrid_ground_state_is_104():
    h = mc_hybrid_hamiltonian()
    energy, vec = exact_ground(h)
    assert energy == pytest.approx(MC_EXACT_GROUND_ENERGY, abs=1e-10)
    amps = np.abs(np.asarray(vec.full()).reshape(-1))
    assert int(np.argmax(amps)) == hybrid_index(*MC_TARGET_QNM, nfocks=(4, 8))
    assert mc_hybrid_energy_tensor()[MC_TARGET_QNM] == pytest.approx(1.0)
