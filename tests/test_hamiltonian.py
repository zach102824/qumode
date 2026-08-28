"""Hamiltonian construction vs Eq. (25) and the hybrid mapping Eq. (26)."""

from __future__ import annotations

import numpy as np
import pytest

from qumode_vqe.hamiltonian import (
    EQ25_IDENTITY,
    EQ25_Z,
    EQ25_ZZ,
    EXACT_GROUND_ENERGY,
    TARGET_BITSTRING,
    TARGET_QNM,
    bits_from_qnm,
    bitstring_from_bits,
    computational_label,
    exact_ground,
    extract_ising_coefficients,
    hybrid_energy_tensor,
    hybrid_hamiltonian,
    hybrid_index,
    qnm_from_bits,
    qubit_hamiltonian_eq25,
    qubit_hamiltonian_from_qubo,
    qubo_energy,
)


def test_target_bitstring_energy_is_minus_twelve():
    bits = bits_from_qnm(*TARGET_QNM)
    assert bitstring_from_bits(bits) == TARGET_BITSTRING
    assert qnm_from_bits(bits) == TARGET_QNM
    assert qubo_energy(bits) == pytest.approx(-12.0)


def test_eq25_coefficients_from_qubo():
    coeffs = extract_ising_coefficients(qubit_hamiltonian_from_qubo())
    assert coeffs["I"] == pytest.approx(EQ25_IDENTITY, abs=1e-8)
    for i, val in EQ25_Z.items():
        assert coeffs["Z"][i] == pytest.approx(val, abs=1e-8)
    for pair, val in EQ25_ZZ.items():
        assert coeffs["ZZ"][pair] == pytest.approx(val, abs=1e-8)


def test_eq25_operator_matches_qubo_operator():
    err = (qubit_hamiltonian_eq25() - qubit_hamiltonian_from_qubo()).norm()
    assert err < 1e-8


def test_seven_qubit_ground_state():
    energy, vec = exact_ground(qubit_hamiltonian_from_qubo())
    assert energy == pytest.approx(EXACT_GROUND_ENERGY, abs=1e-10)
    assert computational_label(vec) == TARGET_BITSTRING


def test_hybrid_ground_state_is_060():
    h = hybrid_hamiltonian()
    energy, vec = exact_ground(h)
    assert energy == pytest.approx(EXACT_GROUND_ENERGY, abs=1e-10)
    amps = np.abs(np.asarray(vec.full()).reshape(-1))
    assert int(np.argmax(amps)) == hybrid_index(*TARGET_QNM)
    assert hybrid_energy_tensor()[TARGET_QNM] == pytest.approx(-12.0)


def test_flattened_qubit_and_hybrid_diagonals_match():
    h7 = np.real(np.diag(qubit_hamiltonian_from_qubo().full()))
    hh = np.real(np.diag(hybrid_hamiltonian().full()))
    np.testing.assert_allclose(h7, hh, atol=1e-10)


def test_optimal_knapsack_feasibility():
    bits = bits_from_qnm(*TARGET_QNM)
    weight = 2.5 * bits[0] + 3 * bits[1] + 4 * bits[2] + 3.5 * bits[3]
    value = 2 * bits[0] + 5 * bits[1] + 7 * bits[2] + 3 * bits[3]
    assert weight == 7
    assert value == 12


def test_paper_knapsack_spectrum_has_penalty_walls():
    from qumode_vqe.hamiltonian import energy_spectrum_stats, hybrid_energy_tensor

    spec = energy_spectrum_stats(hybrid_energy_tensor())
    assert spec["energy_min"] == pytest.approx(-12.0)
    assert spec["gap"] == pytest.approx(2.0)
    # Infeasible packings sit on a high penalty wall.
    assert spec["spread"] > 100.0
    assert spec["n_ground"] == 1


def test_knapsack_sweep_starts_with_paper_and_samples_variants():
    from qumode_vqe.hamiltonian import (
        DEFAULT_VALUES,
        knapsack_packing_stats,
        knapsack_sweep_instance,
        sample_bkp_instance,
    )

    kind, inst = knapsack_sweep_instance(0, 7000)
    assert kind == "paper"
    np.testing.assert_allclose(inst.values, DEFAULT_VALUES)
    rng = np.random.default_rng(1)
    sampled = sample_bkp_instance(rng)
    assert sampled.values.shape == (4,)
    assert sampled.weights.shape == (4,)
    assert 0.5 <= sampled.capacity <= 7.0
    kinds = [knapsack_sweep_instance(i, 7000)[0] for i in range(10)]
    assert len(set(kinds)) == 10
    bits = bits_from_qnm(*TARGET_QNM)
    pack = knapsack_packing_stats(bits, inst)
    assert pack["feasible"]
    assert pack["value"] == pytest.approx(12.0)


def test_knapsack_sweep_qubo_ground_is_feasible():
    from qumode_vqe.hamiltonian import (
        energy_spectrum_stats,
        hybrid_energy_tensor,
        knapsack_packing_stats,
        knapsack_sweep_instance,
        bits_from_qnm,
    )

    for i in range(10):
        _kind, inst = knapsack_sweep_instance(i, 7000)
        spec = energy_spectrum_stats(hybrid_energy_tensor((8, 8), inst))
        pack = knapsack_packing_stats(bits_from_qnm(*spec["ground_qnm"]), inst)
        assert pack["feasible"], (i, _kind, pack, spec["energy_min"])


def test_three_body_z_string_is_pubo_not_qubo():
    from qumode_vqe.hamiltonian import (
        energy_from_z_terms,
        energy_tensor_from_z_terms,
        max_pauli_weight,
        pauli_z_label,
        random_diag_ising_terms,
    )

    terms = [((1, 5, 6), 1.25)]
    assert pauli_z_label(terms[0][0]) == "Z1Z5Z6"
    assert max_pauli_weight(terms) == 3
    zeros = np.zeros(7, dtype=int)
    ones = np.ones(7, dtype=int)
    # Z|0>=+1, Z|1>=-1 so Z1Z5Z6|0000000> = +1 and |1111111> = -1
    assert energy_from_z_terms(zeros, terms) == pytest.approx(1.25)
    assert energy_from_z_terms(ones, terms) == pytest.approx(-1.25)
    flipped = zeros.copy()
    flipped[1] = 1
    assert energy_from_z_terms(flipped, terms) == pytest.approx(-1.25)

    rng = np.random.default_rng(0)
    rand_terms = random_diag_ising_terms(7, rng, n_body={1: 2, 2: 3, 3: 4, 4: 2}, scale=1.0)
    assert max_pauli_weight(rand_terms) == 4
    assert any(len(s) == 3 for s, _ in rand_terms)
    tensor = energy_tensor_from_z_terms(rand_terms, (8, 8))
    assert tensor.shape == (2, 8, 8)
    assert np.isfinite(tensor).all()


def test_tabu_penalty_raises_only_forbidden_entries():
    from qumode_vqe.hamiltonian import tabu_penalized_energy_tensor

    orig = hybrid_energy_tensor()
    penalized = tabu_penalized_energy_tensor([(0, 3, 0), (1, 0, 1)], penalty=50.0)
    assert penalized[0, 3, 0] == pytest.approx(orig[0, 3, 0] + 50.0)
    assert penalized[1, 0, 1] == pytest.approx(orig[1, 0, 1] + 50.0)
    assert penalized[TARGET_QNM] == pytest.approx(orig[TARGET_QNM])
    mask = np.ones(orig.shape, dtype=bool)
    mask[0, 3, 0] = False
    mask[1, 0, 1] = False
    np.testing.assert_allclose(penalized[mask], orig[mask])


def test_knapsack_benchmark_instances_are_feasible_and_not_paper():
    from qumode_vqe.hamiltonian import (
        DEFAULT_CAPACITY,
        DEFAULT_VALUES,
        DEFAULT_WEIGHTS,
        knapsack_benchmark_instance,
        knapsack_packing_stats,
        bits_from_qnm,
        energy_spectrum_stats,
        hybrid_energy_tensor,
    )

    paper_vw = (tuple(DEFAULT_VALUES), tuple(DEFAULT_WEIGHTS), DEFAULT_CAPACITY)
    for i in range(8):
        inst = knapsack_benchmark_instance(i, 8000)
        key = (tuple(inst.values), tuple(inst.weights), inst.capacity)
        assert key != paper_vw
        spec = energy_spectrum_stats(hybrid_energy_tensor((8, 8), inst))
        pack = knapsack_packing_stats(bits_from_qnm(*spec["ground_qnm"]), inst)
        assert pack["feasible"], (i, pack, spec["energy_min"])


def test_ising_benchmark_terms_are_two_body_and_rms_normalized():
    from qumode_vqe.hamiltonian import (
        energy_tensor_from_z_terms,
        ising_benchmark_terms,
        max_pauli_weight,
    )

    terms = ising_benchmark_terms(0, 8000)
    assert max_pauli_weight(terms) == 2
    assert sum(1 for s, _ in terms if len(s) == 1) == 7
    rms = float(np.sqrt(np.mean([c * c for _, c in terms])))
    assert rms == pytest.approx(1.0, rel=1e-8)
    tensor = energy_tensor_from_z_terms(terms, (8, 8))
    assert tensor.shape == (2, 8, 8)
    assert np.isfinite(tensor).all()
