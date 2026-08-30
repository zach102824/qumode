"""Generalized (1, k, k) embedding and even/odd CZ at n=9."""

from __future__ import annotations

import numpy as np
import pytest

from qumode_vqe.hamiltonian import (
    bits_from_qnm,
    bitstring_from_bits,
    bits_from_qubit_index,
    energy_spectrum_stats,
    energy_tensor_from_z_terms,
    hybrid_energy_tensor,
    hybrid_ladder,
    ising_benchmark_terms,
    knapsack_benchmark_instance,
    knapsack_packing_stats,
    qnm_from_bits,
    qubit_index_from_bits,
    slack_from_bits,
)
from qumode_vqe.qaoa import (
    apply_nearest_cz,
    apply_nearest_cz_sequential,
    energy_vector_from_tensor,
    n_hea_params,
    nearest_cz_even_odd_pairs,
    verify_energy_vector,
)


def test_hybrid_ladder_matches_dutta_sizes():
    n7 = hybrid_ladder(7)
    assert n7 == {
        "n_qubits": 7,
        "k": 3,
        "partition": (1, 3, 3),
        "nfocks": (8, 8),
        "n_primary": 4,
        "n_slack": 3,
        "dim": 128,
        "max_capacity": 7.0,
    }
    n9 = hybrid_ladder(9)
    assert n9["partition"] == (1, 4, 4)
    assert n9["nfocks"] == (16, 16)
    assert n9["dim"] == 512
    assert n9["n_primary"] == 5
    assert n9["n_slack"] == 4
    n11 = hybrid_ladder(11)
    assert n11["partition"] == (1, 5, 5)
    assert n11["nfocks"] == (32, 32)
    assert n11["dim"] == 2048
    assert n11["n_primary"] == 6
    assert n11["n_slack"] == 5


def test_n9_binary_fock_map_is_inverse():
    spec = hybrid_ladder(9)
    part = spec["partition"]
    nq = spec["n_qubits"]
    l1, l2 = spec["nfocks"]
    for q in (0, 1):
        for n in (0, 1, 6, 15):
            for m in (0, 3, 15):
                bits = bits_from_qnm(q, n, m, part)
                assert bits.shape == (9,)
                assert qnm_from_bits(bits, part) == (q, n, m)
                idx = qubit_index_from_bits(bits, nq)
                back = bits_from_qubit_index(idx, nq)
                np.testing.assert_array_equal(back, bits)
                assert 0 <= n < l1 and 0 <= m < l2


def test_n9_paper_example_generalizes_msb_fock():
    # Same rule as n=7 |0,6,0> ↔ 0110000: qubit 0 MSB, each cavity MSB-first.
    bits = bits_from_qnm(0, 6, 0, (1, 4, 4))
    assert bitstring_from_bits(bits) == "001100000"
    assert qnm_from_bits(bits, (1, 4, 4)) == (0, 6, 0)
    bits15 = bits_from_qnm(1, 15, 15, (1, 4, 4))
    assert bitstring_from_bits(bits15) == "111111111"


def test_n9_slack_is_four_lsb_bits():
    bits = np.array([1, 0, 0, 0, 0, 1, 1, 0, 1], dtype=int)
    # slack bits [1,1,0,1] → 1 + 2 + 0 + 8 = 11
    assert slack_from_bits(bits, n_primary=5, n_slack=4) == pytest.approx(11.0)


def test_n9_knapsack_and_ising_encoding_ground_matches_tensor():
    spec = hybrid_ladder(9)
    inst = knapsack_benchmark_instance(0, ham_seed=9000, n_primary=5, n_slack=4)
    assert inst.n_primary == 5
    assert inst.n_slack == 4
    knap_t = hybrid_energy_tensor(spec["nfocks"], inst, spec["partition"])
    knap_e = energy_vector_from_tensor(
        knap_t, spec["partition"], n_qubits=9, ground_bitstring=None
    )
    verify_energy_vector(knap_e, knap_t, spec["partition"])
    spec_k = energy_spectrum_stats(knap_t)
    gs_bits = bitstring_from_bits(bits_from_qnm(*spec_k["ground_qnm"], spec["partition"]))
    assert knap_e[int(gs_bits, 2)] == pytest.approx(spec_k["energy_min"], abs=1e-8)
    pack = knapsack_packing_stats(bits_from_qnm(*spec_k["ground_qnm"], spec["partition"]), inst)
    assert pack["feasible"]

    terms = ising_benchmark_terms(0, ham_seed=9000, n_qubits=9)
    assert sum(1 for s, _ in terms if len(s) == 1) == 9
    assert sum(1 for s, _ in terms if len(s) == 2) == 12
    ising_t = energy_tensor_from_z_terms(terms, spec["nfocks"], spec["partition"])
    ising_e = energy_vector_from_tensor(ising_t, spec["partition"], n_qubits=9)
    verify_energy_vector(ising_e, ising_t, spec["partition"])
    gs = energy_spectrum_stats(ising_t)
    gs_bits = bitstring_from_bits(bits_from_qnm(*gs["ground_qnm"], spec["partition"]))
    assert ising_e[int(gs_bits, 2)] == pytest.approx(gs["energy_min"], abs=1e-8)


def test_n7_benchmark_recipe_unchanged():
    inst = knapsack_benchmark_instance(0, ham_seed=8000)
    assert inst.n_primary == 4
    assert inst.n_slack == 3
    terms = ising_benchmark_terms(0, ham_seed=8000)
    assert sum(1 for s, _ in terms if len(s) == 1) == 7
    assert sum(1 for s, _ in terms if len(s) == 2) == 12


def test_even_odd_cz_n9_pairs_and_matches_sequential():
    n_qubits = 9
    even, odd = nearest_cz_even_odd_pairs(n_qubits)
    assert even == [(0, 1), (2, 3), (4, 5), (6, 7)]
    assert odd == [(1, 2), (3, 4), (5, 6), (7, 8)]
    rng = np.random.default_rng(9)
    vec = rng.normal(size=1 << n_qubits) + 1j * rng.normal(size=1 << n_qubits)
    vec = vec / np.linalg.norm(vec)
    packed = apply_nearest_cz(vec.copy(), n_qubits)
    sequential = apply_nearest_cz_sequential(vec.copy(), n_qubits)
    np.testing.assert_allclose(packed, sequential, atol=1e-12, rtol=0.0)
    assert np.linalg.norm(packed) == pytest.approx(1.0, abs=1e-12)


def test_hea_param_count_grows_with_n():
    assert n_hea_params(7, 5) == 42
    assert n_hea_params(9, 5) == 54
    assert n_hea_params(11, 5) == 66
    assert n_hea_params(11, 3) == 44
