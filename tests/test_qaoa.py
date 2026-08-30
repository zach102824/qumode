"""Qubit QAOA / HEA encoding, unitarity, and SPSA smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qumode_vqe.hamiltonian import (
    bits_from_qubit_index,
    bitstring_from_bits,
    hybrid_energy_tensor,
    ising_benchmark_terms,
    energy_tensor_from_z_terms,
    knapsack_benchmark_instance,
    qnm_from_bits,
    qubit_index_from_bits,
)
from qumode_vqe.qaoa import (
    apply_mixer,
    apply_nearest_cz,
    apply_nearest_cz_sequential,
    energy_vector_from_tensor,
    evaluate_histogram,
    hea_statevector,
    n_hea_params,
    nearest_cz_even_odd_pairs,
    plus_state,
    probabilities,
    qaoa_statevector,
    qaoa_unitary_matrix,
    random_qaoa_params,
    optimize_qubit_ansatz,
    verify_energy_vector,
)
from qumode_vqe.vqe import gibbs_objective

REPO = Path(__file__).resolve().parents[1]
HAMILTONIANS = REPO / "results" / "hamiltonians.json"


def _first_of_family(family: str) -> dict:
    metas = json.loads(HAMILTONIANS.read_text(encoding="utf-8"))
    if isinstance(metas, dict):
        metas = metas.get("hamiltonians", [])
    for meta in metas:
        if str(meta.get("family", meta.get("kind"))) == family:
            return meta
    raise AssertionError(f"no {family} Hamiltonian in {HAMILTONIANS}")


@pytest.mark.skipif(not HAMILTONIANS.exists(), reason="stored hamiltonians.json missing")
def test_encoding_matches_stored_knapsack_tensor():
    meta = _first_of_family("knapsack")
    tensor = np.asarray(meta["energy_tensor"], dtype=float)
    e = energy_vector_from_tensor(tensor, ground_bitstring=meta["ground_bitstring"])
    verify_energy_vector(e, tensor, ground_bitstring=meta["ground_bitstring"])
    gidx = int(meta["ground_bitstring"], 2)
    assert e[gidx] == pytest.approx(float(np.min(tensor)), abs=1e-8)
    bits = bits_from_qubit_index(gidx)
    q, n, m = qnm_from_bits(bits)
    assert tensor[q, n, m] == pytest.approx(e[gidx], abs=1e-12)
    assert bitstring_from_bits(bits) == meta["ground_bitstring"]


@pytest.mark.skipif(not HAMILTONIANS.exists(), reason="stored hamiltonians.json missing")
def test_encoding_matches_stored_ising_tensor():
    meta = _first_of_family("ising")
    tensor = np.asarray(meta["energy_tensor"], dtype=float)
    e = energy_vector_from_tensor(tensor, ground_bitstring=meta["ground_bitstring"])
    verify_energy_vector(e, tensor, ground_bitstring=meta["ground_bitstring"])
    assert e[int(meta["ground_bitstring"], 2)] == pytest.approx(float(np.min(tensor)), abs=1e-8)


def test_encoding_matches_generated_knapsack_and_ising_fixtures():
    knap = knapsack_benchmark_instance(0, ham_seed=8000)
    knap_tensor = hybrid_energy_tensor((8, 8), knap)
    knap_e = energy_vector_from_tensor(knap_tensor)
    for idx in (0, 1, 17, 64, 127):
        bits = bits_from_qubit_index(idx)
        q, n, m = qnm_from_bits(bits)
        assert knap_e[idx] == pytest.approx(float(knap_tensor[q, n, m]), abs=1e-12)
        assert qubit_index_from_bits(bits) == idx

    terms = ising_benchmark_terms(0, ham_seed=8000)
    ising_tensor = energy_tensor_from_z_terms(terms, (8, 8))
    ising_e = energy_vector_from_tensor(ising_tensor)
    for idx in range(0, 128, 7):
        bits = bits_from_qubit_index(idx)
        q, n, m = qnm_from_bits(bits)
        assert ising_e[idx] == pytest.approx(float(ising_tensor[q, n, m]), abs=1e-12)


def test_qaoa_p1_two_qubit_ising_is_unitary_and_normalized():
    # Diagonal ZZ: |00>,|11> = +1; |01>,|10> = −1.
    energies = np.array([1.0, -1.0, -1.0, 1.0], dtype=float)
    gamma, beta = 0.37, 0.51
    psi = qaoa_statevector([gamma], [beta], energies, n_qubits=2)
    assert psi.shape == (4,)
    assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-12)
    probs = probabilities(psi)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-12)
    assert np.all(probs >= -1e-15)

    u = qaoa_unitary_matrix([gamma], [beta], energies, n_qubits=2)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-12)
    plus = plus_state(2)
    np.testing.assert_allclose(u @ plus, psi, atol=1e-12)

    # Mixer alone is a product of R_x(2β) and must preserve the norm of |00>.
    basis = np.zeros(4, dtype=complex)
    basis[0] = 1.0
    apply_mixer(basis, beta, 2)
    assert np.linalg.norm(basis) == pytest.approx(1.0, abs=1e-12)


def test_even_odd_cz_matches_sequential_staircase():
    n_qubits = 7
    even, odd = nearest_cz_even_odd_pairs(n_qubits)
    assert even == [(0, 1), (2, 3), (4, 5)]
    assert odd == [(1, 2), (3, 4), (5, 6)]
    rng = np.random.default_rng(7)
    vec = rng.normal(size=1 << n_qubits) + 1j * rng.normal(size=1 << n_qubits)
    vec = vec / np.linalg.norm(vec)
    packed = apply_nearest_cz(vec.copy(), n_qubits)
    sequential = apply_nearest_cz_sequential(vec.copy(), n_qubits)
    np.testing.assert_allclose(packed, sequential, atol=1e-12, rtol=0.0)
    assert np.linalg.norm(packed) == pytest.approx(1.0, abs=1e-12)


def test_hea_probabilities_sum_to_one():
    rng = np.random.default_rng(0)
    n_qubits, n_layers = 3, 2
    x = rng.uniform(0.0, 2.0 * np.pi, size=n_hea_params(n_qubits, n_layers))
    psi = hea_statevector(x, n_qubits, n_layers)
    assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-12)
    assert float(probabilities(psi).sum()) == pytest.approx(1.0, abs=1e-12)
    # |0>^n plus all-zero angles stays |0>.
    psi0 = hea_statevector(np.zeros(n_hea_params(n_qubits, n_layers)), n_qubits, n_layers)
    np.testing.assert_allclose(np.abs(psi0), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_spsa_gibbs_runs_and_does_not_increase_cost_on_tiny_ising():
    # 2-qubit diagonal with unique ground |11>.
    energies = np.array([2.0, 1.0, 1.0, 0.0], dtype=float)
    rng = np.random.default_rng(4)
    x0 = random_qaoa_params(1, rng)
    p0 = probabilities(qaoa_statevector(*_split_qaoa(x0), energies, 2))
    f0 = gibbs_objective(p0, energies, eta=1.0)
    opt = optimize_qubit_ansatz(
        x0,
        energies,
        ansatz="qaoa",
        objective="gibbs",
        p=1,
        n_qubits=2,
        maxiter=8,
        rng=rng,
    )
    assert opt.nit == 8
    assert opt.nfev == 16
    assert opt.x.shape == (2,)
    assert np.isfinite(opt.fun)
    assert opt.eval.probs.sum() == pytest.approx(1.0, abs=1e-12)
    # Smoke: SPSA must run; on this instance it should not make Gibbs worse
    # than the unperturbed start by a large margin.
    assert opt.fun <= f0 + 0.5


def _split_qaoa(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x[:1], x[1:]


def test_success_metric_is_most_likely_ground():
    energies = np.array([0.0, 3.0, 1.0, 4.0], dtype=float)
    hit = np.array([0.6, 0.1, 0.2, 0.1])
    miss = np.array([0.2, 0.1, 0.6, 0.1])
    ev_hit = evaluate_histogram(hit, energies, eta=1.0)
    ev_miss = evaluate_histogram(miss, energies, eta=1.0)
    assert ev_hit.success
    assert ev_hit.most_likely_bitstring == "00"
    assert not ev_miss.success
    assert ev_miss.most_likely_bitstring == "10"
