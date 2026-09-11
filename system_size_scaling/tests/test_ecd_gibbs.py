from __future__ import annotations

import numpy as np
import pytest

from system_size_scaling.ecd import (
    apply_ecd,
    apply_rotation,
    displace_matrix,
    hybrid_energy_tensor,
    n_ecd_parameters,
    qubit_rotation,
    random_ecd_parameters,
    vacuum_prep,
)
from system_size_scaling.embedding import embedding_for_n
from system_size_scaling.four_sat import generate_instance
from system_size_scaling.gibbs import EcdGibbsSim, gibbs_objective, optimize_gibbs_adaptive


def test_rotation_and_displace_are_unitary():
    r = qubit_rotation(0.7, 1.1)
    assert np.allclose(r.conj().T @ r, np.eye(2), atol=1e-12)
    d = displace_matrix(8, 0.4 - 0.2j)
    assert np.allclose(d.conj().T @ d, np.eye(8), atol=1e-10)


def test_ecd_preserves_norm():
    emb = embedding_for_n(7)
    rng = np.random.default_rng(0)
    psi = rng.normal(size=emb.dim) + 1j * rng.normal(size=emb.dim)
    psi = psi / np.linalg.norm(psi)
    out = apply_rotation(psi, emb.dims, 0, 0.5, 0.3)
    out = apply_ecd(out, emb.dims, 0, 1, 0.8 + 0.2j)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-12


def test_n7_param_count_matches_production():
    emb = embedding_for_n(7)
    assert emb.n_prep_params == 5
    assert n_ecd_parameters(3, emb.n_pairs) == 8 * 3
    assert n_ecd_parameters(5, emb.n_pairs) == 40


def test_vacuum_is_all_zero_bitstring():
    emb = embedding_for_n(7)
    sim = EcdGibbsSim(emb, np.zeros(emb.dims), ndepth=1, ground_bitstring="0000000")
    # Identity-ish: L=1 with β=θ=φ=0 is ECD(0)=X on the transmon, not vacuum.
    # Check prep ket is |0,0,0>.
    from system_size_scaling.ecd import prep_to_ket

    ket = prep_to_ket(vacuum_prep(emb), emb)
    assert np.argmax(np.abs(ket) ** 2) == 0


def test_four_sat_unique_and_hybrid_ground():
    rng = np.random.default_rng(27707)
    inst = generate_instance(rng, 7, search_trials=2000)
    assert inst["n_ground"] == 1
    assert inst["energy_min"] == 0
    assert 12 <= int(inst["n_clauses"]) <= 26
    emb = embedding_for_n(7)
    tensor = hybrid_energy_tensor(emb, inst["logical_energies"])
    gs_occ = emb.encode_bits([int(c) for c in inst["ground_bitstring"]])
    assert tensor[gs_occ] == pytest.approx(0.0, abs=1e-12)
    assert float(np.min(tensor)) == pytest.approx(0.0, abs=1e-12)
    assert int(np.sum(np.isclose(inst["logical_energies"], 0.0))) == 1


def test_n7_ecd_matches_production_qutip():
    qutip = pytest.importorskip("qutip")
    del qutip
    from qumode_vqe.circuit import prepare_state
    from qumode_vqe.params import random_parameters

    rng = np.random.default_rng(123)
    emb = embedding_for_n(7)
    x = random_parameters(3, rng)
    psi_prod = np.asarray(prepare_state(x, 3, (8, 8)).full()).reshape(-1)
    from system_size_scaling.ecd import apply_ecd_ansatz, prep_to_ket, vacuum_prep

    psi_ours = apply_ecd_ansatz(prep_to_ket(vacuum_prep(emb), emb), x, emb, 3)
    ov = np.vdot(psi_prod, psi_ours)
    assert abs(ov) ** 2 == pytest.approx(1.0, abs=1e-12)


def test_one_spsa_step_smoke():
    rng = np.random.default_rng(1)
    inst = generate_instance(rng, 7, search_trials=2000)
    emb = embedding_for_n(7)
    energy = hybrid_energy_tensor(emb, inst["logical_energies"])
    x0 = random_ecd_parameters(3, emb.n_pairs, rng)
    result = optimize_gibbs_adaptive(
        emb,
        energy,
        str(inst["ground_bitstring"]),
        x0,
        ndepth=3,
        outer_iter=1,
        rng=rng,
    )
    assert np.isfinite(result.fun)
    assert result.eval_final.most_likely_bitstring
    assert result.nfev >= 2
    # Gibbs on a probability histogram is defined.
    p = np.ones(emb.dims, dtype=float)
    assert np.isfinite(gibbs_objective(p, energy, 1.0))
