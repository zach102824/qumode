"""Readout confusion matrices and shot sampling."""

from __future__ import annotations

import numpy as np
import pytest

from qumode_vqe.hamiltonian import hybrid_energy_tensor, hybrid_hamiltonian
from qumode_vqe.measurement import (
    MeasurementConfig,
    apply_confusion,
    energy_from_histogram,
    identity_confusion,
    is_column_stochastic,
    joint_probabilities,
    nearest_neighbor_fock_confusion,
    qubit_bitflip_confusion,
    sample_shots,
)
from qumode_vqe.vqe import HybridSimulator


def test_confusion_matrices_are_column_stochastic():
    assert is_column_stochastic(identity_confusion(2))
    assert is_column_stochastic(qubit_bitflip_confusion(0.02, 0.07))
    assert is_column_stochastic(nearest_neighbor_fock_confusion(8, 0.1))
    assert is_column_stochastic(nearest_neighbor_fock_confusion(8, 0.0))


def test_identity_confusion_leaves_histogram_unchanged():
    rng = np.random.default_rng(2)
    p = rng.random((2, 8, 8))
    p = p / p.sum()
    out = apply_confusion(p, identity_confusion(2), identity_confusion(8), identity_confusion(8))
    np.testing.assert_allclose(out, p, atol=1e-12)


def test_histogram_energy_equals_tr_h_rho(reference_xvec):
    sim = HybridSimulator()
    psi = sim.statevector(reference_xvec)
    rho = np.outer(np.asarray(psi.full()).reshape(-1), np.asarray(psi.full()).reshape(-1).conj())
    probs = joint_probabilities(rho)
    e_hist = energy_from_histogram(probs, hybrid_energy_tensor())
    e_tr = float(np.real(np.trace(hybrid_hamiltonian().full() @ rho)))
    assert e_hist == pytest.approx(e_tr, abs=1e-10)
    ev = sim.evaluate(reference_xvec)
    assert ev.energy_observed == pytest.approx(ev.energy_physical, abs=1e-10)


def test_finite_shots_converge_to_exact(reference_xvec):
    sim = HybridSimulator()
    exact = sim.evaluate(reference_xvec).measurement.physical_probs.reshape(-1)
    noisy = HybridSimulator(
        measurement=MeasurementConfig(n_shots=20000, seed=3)
    )
    sampled = noisy.evaluate(reference_xvec).measurement.observed_probs.reshape(-1)
    # Chi-squared style: max absolute deviation should shrink with shots.
    assert np.max(np.abs(sampled - exact)) < 0.03
    assert sampled.sum() == pytest.approx(1.0, abs=1e-12)


def test_bitflip_changes_observed_but_not_physical(reference_xvec):
    meas = MeasurementConfig(qubit_c=qubit_bitflip_confusion(0.2, 0.2))
    ev = HybridSimulator(measurement=meas).evaluate(reference_xvec)
    assert ev.energy_physical != pytest.approx(ev.energy_observed, abs=1e-6)
    assert ev.target_prob_observed < ev.target_prob_physical
    assert ev.measurement.observed_probs.sum() == pytest.approx(1.0, abs=1e-12)
