"""Slow noiseless ECD-VQE optimization regression."""

from __future__ import annotations

import numpy as np
import pytest

from qumode_vqe.hamiltonian import TARGET_QNM
from qumode_vqe.params import random_parameters
from qumode_vqe.vqe import HybridSimulator, optimize_vqe


pytestmark = pytest.mark.slow


def test_polish_reference_stays_at_optimum(reference_xvec):
    sim = HybridSimulator()
    opt = optimize_vqe(sim, reference_xvec, method="BFGS", maxiter=20, record_every=5)
    final = sim.evaluate(opt.x)
    assert final.energy_physical < -11.9
    assert final.most_likely == TARGET_QNM
    assert final.target_prob_physical > 0.9


def test_seeded_random_start_resolves_target():
    sim = HybridSimulator()
    rng = np.random.default_rng(2026)
    x0 = random_parameters(5, rng)
    opt = optimize_vqe(sim, x0, method="BFGS", maxiter=80, record_every=10)
    final = sim.evaluate(opt.x)
    assert opt.history[0]["energy_physical"] > final.energy_physical - 1e-6
    assert final.energy_physical < -9.0
    assert final.most_likely == TARGET_QNM
    assert final.target_prob_physical > 0.5
