"""Fig. 14 qualitative photon-loss behaviour: κτ = 0.01 vs 0.1."""

from __future__ import annotations

import pytest

from qumode_vqe.hamiltonian import TARGET_QNM
from qumode_vqe.noise import TimingMode, paper_loss_config
from qumode_vqe.vqe import HybridSimulator, evaluate_fixed_parameters, optimize_vqe


def test_fixed_parameters_loss_threshold(reference_xvec):
    """Even without reoptimization, Fig. 14's qualitative split is visible."""
    ev_lo = evaluate_fixed_parameters(reference_xvec, paper_loss_config(0.01, TimingMode.PER_UER_LAYER))
    ev_hi = evaluate_fixed_parameters(reference_xvec, paper_loss_config(0.1, TimingMode.PER_UER_LAYER))
    assert ev_lo.most_likely == TARGET_QNM
    assert ev_lo.target_prob_physical > 0.5
    assert ev_hi.target_prob_physical < ev_lo.target_prob_physical
    assert ev_hi.physicality["trace_error"] < 1e-8
    assert ev_lo.physicality["min_eig"] > -1e-8


def test_per_ecd_timing_applies_twice_as_many_channels(reference_xvec):
    uer = paper_loss_config(0.01, TimingMode.PER_UER_LAYER)
    ecd = paper_loss_config(0.01, TimingMode.PER_ECD_PAIR)
    p_uer = evaluate_fixed_parameters(reference_xvec, uer).target_prob_physical
    p_ecd = evaluate_fixed_parameters(reference_xvec, ecd).target_prob_physical
    # Same κτ per application but twice as many applications → more loss.
    assert p_ecd < p_uer + 1e-6


@pytest.mark.slow
def test_reoptimization_at_mild_loss_keeps_solution(reference_xvec):
    noise = paper_loss_config(0.01, TimingMode.PER_UER_LAYER)
    sim = HybridSimulator(noise=noise)
    opt = optimize_vqe(sim, reference_xvec, method="BFGS", maxiter=25, record_every=5)
    final = sim.evaluate(opt.x)
    assert final.most_likely == TARGET_QNM
    assert final.target_prob_physical > 0.4


@pytest.mark.slow
def test_reoptimization_at_strong_loss_degrades(reference_xvec):
    noise = paper_loss_config(0.1, TimingMode.PER_UER_LAYER)
    sim = HybridSimulator(noise=noise)
    opt = optimize_vqe(sim, reference_xvec, method="BFGS", maxiter=15, record_every=5)
    final = sim.evaluate(opt.x)
    mild = evaluate_fixed_parameters(reference_xvec, paper_loss_config(0.01))
    assert final.target_prob_physical < mild.target_prob_physical
    assert final.physicality["trace_error"] < 1e-8
