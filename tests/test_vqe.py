"""Optimizer path guards."""

from __future__ import annotations

import numpy as np
import pytest

from qumode_vqe.hamiltonian import hybrid_energy_tensor
from qumode_vqe.measurement import MeasurementConfig, measure
from qumode_vqe.params import paper_bounds
from qumode_vqe.vqe import HybridSimulator, greedy_pair_aas, optimize_gibbs_adaptive, optimize_vqe


def test_bfgs_rejected_for_finite_shots():
    sim = HybridSimulator(measurement=MeasurementConfig(n_shots=32, seed=0))
    with pytest.raises(ValueError, match="stochastic"):
        optimize_vqe(sim, np.zeros(40), method="BFGS", maxiter=1, observed=True)


def test_spsa_runs_on_shot_objective():
    sim = HybridSimulator(measurement=MeasurementConfig(n_shots=200, seed=1))
    rng = np.random.default_rng(0)
    opt = optimize_vqe(
        sim, np.zeros(40), method="SPSA", maxiter=3, observed=True, rng=rng, record_every=1
    )
    assert opt.nit == 3
    assert opt.x.shape == (40,)
    assert np.isfinite(opt.fun)


def test_gibbs_objective_prefers_ground_state_mass():
    from qumode_vqe.hamiltonian import hybrid_energy_tensor
    from qumode_vqe.vqe import gibbs_objective

    energies = hybrid_energy_tensor()
    p_gs = np.zeros_like(energies)
    p_gs[0, 6, 0] = 1.0
    p_trap = np.zeros_like(energies)
    p_trap[0, 3, 0] = 1.0
    p_mix = 0.2 * p_gs + 0.8 * p_trap
    f_gs = gibbs_objective(p_gs, energies)
    f_mix = gibbs_objective(p_mix, energies)
    f_trap = gibbs_objective(p_trap, energies)
    assert f_gs < f_mix < f_trap


def test_gibbs_small_eta_matches_mean_energy():
    from qumode_vqe.hamiltonian import hybrid_energy_tensor
    from qumode_vqe.vqe import gibbs_objective

    energies = hybrid_energy_tensor()
    rng = np.random.default_rng(0)
    p = rng.random(energies.shape)
    p = p / p.sum()
    mean = float(np.sum(p * energies))
    eta = 1e-6
    # f = −ln ⟨e^{−ηE}⟩ = η μ − η² σ²/2 + …  ⇒ f/η → μ
    assert gibbs_objective(p, energies, eta=eta) / eta == pytest.approx(mean, rel=1e-3, abs=1e-3)


def test_optimize_vqe_gibbs_runs():
    sim = HybridSimulator()
    opt = optimize_vqe(
        sim, np.zeros(40), method="BFGS", maxiter=1, objective="gibbs", record_every=0
    )
    assert np.isfinite(opt.fun)
    assert opt.x.shape == (40,)


def test_slice_parameters_keeps_leading_layers():
    from qumode_vqe.params import random_parameters, slice_parameters, unpack

    rng = np.random.default_rng(1)
    x5 = random_parameters(5, rng)
    x3 = slice_parameters(x5, 5, 3)
    p5 = unpack(x5, 5)
    p3 = unpack(x3, 3)
    np.testing.assert_allclose(p3.beta, p5.beta[:3])
    np.testing.assert_allclose(p3.theta, p5.theta[:3])
    np.testing.assert_allclose(p3.phi, p5.phi[:3])


def test_greedy_pair_aas_returns_valid_mask():
    sim = HybridSimulator(ndepth=5, cost_kind="gibbs")
    mask = greedy_pair_aas(sim, np.zeros(40), max_remove=2)
    assert mask.shape == (5, 2)
    assert mask.dtype == bool
    assert int(mask.sum()) >= 1


def test_simulator_uses_non_vacuum_initial_state():
    from qumode_vqe.circuit import ry_coherent_product, vacuum

    init = ry_coherent_product(np.pi / 2.0, 0.0, 0.0, (8, 8))
    sim = HybridSimulator(ndepth=5, initial_state=init)
    psi = sim.statevector(np.zeros(40))
    assert (psi - init).norm() < 1e-10
    assert (psi - vacuum((8, 8))).norm() > 0.1
    rho = sim.density_matrix(np.zeros(40))
    from qumode_vqe.channels import ket_to_dm

    np.testing.assert_allclose(rho, ket_to_dm(init), atol=1e-10)


def test_evaluate_without_target_is_nan():
    sim = HybridSimulator(target_qnm=None)
    ev = sim.evaluate(np.zeros(40))
    assert np.isnan(ev.target_prob_physical)
    assert np.isnan(ev.target_prob_observed)
    p = np.zeros((2, 8, 8))
    p[0, 0, 0] = 1.0
    meas = measure(p, target=None)
    assert np.isnan(meas.target_prob_physical)
    assert meas.most_likely == (0, 0, 0)


def test_bounded_spsa_stays_in_bounds():
    sim = HybridSimulator(ndepth=1, target_qnm=None)
    bounds = paper_bounds(1)
    rng = np.random.default_rng(4)
    x0 = np.full(8, 100.0)
    opt = optimize_vqe(
        sim, x0, method="SPSA", maxiter=2, rng=rng, bounds=bounds, record_every=0
    )
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    assert np.all(opt.x >= lo - 1e-12)
    assert np.all(opt.x <= hi + 1e-12)
    assert opt.nit == 2


def test_adaptive_gibbs_continues_warmed_ansatz():
    """Joint-only (production) and optional freeze continuation share the same warmup."""
    rng = np.random.default_rng(5)
    prep0 = np.array([np.pi / 2.0, 0.1, 0.0, 0.0, 0.2], dtype=float)
    x0 = np.zeros(8)
    continued = optimize_gibbs_adaptive(
        prep0,
        x0,
        ndepth=1,
        nfocks=(4, 4),
        outer_iter=1,
        spsa_iter=0,
        rng=rng,
    )
    np.testing.assert_allclose(continued.x, continued.x_warmup)
    assert continued.nit_warmup == 1
    assert continued.nit == 0
    assert continued.prep.shape == (5,)
    assert np.isfinite(continued.fun)

    rng2 = np.random.default_rng(5)
    two_stage = optimize_gibbs_adaptive(
        prep0,
        x0,
        ndepth=1,
        nfocks=(4, 4),
        outer_iter=1,
        spsa_iter=1,
        rng=rng2,
    )
    assert two_stage.nit_warmup == 1
    assert two_stage.nit == 1
    assert two_stage.x.shape == (8,)
    np.testing.assert_allclose(two_stage.x_warmup, continued.x_warmup)


def test_optimize_gibbs_adaptive_defaults_are_joint_seventy():
    from inspect import signature

    from qumode_vqe.vqe import DEFAULT_ANSATZ_STEPS, DEFAULT_JOINT_STEPS

    sig = signature(optimize_gibbs_adaptive)
    assert DEFAULT_JOINT_STEPS == 70
    assert DEFAULT_ANSATZ_STEPS == 0
    assert sig.parameters["outer_iter"].default == DEFAULT_JOINT_STEPS
    assert sig.parameters["spsa_iter"].default == DEFAULT_ANSATZ_STEPS


def test_sampled_tail_is_shift_invariant_and_falls_back_on_a_peak():
    from qumode_vqe.eta import SampledTailEta, clamp_eta, weighted_quantile

    e = hybrid_energy_tensor()
    p = np.full(e.shape, 1.0)
    p = p / p.sum()
    shifted = e + 40.0
    scaled = 2.0 * e

    s = SampledTailEta()
    eta_s = s.initialize(e, p).eta
    assert SampledTailEta().initialize(shifted, p).eta == pytest.approx(eta_s, rel=1e-5)
    assert SampledTailEta().initialize(scaled, p).eta == pytest.approx(eta_s / 2.0, rel=1e-5)

    peaked = np.zeros_like(e)
    peaked.reshape(-1)[int(np.argmin(e))] = 1.0
    st = SampledTailEta().initialize(e, peaked)
    assert st.fallback in {"degenerate_tail", "clamped"}

    lo, clamped = clamp_eta(1e9)
    assert clamped and lo == 50.0
    q = weighted_quantile(np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.0, 0.0]), 0.5)
    assert q == pytest.approx(0.0)


def test_sampled_eta_held_fixed_within_spsa_pair():
    from qumode_vqe.eta import SampledTailEta
    from qumode_vqe.vqe import HybridSimulator, run_spsa

    sim = HybridSimulator(ndepth=1, target_qnm=None, cost_kind="gibbs")
    policy = SampledTailEta(refresh_every=1)
    p0 = np.ones(sim.energy_tensor.shape)
    policy.initialize(sim.energy_tensor, p0 / p0.sum())
    seen: list[float] = []

    def fun(x):
        seen.append(float(policy.eta))
        return sim.cost(x, objective="gibbs", gibbs_eta=float(policy.eta))

    def before(k, x):
        ev = sim.evaluate(x)
        policy.maybe_update(k, 4, sim.energy_tensor, ev.measurement.physical_probs)

    rng = np.random.default_rng(0)
    run_spsa(fun, np.zeros(8), maxiter=2, rng=rng, bounds=paper_bounds(1), on_before_step=before)
    # two cost evals per step share η
    assert seen[0] == seen[1]
    assert seen[2] == seen[3]


def test_optimize_gibbs_adaptive_uses_sampled_tail():
    rng = np.random.default_rng(6)
    rec = optimize_gibbs_adaptive(
        np.array([0.2, 0.0, 0.0, 0.0, 0.0], dtype=float),
        np.zeros(8),
        ndepth=1,
        nfocks=(4, 4),
        outer_iter=1,
        spsa_iter=1,
        rng=rng,
        energy_tensor=hybrid_energy_tensor((4, 4)),
    )
    assert rec.eta_policy == "sampled_tail"
    assert rec.eta > 0
    assert rec.eta_history


def test_spsa_step_scale_zero_freezes_coordinate():
    from qumode_vqe.vqe import run_spsa

    rng = np.random.default_rng(0)

    def fun(x):
        return float(np.sum(np.square(x)))

    x0 = np.ones(4)
    opt = run_spsa(
        fun,
        x0,
        maxiter=6,
        rng=rng,
        bounds=[(-10.0, 10.0)] * 4,
        step_scale=np.array([0.0, 1.0, 1.0, 1.0]),
    )
    assert opt.x[0] == pytest.approx(1.0)
    assert not np.allclose(opt.x[1:], 1.0)


def test_prep_step_scale_shrinks_warmup_prep_motion():
    prep0 = np.array([np.pi / 2.0, 0.1, 0.0, 0.0, 0.2], dtype=float)
    x0 = np.zeros(8)
    kw = dict(ndepth=1, nfocks=(4, 4), outer_iter=2, spsa_iter=0)
    full = optimize_gibbs_adaptive(prep0, x0, rng=np.random.default_rng(7), prep_step_scale=1.0, **kw)
    tiny = optimize_gibbs_adaptive(prep0, x0, rng=np.random.default_rng(7), prep_step_scale=0.1, **kw)
    d_full = float(np.linalg.norm(full.prep - prep0))
    d_tiny = float(np.linalg.norm(tiny.prep - prep0))
    assert d_tiny < d_full
    assert d_tiny > 0.0
