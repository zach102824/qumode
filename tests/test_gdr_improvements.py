"""Unit tests for regularized GDR, hybrid ZNE+readout, and twin design."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Error_mitigation.advanced import (
    GdrRegConfig,
    fit_gdr_param_reg,
    kernel_identity_tvd,
    readout_then_zne,
    unfold_safe,
    unfold_shrink,
    zne_then_readout,
)
from Error_mitigation.metrics import total_variation
from Error_mitigation.mitigation import (
    apply_transfer,
    binomial_loss_kernel,
    fit_gdr_param,
    observe_histogram,
    params_to_kernels,
    sample_shots_from,
    unfold,
    zne_histogram,
)
from Error_mitigation.noise_models import circuit_noise, readout_spec, scale_noise
from Error_mitigation.twins import build_twins, mag_scale_range_for_twin
from qumode_vqe.hamiltonian import hybrid_energy_tensor
from qumode_vqe.measurement import is_column_stochastic, nearest_neighbor_fock_confusion
from qumode_vqe.params import random_parameters
from qumode_vqe.vqe import HybridSimulator


def _synthetic_binomial_case(*, eta=0.985, n_train=12, shots=4096, seed=0, p_nn=0.0, p01=0.0, p10=0.0):
    from Error_mitigation.twins import product_coherent_histogram

    rng = np.random.default_rng(seed)
    dims = (2, 8, 8)
    b = binomial_loss_kernel(eta, 8)
    cq = np.array([[1.0 - p01, p10], [p01, 1.0 - p10]], dtype=float)
    c_nn = nearest_neighbor_fock_confusion(8, p_nn) if p_nn > 0.0 else np.eye(8)
    p_ideals, q_obs = [], []
    for i in range(n_train):
        qbit = int(rng.integers(0, 2))
        a1 = complex(rng.uniform(0.25, 1.7), rng.uniform(-0.4, 0.4))
        a2 = complex(rng.uniform(0.25, 1.7), rng.uniform(-0.4, 0.4))
        p = product_coherent_histogram(qbit, a1, a2, dims)
        blurred = apply_transfer(p, cq, c_nn @ b, c_nn @ b)
        p_ideals.append(p)
        q_obs.append(sample_shots_from(blurred, shots, seed + 17 * (i + 1)))
    qbit = 1
    p_tgt = product_coherent_histogram(qbit, 1.1 + 0.2j, 0.9 - 0.1j, dims)
    q_tgt = sample_shots_from(apply_transfer(p_tgt, cq, c_nn @ b, c_nn @ b), shots, seed + 999)
    return p_ideals, q_obs, p_tgt, q_tgt, dims


def test_mag_scale_stratified_spans_small_and_large():
    ranges = [mag_scale_range_for_twin(i, 12, "stratified", (0.5, 1.0)) for i in range(12)]
    lows = [r[0] for r in ranges]
    highs = [r[1] for r in ranges]
    assert min(lows) <= 0.15
    assert max(highs) >= 1.5
    assert mag_scale_range_for_twin(0, 12, "uniform", (0.5, 1.0)) == (0.5, 1.0)


def test_stratified_twins_still_product_states():
    sim = HybridSimulator(ndepth=2, nfocks=(8, 8), ansatz="ecd", energy_tensor=hybrid_energy_tensor((8, 8)))
    rng = np.random.default_rng(20260904)
    x = random_parameters(2, rng)
    twins = build_twins(sim, x, rng, n_train=6, n_rank2=0, alpha_policy="stratified")
    assert len(twins) == 6
    for tw in twins:
        assert tw.t_free == 0
        assert tw.product_tvd is not None
        assert tw.product_tvd < 1e-6


def test_regularized_gdr_beats_raw_on_mild_binomial():
    """Success-criterion analogue: κτ=0.003-like η, 12 twins, 4096 shots."""
    eta = float(np.exp(-5 * 0.003))
    p_tr, q_tr, p_tgt, q_tgt, dims = _synthetic_binomial_case(eta=eta, n_train=12, shots=4096, seed=7)
    cfg = circuit_noise("loss", 0.003)
    spec = readout_spec("ideal", n_shots=4096)
    tvd_raw = total_variation(q_tgt, p_tgt)

    theta_rawfit, _ = fit_gdr_param(p_tr, q_tr, cfg, spec, ndepth=5, dims=dims, maxiter=80)
    cq, c1, c2 = params_to_kernels(theta_rawfit, dims)
    tvd_unreg = total_variation(unfold(q_tgt, cq, c1, c2), p_tgt)

    reg = GdrRegConfig(reduced="eta", holdout_frac=0.0, unfold_mode="safe", l2_eta=80.0, cv_l2_eta=None)
    theta_reg, info = fit_gdr_param_reg(p_tr, q_tr, cfg, spec, ndepth=5, dims=dims, maxiter=80, reg=reg)
    cq_r, c1_r, c2_r = params_to_kernels(theta_reg, dims)
    hist_reg = unfold_safe(q_tgt, cq_r, c1_r, c2_r)
    tvd_reg = total_variation(hist_reg, p_tgt)
    assert tvd_reg < tvd_raw
    assert info["fitted"]["eta1"] > 0.9
    # Regularization should not wander far from the physical η.
    assert abs(info["fitted"]["eta1"] - eta) < 0.08
    # Document the unregularized comparison (may or may not beat raw; must not crash).
    assert tvd_unreg >= 0.0


def test_unfold_safe_near_identity_stays_close_to_raw():
    p = np.zeros((2, 8, 8))
    p[1, 2, 3] = 1.0
    q = sample_shots_from(p, 4000, seed=3)
    eye2, eye8 = np.eye(2), np.eye(8)
    rec = unfold_safe(q, eye2, eye8, eye8)
    # Aggressive RL on I would fit shot noise; safe unfold should stay nearer q.
    tvd_to_q = total_variation(rec, q)
    tvd_full = total_variation(unfold(q, eye2, eye8, eye8, n_iter=80), q)
    assert tvd_to_q <= tvd_full + 1e-12
    assert kernel_identity_tvd(eye2, eye8, eye8) == pytest.approx(0.0)


def test_readout_then_zne_beats_zne_idle_under_readout():
    """Detector is not stretched: invert readout per scale, then extrapolate."""
    eta1 = 0.90
    dims = (2, 8, 8)
    p = np.zeros(dims)
    p[0, 4, 3] = 1.0
    spec = readout_spec("readout_strong", n_shots=0, seed=0)
    from Error_mitigation.mitigation import confusion_from_measurement
    from Error_mitigation.noise_models import readout_config

    meas = readout_config(spec, n_fock=8)
    cq, cro1, cro2 = confusion_from_measurement(meas, dims)
    hist_by_scale = {}
    for s, eta in ((1, eta1), (2, eta1**2), (3, eta1**3)):
        b = binomial_loss_kernel(eta, 8)
        phys = apply_transfer(p, np.eye(2), b, b)
        hist_by_scale[s] = apply_transfer(phys, cq, cro1, cro2)
    zne_raw = zne_histogram(hist_by_scale, degree=2)
    hyb = readout_then_zne(hist_by_scale, spec, dims, degree=2)
    post = zne_then_readout(hist_by_scale, spec, dims, degree=2)
    tvd_idle = total_variation(zne_raw, p)
    tvd_hyb = total_variation(hyb, p)
    tvd_post = total_variation(post, p)
    assert tvd_hyb < tvd_idle
    # Post-readout invert should also beat naive ZNE (detector unscaled).
    assert tvd_post < tvd_idle


def test_frozen_readout_recovers_loss_eta():
    eta = 0.92
    p_tr, q_tr, p_tgt, q_tgt, dims = _synthetic_binomial_case(
        eta=eta, n_train=16, shots=8000, seed=11, p_nn=0.03, p01=0.01, p10=0.03
    )
    cfg = circuit_noise("loss", 0.03)
    spec = readout_spec("readout_realistic", n_shots=8000)
    reg = GdrRegConfig(reduced="eta", holdout_frac=0.0, unfold_mode="safe", l2_eta=10.0)
    theta, info = fit_gdr_param_reg(p_tr, q_tr, cfg, spec, ndepth=5, dims=dims, maxiter=80, reg=reg)
    assert info["fitted"]["p01"] == pytest.approx(spec.p01, abs=1e-12)
    assert info["fitted"]["p_nn1"] == pytest.approx(spec.p_nn, abs=1e-12)
    # Cumulative η for κτ=0.03, nd=5 is exp(-0.15)≈0.86; here true synthetic η=0.92.
    # Frozen readout should still land in the right ballpark, not collapse to p_nn.
    assert 0.7 < info["fitted"]["eta1"] < 1.0


def test_reg_kernels_column_stochastic():
    eta = 0.9
    p_tr, q_tr, _, _, dims = _synthetic_binomial_case(eta=eta, n_train=8, shots=2000, seed=2)
    cfg = circuit_noise("loss", 0.03)
    spec = readout_spec("readout_realistic", n_shots=2000)
    theta, _ = fit_gdr_param_reg(
        p_tr, q_tr, cfg, spec, 5, dims, maxiter=40, reg=GdrRegConfig(reduced="eta_nth", holdout_frac=0.0)
    )
    cq, c1, c2 = params_to_kernels(theta, dims)
    assert is_column_stochastic(cq)
    assert is_column_stochastic(c1)
    assert is_column_stochastic(c2)


def test_unfold_shrink_mixes_toward_raw_when_kernel_is_identity():
    p = np.zeros((2, 8, 8))
    p[0, 1, 1] = 1.0
    spec = readout_spec("ideal", n_shots=2000)
    q = sample_shots_from(p, 2000, seed=5)
    mixed = unfold_shrink(q, np.eye(2), np.eye(8), np.eye(8), spec, (2, 8, 8))
    assert total_variation(mixed, q) < 1e-12
