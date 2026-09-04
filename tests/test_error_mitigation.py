"""Fast checks for Error_mitigation kernels, twins, and readout."""

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

from Error_mitigation.metrics import total_variation
from Error_mitigation.mitigation import (
    binomial_loss_kernel,
    fock_kernel,
    observe_histogram,
    oracle_kernels,
    richardson_lucy,
    run_readout_only,
    thermal_loss_kernel,
)
from Error_mitigation.noise_models import (
    circuit_noise,
    is_trivial_readout,
    readout_config,
    readout_spec,
)
from Error_mitigation.twins import build_twins, product_coherent_histogram, truncated_poisson
from qumode_vqe.hamiltonian import hybrid_energy_tensor
from qumode_vqe.measurement import is_column_stochastic, nearest_neighbor_fock_confusion
from qumode_vqe.params import random_parameters, random_snap_parameters
from qumode_vqe.vqe import HybridSimulator


def test_kernels_are_column_stochastic():
    for eta in (1.0, 0.9, 0.5):
        b = binomial_loss_kernel(eta, 8)
        t = thermal_loss_kernel(eta, 0.0, 8)
        f = fock_kernel(eta, 0.05, 0.01, 0.01, 0.02, 0.03, 8)
        assert is_column_stochastic(b)
        assert is_column_stochastic(t)
        assert is_column_stochastic(f)


def test_pure_loss_thermal_matches_binomial():
    eta = 0.8
    b = binomial_loss_kernel(eta, 8)
    t = thermal_loss_kernel(eta, 0.0, 8)
    # Truncation at n=L-1 makes the last column differ slightly.
    assert np.allclose(b[:, :7], t[:, :7], atol=0.02)


def test_readout_confusion_column_stochastic():
    for level in ("ideal", "readout_realistic", "readout_strong"):
        spec = readout_spec(level, n_shots=100, seed=0)
        cfg = readout_config(spec, n_fock=8)
        if cfg.qubit_c is not None:
            assert is_column_stochastic(cfg.qubit_c)
            assert is_column_stochastic(cfg.fock1_c)
            assert is_column_stochastic(cfg.fock2_c)
        assert is_column_stochastic(nearest_neighbor_fock_confusion(8, spec.p_nn))


def test_readout_only_skipped_on_ideal():
    spec = readout_spec("ideal", n_shots=100)
    assert is_trivial_readout(spec)
    p = np.zeros((2, 8, 8))
    p[0, 0, 0] = 1.0
    assert run_readout_only(p, spec, (2, 8, 8)) is None


def test_readout_only_reduces_confusion():
    spec = readout_spec("readout_strong", n_shots=0, seed=1)
    rng = np.random.default_rng(0)
    p = rng.random((2, 8, 8))
    p = p / p.sum()
    q = observe_histogram(p, spec, (2, 8, 8), seed=1)
    mit = run_readout_only(q, spec, (2, 8, 8))
    assert mit is not None
    assert total_variation(mit.histogram, p) < total_variation(q, p)


def test_rl_inverts_binomial_blur():
    eta = 0.85
    b = binomial_loss_kernel(eta, 8)
    p = np.zeros((2, 8, 8))
    p[1, 3, 2] = 1.0
    q = np.zeros_like(p)
    for n in range(8):
        for m in range(8):
            q[1, n, m] = b[n, 3] * b[m, 2]
    rec = richardson_lucy(q, np.eye(2), b, b, n_iter=40)
    assert rec[1, 3, 2] > 0.8


def test_ecd_gaussian_twins_match_poisson():
    sim = HybridSimulator(ndepth=2, nfocks=(8, 8), ansatz="ecd", energy_tensor=hybrid_energy_tensor((8, 8)))
    rng = np.random.default_rng(2026)
    x = random_parameters(2, rng)
    twins = build_twins(sim, x, rng, n_train=4, n_rank2=0)
    assert len(twins) == 4
    for tw in twins:
        assert tw.t_free == 0
        assert tw.product_tvd is not None
        assert tw.product_tvd < 1e-6
        assert tw.poisson_tvd is not None
        assert tw.poisson_tvd < 1e-6
        q, a1, a2 = tw.qubit, tw.alpha[0], tw.alpha[1]
        poisson = product_coherent_histogram(q, a1, a2, (2, 8, 8))
        # p_analytic is the truncated product evolution, not the Poisson formula.
        assert total_variation(poisson, tw.p_ideal) < 1e-6


def test_snap_gaussian_twins_match_poisson():
    sim = HybridSimulator(ndepth=1, nfocks=(8, 8), ansatz="snap", energy_tensor=hybrid_energy_tensor((8, 8)))
    rng = np.random.default_rng(7)
    x = random_snap_parameters(1, (8, 8), rng)
    twins = build_twins(sim, x, rng, n_train=3, n_rank2=0)
    for tw in twins:
        assert tw.product_tvd < 1e-6
        assert tw.poisson_tvd is not None


def test_oracle_kernels_stochastic():
    cfg = circuit_noise("loss", 0.03)
    spec = readout_spec("readout_realistic", n_shots=100)
    cq, c1, c2 = oracle_kernels(cfg, spec, ndepth=5, dims=(2, 8, 8))
    assert is_column_stochastic(cq)
    assert is_column_stochastic(c1)
    assert is_column_stochastic(c2)


def test_truncated_poisson_vacuum():
    p = truncated_poisson(0.0, 8)
    assert p[0] == pytest.approx(1.0)
    assert p.sum() == pytest.approx(1.0)
