"""Kraus completeness, CPTP, identity limits, and noisy vs statevector agreement."""

from __future__ import annotations

import numpy as np
import pytest

from qumode_vqe.channels import (
    density_physicality,
    is_cptp_kraus,
    kraus_completeness_error,
    paper_amplitude_damping_kraus,
)
from qumode_vqe.noise import (
    ChannelCache,
    LossModel,
    NoiseConfig,
    TimingMode,
    comprehensive_config,
    paper_loss_config,
)
from qumode_vqe.vqe import HybridSimulator


def test_paper_kraus_complete_and_cptp():
    for kt in (0.0, 1e-6, 0.01, 0.1, 1.0):
        kraus = paper_amplitude_damping_kraus(kt, 8)
        assert kraus_completeness_error(kraus) < 1e-10
        assert is_cptp_kraus(kraus)


def test_zero_rate_channel_is_identity():
    kraus = paper_amplitude_damping_kraus(0.0, 8)
    assert len(kraus) == 1
    np.testing.assert_allclose(kraus[0], np.eye(8), atol=1e-14)


def test_output_physicality_after_loss(reference_xvec):
    sim = HybridSimulator(noise=paper_loss_config(0.05))
    rho = sim.density_matrix(reference_xvec)
    phys = density_physicality(rho)
    assert phys["trace_error"] < 1e-10
    assert phys["hermiticity"] < 1e-10
    assert phys["min_eig"] > -1e-10
    assert 0.0 <= phys["purity"] <= 1.0 + 1e-8


def test_all_zero_noise_matches_statevector(reference_xvec):
    ideal = HybridSimulator()
    noisy = HybridSimulator(
        noise=NoiseConfig(loss_model=LossModel.NONE, enable_transmon=False)
    )
    ev_i = ideal.evaluate(reference_xvec)
    ev_n = noisy.evaluate(reference_xvec)
    assert ev_i.energy_physical == pytest.approx(ev_n.energy_physical, abs=1e-10)
    np.testing.assert_allclose(
        ev_i.measurement.physical_probs, ev_n.measurement.physical_probs, atol=1e-12
    )


def test_paper_kraus_zero_kappa_matches_ideal(reference_xvec):
    from qumode_vqe.channels import ket_to_dm

    sim_i = HybridSimulator()
    sim_k = HybridSimulator(noise=paper_loss_config(0.0))
    rho_i = ket_to_dm(sim_i.statevector(reference_xvec))
    rho_k = sim_k.density_matrix(reference_xvec)
    assert np.linalg.norm(rho_i - rho_k) < 1e-10


def test_mutually_exclusive_loss_paths():
    with pytest.raises(ValueError, match="PAPER_KRAUS"):
        NoiseConfig(loss_model=LossModel.PAPER_KRAUS, nth_cav=0.1)
    with pytest.raises(NotImplementedError, match="leakage"):
        NoiseConfig(transmon_leakage=True)


def test_channel_cache_builds_for_lindblad():
    cfg = NoiseConfig(
        timing=TimingMode.PER_ECD_PAIR,
        loss_model=LossModel.LINDBLAD,
        kappa_tau=0.01,
        nth_cav=0.02,
        kappa_phi=1e3,
        enable_transmon=True,
    )
    cache = ChannelCache(cfg)
    assert cache.kraus_cav1 and cache.kraus_qubit
    assert kraus_completeness_error(cache.kraus_cav1) < 1e-8
    assert kraus_completeness_error(cache.kraus_qubit) < 1e-8


def test_comprehensive_config_enables_all_device_channels():
    cfg = comprehensive_config()
    assert cfg.loss_model is LossModel.LINDBLAD
    assert cfg.timing is TimingMode.PER_ECD_PAIR
    assert cfg.enable_transmon
    assert cfg.nth_cav == pytest.approx(0.01)
    assert cfg.rotation_rel_error == pytest.approx(0.01)
    assert cfg.ecd_amp_rel_error == pytest.approx(0.01)
    assert cfg.kerr == pytest.approx(2.0 * np.pi * 500.0)
    assert not cfg.is_identity()
    cache = ChannelCache(cfg)
    assert cache.kraus_cav1 and cache.kraus_cav2 and cache.kraus_qubit
    assert cache.kerr_u1 is not None


def test_comprehensive_config_accepts_kappa_tau_override():
    cfg = comprehensive_config(kappa_tau=0.03)
    assert cfg.kappa_tau_used() == pytest.approx(0.03)
