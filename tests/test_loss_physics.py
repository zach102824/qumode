"""Physical checks of amplitude damping vs analytic decay and QuTiP mesolve."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from qumode_vqe.channels import (
    apply_kraus_local,
    destroy_matrix,
    lindblad_kraus,
    num_matrix,
    paper_amplitude_damping_kraus,
)
from qumode_vqe.noise import LossModel, NoiseConfig, TimingMode
from qumode_vqe.vqe import HybridSimulator


def _apply_single_mode(rho: np.ndarray, kraus: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(rho)
    for k in kraus:
        out = out + k @ rho @ k.conj().T
    return out


def test_mean_photon_exponential_decay_fock_and_coherent():
    n_fock = 16
    kappa_tau = 0.35
    kraus = paper_amplitude_damping_kraus(kappa_tau, n_fock)
    n_op = num_matrix(n_fock)

    rho_fock = np.zeros((n_fock, n_fock), dtype=complex)
    rho_fock[5, 5] = 1.0
    rho_f = _apply_single_mode(rho_fock, kraus)
    n_f = float(np.real(np.trace(n_op @ rho_f)))
    assert n_f == pytest.approx(5.0 * np.exp(-kappa_tau), rel=1e-3, abs=1e-3)

    alpha = 1.4
    psi = qt.coherent(n_fock, alpha)
    rho_c = np.asarray((psi * psi.dag()).full())
    rho_after = _apply_single_mode(rho_c, kraus)
    n_c = float(np.real(np.trace(n_op @ rho_after)))
    assert n_c == pytest.approx((abs(alpha) ** 2) * np.exp(-kappa_tau), rel=2e-3, abs=2e-3)


def test_paper_kraus_agrees_with_mesolve():
    n_fock = 12
    kappa = 0.2
    tau = 1.0
    kraus = paper_amplitude_damping_kraus(kappa * tau, n_fock)
    psi = qt.coherent(n_fock, 1.1) + qt.basis(n_fock, 3)
    psi = psi.unit()
    rho0 = psi * psi.dag()
    rho_k = qt.Qobj(_apply_single_mode(np.asarray(rho0.full()), kraus))
    result = qt.mesolve(qt.qzero(n_fock), rho0, [0.0, tau], c_ops=[np.sqrt(kappa) * qt.destroy(n_fock)])
    rho_me = result.states[-1]
    fid = qt.fidelity(rho_k, rho_me)
    assert fid > 0.995


def test_lindblad_zero_temperature_matches_paper_kraus():
    n_fock = 8
    kappa_tau = 0.08
    paper = paper_amplitude_damping_kraus(kappa_tau, n_fock)
    a = destroy_matrix(n_fock)
    lindblad = lindblad_kraus([np.sqrt(1.0) * a], kappa_tau, n_fock)

    rng = np.random.default_rng(0)
    vec = rng.normal(size=n_fock) + 1j * rng.normal(size=n_fock)
    vec = vec / np.linalg.norm(vec)
    rho = np.outer(vec, vec.conj())
    rho_p = _apply_single_mode(rho, paper)
    rho_l = _apply_single_mode(rho, lindblad)
    assert np.linalg.norm(rho_p - rho_l) < 5e-3


def test_two_mode_loss_on_hybrid_vacuum_stays_vacuum():
    kraus = paper_amplitude_damping_kraus(0.2, 8)
    vac = np.zeros((128, 128), dtype=complex)
    vac[0, 0] = 1.0
    out = apply_kraus_local(vac, kraus, 1, (2, 8, 8))
    out = apply_kraus_local(out, kraus, 2, (2, 8, 8))
    np.testing.assert_allclose(out, vac, atol=1e-12)
