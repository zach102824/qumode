"""Fock-space truncation diagnostics for L = 8 vs a larger cutoff."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from qumode_vqe.circuit import ecd_gate
from qumode_vqe.channels import paper_amplitude_damping_kraus


def _fock_probs(psi: qt.Qobj, n_fock: int) -> np.ndarray:
    cav = psi.ptrace(1)
    return np.real(np.diag(cav.full())[:n_fock])


def test_displacement_boundary_occupation_small_for_paper_betas():
    """Typical optimized |β| values in the notebook are O(1); L=8 is then safe."""
    beta = 1.3 + 0.0j
    l_small, l_large = 8, 20
    vac_s = qt.tensor(qt.basis(2, 0), qt.basis(l_small, 0), qt.basis(4, 0))
    vac_l = qt.tensor(qt.basis(2, 0), qt.basis(l_large, 0), qt.basis(4, 0))
    out_s = ecd_gate(beta, 0, (l_small, 4)) * vac_s
    out_l = ecd_gate(beta, 0, (l_large, 4)) * vac_l
    p_s = _fock_probs(out_s, l_small)
    p_l = _fock_probs(out_l, l_large)
    assert p_s[-1] < 0.02
    np.testing.assert_allclose(p_s[:7], p_l[:7], atol=5e-3)


def test_large_displacement_is_flagged_by_boundary_occupation():
    beta = 8.0
    l_small = 8
    vac = qt.tensor(qt.basis(2, 0), qt.basis(l_small, 0), qt.basis(4, 0))
    out = ecd_gate(beta, 0, (l_small, 4)) * vac
    p = _fock_probs(out, l_small)
    assert p[-1] > 0.05


def test_loss_channel_on_high_fock_agrees_when_cutoff_grows():
    kt = 0.2
    k8 = paper_amplitude_damping_kraus(kt, 8)
    k16 = paper_amplitude_damping_kraus(kt, 16)
    rho8 = np.zeros((8, 8), dtype=complex)
    rho8[4, 4] = 1.0
    rho16 = np.zeros((16, 16), dtype=complex)
    rho16[4, 4] = 1.0

    def apply(rho, ks):
        out = np.zeros_like(rho)
        for k in ks:
            out = out + k @ rho @ k.conj().T
        return out

    p8 = np.real(np.diag(apply(rho8, k8)))
    p16 = np.real(np.diag(apply(rho16, k16)))[:8]
    np.testing.assert_allclose(p8, p16, atol=2e-3)
