"""Unitarity and analytic action of the ECD-rotation ansatz."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from qumode_vqe.circuit import (
    ecd_ansatz_unitary,
    ecd_gate,
    ecd_rotation_pair,
    prep_params_to_ket,
    qubit_rotation,
    ry_coherent_product,
    uer_layer,
    vacuum,
    vacuum_prep_params,
)
from qumode_vqe.params import UnpackedParams, pack, unpack
from qumode_vqe.vqe import HybridSimulator


NFOCKS = (8, 8)


def _is_unitary(op: qt.Qobj, tol: float = 1e-10) -> bool:
    ident = qt.tensor(qt.qeye(2), qt.qeye(NFOCKS[0]), qt.qeye(NFOCKS[1]))
    if op.shape != ident.shape:
        ident = qt.qeye(op.shape[0])
    err = (op.dag() * op - ident).norm()
    return err < tol


def test_qubit_rotation_is_unitary():
    r = qubit_rotation(0.7, 1.2)
    ident = qt.qeye(2)
    assert (r.dag() * r - ident).norm() < 1e-12
    det = np.linalg.det(np.asarray(r.full()))
    assert abs(abs(det) - 1.0) < 1e-10


def test_vacuum_prep_params_are_the_hybrid_vacuum():
    ket = prep_params_to_ket(vacuum_prep_params(), NFOCKS)
    assert (ket - vacuum(NFOCKS)).norm() < 1e-12


def test_ecd_is_unitary():
    assert _is_unitary(ecd_gate(0.8 - 0.3j, 0, NFOCKS))
    assert _is_unitary(ecd_gate(1.1 + 0.4j, 1, NFOCKS))


def test_uer_and_ansatz_are_unitary():
    beta = np.array([0.4 + 0.2j, -0.7 + 0.1j])
    theta = np.array([0.3, 1.1])
    phi = np.array([0.2, 2.0])
    layer = uer_layer(beta, theta, phi, NFOCKS)
    assert _is_unitary(layer)

    params = UnpackedParams(
        beta=np.array([[0.2, 0.3j], [0.4, -0.5], [0.1 - 0.2j, 0.6]]),
        theta=np.array([[0.1, 0.5], [0.7, 0.2], [1.0, 0.3]]),
        phi=np.array([[0.4, 1.1], [0.0, 0.8], [0.2, 0.9]]),
    )
    assert _is_unitary(ecd_ansatz_unitary(params, NFOCKS))


def test_ecd_analytic_action_on_ground_and_excited():
    beta = 0.9 + 0.25j
    l = 12
    ecd = ecd_gate(beta, 0, (l, 4))
    vac_c = qt.basis(l, 0)
    # ECD |0> ⊗ |0> ⊗ |0> = |1> ⊗ D(β/2)|0> ⊗ |0>
    psi0 = qt.tensor(qt.basis(2, 0), vac_c, qt.basis(4, 0))
    out0 = ecd * psi0
    expected0 = qt.tensor(qt.basis(2, 1), qt.displace(l, beta / 2) * vac_c, qt.basis(4, 0))
    assert (out0 - expected0).norm() < 1e-10

    psi1 = qt.tensor(qt.basis(2, 1), vac_c, qt.basis(4, 0))
    out1 = ecd * psi1
    expected1 = qt.tensor(qt.basis(2, 0), qt.displace(l, -beta / 2) * vac_c, qt.basis(4, 0))
    assert (out1 - expected1).norm() < 1e-10


def test_zero_parameter_uer_is_identity_and_returns_vacuum():
    zeros = np.zeros(8 * 5)
    sim = HybridSimulator(ndepth=5)
    psi = sim.statevector(zeros)
    assert (psi - vacuum(NFOCKS)).norm() < 1e-10
    layer = ecd_rotation_pair(0.0, 0.0, 0.0, 0, NFOCKS)
    # A single pair with β=θ=0 is σx on the qubit; two pairs cancel.
    uer = uer_layer([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], NFOCKS)
    ident = qt.tensor(qt.qeye(2), qt.qeye(8), qt.qeye(8))
    assert (uer - ident).norm() < 1e-10
    assert (layer.dag() * layer - ident).norm() < 1e-10


def test_empty_pair_mask_is_vacuum():
    rng = np.random.default_rng(2)
    x = rng.normal(size=40)
    p = unpack(x, 5)
    mask = np.zeros((5, 2), dtype=bool)
    uni = ecd_ansatz_unitary(p, NFOCKS, pair_mask=mask)
    ident = qt.tensor(qt.qeye(2), qt.qeye(8), qt.qeye(8))
    assert (uni - ident).norm() < 1e-10
    sim = HybridSimulator(ndepth=5, pair_mask=mask)
    assert (sim.statevector(x) - vacuum(NFOCKS)).norm() < 1e-10


def test_all_true_pair_mask_matches_unmasked():
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    p = unpack(x, 5)
    masked = ecd_ansatz_unitary(p, NFOCKS, pair_mask=np.ones((5, 2), dtype=bool))
    full = ecd_ansatz_unitary(p, NFOCKS, pair_mask=None)
    assert (masked - full).norm() < 1e-10


def test_parameter_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    p = unpack(x, 5)
    x2 = pack(beta=p.beta, theta=p.theta, phi=p.phi)
    p2 = unpack(x2, 5)
    np.testing.assert_allclose(p.theta, p2.theta)
    np.testing.assert_allclose(p.phi, p2.phi)
    np.testing.assert_allclose(np.abs(p.beta), np.abs(p2.beta), atol=1e-12)


def test_ry_coherent_product_is_normalized():
    psi = ry_coherent_product(0.3, 0.8 - 0.4j, 1.2 + 0.1j, NFOCKS)
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)


def test_zero_prep_is_vacuum():
    psi = ry_coherent_product(0.0, 0.0, 0.0, NFOCKS)
    assert (psi - vacuum(NFOCKS)).norm() < 1e-10


def test_ry_pi2_is_plus_on_qubit():
    psi = ry_coherent_product(np.pi / 2.0, 0.0, 0.0, NFOCKS)
    vec = np.asarray(psi.full(), dtype=complex).reshape(2, NFOCKS[0], NFOCKS[1])
    p0 = float(np.sum(np.abs(vec[0]) ** 2))
    p1 = float(np.sum(np.abs(vec[1]) ** 2))
    assert p0 == pytest.approx(0.5, abs=1e-10)
    assert p1 == pytest.approx(0.5, abs=1e-10)
