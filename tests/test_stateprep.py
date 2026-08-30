"""Single-mode ECD, cat parity, and HEA parameter-count tests."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from qumode_vqe.qaoa import n_hea_params
from qumode_vqe.stateprep import (
    apply_ecd,
    choose_cutoff,
    constructive_even_cat,
    ecd_gate_single,
    ecd_statevector,
    embed_fock_in_qubits,
    even_cat_amplitudes,
    even_parity_weight,
    fock_amplitudes,
    matched_hea_layers,
    n_ecd_params,
    n_qubits_for_cutoff,
    state_fidelity,
    vacuum_hybrid,
)


def test_ecd_ground_is_excited_displaced_plus_half():
    """ECD(β)|g, 0⟩ = |e⟩|β/2⟩ (Eickbusch / circuit.py: D(β/2)|e⟩⟨g| + …)."""
    l = 20
    beta = 0.8 - 0.3j
    out = apply_ecd(vacuum_hybrid(l), beta, l)
    comps = out.reshape(2, l)
    assert np.linalg.norm(comps[0]) == pytest.approx(0.0, abs=1e-12)
    expected = np.asarray((qt.displace(l, beta / 2.0) * qt.basis(l, 0)).full(), dtype=complex).reshape(-1)
    expected = expected / np.linalg.norm(expected)
    got = comps[1] / np.linalg.norm(comps[1])
    # Global phase from the truncated generator is allowed.
    phase = np.vdot(expected, got)
    phase = phase / abs(phase)
    np.testing.assert_allclose(got, expected * phase, atol=1e-10)

    gate = ecd_gate_single(beta, l)
    ket = qt.tensor(qt.basis(2, 0), qt.basis(l, 0))
    qout = np.asarray((gate * ket).full(), dtype=complex).reshape(-1)
    qout = qout / np.linalg.norm(qout)
    phase2 = np.vdot(qout, out)
    phase2 = phase2 / abs(phase2)
    np.testing.assert_allclose(out, qout * phase2, atol=1e-10)


def test_even_cat_is_even_parity():
    for alpha in (1.0, 1.5, 2.0, 3.0):
        l, f_trunc = choose_cutoff(alpha)
        amps = even_cat_amplitudes(alpha, l)
        assert even_parity_weight(amps) == pytest.approx(1.0, abs=1e-12)
        odd = 1.0 - even_parity_weight(amps)
        assert odd < 1e-12
        assert f_trunc > 1.0 - 1e-4
        assert abs(amps[1]) ** 2 + abs(amps[3]) ** 2 < 1e-14


def test_hea_param_count_formula():
    """HEA has n_qubits * (n_layers + 1) real parameters (final Ry layer included)."""
    assert n_hea_params(5, 0) == 5
    assert n_hea_params(5, 1) == 10
    assert n_hea_params(6, 0) == 6
    assert n_hea_params(6, 5) == 36
    assert n_hea_params(7, 5) == 42
    for n_qubits, n_layers in ((3, 2), (6, 1), (8, 5)):
        assert n_hea_params(n_qubits, n_layers) == n_qubits * (n_layers + 1)


def test_ecd_param_count_is_four_per_layer():
    assert n_ecd_params(1) == 4
    assert n_ecd_params(2) == 8
    assert n_ecd_params(3) == 12
    assert n_ecd_params(2, terminal_rotation=True) == 10


def test_matched_hea_layers_tracks_ecd_budget():
    # Primary match: N_d=2 → 8 ECD params.
    assert n_hea_params(5, matched_hea_layers(5, 8)) in (5, 10)
    assert abs(n_hea_params(5, matched_hea_layers(5, 8)) - 8) <= abs(n_hea_params(5, 0) - 8)
    assert n_hea_params(6, matched_hea_layers(6, 8)) == 6  # 6 closer to 8 than 12
    assert n_qubits_for_cutoff(25) == 5
    assert n_qubits_for_cutoff(64) == 6


def test_constructive_even_cat_high_fidelity():
    l, _ = choose_cutoff(1.5)
    result = constructive_even_cat(1.5, l)
    assert result.fidelity > 0.99
    assert 0.2 < result.success_probability < 0.8
    assert even_parity_weight(result.cavity) == pytest.approx(1.0, abs=1e-10)


def test_embed_pads_unused_levels():
    amps = fock_amplitudes(3, 10)
    embedded = embed_fock_in_qubits(amps)
    assert embedded.size == 16
    assert embedded[3] == pytest.approx(1.0)
    assert np.linalg.norm(embedded[10:]) == pytest.approx(0.0)
    assert state_fidelity(embedded, embed_fock_in_qubits(amps, 4)) == pytest.approx(1.0)


def test_ecd_circuit_layer_order_is_rotation_then_ecd():
    """One layer: R then ECD. Zero rotation + β leaves |e⟩|β/2⟩ from |g,0⟩."""
    l = 16
    beta = 0.6
    x = np.array([beta, 0.0, 0.0, 0.0], dtype=float)
    psi = ecd_statevector(x, l, n_layers=1)
    comps = psi.reshape(2, l)
    assert np.linalg.norm(comps[0]) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(comps[1]) == pytest.approx(1.0, abs=1e-12)
