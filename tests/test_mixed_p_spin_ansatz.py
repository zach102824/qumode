"""Mixed p-spin loader, SNAP+displacement circuit, and shallow hybrid sweeps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import qutip as qt

from qumode_vqe.circuit import (
    N_PREP_PARAMS,
    displacement_on_hybrid,
    snap_ansatz_unitary,
    snap_displacement_pair,
    snap_gate,
    vacuum,
)
from qumode_vqe.hamiltonian import (
    DEFAULT_NFOCKS,
    N_QUBITS,
    bits_from_qnm,
    bits_from_qubit_index,
    energy_from_z_terms,
    energy_tensor_from_z_terms,
    load_mixed_p_spin_instances,
    qnm_from_bits,
    z_terms_from_mixed_p_spin_npz,
)
from qumode_vqe.params import (
    UnpackedSnapParams,
    ansatz_inventory,
    n_parameters,
    n_snap_parameters,
    pack_snap,
    random_parameters,
    random_snap_parameters,
    unpack_snap,
)
from qumode_vqe.vqe import HybridSimulator, optimize_gibbs_adaptive

ROOT = Path(__file__).resolve().parents[1]
HAM_DIR = ROOT / "Hamiltonians" / "mixed_p_spin"
NFOCKS = DEFAULT_NFOCKS


def _first_npz() -> Path:
    paths = sorted(HAM_DIR.glob("mixed_p_spin_p2-4_[0-9][0-9][0-9].npz"))
    if not paths:
        pytest.skip(f"no mixed p-spin NPZ files in {HAM_DIR}")
    return paths[0]


def test_mixed_p_spin_tensor_matches_z_bitstrings():
    path = _first_npz()
    terms, meta = z_terms_from_mixed_p_spin_npz(path)
    assert meta["num_spins"] == N_QUBITS
    tensor = energy_tensor_from_z_terms(terms, NFOCKS, tilt=0.0)
    assert tensor.shape == (2, NFOCKS[0], NFOCKS[1])
    for idx in range(1 << N_QUBITS):
        bits = bits_from_qubit_index(idx)
        q, n, m = qnm_from_bits(bits)
        decoded = bits_from_qnm(q, n, m)
        np.testing.assert_array_equal(decoded, bits)
        expected = energy_from_z_terms(bits, terms)
        assert tensor[q, n, m] == pytest.approx(expected, abs=1e-12)


def test_load_mixed_p_spin_instances_hybrid_shape():
    instances = load_mixed_p_spin_instances(HAM_DIR, max_hamiltonians=2)
    assert len(instances) == 2
    for inst in instances:
        tensor = inst["energy_tensor"]
        assert tensor.shape == (2, 8, 8)
        assert inst["family"] == "mixed_p_spin"
        assert inst["n_terms"] == 91
        assert inst["energy_min"] <= inst["energy_max"]


def test_ecd_and_snap_resource_counts_match_plan():
    ecd1 = ansatz_inventory("ecd", 1, NFOCKS)
    ecd5 = ansatz_inventory("ecd", 5, NFOCKS)
    snap1 = ansatz_inventory("snap", 1, NFOCKS)
    snap2 = ansatz_inventory("snap", 2, NFOCKS)
    assert ecd1["n_ansatz_params"] == 8
    assert ecd1["n_primitive_gates"] == 4
    assert ecd5["n_ansatz_params"] == 40
    assert ecd5["n_primitive_gates"] == 20
    assert ecd5["n_params"] == 45
    assert snap1["n_ansatz_params"] == 18
    assert snap1["n_primitive_gates"] == 4
    assert snap2["n_ansatz_params"] == 36
    assert snap2["n_primitive_gates"] == 8
    assert snap2["n_params"] == 41
    assert n_parameters(5) == 40
    assert n_snap_parameters(2, NFOCKS) == 36
    assert n_snap_parameters(3, NFOCKS) == 54


def test_snap_operator_and_pairs_are_unitary():
    ident = qt.tensor(qt.qeye(2), qt.qeye(NFOCKS[0]), qt.qeye(NFOCKS[1]))
    phases = np.linspace(0.0, 1.3, NFOCKS[0])
    phases[0] = 0.0
    snap = snap_gate(phases, 0, NFOCKS)
    assert (snap.dag() * snap - ident).norm() < 1e-10
    disp = displacement_on_hybrid(0.4 - 0.2j, 1, NFOCKS)
    assert (disp.dag() * disp - ident).norm() < 1e-10
    pair = snap_displacement_pair(0.5 + 0.1j, phases, 0, NFOCKS)
    assert (pair.dag() * pair - ident).norm() < 1e-10


def test_snap_level0_gauge_is_global_phase():
    rng = np.random.default_rng(4)
    x = random_snap_parameters(1, NFOCKS, rng)
    params = unpack_snap(x, 1, NFOCKS)
    delta = 0.37
    shifted = params.phases.copy()
    shifted[0, 0, :] += delta
    shifted[0, 1, :] += delta
    u0 = snap_ansatz_unitary(params, NFOCKS)
    u1 = snap_ansatz_unitary(UnpackedSnapParams(alpha=params.alpha, phases=shifted), NFOCKS)
    psi = vacuum(NFOCKS)
    overlap = (u0 * psi).overlap(u1 * psi)
    assert abs(abs(overlap) - 1.0) < 1e-10


def test_gauge_fixed_unpack_zeros_fock0_phase():
    rng = np.random.default_rng(5)
    x = random_snap_parameters(2, NFOCKS, rng)
    params = unpack_snap(x, 2, NFOCKS)
    np.testing.assert_allclose(params.phases[:, :, 0], 0.0)
    x2 = pack_snap(params.alpha, params.phases, NFOCKS)
    p2 = unpack_snap(x2, 2, NFOCKS)
    np.testing.assert_allclose(np.abs(params.alpha), np.abs(p2.alpha), atol=1e-12)
    np.testing.assert_allclose(params.phases, p2.phases, atol=1e-12)


def test_zero_snap_displacement_returns_vacuum():
    x = np.zeros(n_snap_parameters(1, NFOCKS))
    sim = HybridSimulator(ndepth=1, ansatz="snap")
    psi = sim.statevector(x)
    assert (psi - vacuum(NFOCKS)).norm() < 1e-10


def test_optimize_gibbs_adaptive_accepts_both_ansatze():
    instances = load_mixed_p_spin_instances(HAM_DIR, max_hamiltonians=1)
    tensor = instances[0]["energy_tensor"]
    rng = np.random.default_rng(6)
    prep0 = np.array([0.4, 0.2, -0.1, 0.15, 0.05], dtype=float)
    ecd_opt = optimize_gibbs_adaptive(
        prep0,
        random_parameters(1, rng),
        ndepth=1,
        outer_iter=1,
        spsa_iter=0,
        rng=rng,
        energy_tensor=tensor,
        ansatz="ecd",
    )
    snap_opt = optimize_gibbs_adaptive(
        prep0,
        random_snap_parameters(1, NFOCKS, rng),
        ndepth=1,
        outer_iter=1,
        spsa_iter=0,
        rng=rng,
        energy_tensor=tensor,
        ansatz="snap",
    )
    assert ecd_opt.x.shape == (8,)
    assert snap_opt.x.shape == (18,)
    assert np.isfinite(ecd_opt.fun)
    assert np.isfinite(snap_opt.fun)
    assert ecd_opt.nfev >= 1
    assert snap_opt.nfev >= 1


def test_mixed_p_spin_script_smoke(tmp_path):
    import importlib.util
    import sys

    ecd_path = ROOT / "scripts" / "Gibbs_and_adaptive_optim_ECD.py"
    spec = importlib.util.spec_from_file_location("gibbs_ecd_smoke", ecd_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    payload = mod.run_mixed_p_spin_suite(
        ham_dir=HAM_DIR,
        n_trials=1,
        outer_iter=1,
        spsa_iter=0,
        workers=1,
        seed_base=11,
        outdir=tmp_path,
        ndepths=(1,),
        nfocks=NFOCKS,
        spsa_a=0.2,
        spsa_c=0.15,
        spsa_A=10.0,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        ansatz="ecd",
        max_hamiltonians=1,
    )
    rec = payload["trials"][0]
    assert rec["n_ansatz_params"] == 8
    assert rec["n_primitive_gates"] == 4
    assert rec["n_params"] == 8 + N_PREP_PARAMS
    assert rec["nit_total"] == 1
    assert Path(tmp_path / "gibbs_mixed_p_spin_ecd.json").is_file()

    snap_payload = mod.run_mixed_p_spin_suite(
        ham_dir=HAM_DIR,
        n_trials=1,
        outer_iter=1,
        spsa_iter=0,
        workers=1,
        seed_base=11,
        outdir=tmp_path,
        ndepths=(1,),
        nfocks=NFOCKS,
        spsa_a=0.2,
        spsa_c=0.15,
        spsa_A=10.0,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        ansatz="snap",
        output=tmp_path / "gibbs_mixed_p_spin_snap.json",
        max_hamiltonians=1,
    )
    snap_rec = snap_payload["trials"][0]
    assert snap_rec["n_ansatz_params"] == 18
    assert snap_rec["n_primitive_gates"] == 4
    assert snap_rec["n_params"] == 23
    assert snap_payload["ansatz"] == "snap"
