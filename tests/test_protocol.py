"""Default noiseless+noisy protocol: κ_φ, step traces, gdr-in-loop, 4-SAT CLI."""

from __future__ import annotations

import json
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

from Error_mitigation.gdr_in_loop import fit_gdr_param_policy_b, make_gdr_forward
from Error_mitigation.noise_models import circuit_noise
from qumode_vqe.circuit import vacuum_prep_params
from qumode_vqe.hamiltonian import DEFAULT_NFOCKS, load_four_sat_instances
from qumode_vqe.params import random_parameters
from qumode_vqe.vqe import optimize_gibbs_adaptive

HAM_DIR = ROOT / "Hamiltonians" / "four_sat"
NFOCKS = DEFAULT_NFOCKS


def test_circuit_noise_comprehensive_includes_number_dephasing():
    cfg = circuit_noise("comprehensive", 0.1)
    assert cfg.kappa_phi * cfg.tau_application == pytest.approx(0.05)
    thermal = circuit_noise("loss_thermal_dephasing", 0.1)
    assert thermal.kappa_phi * thermal.tau_application == pytest.approx(0.05)


def test_four_sat_noiseless_suite_smoke(tmp_path):
    import importlib.util

    ecd_path = ROOT / "scripts" / "Gibbs_and_adaptive_optim_ECD.py"
    spec = importlib.util.spec_from_file_location("gibbs_ecd_four_sat_smoke", ecd_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    payload = mod.run_saved_hamiltonian_suite(
        ham_dir=HAM_DIR,
        n_trials=1,
        outer_iter=1,
        spsa_iter=0,
        workers=1,
        seed_base=4000,
        outdir=tmp_path,
        ndepths=(1,),
        nfocks=NFOCKS,
        spsa_a=0.2,
        spsa_c=0.15,
        spsa_A=10.0,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        ansatz="ecd",
        family="four_sat",
        hamiltonian_ids=(0,),
        output=tmp_path / "gibbs_four_sat_ecd.json",
    )
    rec = payload["trials"][0]
    assert payload["family"] == "four_sat"
    assert rec["ndepth"] == 1
    assert payload["success_metric"] == "most_likely_bitstring == ground_bitstring"
    assert rec["step_trace"]
    assert rec["step_trace"][0]["step"] == 0
    assert rec["step_trace"][-1]["step"] == 1
    assert "p_gs" in rec["step_trace"][0]
    assert rec["ground_bitstring"]
    assert "success" in rec
    assert Path(tmp_path / "gibbs_four_sat_ecd.json").is_file()


def test_gdr_in_loop_one_spsa_step(tmp_path):
    instances = load_four_sat_instances(HAM_DIR, max_hamiltonians=1)
    inst = instances[0]
    tensor = inst["energy_tensor"]
    fit = fit_gdr_param_policy_b(
        ansatz="ecd",
        ndepth=1,
        hid=int(inst["hamiltonian_id"]),
        kappa_tau=0.003,
        energy_tensor=tensor,
        nfocks=NFOCKS,
        n_train=4,
        n_shots=128,
        seed=4000,
        cache_dir=tmp_path,
        fit_maxiter=15,
    )
    assert fit["m_policy"] == "B"
    assert fit["method"] == "gdr_param"
    assert len(fit["theta"]) == 11
    noise = circuit_noise("comprehensive", 0.003)
    forward = make_gdr_forward(
        theta=np.asarray(fit["theta"], dtype=float),
        dims=(2, int(NFOCKS[0]), int(NFOCKS[1])),
        energy_tensor=tensor,
    )
    rng = np.random.default_rng(0)
    rec = optimize_gibbs_adaptive(
        vacuum_prep_params(),
        random_parameters(1, rng),
        ndepth=1,
        nfocks=NFOCKS,
        outer_iter=1,
        spsa_iter=0,
        rng=rng,
        noise=noise,
        energy_tensor=tensor,
        ansatz="ecd",
        forward=forward,
        record_steps=True,
    )
    assert len(rec.step_trace) == 2
    assert rec.step_trace[0]["step"] == 0
    assert rec.step_trace[-1]["step"] == 1
    assert "p_gs" in rec.step_trace[-1]
    assert rec.eval_final.get("p_gs_raw") is not None or rec.step_trace[-1].get("p_gs_raw") is not None
    assert np.isfinite(rec.fun)
    # Policy B cache replay
    again = fit_gdr_param_policy_b(
        ansatz="ecd",
        ndepth=1,
        hid=int(inst["hamiltonian_id"]),
        kappa_tau=0.003,
        energy_tensor=tensor,
        nfocks=NFOCKS,
        n_train=4,
        n_shots=128,
        seed=4000,
        cache_dir=tmp_path,
        fit_maxiter=15,
    )
    assert again["from_cache"] is True
    np.testing.assert_allclose(again["theta"], fit["theta"])
