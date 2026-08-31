"""Guards for the fixed-300 SPSA HEA vs joint ECD suite."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qumode_vqe.qaoa import n_hea_params, optimize_qubit_ansatz
from qumode_vqe.vqe import optimize_gibbs_adaptive, run_spsa

REPO = Path(__file__).resolve().parents[1]
import sys

_SCRIPTS = REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _load_spsa300():
    scripts = REPO / "scripts"
    import sys

    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("spsa300", scripts / "spsa300.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_seed_base_5000_is_distinct_from_70_step_suites():
    from size_sweep import ecd_ansatz_seed, ecd_prep_seed, hea_trial_seed

    # 70-step n=7 used 3000; n=9 used 4000. This suite uses 5000.
    hea7_70 = hea_trial_seed(3000, "knapsack", 0, 0)
    hea9_70 = hea_trial_seed(4000, "knapsack", 0, 0)
    hea300 = hea_trial_seed(5000, "knapsack", 0, 0)
    assert hea7_70 == 35000
    assert hea9_70 == 36000
    assert hea300 == 37000
    assert len({hea7_70, hea9_70, hea300}) == 3
    assert ecd_ansatz_seed(5000, "ising", 3, 0) != ecd_ansatz_seed(4000, "ising", 3, 0)
    assert ecd_prep_seed(5000, "knapsack", 1, 0) != ecd_prep_seed(3000, "knapsack", 1, 0)
    assert ecd_ansatz_seed(5000, "ising", 0, 0) == 15000
    assert ecd_prep_seed(5000, "ising", 0, 0) == 65000


def test_run_spsa_always_uses_full_budget():
    rng = np.random.default_rng(1)
    hits = {"n": 0}

    def fun(x):
        hits["n"] += 1
        return float(np.dot(x, x))

    opt = run_spsa(fun, np.ones(4), maxiter=11, rng=rng)
    assert opt.nit == 11
    assert opt.nfev == 22
    assert hits["n"] == 23  # 2 per step + final fun(x)


def test_hea_first_success_step_is_zero_when_start_is_ground():
    energies = np.array([0.0, 3.0, 1.0, 4.0], dtype=float)
    rng = np.random.default_rng(0)
    x0 = np.zeros(n_hea_params(2, 1))
    opt = optimize_qubit_ansatz(
        x0,
        energies,
        ansatz="hea",
        objective="gibbs",
        n_qubits=2,
        n_layers=1,
        maxiter=4,
        rng=rng,
    )
    assert opt.nit == 4
    assert opt.nfev == 8
    assert opt.eval.success
    assert opt.first_success_step == 0


def test_hea_first_success_null_when_never_on_ground():
    # Unique ground |11>; all-zero HEA stays on |00>.
    energies = np.array([2.0, 1.0, 1.0, 0.0], dtype=float)
    rng = np.random.default_rng(1)
    x0 = np.zeros(n_hea_params(2, 1))
    opt = optimize_qubit_ansatz(
        x0,
        energies,
        ansatz="hea",
        objective="gibbs",
        n_qubits=2,
        n_layers=1,
        maxiter=3,
        rng=rng,
    )
    assert opt.nit == 3
    if not opt.eval.success:
        assert opt.first_success_step is None


def test_ecd_joint_records_warmup_steps_and_first_success_field():
    rng = np.random.default_rng(5)
    prep0 = np.array([np.pi / 2.0, 0.1, 0.0, 0.0, 0.2], dtype=float)
    result = optimize_gibbs_adaptive(
        prep0,
        np.zeros(8),
        ndepth=1,
        nfocks=(4, 4),
        outer_iter=3,
        spsa_iter=0,
        rng=rng,
    )
    assert result.nit_warmup == 3
    assert result.nit == 0
    assert result.first_success_step is None or 0 <= int(result.first_success_step) <= 3


def test_spsa300_refuses_protected_outputs_and_70_step_seeds():
    mod = _load_spsa300()
    assert mod.SEED_BASE == 5000
    assert mod.MAXITER == 300
    assert mod.OUTPUTS["hea7"] == "hea_gibbs_n7_spsa300.json"
    assert mod.OUTPUTS["ecd7"] == "ecd_joint_n7_spsa300.json"
    assert mod.OUTPUTS["hea9"] == "hea_gibbs_n9_spsa300.json"
    assert mod.OUTPUTS["ecd9"] == "ecd_joint_n9_spsa300.json"
    with pytest.raises(RuntimeError, match="protected"):
        mod._assert_new_path(Path("results/hea_gibbs.json"))
    with pytest.raises(RuntimeError, match="protected"):
        mod._assert_new_path(Path("results/size_sweep_n9_ecd.json"))
    with pytest.raises(RuntimeError, match="protected"):
        mod._assert_new_path(Path("results/gibbs_schedule_abc.json"))
    mod._assert_new_path(Path("results/hea_gibbs_n7_spsa300.json"))
    with pytest.raises(SystemExit):
        mod.main(["--seed-base", "3000", "--phase", "report"])
    with pytest.raises(SystemExit):
        mod.main(["--seed-base", "4000", "--phase", "report"])
