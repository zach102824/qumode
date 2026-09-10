#!/usr/bin/env python3
"""ECD histogram figures for optimized H004 |016> (ground state).

Replay from the multi_h cache. Writes:

  case_histogram_ecd_gaussian  — t_free=0 twins only
  case_histogram_ecd_016       — official 30+10 mix
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from Error_mitigation.metrics import compare_histograms
from Error_mitigation.mitigation import fit_gdr_param, observe_histogram, params_to_kernels, unfold
from Error_mitigation.noise_models import circuit_noise, readout_spec
from Error_mitigation.run_mitigation_experiment import DIMS, SEED_BASE, case_seed, load_instance

from generate_analysis import METHOD, OUT, as_prob, case_histogram, style

CACHE = (
    ROOT
    / "Error_mitigation/out_research/multi_h/cache"
    / "ecd_optimized_comprehensive_kt0.003_n40_default_nr10_lo0.25_hi1.35_x0_h004.npz"
)
FAMILY = "comprehensive"
KT = 0.003
ANSATZ = "ecd"
PSET = "optimized"
HID = 4
EXPECTED_PEAK = (0, 1, 6)


def _observe(phys: dict, spec, keep: np.ndarray):
    q_obs = observe_histogram(
        phys["p_phys"],
        spec,
        DIMS,
        case_seed("obs", ANSATZ, PSET, FAMILY, KT, spec.level, SEED_BASE, "target", "s8192"),
    )
    q_twins = [
        observe_histogram(
            phys["twin_phys"][int(i)],
            spec,
            DIMS,
            case_seed("obs", ANSATZ, PSET, FAMILY, KT, spec.level, SEED_BASE, "twin", int(i), "s8192"),
        )
        for i in keep
    ]
    p_twin = [phys["twin_p_ideal"][int(i)] for i in keep]
    return q_obs, q_twins, p_twin


def replay(gauss_only: bool) -> dict:
    phys = dict(np.load(CACHE, allow_pickle=False))
    t_free = np.asarray(phys["twin_t_free"])
    keep = np.where(t_free == 0)[0] if gauss_only else np.arange(t_free.size)
    inst = load_instance(HID)
    energy_tensor = np.asarray(inst["energy_tensor"], dtype=float)
    ground = tuple(int(v) for v in inst["ground_qnm"])
    cfg = circuit_noise(FAMILY, KT, dims=DIMS)
    spec = readout_spec("readout_strong", 8192, seed=None)
    q_obs, q_twins, p_twin = _observe(phys, spec, keep)
    theta, _info = fit_gdr_param(p_twin, q_twins, cfg, spec, 5, DIMS, maxiter=200)
    cq, c1, c2 = params_to_kernels(theta, DIMS)
    p_mit = unfold(q_obs, cq, c1, c2)
    p_ideal = as_prob(phys["p_ideal"])
    peak = tuple(int(v) for v in np.unravel_index(int(p_ideal.argmax()), p_ideal.shape))
    if peak != EXPECTED_PEAK:
        raise RuntimeError(f"expected peak |016>, got |{peak[0]}{peak[1]}{peak[2]}>")
    train = "fully Gaussian twins" if gauss_only else "official twin mix"
    return {
        "ansatz": ANSATZ,
        "params": PSET,
        "family": FAMILY,
        "kappa_tau": KT,
        "readout": "readout_strong",
        "p_ideal": p_ideal,
        "hists": {"raw": as_prob(q_obs), METHOD: as_prob(p_mit)},
        "metrics": {
            "raw": compare_histograms(q_obs, p_ideal, energy_tensor, ground),
            METHOD: compare_histograms(p_mit, p_ideal, energy_tensor, ground),
        },
        "figure_title": (
            rf"ECD optimized, $|016\rangle$ (H004 GS), {train}, comprehensive, "
            rf"$\kappa\tau=0.003$, strong readout error"
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    jobs = (
        (True, "case_histogram_ecd_gaussian"),
        (False, "case_histogram_ecd_016"),
    )
    for gauss_only, stem in jobs:
        case = replay(gauss_only)
        case_histogram(case, stem)
        print(
            f"{stem}: TVD {case['metrics']['raw']['tvd']:.3f} -> {case['metrics'][METHOD]['tvd']:.3f}  "
            f"|dE| {case['metrics']['raw']['dE']:.2f} -> {case['metrics'][METHOD]['dE']:.2f}"
        )


if __name__ == "__main__":
    main()
