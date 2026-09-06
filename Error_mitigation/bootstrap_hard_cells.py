#!/usr/bin/env python3
"""Resample shot histograms on a handful of hard cells (research only)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from Error_mitigation.metrics import compare_histograms
from Error_mitigation.noise_models import circuit_noise, readout_spec
from Error_mitigation.run_ablation import DIMS, _observe_block, load_cache, mitigate_research
from Error_mitigation.run_mitigation_experiment import ANSATZ_SPEC, SEED_BASE, json_ready, load_instance

CACHE = ROOT / "Error_mitigation" / "out_research" / "cache"
OUT = ROOT / "Error_mitigation" / "out_research" / "leftover_bootstrap"

CELLS = [
    ("ecd", "random", "loss", 0.1, "span_nr10_lo0.25_hi1.35_x0", "ideal"),
    ("ecd", "random", "comprehensive", 0.1, "span_nr10_lo0.25_hi1.35_x0", "ideal"),
    ("ecd", "optimized", "comprehensive", 0.1, "default_nr10_lo0.25_hi1.35_x0", "ideal"),
    ("ecd", "optimized", "loss", 0.1, "default_nr10_lo0.25_hi1.35_x0", "ideal"),
    ("snap", "random", "comprehensive", 0.003, "span_nr10_lo0.25_hi1.35_x0", "ideal"),
]
N_BOOT = 8
SHOTS = 8192
METHODS = ("raw", "gdr_param", "gdr_damped", "gdr_select")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inst = load_instance(0)
    energy_tensor = np.asarray(inst["energy_tensor"], float)
    ground_qnm = tuple(int(v) for v in inst["ground_qnm"])
    rows = []
    t0 = time.time()
    for ansatz, pset, family, kt, tag, ro in CELLS:
        key = f"{ansatz}_{pset}_{family}_kt{kt:g}_n40_{tag}"
        phys = load_cache(CACHE / f"{key}.npz")
        if phys is None:
            print(f"MISS {key}", flush=True)
            continue
        cfg = circuit_noise(family, float(kt), dims=DIMS)
        ndepth = int(ANSATZ_SPEC[ansatz]["ndepth"])
        tvds = {m: [] for m in METHODS}
        for b in range(N_BOOT):
            spec = readout_spec(ro, SHOTS, seed=None)
            seed = int(SEED_BASE) + 10007 * (b + 1)
            q_obs, q_twins, hist_by_scale = _observe_block(
                phys, spec, ansatz, pset, family, kt, seed, f"boot{b}_s{SHOTS}"
            )
            mitigated = mitigate_research(
                phys=phys,
                q_obs=q_obs,
                q_twins=q_twins,
                hist_by_scale=hist_by_scale,
                cfg=cfg,
                spec=spec,
                ndepth=ndepth,
                energy_tensor=energy_tensor,
                methods=METHODS,
                fit_maxiter=80,
                circuit_kind=pset,
                family=family,
                kappa_tau=float(kt),
            )
            for name, blob in mitigated.items():
                met = compare_histograms(
                    blob.get("hist"),
                    phys["p_ideal"],
                    energy_tensor,
                    ground_qnm,
                    energy_mit=blob.get("energy"),
                )
                if met.get("tvd") is not None:
                    tvds[name].append(float(met["tvd"]))
            print(
                f"  {ansatz} {pset} {family} kt={kt} boot={b} "
                f"raw={tvds['raw'][-1]:.4f} gdr={tvds['gdr_param'][-1]:.4f} "
                f"sel={tvds['gdr_select'][-1]:.4f}",
                flush=True,
            )
        rec = {
            "ansatz": ansatz,
            "params": pset,
            "family": family,
            "kappa_tau": float(kt),
            "readout": ro,
            "n_boot": N_BOOT,
            "methods": {
                m: {
                    "mean": float(np.mean(v)) if v else None,
                    "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                    "tvds": v,
                }
                for m, v in tvds.items()
            },
        }
        rows.append(rec)
        g = rec["methods"]["gdr_select"]
        print(
            f"CELL {ansatz} {pset} {family} {kt}: select {g['mean']:.4f} ± {g['std']:.4f}",
            flush=True,
        )
    payload = {"n_boot": N_BOOT, "shots": SHOTS, "elapsed_s": time.time() - t0, "records": rows}
    (OUT / "bootstrap.json").write_text(json.dumps(json_ready(payload), indent=2))
    lines = [
        "# Bootstrap TVD (8 resampled 8192-shot histograms)",
        "",
        "| cell | raw | gdr_param | select |",
        "|---|---:|---:|---:|",
    ]
    for rec in rows:
        def fmt(name):
            b = rec["methods"][name]
            if b["mean"] is None:
                return "—"
            return f"{b['mean']:.3f} ± {b['std']:.3f}"

        lines.append(
            f"| {rec['ansatz']} {rec['params']} {rec['family']} {rec['kappa_tau']:g} | "
            f"{fmt('raw')} | {fmt('gdr_param')} | {fmt('gdr_select')} |"
        )
    (OUT / "bootstrap.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'bootstrap.md'} in {payload['elapsed_s']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
