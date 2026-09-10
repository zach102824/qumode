#!/usr/bin/env python3
"""One extra H001 ECD opt budget: warm-start from H000 plus two random restarts."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from Error_mitigation.run_mitigation_experiment import (  # noqa: E402
    ANSATZ_SPEC,
    SEED_BASE,
    json_ready,
    load_instance,
    make_sim,
)
from qumode_vqe.params import random_parameters  # noqa: E402
from qumode_vqe.vqe import optimize_vqe  # noqa: E402

OUT = ROOT / "Error_mitigation" / "out_research"
SRC = OUT / "optimized_params_ecd_h001_nd5.json"
BAK = OUT / "optimized_params_ecd_h001_nd5_budget3.json"
DEST = OUT / "leftover_xfer_h001_opt2"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    if SRC.is_file() and not BAK.is_file():
        shutil.copy2(SRC, BAK)
        print(f"backed up {SRC.name} -> {BAK.name}", flush=True)
    src_path = BAK if BAK.is_file() else SRC
    old = json.loads(src_path.read_text())
    print(f"incumbent E={old['fun']:.6f} restarts={old['n_restarts']}", flush=True)
    inst = load_instance(1)
    energy_tensor = np.asarray(inst["energy_tensor"], float)
    ground_qnm = tuple(int(v) for v in inst["ground_qnm"])
    ndepth = int(ANSATZ_SPEC["ecd"]["ndepth"])
    sim = make_sim("ecd", ndepth, energy_tensor, ground_qnm)
    best_x = np.asarray(old["x"], float)
    best_e = float(old["fun"])
    history = list(old.get("restarts") or [])
    h000 = json.loads((OUT / "optimized_params_ecd_h000_nd5.json").read_text())
    starts = [("warm_h000", np.asarray(h000["x"], float))]
    for r in (3, 4):
        rng = np.random.default_rng(int(SEED_BASE) + 1009 * r)
        starts.append((f"random_{r}", random_parameters(ndepth, rng)))
    t0 = time.time()
    for name, x0 in starts:
        print(f"  start {name} ...", flush=True)
        opt = optimize_vqe(sim, x0, method="L-BFGS-B", maxiter=200, record_every=0, verbose=False)
        rec = {"restart": name, "fun": float(opt.fun), "success": bool(opt.success), "nfev": int(opt.nfev)}
        history.append(rec)
        print(f"  {name}: E={opt.fun:.6f} nfev={opt.nfev}", flush=True)
        if float(opt.fun) < best_e:
            best_e = float(opt.fun)
            best_x = np.asarray(opt.x, float)
    elapsed = time.time() - t0
    payload = {
        **{k: old[k] for k in old if k not in ("x", "restarts", "fun", "nfev", "elapsed_s", "n_restarts")},
        "n_restarts": 6,
        "fun": best_e,
        "elapsed_s": float(old.get("elapsed_s", 0)) + elapsed,
        "restarts": history,
        "x": best_x.tolist(),
        "note": "budget3 plus warm-H000 and two extra random restarts",
    }
    (DEST / "opt_extra.json").write_text(json.dumps(json_ready(payload), indent=2))
    print(f"BEST E={best_e:.6f} (was {old['fun']:.6f}) extra {elapsed:.1f}s", flush=True)
    improved = best_e < float(old["fun"]) - 0.05
    (DEST / "IMPROVED").write_text("1\n" if improved else "0\n")
    if improved:
        (DEST / "optimized_params_ecd_h001_nd5.json").write_text(json.dumps(json_ready(payload), indent=2))
        print("wrote improved params under leftover_xfer_h001_opt2/", flush=True)
    else:
        print("no meaningful improvement; skip second physics pass", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
