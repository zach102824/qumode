#!/usr/bin/env python3
"""Serious noiseless ECD opt on mixed-p-spin H002.

Five random L-BFGS-B restarts (maxiter=200) plus a warm start from the
H000 ECD optimum. Writes only under ``Error_mitigation/out_research/``.
Promotes ``optimized_params_ecd_h002_nd5.json`` when finished so
``run_ablation.py --instance 2`` can load it.
"""

from __future__ import annotations

import json
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
DEST = OUT / "leftover_xfer_h002"
CACHE = OUT / "optimized_params_ecd_h002_nd5.json"
H000 = OUT / "optimized_params_ecd_h000_nd5.json"
WALL_S = 2 * 3600
NEAR_E0 = 0.3


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2))


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    inst = load_instance(2)
    energy_tensor = np.asarray(inst["energy_tensor"], float)
    ground_qnm = tuple(int(v) for v in inst["ground_qnm"])
    e0 = float(inst["energy_min"])
    gap = float(inst.get("gap") or 0.0)
    ndepth = int(ANSATZ_SPEC["ecd"]["ndepth"])
    sim = make_sim("ecd", ndepth, energy_tensor, ground_qnm)
    print(
        f"H002 file={inst.get('file')}  E0={e0:.6f}  gap={gap:.6f}  "
        f"ground={ground_qnm}  N_d={ndepth}",
        flush=True,
    )

    starts: list[tuple[str, np.ndarray]] = []
    if H000.is_file():
        h000 = json.loads(H000.read_text())
        starts.append(("warm_h000", np.asarray(h000["x"], float)))
        print(f"warm-start from H000 E={h000.get('fun')}", flush=True)
    for r in range(5):
        rng = np.random.default_rng(int(SEED_BASE) + 1009 * r + 2)
        starts.append((f"random_{r}", random_parameters(ndepth, rng)))

    best_x: np.ndarray | None = None
    best_e = float("inf")
    history: list[dict] = []
    t0 = time.time()
    for i, (name, x0) in enumerate(starts):
        elapsed = time.time() - t0
        if elapsed > WALL_S and i > 0:
            print(f"wall {elapsed:.0f}s > {WALL_S}s; stop before {name}", flush=True)
            break
        print(f"  start {i + 1}/{len(starts)} {name} ...", flush=True)
        opt = optimize_vqe(
            sim, x0, method="L-BFGS-B", maxiter=200, record_every=0, verbose=False
        )
        rec = {
            "restart": name,
            "fun": float(opt.fun),
            "success": bool(opt.success),
            "nfev": int(opt.nfev),
            "elapsed_s": time.time() - t0,
        }
        history.append(rec)
        print(
            f"  {name}: E={opt.fun:.6f} deficit={float(opt.fun) - e0:.6f} "
            f"nfev={opt.nfev}",
            flush=True,
        )
        if float(opt.fun) < best_e:
            best_e = float(opt.fun)
            best_x = np.asarray(opt.x, float)
        progress = {
            "ansatz": "ecd",
            "ndepth": ndepth,
            "hamiltonian_id": 2,
            "seed_base": int(SEED_BASE),
            "maxiter": 200,
            "n_restarts": len(history),
            "fun": best_e,
            "elapsed_s": time.time() - t0,
            "restarts": history,
            "x": None if best_x is None else best_x.tolist(),
            "energy_min": e0,
            "gap": gap,
            "deficit": best_e - e0,
            "near_e0": (best_e - e0) <= NEAR_E0,
            "note": "H002 ECD: 5 random + optional H000 warm start, maxiter=200",
        }
        _write(DEST / "opt_progress.json", progress)

    assert best_x is not None
    elapsed = time.time() - t0
    deficit = best_e - e0
    payload = {
        "ansatz": "ecd",
        "ndepth": ndepth,
        "hamiltonian_id": 2,
        "seed_base": int(SEED_BASE),
        "maxiter": 200,
        "n_restarts": max(len(history), 3),
        "fun": best_e,
        "nfev": int(min(history, key=lambda r: r["fun"])["nfev"]) if history else 0,
        "elapsed_s": elapsed,
        "restarts": history,
        "x": best_x.tolist(),
        "energy_min": e0,
        "gap": gap,
        "deficit": deficit,
        "near_e0": deficit <= NEAR_E0,
        "note": "H002 ECD: 5 random + optional H000 warm start, maxiter=200",
    }
    _write(DEST / "opt_extra.json", payload)
    _write(CACHE, payload)
    (DEST / "NEAR_E0").write_text("1\n" if deficit <= NEAR_E0 else "0\n")
    print(
        f"BEST E={best_e:.6f}  E0={e0:.6f}  deficit={deficit:.6f}  "
        f"near_e0={deficit <= NEAR_E0}  wall={elapsed:.1f}s  wrote {CACHE.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
