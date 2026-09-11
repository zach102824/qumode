#!/usr/bin/env python3
"""Run matched 4-SAT GDR @8192 (2-wide).

Default queue is SNAP L3 + ECD L4 into the original ``{ansatz}_hXXX_s8192``
dirs. Pass ``--suite l2l3`` for SNAP L2 + ECD L3, or ``--suite l1l2`` for
SNAP L1 + ECD L2, into ``{ansatz}_l{d}_hXXX_*``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LOG = OUT / "COMMANDS.log"
GDRLOG = OUT / "gdr_run.log"
MAX_WORKERS = 2

SUITES = {
    "l3l4": (("snap", 3, "gibbs_four_sat_snap_matched_n10.json", ""), ("ecd", 4, "gibbs_four_sat_ecd_matched_n10.json", "")),
    "l2l3": (("snap", 2, "gibbs_four_sat_snap_matched_n10_L2.json", "l2_"), ("ecd", 3, "gibbs_four_sat_ecd_matched_n10_L3.json", "l3_")),
    "l1l2": (("snap", 1, "gibbs_four_sat_snap_matched_n10_L1.json", "l1_"), ("ecd", 2, "gibbs_four_sat_ecd_matched_n10_L2.json", "l2_")),
}


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
    print(line, flush=True)
    for path in (LOG, GDRLOG):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def run_one(ansatz: str, hid: int, ndepth: int, gibbs_name: str, prefix: str) -> tuple[str, int, int, float]:
    tag = f"{ansatz}_{prefix}h{hid:03d}_s8192"
    outdir = OUT / tag
    results = outdir / "results.json"
    if results.is_file():
        log(f"SKIP existing {outdir.relative_to(ROOT)}")
        return ansatz, hid, 0, 0.0
    gibbs = ROOT / "results" / gibbs_name
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "Error_mitigation" / "run_mitigation_experiment.py"),
        "--preset",
        "full",
        "--family",
        "four_sat",
        "--instance",
        str(hid),
        "--ansatz",
        ansatz,
        "--ndepth",
        str(ndepth),
        "--families",
        "comprehensive",
        "--readout",
        "readout_realistic",
        "--kappa-tau",
        "0.003,0.03,0.1",
        "--gibbs-json",
        str(gibbs),
        "--gibbs-pick",
        "success_then_cost",
        "--twin-design",
        "adaptive",
        "--shots",
        "8192",
        "--n-train",
        "40",
        "--params",
        "both",
        "--outdir",
        str(outdir),
    ]
    log(f"START {ansatz} H{hid} nd={ndepth} s8192 shots=8192 n_train=40")
    t0 = time.time()
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT / "src")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log(f"FAIL {ansatz} H{hid} s8192 rc={proc.returncode} {elapsed:.1f}s")
    else:
        log(f"DONE {ansatz} H{hid} s8192 {elapsed:.1f}s")
    return ansatz, hid, int(proc.returncode), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="l3l4")
    args = parser.parse_args()
    spec = SUITES[args.suite]
    jobs = [(ansatz, hid, nd, gibbs, prefix) for ansatz, nd, gibbs, prefix in spec for hid in range(20)]
    Path("/tmp/mpl").mkdir(parents=True, exist_ok=True)
    log(f"GDR 8192 suite={args.suite} queue {len(jobs)} jobs workers={MAX_WORKERS}")
    rc = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(run_one, a, h, d, g, p) for a, h, d, g, p in jobs]
        for fut in as_completed(futs):
            _ansatz, _hid, code, _elapsed = fut.result()
            if code != 0:
                rc = code
    log(f"GDR 8192 suite={args.suite} finished rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
