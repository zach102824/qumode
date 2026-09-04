#!/usr/bin/env python3
"""Sequential cheap GDR research loops (one heavy process).

Writes only under ``Error_mitigation/out_research/``. Never touches
``out_smoke/`` or ``out/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from Error_mitigation.run_mitigation_experiment import parse_args, run
from Error_mitigation.summarize import records_from, write_markdown

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_research"

# Phase-1 matrix: one ansatz at a time, random params, small n_train.
JOBS = [
    {
        "name": "ecd_loss_mild_uniform",
        "argv": [
            "--preset", "diag",
            "--ansatz", "ecd",
            "--param-set", "random",
            "--families", "loss",
            "--kappa-tau", "0.003",
            "--readout", "ideal",
            "--n-train", "12",
            "--shots", "4096",
            "--alpha-policy", "uniform",
        ],
    },
    {
        "name": "ecd_loss_mild_uniform_ro",
        "argv": [
            "--preset", "diag",
            "--ansatz", "ecd",
            "--param-set", "random",
            "--families", "loss",
            "--kappa-tau", "0.003",
            "--readout", "readout_realistic",
            "--n-train", "12",
            "--shots", "4096",
            "--alpha-policy", "uniform",
        ],
    },
    {
        "name": "ecd_loss_mild_stratified",
        "argv": [
            "--preset", "diag",
            "--ansatz", "ecd",
            "--param-set", "random",
            "--families", "loss",
            "--kappa-tau", "0.003",
            "--readout", "ideal",
            "--n-train", "12",
            "--shots", "4096",
            "--alpha-policy", "stratified",
        ],
    },
    {
        "name": "ecd_loss_kt03",
        "argv": [
            "--preset", "diag",
            "--ansatz", "ecd",
            "--param-set", "random",
            "--families", "loss",
            "--kappa-tau", "0.03",
            "--readout", "ideal",
            "--n-train", "12",
            "--shots", "4096",
            "--alpha-policy", "uniform",
        ],
    },
    {
        "name": "ecd_thermal_mild",
        "argv": [
            "--preset", "diag",
            "--ansatz", "ecd",
            "--param-set", "random",
            "--families", "loss_thermal_dephasing",
            "--kappa-tau", "0.003",
            "--readout", "ideal",
            "--n-train", "12",
            "--shots", "4096",
        ],
    },
]


def _merge_scoreboard() -> None:
    recs = []
    for job in JOBS:
        path = OUT / job["name"] / "results.json"
        if path.is_file():
            recs.extend(records_from(path))
    if recs:
        write_markdown(recs, OUT / "phase1_scoreboard.md", title="Phase 1 GDR diagnostics")
        print(f"wrote {OUT / 'phase1_scoreboard.md'}  ({len(recs)} cases)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default=None, help="Run a single JOBS name.")
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)
    if args.list:
        for job in JOBS:
            print(job["name"])
        return 0
    jobs = JOBS
    if args.only:
        jobs = [j for j in JOBS if j["name"] == args.only]
        if not jobs:
            print(f"unknown job {args.only!r}", file=sys.stderr)
            return 2
    OUT.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        outdir = OUT / job["name"]
        if (outdir / "results.json").is_file():
            print(f"skip {job['name']} (results.json exists)")
            continue
        argv_job = list(job["argv"]) + ["--outdir", str(outdir), "--seed", "2026"]
        print(f"\n===== {job['name']} =====\n{' '.join(argv_job)}")
        t0 = time.time()
        ns = parse_args(argv_job)
        run(ns)
        print(f"===== {job['name']} done in {time.time() - t0:.1f}s =====")
        _merge_scoreboard()
    _merge_scoreboard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
