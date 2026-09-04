#!/usr/bin/env python3
"""Noiseless Gibbs VQE: joint SPSA on preparation and SNAP+displacement.

Same mixed p-spin Hamiltonians, seeds, vacuum initial state,
sampled-tail Gibbs cost, and SPSA gains as ``Gibbs_and_adaptive_optim_ECD.py``.
Each SNAP+displacement layer is

    SNAP(θ) D(α) on qumode 1, then SNAP(θ) D(α) on qumode 2

with the Fock-0 SNAP phase gauge-fixed to zero (7 trainable phases per
mode at L=8). That is 18 ansatz parameters and 4 primitive gates per layer,
so depths 1–2 stay at 18 and 36 ansatz parameters (under the ~40 cap).

Default budget is 200 joint SPSA steps; prep is never frozen.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_ECD_PATH = Path(__file__).resolve().parent / "Gibbs_and_adaptive_optim_ECD.py"
_SPEC = importlib.util.spec_from_file_location("gibbs_and_adaptive_optim_ecd", _ECD_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"could not load {_ECD_PATH}")
ecd = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ecd
_SPEC.loader.exec_module(ecd)

OUTDIR = ecd.OUTDIR
HAM_DIR = ecd.MIXED_P_SPIN_DIR
OUTPUT_JSON = "gibbs_mixed_p_spin_snap.json"
N_TRIALS_DEFAULT = 1
WORKERS = ecd.WORKERS
SEED_BASE = ecd.SEED_BASE
MIXED_P_SPIN_STEPS = ecd.MIXED_P_SPIN_STEPS
MIXED_P_SPIN_SNAP_DEPTHS = ecd.MIXED_P_SPIN_SNAP_DEPTHS
NFOCKS = ecd.NFOCKS
SPSA_A = ecd.SPSA_A
SPSA_C = ecd.SPSA_C
SPSA_A_STAB = ecd.SPSA_A_STAB
SPSA_ALPHA = ecd.SPSA_ALPHA
SPSA_GAMMA = ecd.SPSA_GAMMA
SPSA_ITERATIONS = ecd.SPSA_ITERATIONS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument(
        "--outer-iter",
        type=int,
        default=MIXED_P_SPIN_STEPS,
        help="Joint SPSA steps on (prep, SNAP+D). Default 200; prep stays live.",
    )
    parser.add_argument(
        "--spsa-iter",
        type=int,
        default=SPSA_ITERATIONS,
        help="Optional ansatz-only steps after freezing prep. Default 0 (never freeze).",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--ham-dir", type=Path, default=HAM_DIR)
    parser.add_argument(
        "--ndepths",
        type=int,
        nargs="+",
        default=None,
        help="SNAP+D layer counts (default: 1 2, i.e. 18 and 36 ansatz parameters).",
    )
    parser.add_argument(
        "--max-hamiltonians",
        type=int,
        default=None,
        help="Optional cap on how many mixed p-spin NPZ files to load.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON path (default: <outdir>/gibbs_mixed_p_spin_snap.json).",
    )
    parser.add_argument("--spsa-a", type=float, default=SPSA_A)
    parser.add_argument("--spsa-c", type=float, default=SPSA_C)
    parser.add_argument("--spsa-A", type=float, default=SPSA_A_STAB)
    parser.add_argument("--spsa-alpha", type=float, default=SPSA_ALPHA)
    parser.add_argument("--spsa-gamma", type=float, default=SPSA_GAMMA)
    args = parser.parse_args(argv)
    ecd.run_mixed_p_spin_suite(
        ham_dir=args.ham_dir,
        n_trials=args.n_trials,
        outer_iter=args.outer_iter,
        spsa_iter=args.spsa_iter,
        workers=args.workers,
        seed_base=args.seed_base,
        outdir=args.outdir,
        ndepths=tuple(args.ndepths) if args.ndepths else MIXED_P_SPIN_SNAP_DEPTHS,
        nfocks=NFOCKS,
        spsa_a=args.spsa_a,
        spsa_c=args.spsa_c,
        spsa_A=args.spsa_A,
        spsa_alpha=args.spsa_alpha,
        spsa_gamma=args.spsa_gamma,
        ansatz="snap",
        output=args.output or Path(OUTPUT_JSON),
        max_hamiltonians=args.max_hamiltonians,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
