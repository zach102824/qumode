#!/usr/bin/env python3
"""Generate 20 unique-planted 4-SAT Hamiltonians per system size."""

from __future__ import annotations

import argparse
import json

from .config import LADDER_NS, N_HAMILTONIANS, SEARCH_TRIALS, clause_window, ham_dir, ham_seed
from .four_sat import generate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None, help="Single n in 7…11.")
    parser.add_argument("--all", action="store_true", help="Generate n=8…11 (live ladder).")
    parser.add_argument("--n-hamiltonians", type=int, default=N_HAMILTONIANS)
    parser.add_argument("--search-trials", type=int, default=SEARCH_TRIALS)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    if args.all:
        ns = list(LADDER_NS)
    elif args.n is not None:
        ns = [int(args.n)]
    else:
        parser.error("pass --n N or --all")
    for n in ns:
        lo, target, hi = clause_window(n)
        print(
            f"=== n={n}  seed={args.seed if args.seed is not None else ham_seed(n)}  "
            f"window={lo}–{hi} target={target} → {ham_dir(n)} ===",
            flush=True,
        )
        manifest = generate_dataset(
            n,
            n_hamiltonians=args.n_hamiltonians,
            search_trials=args.search_trials,
            seed=args.seed,
        )
        print(json.dumps({"n": n, "n_saved": len(manifest), "dir": str(ham_dir(n))}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
