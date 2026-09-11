#!/usr/bin/env python3
"""n=7 → 8 → 9 → 10 → 11 depth-sweep ladder."""

from __future__ import annotations

import argparse

from .config import L_MAX, L_START, LADDER_NS, N_HAMILTONIANS, N_TRIALS, OUTER_ITER, SEARCH_TRIALS, ham_dir
from .four_sat import generate_dataset
from .io_util import write_conclusion
from .run_one_n import run_sweep


def ensure_hamiltonians(n: int, n_hamiltonians: int, search_trials: int) -> None:
    d = ham_dir(n)
    have = sorted(d.glob("four_sat_[0-9][0-9][0-9].npz"))
    if len(have) >= int(n_hamiltonians):
        print(f"n={n}: reusing {len(have)} Hamiltonians in {d}", flush=True)
        return
    print(f"n={n}: generating {n_hamiltonians} Hamiltonians in {d}", flush=True)
    generate_dataset(n, n_hamiltonians=n_hamiltonians, search_trials=search_trials)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-n", type=int, default=7)
    parser.add_argument("--to-n", type=int, default=11)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--n-hamiltonians", type=int, default=N_HAMILTONIANS)
    parser.add_argument("--search-trials", type=int, default=SEARCH_TRIALS)
    parser.add_argument("--outer-iter", type=int, default=OUTER_ITER)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--l-start", type=int, default=L_START)
    parser.add_argument("--l-max", type=int, default=L_MAX)
    args = parser.parse_args(argv)
    ns = [n for n in LADDER_NS if int(args.from_n) <= n <= int(args.to_n)]
    if not ns:
        parser.error("empty n range")
    write_conclusion("Ladder started; cells fill as each n finishes.")
    for n in ns:
        ensure_hamiltonians(n, args.n_hamiltonians, args.search_trials)
        summary = run_sweep(
            n,
            l_start=args.l_start,
            l_max=args.l_max,
            n_trials=args.n_trials,
            max_hamiltonians=args.n_hamiltonians,
            outer_iter=args.outer_iter,
            workers=args.workers,
        )
        print(
            f"FINISHED n={n}: L*={summary.get('L_star')}  "
            f"{summary.get('success_fraction')}  status={summary.get('status')}",
            flush=True,
        )
    write_conclusion("Ladder command completed for the requested n range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
