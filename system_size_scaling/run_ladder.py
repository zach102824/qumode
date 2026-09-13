#!/usr/bin/env python3
"""n=12 → 13 → 14 → 15 depth-sweep ladder (L starts at 4, 200 joint SPSA).

n=7 is prior PR #14 data. n=8…11 is the finished PR #15 ladder — do not rerun.
Soft cap L=12 for this extension (deeper L at 200 SPSA collapsed on n=8).
"""

from __future__ import annotations

import argparse

from .config import (
    HIGHER_NS,
    L_MAX_HIGHER,
    L_START,
    N_HAMILTONIANS,
    N_TRIALS,
    OUTER_ITER,
    SEARCH_TRIALS,
    ham_dir,
)
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
    parser.add_argument("--from-n", type=int, default=12)
    parser.add_argument("--to-n", type=int, default=15)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--n-hamiltonians", type=int, default=N_HAMILTONIANS)
    parser.add_argument("--search-trials", type=int, default=SEARCH_TRIALS)
    parser.add_argument("--outer-iter", type=int, default=OUTER_ITER)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--l-start", type=int, default=L_START)
    parser.add_argument("--l-max", type=int, default=L_MAX_HIGHER)
    parser.add_argument(
        "--scout",
        action="store_true",
        help="5 H × 4 trials per cell (timing / sanity) before a full 20×10.",
    )
    args = parser.parse_args(argv)
    ns = [n for n in HIGHER_NS if int(args.from_n) <= n <= int(args.to_n)]
    if not ns:
        parser.error("empty n range (this ladder is n=12…15; n=8…11 is PR #15)")
    n_h = 5 if args.scout and args.n_hamiltonians == N_HAMILTONIANS else args.n_hamiltonians
    n_t = 4 if args.scout and args.n_trials == N_TRIALS else args.n_trials
    write_conclusion(
        f"Higher-n ladder started under {args.outer_iter}-SPSA protocol; "
        f"n={ns[0]}…{ns[-1]}, L={args.l_start}…{args.l_max}"
        f"{' (scout 5H×4trial)' if args.scout else ''}."
    )
    for n in ns:
        # Always materialize the full 20-H fleet even if this pass is a scout.
        ensure_hamiltonians(n, args.n_hamiltonians, args.search_trials)
        if n <= 11:
            print(f"REFUSING to rerun n={n} (PR #15 / #14). Skip.", flush=True)
            continue
        summary = run_sweep(
            n,
            l_start=args.l_start,
            l_max=args.l_max,
            n_trials=n_t,
            max_hamiltonians=n_h,
            outer_iter=args.outer_iter,
            workers=args.workers,
        )
        print(
            f"FINISHED n={n}: L*={summary.get('L_star')}  "
            f"{summary.get('success_fraction')}  status={summary.get('status')}",
            flush=True,
        )
        if str(summary.get("status", "")).startswith("capped_"):
            print(
                f"SOFT CAP: n={n} never reached 90% through L={args.l_max}. "
                "Recorded and stopping this n as requested.",
                flush=True,
            )
    write_conclusion("Higher-n ladder command completed for the requested n range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
