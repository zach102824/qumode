#!/usr/bin/env python3
"""Depth sweep for one n: ECD L=4…40 until bitstring success ≥ 90%.

Canonical protocol: 200 joint SPSA, vacuum prep, sampled_tail η, noiseless.
Old 70-SPSA cells in results_70spsa_superseded/ are not resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .config import (
    L_MAX,
    L_START,
    N_HAMILTONIANS,
    N_TRIALS,
    OUTER_ITER,
    PREP_STEP_SCALE,
    PROTOCOL_TAG,
    SPSA_A,
    SPSA_A_STAB,
    SPSA_ALPHA,
    SPSA_C,
    SPSA_GAMMA,
    SUCCESS_THRESHOLD,
    ham_dir,
    is_canonical_cell,
    results_path,
    summary_path,
    trial_seed,
)
from .ecd import hybrid_energy_tensor, random_ecd_parameters, vacuum_prep
from .embedding import embedding_for_n, hardware_idle_modes
from .four_sat import load_instance
from .gibbs import optimize_gibbs_adaptive
from .io_util import write_conclusion, write_json


def _limit_blas(n: int = 1) -> None:
    n_s = str(int(n))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "QUTIP_NUM_PROCESSES",
    ):
        os.environ[key] = n_s


_limit_blas(1)


def _load_jobs(n: int, max_hamiltonians: int | None) -> list[dict]:
    d = ham_dir(n)
    paths = sorted(d.glob("four_sat_[0-9][0-9][0-9].npz"))
    if not paths:
        raise FileNotFoundError(f"no Hamiltonians in {d}; run generate_hamiltonians --n {n}")
    if max_hamiltonians is not None:
        paths = paths[: max(int(max_hamiltonians), 0)]
    return [load_instance(p) for p in paths]


def _run_trial(job: dict) -> dict:
    _limit_blas(1)
    n = int(job["n"])
    hid = int(job["hid"])
    trial = int(job["trial"])
    ndepth = int(job["L"])
    inst = load_instance(job["path"])
    emb = embedding_for_n(n)
    energy = hybrid_energy_tensor(emb, inst["logical_energies"])
    rng = np.random.default_rng(int(job["seed"]))
    x0 = random_ecd_parameters(ndepth, emb.n_pairs, rng)
    t0 = time.perf_counter()
    result = optimize_gibbs_adaptive(
        emb,
        energy,
        inst["ground_bitstring"],
        x0,
        ndepth=ndepth,
        prep0=vacuum_prep(emb),
        outer_iter=int(job["outer_iter"]),
        rng=rng,
        a=float(job["spsa_a"]),
        c=float(job["spsa_c"]),
        A=float(job["spsa_A"]),
        alpha=float(job["spsa_alpha"]),
        gamma=float(job["spsa_gamma"]),
        prep_step_scale=float(job["prep_step_scale"]),
    )
    ev = result.eval_final
    elapsed = time.perf_counter() - t0
    return {
        "n": n,
        "L": ndepth,
        "hamiltonian_id": hid,
        "trial": trial,
        "seed": int(job["seed"]),
        "success": bool(ev.success),
        "most_likely_bitstring": ev.most_likely_bitstring,
        "ground_bitstring": ev.ground_bitstring,
        "gibbs_cost": float(ev.gibbs_cost),
        "energy": float(ev.energy),
        "p_most_likely": float(ev.p_most_likely),
        "p_ground": float(ev.p_ground),
        "eta": float(result.eta),
        "eta0": float(result.eta0),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "n_eta_clamps": int(result.n_eta_clamps),
        "n_eta_fallbacks": int(result.n_eta_fallbacks),
        "elapsed_s": float(elapsed),
        "file": inst["file"],
    }


def _protocol_block(outer_iter: int) -> dict:
    return {
        "tag": PROTOCOL_TAG,
        "ansatz": "ecd",
        "objective": "gibbs",
        "eta": "sampled_tail",
        "noise": None,
        "snap": False,
        "gdr": False,
        "success": "most_likely_bitstring == ground_bitstring",
        "spsa": {
            "a": SPSA_A,
            "c": SPSA_C,
            "A": SPSA_A_STAB,
            "alpha": SPSA_ALPHA,
            "gamma": SPSA_GAMMA,
            "outer_iter": int(outer_iter),
            "prep_step_scale": PREP_STEP_SCALE,
            "prep_init": "vacuum",
        },
    }


def _payload(
    n: int,
    depth: int,
    trials: list[dict],
    *,
    instances: list[dict],
    n_trials: int,
    outer_iter: int,
    emb,
    wall_s: float,
) -> dict:
    trials = sorted(trials, key=lambda r: (int(r["hamiltonian_id"]), int(r["trial"])))
    k = int(sum(bool(r["success"]) for r in trials))
    ntot = len(trials)
    return {
        "n": n,
        "L": int(depth),
        "k": k,
        "n_total": ntot,
        "success_fraction": f"{k}/{ntot}",
        "success_prob": k / max(ntot, 1),
        "wall_s": wall_s,
        "n_hamiltonians": len(instances),
        "n_trials_per_h": int(n_trials),
        "outer_iter": int(outer_iter),
        "embedding": emb.as_dict(),
        "idle_hardware_modes": hardware_idle_modes(n),
        "protocol": _protocol_block(outer_iter),
        "trials": trials,
    }


def run_depth(
    n: int,
    depth: int,
    *,
    n_trials: int = N_TRIALS,
    max_hamiltonians: int | None = None,
    outer_iter: int = OUTER_ITER,
    workers: int = 1,
    resume: bool = True,
) -> dict:
    instances = _load_jobs(n, max_hamiltonians)
    emb = embedding_for_n(n)
    out_path = results_path(n, depth)
    existing: dict[tuple[int, int], dict] = {}
    expected = len(instances) * int(n_trials)
    if resume and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if not is_canonical_cell(prev, outer_iter=outer_iter):
            print(
                f"=== n={n}  L={depth}  ignore non-canonical "
                f"outer_iter={prev.get('outer_iter')} tag={(prev.get('protocol') or {}).get('tag')} "
                f"(need {outer_iter} / {PROTOCOL_TAG}) ===",
                flush=True,
            )
        else:
            for rec in prev.get("trials", []):
                existing[(int(rec["hamiltonian_id"]), int(rec["trial"]))] = rec
            if len(existing) >= expected:
                print(
                    f"=== n={n}  L={depth}  skip complete {len(existing)}/{expected}  "
                    f"outer_iter={prev.get('outer_iter')}  → {out_path} ===",
                    flush=True,
                )
                return prev

    jobs = []
    for inst in instances:
        hid = int(inst["hamiltonian_id"])
        for t in range(int(n_trials)):
            if (hid, t) in existing:
                continue
            jobs.append(
                {
                    "n": n,
                    "L": int(depth),
                    "hid": hid,
                    "trial": t,
                    "path": inst["path"],
                    "seed": trial_seed(n, hid, t),
                    "outer_iter": int(outer_iter),
                    "spsa_a": SPSA_A,
                    "spsa_c": SPSA_C,
                    "spsa_A": SPSA_A_STAB,
                    "spsa_alpha": SPSA_ALPHA,
                    "spsa_gamma": SPSA_GAMMA,
                    "prep_step_scale": PREP_STEP_SCALE,
                }
            )

    trials = list(existing.values())
    t0 = time.perf_counter()
    print(
        f"=== n={n}  L={depth}  {len(instances)} H × {n_trials} trials  "
        f"todo={len(jobs)} cached={len(existing)}  dim={emb.dim}  "
        f"pairs={emb.n_pairs}  workers={workers}  SPSA={outer_iter} ===",
        flush=True,
    )

    def _commit(extra_wall: float = 0.0) -> dict:
        wall = time.perf_counter() - t0 + extra_wall
        payload = _payload(
            n,
            depth,
            trials,
            instances=instances,
            n_trials=n_trials,
            outer_iter=outer_iter,
            emb=emb,
            wall_s=wall,
        )
        write_json(out_path, payload)
        return payload

    if jobs:
        workers = max(1, min(int(workers), len(jobs)))
        if workers == 1:
            for i, job in enumerate(jobs, start=1):
                rec = _run_trial(job)
                trials.append(rec)
                print(
                    f"  [{i}/{len(jobs)}] H{rec['hamiltonian_id']} t{rec['trial']}: "
                    f"{'HIT' if rec['success'] else 'miss'}  "
                    f"{rec['most_likely_bitstring']} vs {rec['ground_bitstring']}  "
                    f"cost={rec['gibbs_cost']:.3f}  {rec['elapsed_s']:.2f}s",
                    flush=True,
                )
                if i == len(jobs) or i % 10 == 0:
                    _commit()
        else:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_limit_blas,
                initargs=(1,),
            ) as pool:
                futs = [pool.submit(_run_trial, job) for job in jobs]
                done = 0
                for fut in as_completed(futs):
                    rec = fut.result()
                    trials.append(rec)
                    done += 1
                    print(
                        f"  [{done}/{len(jobs)}] H{rec['hamiltonian_id']} t{rec['trial']}: "
                        f"{'HIT' if rec['success'] else 'miss'}  "
                        f"{rec['most_likely_bitstring']} vs {rec['ground_bitstring']}  "
                        f"cost={rec['gibbs_cost']:.3f}  {rec['elapsed_s']:.2f}s",
                        flush=True,
                    )
                    if done == len(jobs) or done % 10 == 0:
                        _commit()

    payload = _commit()
    print(
        f"n={n} L={depth}: {payload['k']}/{payload['n_total']} = {payload['success_prob']:.3f}  "
        f"wall={payload['wall_s']:.1f}s  SPSA={outer_iter}  → {out_path}",
        flush=True,
    )
    return payload


def summarize_n(n: int, curve: list[dict], wall_s: float) -> dict:
    curve = [c for c in curve if int(c["L"]) >= L_START]
    hit = next((c for c in curve if float(c["success_prob"]) >= SUCCESS_THRESHOLD), None)
    best = max(curve, key=lambda r: (float(r["success_prob"]), -int(r["L"]))) if curve else None
    chosen = hit or best
    last_l = int(curve[-1]["L"]) if curve else 0
    if hit is not None:
        status = "hit_threshold"
    elif curve and last_l >= L_MAX:
        status = "capped_L40_below_threshold"
    else:
        status = "in_progress"
    summary = {
        "n": int(n),
        "L_star": None if hit is None else int(hit["L"]),
        "k": 0 if chosen is None else int(chosen["k"]),
        "n_total": 0 if chosen is None else int(chosen["n_total"]),
        "success_prob": 0.0 if chosen is None else float(chosen["success_prob"]),
        "success_fraction": "0/0" if chosen is None else f"{int(chosen['k'])}/{int(chosen['n_total'])}",
        "wall_s": float(wall_s),
        "status": status,
        "outer_iter": OUTER_ITER,
        "protocol_tag": PROTOCOL_TAG,
        "l_max": L_MAX,
        "curve": [
            {
                "L": int(c["L"]),
                "k": int(c["k"]),
                "n_total": int(c["n_total"]),
                "success_prob": float(c["success_prob"]),
                "wall_s": float(c.get("wall_s", 0.0)),
                "outer_iter": int(c.get("outer_iter", OUTER_ITER)),
            }
            for c in curve
        ],
        "embedding": embedding_for_n(n).as_dict(),
        "idle_hardware_modes": hardware_idle_modes(n),
    }
    write_json(summary_path(n), summary)
    write_conclusion()
    return summary


def run_sweep(
    n: int,
    *,
    l_start: int = L_START,
    l_max: int = L_MAX,
    n_trials: int = N_TRIALS,
    max_hamiltonians: int | None = None,
    outer_iter: int = OUTER_ITER,
    workers: int = 1,
    no_sweep: bool = False,
) -> dict:
    n = int(n)
    t0 = time.perf_counter()
    curve: list[dict] = []
    depths = [int(l_start)] if no_sweep else list(range(int(l_start), int(l_max) + 1))
    for depth in depths:
        rec = run_depth(
            n,
            depth,
            n_trials=n_trials,
            max_hamiltonians=max_hamiltonians,
            outer_iter=outer_iter,
            workers=workers,
        )
        curve.append(rec)
        write_conclusion(f"Live: n={n} L={depth} → {rec['success_fraction']} ({PROTOCOL_TAG}).")
        if float(rec["success_prob"]) >= SUCCESS_THRESHOLD:
            print(f"n={n}: L*={depth} hits ≥{SUCCESS_THRESHOLD:.0%}", flush=True)
            break
        if depth >= int(l_max):
            print(
                f"n={n}: still below {SUCCESS_THRESHOLD:.0%} at soft cap L={depth}. "
                f"Stopping this n (do not silently treat L=20 as a hard cap).",
                flush=True,
            )
            break
        if not no_sweep:
            print(f"n={n} L={depth} below threshold; trying L={depth + 1}", flush=True)
    wall = time.perf_counter() - t0
    return summarize_n(n, curve, wall)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--L", type=int, default=L_START, dest="depth")
    parser.add_argument("--l-max", type=int, default=L_MAX)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--max-hamiltonians", type=int, default=None)
    parser.add_argument("--outer-iter", type=int, default=OUTER_ITER)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="1 = in-process (fast). >1 uses a spawn pool; BLAS is pinned to 1 thread.",
    )
    parser.add_argument("--no-sweep", action="store_true", help="Run only --L, do not increment.")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if args.no_resume:
        path = results_path(args.n, args.depth)
        if path.exists():
            path.unlink()
    summary = run_sweep(
        args.n,
        l_start=args.depth,
        l_max=args.l_max,
        n_trials=args.n_trials,
        max_hamiltonians=args.max_hamiltonians,
        outer_iter=args.outer_iter,
        workers=args.workers,
        no_sweep=args.no_sweep,
    )
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
