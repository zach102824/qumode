#!/usr/bin/env python3
"""ECD ndepth=5 Gibbs η ablation on 4-SAT (diagnostic; not a GDR default).

Holds the production path (sampled_tail adaptive) as one arm and compares
fixed η values from paper_result/run_eta_ablation.py plus typical final η
logged on mixed p-spin ECD and 4-SAT SNAP/ECD fleets.

Default: 4-SAT H000, 5 paired trials, 200 joint SPSA, same seeds across arms.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qumode_vqe.hamiltonian import EXACT_GROUND_ENERGY, load_four_sat_instances

HERE = Path(__file__).resolve().parent
GATE = 0.5
PAPER_KNAPSACK_ETA = 1.0 / (0.05 * abs(EXACT_GROUND_ENERGY))
SHARP_ETA = 26.0
MIXED_ECD_L5_ETA = 3.0
SAT_ECD_FLEET_ETA = 7.2
SAT_SNAP_FLEET_ETA = 12.5
NDEPTH = 5
NFOCKS = (8, 8)


def _load_ecd():
    path = ROOT / "scripts" / "Gibbs_and_adaptive_optim_ECD.py"
    spec = importlib.util.spec_from_file_location("gibbs_ecd_eta_ablation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ecd = _load_ecd()

POLICIES = (
    {
        "name": "tail_adaptive",
        "label": "sampled_tail adaptive (production)",
        "eta_adaptive": True,
        "fixed_eta": None,
    },
    {
        "name": "tail_frozen",
        "label": "sampled_tail frozen at t=0",
        "eta_adaptive": False,
        "fixed_eta": None,
    },
    {
        "name": "fixed_soft",
        "label": "fixed η=0.1 (too soft)",
        "eta_adaptive": False,
        "fixed_eta": 0.1,
    },
    {
        "name": "fixed_paper",
        "label": f"fixed η={PAPER_KNAPSACK_ETA:.3f} (paper BKP)",
        "eta_adaptive": False,
        "fixed_eta": PAPER_KNAPSACK_ETA,
    },
    {
        "name": "fixed_mixed_l5",
        "label": f"fixed η={MIXED_ECD_L5_ETA:.1f} (mixed p-spin ECD L5 mean)",
        "eta_adaptive": False,
        "fixed_eta": MIXED_ECD_L5_ETA,
    },
    {
        "name": "fixed_sat_ecd",
        "label": f"fixed η={SAT_ECD_FLEET_ETA:.1f} (4-SAT ECD fleet mean)",
        "eta_adaptive": False,
        "fixed_eta": SAT_ECD_FLEET_ETA,
    },
    {
        "name": "fixed_sat_snap",
        "label": f"fixed η={SAT_SNAP_FLEET_ETA:.1f} (4-SAT SNAP fleet mean)",
        "eta_adaptive": False,
        "fixed_eta": SAT_SNAP_FLEET_ETA,
    },
    {
        "name": "fixed_sharp",
        "label": f"fixed η={SHARP_ETA:.0f} (paper sharp)",
        "eta_adaptive": False,
        "fixed_eta": SHARP_ETA,
    },
)
POLICY_BY_NAME = {p["name"]: p for p in POLICIES}


def _json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def _lite(rec: dict) -> dict:
    skip = {"x0", "x", "x_warmup", "prep0", "prep", "prep_unit", "warmup", "final", "eta_history"}
    out = {k: v for k, v in rec.items() if k not in skip}
    out["deficit"] = float(rec["energy_physical"]) - float(rec.get("energy_min") or 0.0)
    out["near_e0"] = bool(out["deficit"] <= GATE)
    hist = rec.get("eta_history") or []
    if hist:
        out["eta_history_ends"] = [
            {"step": int(hist[0]["step"]), "eta": float(hist[0]["eta"])},
            {"step": int(hist[-1]["step"]), "eta": float(hist[-1]["eta"])},
        ]
    return out


def run_one(job: dict) -> dict:
    rec = ecd.run_trial(job)
    rec["eta_policy_name"] = str(job["eta_policy_name"])
    rec["eta_label"] = str(job["eta_label"])
    rec["fixed_eta"] = job.get("fixed_eta")
    rec["deficit"] = float(rec["energy_physical"]) - float(rec.get("energy_min") or 0.0)
    rec["near_e0"] = bool(rec["deficit"] <= GATE)
    return rec


def _summarize(records: list[dict]) -> dict:
    n = len(records)
    deficits = [float(r["deficit"]) for r in records]
    ephys = [float(r["energy_physical"]) for r in records]
    etas = [float(r["eta"]) for r in records if r.get("eta") is not None]
    n_gate = int(sum(bool(r["near_e0"]) for r in records))
    n_gs = int(sum(bool(r.get("success")) for r in records))
    best = min(records, key=lambda r: float(r["deficit"])) if records else None
    return {
        "n": n,
        "n_near_e0": n_gate,
        "near_e0_rate": n_gate / max(n, 1),
        "n_success_gs": n_gs,
        "mean_energy_physical": float(np.mean(ephys)) if ephys else None,
        "best_energy_physical": float(np.min(ephys)) if ephys else None,
        "mean_deficit": float(np.mean(deficits)) if deficits else None,
        "best_deficit": float(np.min(deficits)) if deficits else None,
        "mean_eta": float(np.mean(etas)) if etas else None,
        "best_trial": None
        if best is None
        else {
            "trial": int(best["trial"]),
            "hamiltonian_id": int(best["hamiltonian_id"]),
            "energy_physical": float(best["energy_physical"]),
            "deficit": float(best["deficit"]),
            "success": bool(best.get("success")),
            "eta": best.get("eta"),
            "most_likely_bitstring": best.get("most_likely_bitstring"),
        },
    }


def run_ablation(
    *,
    hamiltonian_ids: list[int],
    n_trials: int,
    outer_iter: int,
    workers: int,
    seed_base: int,
    policy_names: list[str] | None = None,
) -> dict:
    policies = [POLICY_BY_NAME[n] for n in (policy_names or [p["name"] for p in POLICIES])]
    instances = load_four_sat_instances(
        ROOT / "Hamiltonians" / "four_sat",
        nfocks=NFOCKS,
    )
    wanted = {int(i) for i in hamiltonian_ids}
    instances = [inst for inst in instances if int(inst["hamiltonian_id"]) in wanted]
    if len(instances) != len(wanted):
        found = {int(inst["hamiltonian_id"]) for inst in instances}
        raise FileNotFoundError(f"missing 4-SAT ids {sorted(wanted - found)}")

    spsa = ecd._spsa_job_fields(
        ndepth=NDEPTH,
        nfocks=NFOCKS,
        outer_iter=int(outer_iter),
        spsa_iter=0,
        spsa_a=ecd.SPSA_A,
        spsa_c=ecd.SPSA_C,
        spsa_A=ecd.SPSA_A_STAB,
        spsa_alpha=ecd.SPSA_ALPHA,
        spsa_gamma=ecd.SPSA_GAMMA,
    )
    jobs: list[dict] = []
    for inst in instances:
        hid = int(inst["hamiltonian_id"])
        for trial in range(int(n_trials)):
            unit, prep0 = ecd.vacuum_start()
            for policy in policies:
                jobs.append(
                    {
                        **spsa,
                        "trial": trial,
                        "hamiltonian_id": hid,
                        "family": "four_sat",
                        "kind": "four_sat",
                        "file": inst["file"],
                        "ansatz": "ecd",
                        "ndepth": NDEPTH,
                        "seed_base": int(seed_base) + 100 * hid,
                        "prep0": prep0,
                        "prep_unit": unit,
                        "energy_tensor": inst["energy_tensor"],
                        "fixed_eta": policy["fixed_eta"],
                        "eta_adaptive": policy["eta_adaptive"],
                        "eta_policy_name": policy["name"],
                        "eta_label": policy["label"],
                    }
                )

    print(
        f"=== ECD L{NDEPTH} η ablation  H={sorted(wanted)}  "
        f"{n_trials} trials × {len(policies)} arms  joint SPSA {outer_iter}  "
        f"workers={workers} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    records: list[dict] = []
    n = len(jobs)
    if workers <= 1 or n <= 1:
        for i, job in enumerate(jobs, 1):
            rec = run_one(job)
            records.append(rec)
            print(
                f"[{i}/{n}] H{int(rec['hamiltonian_id']):03d} t{rec['trial']}  "
                f"{rec['eta_policy_name']:<16}  ⟨H⟩={rec['energy_physical']:.4f}  "
                f"def={rec['deficit']:.4f}  nearE0={rec['near_e0']}  "
                f"η={float(rec['eta']):.3f}  gs={rec.get('success')}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(run_one, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                done += 1
                print(
                    f"[{done}/{n}] H{int(rec['hamiltonian_id']):03d} t{rec['trial']}  "
                    f"{rec['eta_policy_name']:<16}  ⟨H⟩={rec['energy_physical']:.4f}  "
                    f"def={rec['deficit']:.4f}  nearE0={rec['near_e0']}  "
                    f"η={float(rec['eta']):.3f}  gs={rec.get('success')}",
                    flush=True,
                )
    elapsed = time.perf_counter() - t0
    records.sort(
        key=lambda r: (
            int(r["hamiltonian_id"]),
            str(r["eta_policy_name"]),
            int(r["trial"]),
        )
    )
    by_policy: dict[str, list[dict]] = {p["name"]: [] for p in policies}
    for rec in records:
        by_policy.setdefault(str(rec["eta_policy_name"]), []).append(rec)
    summaries = {name: _summarize(by_policy.get(name, [])) for name, _ in by_policy.items() if name in POLICY_BY_NAME}
    any_pass = any(int(s["n_near_e0"]) > 0 for s in summaries.values())
    best_arm = min(
        summaries.items(),
        key=lambda kv: (float("inf") if kv[1]["best_deficit"] is None else kv[1]["best_deficit"]),
    )
    return {
        "ansatz": "ecd",
        "family": "four_sat",
        "ndepth": NDEPTH,
        "n_trials": int(n_trials),
        "outer_iter": int(outer_iter),
        "seed_base": int(seed_base),
        "hamiltonian_ids": sorted(wanted),
        "gate": GATE,
        "workers": int(workers),
        "elapsed_sec": elapsed,
        "policies": list(policies),
        "any_near_e0": bool(any_pass),
        "best_arm": best_arm[0],
        "best_deficit": best_arm[1]["best_deficit"],
        "summaries": summaries,
        "trials": [_lite(r) for r in records],
    }


def write_text(payload: dict) -> str:
    lines = [
        "# ECD L5 4-SAT η ablation",
        "",
        f"H={payload['hamiltonian_ids']}  trials={payload['n_trials']}  "
        f"joint SPSA={payload['outer_iter']}  gate={payload['gate']}",
        f"any arm with energy_physical−E0 ≤ 0.5: {payload['any_near_e0']}",
        f"best arm: {payload['best_arm']}  best deficit={payload['best_deficit']:.4f}",
        "",
        f"{'policy':<16} {'nearE0':>8} {'best⟨H⟩':>8} {'mean⟨H⟩':>8} {'best def':>9} {'mean η':>8}",
    ]
    for policy in payload["policies"]:
        name = policy["name"]
        s = payload["summaries"][name]
        lines.append(
            f"{name:<16} {s['n_near_e0']:3d}/{s['n']:<4d} "
            f"{s['best_energy_physical']:8.3f} {s['mean_energy_physical']:8.3f} "
            f"{s['best_deficit']:9.3f} {s['mean_eta']:8.3f}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamiltonian-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--outer-iter", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=6100)
    parser.add_argument("--outdir", type=Path, default=HERE)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help=f"Subset of {[p['name'] for p in POLICIES]}.",
    )
    parser.add_argument("--tag", type=str, default="h000")
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = run_ablation(
        hamiltonian_ids=list(args.hamiltonian_ids),
        n_trials=int(args.n_trials),
        outer_iter=int(args.outer_iter),
        workers=int(args.workers),
        seed_base=int(args.seed_base),
        policy_names=args.policies,
    )
    json_path = outdir / f"ecd_l5_eta_ablation_{args.tag}.json"
    txt_path = outdir / f"ecd_l5_eta_ablation_{args.tag}.txt"
    json_path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")
    text = write_text(payload)
    txt_path.write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    print(f"Wrote {json_path} and {txt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
