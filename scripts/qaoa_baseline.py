#!/usr/bin/env python3
"""Qubit QAOA / HEA baselines on the stored 40-Hamiltonian Gibbs suite.

Loads ``results/hamiltonians.json`` (does not regenerate instances). For each
Hamiltonian, builds the 7-qubit diagonal E[z] from the hybrid energy tensor
using the existing MSB-first embedding, then runs noiseless statevector
circuits with the same SPSA gains and sampled_tail Gibbs protocol as the
ECD mixed suite.

Primary fair match: QAOA p=20 (40 real parameters), |+⟩^{⊗ 7}. Also runs
QAOA p=20 with ⟨H⟩, a 42-parameter HEA with Gibbs, and optional QAOA p=22
(44 params; extra sensitivity check vs ECD's 45 coordinates).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from qumode_vqe.hamiltonian import (
    BKPInstance,
    knapsack_packing_stats,
    qnm_from_bits,
)
from qumode_vqe.qaoa import (
    DEFAULT_HEA_LAYERS,
    DEFAULT_QAOA_P,
    N_QUBITS,
    energy_vector_from_tensor,
    n_hea_params,
    n_qaoa_params,
    optimize_qubit_ansatz,
    random_hea_params,
    random_qaoa_params,
    verify_energy_vector,
)

OUTDIR = Path("results")
HAMILTONIANS_JSON = "hamiltonians.json"
MIXED_JSON = "gibbs_mixed_40.json"
ENERGY_JSON = "energy_spsa_baseline.json"
SCHEDULE_JSON = "gibbs_schedule_abc.json"

SEED_BASE = 3000
WORKERS = 7
MAXITER = 70
N_TRIALS = 1

# Distinct from ECD ansatz seeds so each protocol has its own random start.
# Mixed-suite convention: seed_base=3000; knapsack hid offset 100*hid,
# Ising hid offset 10_000 + 100*hid (same as energy_spsa_baseline.py).
PROTOCOL_SEED = {
    "qaoa_gibbs_p20": 30_000,
    "qaoa_energy_p20": 31_000,
    "hea_gibbs": 32_000,
    "qaoa_gibbs_p22": 33_000,
}

SPSA_A = 0.2
SPSA_C = 0.15
SPSA_A_STAB = 10.0
SPSA_ALPHA = 0.602
SPSA_GAMMA = 0.101

OUTPUT_FILES = {
    "qaoa_gibbs_p20": "qaoa_gibbs_p20.json",
    "qaoa_energy_p20": "qaoa_energy_p20.json",
    "hea_gibbs": "hea_gibbs.json",
    "qaoa_gibbs_p22": "qaoa_gibbs_p22.json",
}


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


def _family_hid_seed(seed_base: int, family: str, hid: int) -> int:
    if str(family) == "ising":
        return int(seed_base) + 10_000 + 100 * int(hid)
    return int(seed_base) + 100 * int(hid)


def trial_seed(seed_base: int, protocol: str, family: str, hid: int, trial: int) -> int:
    return _family_hid_seed(seed_base, family, hid) + int(PROTOCOL_SEED[protocol]) + int(trial)


def load_hamiltonians(path: Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("hamiltonians", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"expected a non-empty Hamiltonian list in {path}")
    return raw


def prepare_instance(meta: dict, partition: tuple[int, int, int] = (1, 3, 3)) -> dict:
    tensor = np.asarray(meta["energy_tensor"], dtype=float)
    ground = meta.get("ground_bitstring")
    energies = energy_vector_from_tensor(
        tensor,
        partition,
        ground_bitstring=None if ground is None else str(ground),
    )
    verify_energy_vector(energies, tensor, partition, ground_bitstring=ground)
    out = {
        "family": str(meta.get("family", meta.get("kind", "custom"))),
        "kind": str(meta.get("kind", meta.get("family", "custom"))),
        "hamiltonian_id": int(meta["hamiltonian_id"]),
        "energy_tensor": tensor,
        "energies": energies,
        "ground_bitstring": None if ground is None else str(ground),
        "instance": meta.get("instance"),
        "partition": partition,
    }
    return out


def _summarize_group(recs: list[dict]) -> dict:
    n = len(recs)
    n_success = int(sum(bool(r.get("success")) for r in recs))
    n_feas = int(sum(bool(r.get("packing", {}).get("feasible")) for r in recs))
    etas = [float(r["eta"]) for r in recs if r.get("eta") is not None]
    return {
        "n": n,
        "n_success": n_success,
        "success_rate": n_success / max(n, 1),
        "n_feasible": n_feas,
        "mean_rel_gap": float(np.mean([r["rel_gap"] for r in recs])) if n else float("nan"),
        "mean_gap_to_min": float(np.mean([r["gap_to_min"] for r in recs])) if n else float("nan"),
        "mean_energy_diag": float(np.mean([r["energy_diag"] for r in recs])) if n else float("nan"),
        "mean_p_ground": float(np.mean([r["p_ground"] for r in recs])) if n else float("nan"),
        "mean_eta": float(np.mean(etas)) if etas else None,
    }


def _print_job_line(i: int, n: int, rec: dict) -> None:
    feas = rec.get("packing", {}).get("feasible")
    feas_s = "" if feas is None else f"  feas={feas}"
    eta = rec.get("eta")
    eta_s = "" if eta is None else f"  η={float(eta):.3f}"
    print(
        f"[{i}/{n}] {rec.get('protocol')}  {rec.get('family')}/H{rec.get('hamiltonian_id')}  "
        f"cost={rec['cost']:.4f}  ⟨H⟩={rec['energy_physical']:.3f}  "
        f"E={rec['energy_diag']:.3f} (min {rec['energy_min']:.3f}, rel={rec['rel_gap']:.3f})  "
        f"P(gs)={rec['p_ground']:.3f}  gs={rec['success']}{feas_s}{eta_s}  "
        f"bits={rec['most_likely_bitstring']}",
        flush=True,
    )


def run_one(job: dict) -> dict:
    protocol = str(job["protocol"])
    ansatz = str(job["ansatz"])
    objective = str(job["objective"])
    n_qubits = int(job["n_qubits"])
    p = int(job["p"])
    n_layers = int(job["n_layers"])
    energies = np.asarray(job["energies"], dtype=float)
    rng = np.random.default_rng(int(job["seed"]))
    if ansatz == "qaoa":
        x0 = random_qaoa_params(p, rng)
    elif ansatz == "hea":
        x0 = random_hea_params(n_qubits, n_layers, rng)
    else:
        raise ValueError(f"unknown ansatz {ansatz!r}")
    result = optimize_qubit_ansatz(
        x0,
        energies,
        ansatz=ansatz,
        objective=objective,
        p=p,
        n_qubits=n_qubits,
        n_layers=n_layers,
        maxiter=int(job["maxiter"]),
        rng=rng,
        a=float(job["spsa_a"]),
        c=float(job["spsa_c"]),
        A=float(job["spsa_A"]),
        alpha=float(job["spsa_alpha"]),
        gamma=float(job["spsa_gamma"]),
    )
    rec = {
        "trial": int(job["trial"]),
        "protocol": protocol,
        "method": protocol,
        "ansatz": ansatz,
        "objective": objective,
        "family": str(job["family"]),
        "kind": str(job["kind"]),
        "hamiltonian_id": int(job["hamiltonian_id"]),
        "n_qubits": n_qubits,
        "p": p if ansatz == "qaoa" else None,
        "n_layers": n_layers if ansatz == "hea" else None,
        "n_params": int(job["n_params"]),
        "initial_state": str(job["initial_state"]),
        "seed": int(job["seed"]),
        "x0": np.asarray(result.x0, dtype=float),
        "x": np.asarray(result.x, dtype=float),
        "eta_policy": result.eta_policy,
        "n_eta_clamps": int(result.n_eta_clamps),
        "n_eta_fallbacks": int(result.n_eta_fallbacks),
    }
    rec.update(result.as_score_dict())
    stored_gs = job.get("ground_bitstring")
    if stored_gs:
        rec["ground_bitstring"] = str(stored_gs)
    bits = np.array([int(ch) for ch in rec["most_likely_bitstring"]], dtype=int)
    rec["most_likely"] = [int(v) for v in qnm_from_bits(bits)]
    gs_bits = np.array([int(ch) for ch in rec["ground_bitstring"]], dtype=int)
    rec["ground_qnm"] = [int(v) for v in qnm_from_bits(gs_bits)]
    inst_d = job.get("instance")
    if inst_d is not None:
        inst = BKPInstance(
            inst_d["values"], inst_d["weights"], inst_d["capacity"], inst_d["penalty"]
        )
        rec["packing"] = knapsack_packing_stats(bits, inst)
    return rec


def _run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    records: list[dict] = []
    if workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs, 1):
            rec = run_one(job)
            records.append(rec)
            _print_job_line(i, len(jobs), rec)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(run_one, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                done += 1
                _print_job_line(done, len(jobs), rec)
    records.sort(
        key=lambda r: (str(r.get("family", "")), int(r.get("hamiltonian_id", 0)), int(r["trial"]))
    )
    return records


def _protocol_spec(name: str) -> dict:
    if name == "qaoa_gibbs_p20":
        return dict(
            protocol=name,
            ansatz="qaoa",
            objective="gibbs",
            p=DEFAULT_QAOA_P,
            n_layers=0,
            n_params=n_qaoa_params(DEFAULT_QAOA_P),
            initial_state="plus",
            extra=False,
        )
    if name == "qaoa_energy_p20":
        return dict(
            protocol=name,
            ansatz="qaoa",
            objective="energy",
            p=DEFAULT_QAOA_P,
            n_layers=0,
            n_params=n_qaoa_params(DEFAULT_QAOA_P),
            initial_state="plus",
            extra=False,
        )
    if name == "hea_gibbs":
        return dict(
            protocol=name,
            ansatz="hea",
            objective="gibbs",
            p=0,
            n_layers=DEFAULT_HEA_LAYERS,
            n_params=n_hea_params(N_QUBITS, DEFAULT_HEA_LAYERS),
            initial_state="zero",
            extra=False,
        )
    if name == "qaoa_gibbs_p22":
        return dict(
            protocol=name,
            ansatz="qaoa",
            objective="gibbs",
            p=22,
            n_layers=0,
            n_params=n_qaoa_params(22),
            initial_state="plus",
            extra=True,
        )
    raise ValueError(f"unknown protocol {name!r}")


def build_jobs(
    instances: list[dict],
    protocol: str,
    *,
    seed_base: int,
    maxiter: int,
    n_trials: int,
    n_qubits: int,
    ids: set[tuple[str, int]] | None = None,
) -> list[dict]:
    spec = _protocol_spec(protocol)
    jobs: list[dict] = []
    for inst in instances:
        key = (inst["family"], inst["hamiltonian_id"])
        if ids is not None and key not in ids:
            continue
        for trial in range(int(n_trials)):
            seed = trial_seed(seed_base, protocol, inst["family"], inst["hamiltonian_id"], trial)
            jobs.append(
                {
                    **spec,
                    "trial": trial,
                    "family": inst["family"],
                    "kind": inst["kind"],
                    "hamiltonian_id": inst["hamiltonian_id"],
                    "energies": inst["energies"],
                    "ground_bitstring": inst["ground_bitstring"],
                    "instance": inst["instance"],
                    "n_qubits": n_qubits,
                    "maxiter": maxiter,
                    "seed": seed,
                    "spsa_a": SPSA_A,
                    "spsa_c": SPSA_C,
                    "spsa_A": SPSA_A_STAB,
                    "spsa_alpha": SPSA_ALPHA,
                    "spsa_gamma": SPSA_GAMMA,
                }
            )
    return jobs


def run_protocol(
    instances: list[dict],
    protocol: str,
    *,
    seed_base: int,
    maxiter: int,
    n_trials: int,
    n_qubits: int,
    workers: int,
    outdir: Path,
    ids: set[tuple[str, int]] | None = None,
) -> dict:
    spec = _protocol_spec(protocol)
    jobs = build_jobs(
        instances,
        protocol,
        seed_base=seed_base,
        maxiter=maxiter,
        n_trials=n_trials,
        n_qubits=n_qubits,
        ids=ids,
    )
    label = "extra" if spec["extra"] else "primary"
    print(
        f"=== {protocol} ({label}): ansatz={spec['ansatz']}  obj={spec['objective']}  "
        f"params={spec['n_params']}  init={spec['initial_state']}  "
        f"steps={maxiter}  jobs={len(jobs)}  workers={workers} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    records = _run_jobs(jobs, workers)
    elapsed = time.perf_counter() - t0
    by_family: dict[str, list[dict]] = {}
    for rec in records:
        by_family.setdefault(str(rec.get("family", "custom")), []).append(rec)
    family_summary = {k: _summarize_group(v) for k, v in by_family.items()}
    n_success = int(sum(bool(r.get("success")) for r in records))
    etas = [float(r["eta"]) for r in records if r.get("eta") is not None]
    payload = {
        "method": protocol,
        "protocol": protocol,
        "ansatz": spec["ansatz"],
        "objective": spec["objective"],
        "extra": bool(spec["extra"]),
        "initial_state": spec["initial_state"],
        "n_qubits": n_qubits,
        "p": spec["p"] if spec["ansatz"] == "qaoa" else None,
        "n_layers": spec["n_layers"] if spec["ansatz"] == "hea" else None,
        "n_params": spec["n_params"],
        "n_hamiltonians": len({(r["family"], r["hamiltonian_id"]) for r in records}),
        "n_trials_per_hamiltonian": n_trials,
        "maxiter": maxiter,
        "seed_base": seed_base,
        "seed_protocol_offset": PROTOCOL_SEED[protocol],
        "workers": workers,
        "elapsed_sec": elapsed,
        "eta_policy": "sampled_tail" if spec["objective"] == "gibbs" else None,
        "n_success": n_success,
        "success_rate": n_success / max(len(records), 1),
        "mean_rel_gap": float(np.mean([r["rel_gap"] for r in records])) if records else float("nan"),
        "mean_p_ground": float(np.mean([r["p_ground"] for r in records])) if records else float("nan"),
        "mean_eta": float(np.mean(etas)) if etas else None,
        "by_family": family_summary,
        "hamiltonian_file": "results/hamiltonians.json",
        "trials": records,
        "spsa": {
            "a": SPSA_A,
            "c": SPSA_C,
            "A": SPSA_A_STAB,
            "alpha": SPSA_ALPHA,
            "gamma": SPSA_GAMMA,
        },
        "param_init": (
            "gamma~U(0,2pi), beta~U(0,pi)"
            if spec["ansatz"] == "qaoa"
            else "Ry angles ~U(0,2pi)"
        ),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / OUTPUT_FILES[protocol]
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    print(
        f"  {protocol} exact-GS hits {n_success}/{len(records)}  "
        f"mean rel-gap {payload['mean_rel_gap']:.3f}  "
        f"mean P(gs) {payload['mean_p_ground']:.3f}  "
        f"elapsed {elapsed:.1f}s",
        flush=True,
    )
    for k, s in family_summary.items():
        extra = ""
        if k == "knapsack":
            extra = f"  feas {s['n_feasible']}/{s['n']}"
        eta_s = "" if s["mean_eta"] is None else f"  mean η {s['mean_eta']:.3f}"
        print(
            f"  {k}: {s['n_success']}/{s['n']}{extra}  "
            f"mean rel-gap {s['mean_rel_gap']:.3f}{eta_s}",
            flush=True,
        )
    return payload


def _hits_from_payload(payload: dict) -> dict:
    by = payload.get("by_family") or payload.get("by_schedule") or {}
    out = {
        "n_success": int(payload.get("n_success", 0)),
        "n": int(payload.get("n_hamiltonians", payload.get("n", 0)) or 0),
        "success_rate": float(payload.get("success_rate", float("nan"))),
        "by_family": {},
    }
    if out["n"] == 0:
        trials = payload.get("trials") or []
        out["n"] = len(trials)
    for fam, slot in by.items():
        if not isinstance(slot, dict):
            continue
        if "n_success" not in slot:
            continue
        out["by_family"][str(fam)] = {
            "n_success": int(slot["n_success"]),
            "n": int(slot.get("n", 0)),
            "success_rate": float(slot.get("success_rate", float("nan"))),
        }
    return out


def _load_ecd_reference(outdir: Path) -> dict:
    refs: dict[str, dict] = {}
    mixed = outdir / MIXED_JSON
    if mixed.exists():
        payload = json.loads(mixed.read_text(encoding="utf-8"))
        refs["ecd_gibbs_freeze_20_50"] = {
            "source": str(mixed),
            "label": "ECD Gibbs freeze 20+50",
            **_hits_from_payload(payload),
        }
    energy = outdir / ENERGY_JSON
    if energy.exists():
        payload = json.loads(energy.read_text(encoding="utf-8"))
        refs["ecd_energy_spsa_vacuum_70"] = {
            "source": str(energy),
            "label": "ECD energy SPSA vacuum 70",
            **_hits_from_payload(payload),
        }
    schedule = outdir / SCHEDULE_JSON
    if schedule.exists():
        payload = json.loads(schedule.read_text(encoding="utf-8"))
        joint = (payload.get("by_schedule") or {}).get("joint70")
        if joint:
            refs["ecd_gibbs_joint_70"] = {
                "source": str(schedule),
                "label": "ECD Gibbs joint-70",
                "n_success": int(joint["n_success"]),
                "n": int(joint.get("n", 40)),
                "success_rate": float(joint.get("success_rate", float("nan"))),
                "by_family": {
                    fam: {
                        "n_success": int(slot["n_success"]),
                        "n": int(slot.get("n", 0)),
                        "success_rate": float(slot.get("success_rate", float("nan"))),
                    }
                    for fam, slot in (joint.get("by_family") or {}).items()
                },
            }
    return refs


def comparison_rows(ecd: dict, qubit: dict[str, dict]) -> list[dict]:
    rows = []
    order = [
        ("ecd_gibbs_freeze_20_50", None),
        ("ecd_gibbs_joint_70", None),
        ("ecd_energy_spsa_vacuum_70", None),
        ("qaoa_gibbs_p20", "qaoa_gibbs_p20"),
        ("qaoa_energy_p20", "qaoa_energy_p20"),
        ("hea_gibbs", "hea_gibbs"),
        ("qaoa_gibbs_p22", "qaoa_gibbs_p22"),
    ]
    labels = {
        "ecd_gibbs_freeze_20_50": "ECD Gibbs freeze 20+50",
        "ecd_gibbs_joint_70": "ECD Gibbs joint-70",
        "ecd_energy_spsa_vacuum_70": "ECD energy SPSA vacuum 70",
        "qaoa_gibbs_p20": "QAOA p=20 Gibbs 70",
        "qaoa_energy_p20": "QAOA p=20 energy 70",
        "hea_gibbs": "HEA L=5 Gibbs 70 (42 params)",
        "qaoa_gibbs_p22": "QAOA p=22 Gibbs 70 (extra)",
    }
    for key, qname in order:
        if qname is not None:
            payload = qubit.get(qname)
            if payload is None:
                continue
            slot = _hits_from_payload(payload)
            label = labels[key]
        else:
            slot = ecd.get(key)
            if slot is None:
                continue
            label = slot.get("label", labels[key])
        knap = slot.get("by_family", {}).get("knapsack", {})
        ising = slot.get("by_family", {}).get("ising", {})
        rows.append(
            {
                "name": label,
                "all": f"{slot['n_success']}/{slot['n']}",
                "knapsack": f"{knap.get('n_success', '?')}/{knap.get('n', '?')}",
                "ising": f"{ising.get('n_success', '?')}/{ising.get('n', '?')}",
            }
        )
    return rows


def format_table(rows: list[dict]) -> str:
    headers = ["Run", "All", "Knapsack", "Ising"]
    data = [[r["name"], r["all"], r["knapsack"], r["ising"]] for r in rows]
    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    def fmt(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    lines = [fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(fmt(r) for r in data)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamiltonians", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--limit", type=int, default=0, help="If >0, only the first N instances (smoke).")
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["qaoa_gibbs_p20", "qaoa_energy_p20", "hea_gibbs", "qaoa_gibbs_p22"],
        choices=list(OUTPUT_FILES),
    )
    parser.add_argument(
        "--skip-p22",
        action="store_true",
        help="Drop the optional QAOA p=22 extra run.",
    )
    args = parser.parse_args(argv)

    ham_path = args.hamiltonians or (args.outdir / HAMILTONIANS_JSON)
    metas = load_hamiltonians(ham_path)
    print(f"Loaded {len(metas)} Hamiltonians from {ham_path}", flush=True)
    instances = [prepare_instance(m) for m in metas]
    print(
        f"Encoding check passed for {len(instances)} instances "
        f"(E[ground_bitstring] == energy_min, tensor ↔ 7-bit diagonal).",
        flush=True,
    )
    if int(args.limit) > 0:
        instances = instances[: int(args.limit)]
        print(f"Smoke limit: {len(instances)} instance(s).", flush=True)

    protocols = list(args.protocols)
    if args.skip_p22:
        protocols = [p for p in protocols if p != "qaoa_gibbs_p22"]

    payloads: dict[str, dict] = {}
    for name in protocols:
        payloads[name] = run_protocol(
            instances,
            name,
            seed_base=args.seed_base,
            maxiter=args.maxiter,
            n_trials=args.n_trials,
            n_qubits=N_QUBITS,
            workers=args.workers,
            outdir=args.outdir,
        )

    ecd = _load_ecd_reference(args.outdir)
    rows = comparison_rows(ecd, payloads)
    table = format_table(rows)
    print("\nComparison (success = most-likely bitstring is an exact ground, atol 1e-8)\n", flush=True)
    print(table, flush=True)
    if "ecd_gibbs_joint_70" in ecd:
        print(
            "\nECD joint-70 numbers are from results/gibbs_schedule_abc.json "
            "(not gibbs_joint_step_sweep.json, which has budgets 40/50/80/100/150/200).",
            flush=True,
        )
    elif "ecd_gibbs_freeze_20_50" in ecd:
        print(
            "\nNo joint-70 block found; paired against stored freeze 20+50 in "
            "results/gibbs_mixed_40.json.",
            flush=True,
        )
    summary_path = args.outdir / "qaoa_baseline_summary.json"
    summary_path.write_text(
        json.dumps(
            _json_ready({"ecd": ecd, "qubit": {k: _hits_from_payload(v) for k, v in payloads.items()}, "table": rows}),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
