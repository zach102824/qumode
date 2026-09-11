#!/usr/bin/env python3
"""Default noiseless + noisy-in-loop protocol for n=7 4-SAT.

Canonical recipe (see docs/DEFAULT_PROTOCOL.md):

* ECD L4 and SNAP L3
* four_sat_000 … four_sat_019, 10 trials, seed_base=4000
* vacuum start, sampled_tail Gibbs, 200 joint SPSA
* noiseless: success = most_likely_bitstring == ground
* noisy: comprehensive + κ_φ τ = 0.5 κτ + readout_realistic
* GDR inside every SPSA step (gdr_param only); M policy B (fit once)

Usage
-----
    export PYTHONPATH=src
    python -u scripts/run_default_protocol.py smoke
    python -u scripts/run_default_protocol.py noiseless --ansatz ecd
    python -u scripts/run_default_protocol.py noiseless --ansatz snap
    python -u scripts/run_default_protocol.py noisy --ansatz ecd --kappa-tau 0.003
    python -u scripts/run_default_protocol.py plots
    python -u scripts/run_default_protocol.py conclude
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from qumode_vqe.circuit import vacuum_prep_params
from qumode_vqe.hamiltonian import DEFAULT_NFOCKS, ground_qnm_from_tensor, load_four_sat_instances
from qumode_vqe.params import ansatz_inventory, random_parameters, random_snap_parameters
from qumode_vqe.vqe import optimize_gibbs_adaptive

from Error_mitigation.gdr_in_loop import (
    PROTOCOL_FAMILY,
    PROTOCOL_KAPPA_TAU,
    PROTOCOL_N_TRAIN,
    PROTOCOL_READOUT,
    PROTOCOL_SEED_BASE,
    PROTOCOL_TWIN_SHOTS,
    fit_gdr_param_policy_b,
    make_gdr_forward,
)
from Error_mitigation.noise_models import circuit_noise

import importlib.util

_ECD_PATH = ROOT / "scripts" / "Gibbs_and_adaptive_optim_ECD.py"
_SPEC = importlib.util.spec_from_file_location("gibbs_ecd_protocol", _ECD_PATH)
ecd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ecd)

OUTDIR = ROOT / "results" / "protocol"
NFOCKS = DEFAULT_NFOCKS
DIMS = (2, int(NFOCKS[0]), int(NFOCKS[1]))
N_TRIALS = 10
N_STEPS = 200
ECD_DEPTH = 4
SNAP_DEPTH = 3
WORKERS = min(4, os.cpu_count() or 4)
HAM_DIR = ROOT / "Hamiltonians" / "four_sat"

SPSA_A = 0.2
SPSA_C = 0.15
SPSA_A_STAB = 10.0
SPSA_ALPHA = 0.602
SPSA_GAMMA = 0.101


def _json_ready(obj):
    return ecd._json_ready(obj)


def _ham_meta(inst: dict) -> dict:
    return {k: v for k, v in inst.items() if k != "energy_tensor"}


def _load_instances(hamiltonian_ids=None, max_hamiltonians=None):
    return ecd._load_family_instances(
        "four_sat",
        HAM_DIR,
        NFOCKS,
        max_hamiltonians=max_hamiltonians,
        hamiltonian_ids=hamiltonian_ids,
    )


def run_noiseless_trial(job: dict) -> dict:
    return ecd.run_trial(job)


def run_noisy_trial(job: dict) -> dict:
    t0 = time.perf_counter()
    trial = int(job["trial"])
    ndepth = int(job["ndepth"])
    ansatz = str(job["ansatz"]).lower()
    hid = int(job["hamiltonian_id"])
    nfocks = (int(job["nfocks"][0]), int(job["nfocks"][1]))
    rng = np.random.default_rng(int(job["seed_base"]) + trial)
    prep0 = np.asarray(job["prep0"], dtype=float)
    if ansatz == "snap":
        x0 = random_snap_parameters(ndepth, nfocks, rng)
    else:
        x0 = random_parameters(ndepth, rng)
    energy_tensor = np.asarray(job["energy_tensor"], dtype=float)
    kt = float(job["kappa_tau"])
    noise = circuit_noise(PROTOCOL_FAMILY, kt, dims=DIMS)
    theta = np.asarray(job["gdr_theta"], dtype=float)
    forward = make_gdr_forward(
        theta=theta,
        dims=DIMS,
        energy_tensor=energy_tensor,
        readout_level=PROTOCOL_READOUT,
        n_shots=None,
    )
    result = optimize_gibbs_adaptive(
        prep0,
        x0,
        ndepth=ndepth,
        nfocks=nfocks,
        outer_iter=int(job["outer_iter"]),
        spsa_iter=0,
        rng=rng,
        noise=noise,
        a=float(job["spsa_a"]),
        c=float(job["spsa_c"]),
        A=float(job["spsa_A"]),
        alpha=float(job["spsa_alpha"]),
        gamma=float(job["spsa_gamma"]),
        energy_tensor=energy_tensor,
        ansatz=ansatz,
        forward=forward,
        record_steps=True,
    )
    final = result.eval_final
    ml = [int(v) for v in final.get("most_likely", [0, 0, 0])]
    bits = str(final.get("most_likely_bitstring", ""))
    from qumode_vqe.hamiltonian import bitstring_from_bits, bits_from_qnm

    gs_qnm = list(ground_qnm_from_tensor(energy_tensor))
    gs_bits = bitstring_from_bits(bits_from_qnm(*gs_qnm))
    score = ecd._score_tensor(ml or gs_qnm, energy_tensor)
    score["success"] = bits == gs_bits
    inventory = ansatz_inventory(ansatz, ndepth, nfocks, n_prep_params=5)
    rec = {
        "trial": trial,
        "ansatz": ansatz,
        "ndepth": ndepth,
        **inventory,
        "hamiltonian_id": hid,
        "family": "four_sat",
        "kind": "four_sat",
        "file": str(job.get("file", "")),
        "kappa_tau": kt,
        "m_policy": "B",
        "mitigation": "gdr_param",
        "readout": PROTOCOL_READOUT,
        "noise_family": PROTOCOL_FAMILY,
        "prep": np.asarray(result.prep, dtype=float),
        "x": np.asarray(result.x, dtype=float),
        "x0": np.asarray(result.x0, dtype=float),
        "cost": float(result.fun),
        "energy_physical": float(final.get("energy_physical", result.fun)),
        "p_gs": None if final.get("p_gs") is None else float(final["p_gs"]),
        "p_gs_raw": None if final.get("p_gs_raw") is None else float(final["p_gs_raw"]),
        "p_gs_mit": None if final.get("p_gs_mit") is None else float(final["p_gs_mit"]),
        "most_likely": ml,
        "most_likely_bitstring": bits,
        "most_likely_bitstring_raw": final.get("most_likely_bitstring_raw"),
        "ground_bitstring": gs_bits,
        "success": bits == gs_bits,
        "success_raw": bool(final.get("success_raw")),
        "success_mit": bits == gs_bits,
        "eta": float(result.eta),
        "nfev": int(result.nfev),
        "nit_total": int(result.nit_warmup) + int(result.nit),
        "outer_iter": int(job["outer_iter"]),
        "elapsed_s": float(time.perf_counter() - t0),
        "step_trace": list(result.step_trace or []),
        "gdr_fit": job.get("gdr_fit"),
        **score,
    }
    return rec


def _print_line(i: int, n: int, rec: dict, kind: str) -> None:
    extra = ""
    if "success" in rec:
        extra = (
            f"  H{rec.get('hamiltonian_id', '?')}  η={rec.get('eta', float('nan')):.3f}  "
            f"E={rec.get('energy_physical', float('nan')):.3f}  "
            f"pGS={rec.get('p_gs', float('nan'))}  gs={rec.get('success')}"
        )
        if rec.get("success_raw") is not None:
            extra += f" raw={rec.get('success_raw')}"
    print(
        f"[{i}/{n}] {kind} trial {rec['trial']} {rec.get('ansatz', '?')} "
        f"L{rec.get('ndepth', '?')}{extra}  bits={rec.get('most_likely_bitstring')}",
        flush=True,
    )


def _run_pool(jobs: list[dict], workers: int, fn, kind: str) -> list[dict]:
    records: list[dict] = []
    if workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs, 1):
            rec = fn(job)
            records.append(rec)
            _print_line(i, len(jobs), rec, kind)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(fn, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                done += 1
                _print_line(done, len(jobs), rec, kind)
    records.sort(
        key=lambda r: (int(r.get("hamiltonian_id", 0)), int(r["trial"]), str(r.get("kappa_tau", "")))
    )
    return records


def _write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compact: 200 trials × 201-step traces are tens of MB with indent=2.
    path.write_text(json.dumps(_json_ready(payload), separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    return path


def run_noiseless(args: argparse.Namespace) -> dict:
    ansatz = str(args.ansatz)
    ndepth = int(args.ndepth if args.ndepth is not None else (ECD_DEPTH if ansatz == "ecd" else SNAP_DEPTH))
    instances = _load_instances(args.hamiltonian_ids, args.max_hamiltonians)
    n_trials = int(args.n_trials)
    outer_iter = int(args.outer_iter)
    workers = int(args.workers)
    seed_base = int(args.seed_base)
    jobs = []
    spsa_fields = ecd._spsa_job_fields(
        ndepth=ndepth,
        nfocks=NFOCKS,
        outer_iter=outer_iter,
        spsa_iter=0,
        spsa_a=SPSA_A,
        spsa_c=SPSA_C,
        spsa_A=SPSA_A_STAB,
        spsa_alpha=SPSA_ALPHA,
        spsa_gamma=SPSA_GAMMA,
    )
    for inst in instances:
        hid = int(inst["hamiltonian_id"])
        for t in range(n_trials):
            unit, prep0 = ecd.vacuum_start()
            jobs.append(
                {
                    **spsa_fields,
                    "trial": t,
                    "hamiltonian_id": hid,
                    "family": "four_sat",
                    "kind": "four_sat",
                    "file": inst["file"],
                    "ansatz": ansatz,
                    "ndepth": ndepth,
                    "seed_base": seed_base + 1000 * ndepth + 100 * hid,
                    "prep0": prep0,
                    "prep_unit": unit,
                    "energy_tensor": inst["energy_tensor"],
                }
            )
    print(
        f"=== noiseless four_sat {ansatz} L{ndepth}: {len(instances)} H × {n_trials} "
        f"trials, {outer_iter} SPSA, workers={workers} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    records = _run_pool(jobs, workers, run_noiseless_trial, "noiseless")
    elapsed = time.perf_counter() - t0
    n_success = int(sum(bool(r.get("success")) for r in records))
    inv = ansatz_inventory(ansatz, ndepth, NFOCKS, n_prep_params=5)
    payload = {
        "protocol": "default_noiseless",
        "ansatz": ansatz,
        "family": "four_sat",
        "ndepth": ndepth,
        "inventory": inv,
        "n_hamiltonians": len(instances),
        "n_trials_per_hamiltonian": n_trials,
        "n_trials_total": len(records),
        "outer_iterations": outer_iter,
        "seed_base": seed_base,
        "eta_policy": "sampled_tail",
        "initial_state": "vacuum",
        "success_metric": "most_likely_bitstring == ground_bitstring",
        "n_success": n_success,
        "success_rate": n_success / max(len(records), 1),
        "k_over_n": f"{n_success}/{len(records)}",
        "elapsed_sec": elapsed,
        "by_hamiltonian": ecd._summarize_by_hamiltonian(records),
        "summary": ecd._summarize_group(records),
        "hamiltonians": [_ham_meta(inst) for inst in instances],
        "trials": records,
        "spsa": {"a": SPSA_A, "c": SPSA_C, "A": SPSA_A_STAB, "alpha": SPSA_ALPHA, "gamma": SPSA_GAMMA},
    }
    out = Path(args.outdir) / f"gibbs_four_sat_{ansatz}_l{ndepth}_noiseless.json"
    if args.output:
        out = Path(args.output)
    _write_json(out, payload)
    print(
        f"  noiseless {ansatz} L{ndepth}: {n_success}/{len(records)}  "
        f"({payload['success_rate']:.3f})  wall {elapsed:.1f}s",
        flush=True,
    )
    return payload


def run_noisy(args: argparse.Namespace) -> dict:
    ansatz = str(args.ansatz)
    ndepth = int(args.ndepth if args.ndepth is not None else (ECD_DEPTH if ansatz == "ecd" else SNAP_DEPTH))
    instances = _load_instances(args.hamiltonian_ids, args.max_hamiltonians)
    n_trials = int(args.n_trials)
    outer_iter = int(args.outer_iter)
    workers = int(args.workers)
    seed_base = int(args.seed_base)
    kappas = tuple(float(x) for x in args.kappa_tau)
    n_train = int(args.n_train)
    cache_dir = Path(args.outdir) / "gdr_M"
    cache_dir.mkdir(parents=True, exist_ok=True)

    m_fits: dict[tuple[int, float], dict] = {}
    print(
        f"=== fit gdr_param M (policy B) for {ansatz} L{ndepth} × {len(instances)} H × "
        f"{len(kappas)} κτ, n_train={n_train} ===",
        flush=True,
    )
    for inst in instances:
        hid = int(inst["hamiltonian_id"])
        for kt in kappas:
            print(f"  fitting M  {ansatz} L{ndepth} H{hid:03d} κτ={kt} ...", flush=True)
            t0 = time.perf_counter()
            fit = fit_gdr_param_policy_b(
                ansatz=ansatz,
                ndepth=ndepth,
                hid=hid,
                kappa_tau=kt,
                energy_tensor=inst["energy_tensor"],
                nfocks=NFOCKS,
                n_train=n_train,
                n_shots=int(args.twin_shots),
                seed=seed_base,
                cache_dir=cache_dir,
                fit_maxiter=int(args.fit_maxiter),
            )
            # Drop kernels (ndarray) before storing in JSON-bound job dict; keep theta.
            slim = {k: v for k, v in fit.items() if k != "kernels"}
            m_fits[(hid, kt)] = slim
            print(
                f"    done in {time.perf_counter() - t0:.1f}s  cache={fit.get('from_cache')}  "
                f"η1={fit.get('fit', {}).get('fitted', {}).get('eta1')}",
                flush=True,
            )

    jobs = []
    spsa_fields = ecd._spsa_job_fields(
        ndepth=ndepth,
        nfocks=NFOCKS,
        outer_iter=outer_iter,
        spsa_iter=0,
        spsa_a=SPSA_A,
        spsa_c=SPSA_C,
        spsa_A=SPSA_A_STAB,
        spsa_alpha=SPSA_ALPHA,
        spsa_gamma=SPSA_GAMMA,
    )
    for inst in instances:
        hid = int(inst["hamiltonian_id"])
        for kt in kappas:
            fit = m_fits[(hid, kt)]
            for t in range(n_trials):
                jobs.append(
                    {
                        **spsa_fields,
                        "trial": t,
                        "hamiltonian_id": hid,
                        "family": "four_sat",
                        "file": inst["file"],
                        "ansatz": ansatz,
                        "ndepth": ndepth,
                        "kappa_tau": kt,
                        "seed_base": seed_base + 1000 * ndepth + 100 * hid + int(round(10000 * kt)),
                        "prep0": vacuum_prep_params(),
                        "energy_tensor": inst["energy_tensor"],
                        "gdr_theta": fit["theta"],
                        "gdr_fit": fit.get("fit"),
                    }
                )
    print(
        f"=== noisy-in-loop four_sat {ansatz} L{ndepth}: {len(instances)} H × {n_trials} "
        f"trials × {len(kappas)} κτ, {outer_iter} SPSA, workers={workers} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    records = _run_pool(jobs, workers, run_noisy_trial, "noisy")
    elapsed = time.perf_counter() - t0

    by_kt: dict[str, list[dict]] = {}
    for rec in records:
        by_kt.setdefault(str(rec["kappa_tau"]), []).append(rec)
    kt_summary = {}
    for kt, recs in by_kt.items():
        n_mit = int(sum(bool(r.get("success_mit") or r.get("success")) for r in recs))
        n_raw = int(sum(bool(r.get("success_raw")) for r in recs))
        kt_summary[kt] = {
            **ecd._summarize_group(recs),
            "n_success_mit": n_mit,
            "n_success_raw": n_raw,
            "k_over_n_mit": f"{n_mit}/{len(recs)}",
            "k_over_n_raw": f"{n_raw}/{len(recs)}",
        }

    n_success = int(sum(bool(r.get("success")) for r in records))
    payload = {
        "protocol": "default_noisy_in_loop",
        "ansatz": ansatz,
        "family": "four_sat",
        "ndepth": ndepth,
        "inventory": ansatz_inventory(ansatz, ndepth, NFOCKS, n_prep_params=5),
        "noise_family": PROTOCOL_FAMILY,
        "readout": PROTOCOL_READOUT,
        "mitigation": "gdr_param",
        "m_policy": "B",
        "kappa_tau": list(kappas),
        "n_train": n_train,
        "twin_shots": int(args.twin_shots),
        "in_loop_shots": None,
        "n_hamiltonians": len(instances),
        "n_trials_per_hamiltonian": n_trials,
        "n_trials_total": len(records),
        "outer_iterations": outer_iter,
        "seed_base": seed_base,
        "success_metric": "most_likely_bitstring == ground_bitstring",
        "n_success": n_success,
        "success_rate": n_success / max(len(records), 1),
        "k_over_n": f"{n_success}/{len(records)}",
        "elapsed_sec": elapsed,
        "by_kappa_tau": kt_summary,
        "by_hamiltonian": ecd._summarize_by_hamiltonian(records),
        "m_fits": {f"h{hid:03d}_kt{kt}": fit for (hid, kt), fit in m_fits.items()},
        "hamiltonians": [_ham_meta(inst) for inst in instances],
        "trials": records,
        "spsa": {"a": SPSA_A, "c": SPSA_C, "A": SPSA_A_STAB, "alpha": SPSA_ALPHA, "gamma": SPSA_GAMMA},
    }
    ktag = "-".join(str(k) for k in kappas)
    out = Path(args.outdir) / f"gibbs_four_sat_{ansatz}_l{ndepth}_noisy_kt{ktag}.json"
    if args.output:
        out = Path(args.output)
    _write_json(out, payload)
    print(
        f"  noisy-in-loop {ansatz} L{ndepth}: {n_success}/{len(records)}  wall {elapsed:.1f}s",
        flush=True,
    )
    for kt, s in kt_summary.items():
        print(
            f"    κτ={kt}: mit {s['k_over_n_mit']}  raw {s['k_over_n_raw']}  "
            f"⟨H⟩={s['mean_energy_physical']:.3f}",
            flush=True,
        )
    return payload


def _iter_protocol_json(outdir: Path):
    outdir = Path(outdir)
    if not outdir.is_dir():
        return
    for path in sorted(outdir.glob("*.json")):
        if path.name.startswith("gdr_param_"):
            continue
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue


def make_plots(args: argparse.Namespace) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path, payload in _iter_protocol_json(outdir):
        trials = payload.get("trials") or []
        traces = [t for t in trials if t.get("step_trace")]
        if not traces:
            continue
        # A few example trials: first success and first failure if present.
        picks = []
        succ = [t for t in traces if t.get("success")]
        fail = [t for t in traces if not t.get("success")]
        picks.extend(succ[:2])
        picks.extend(fail[:1])
        if not picks:
            picks = traces[:2]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
        for rec in picks:
            tr = rec["step_trace"]
            steps = [r["step"] for r in tr]
            ephys = [r.get("energy_physical") for r in tr]
            pgs = [r.get("p_gs") for r in tr]
            label = f"H{int(rec['hamiltonian_id']):03d} t{rec['trial']}"
            axes[0].plot(steps, ephys, lw=1.2, label=label)
            axes[1].plot(steps, pgs, lw=1.2, label=label)
        axes[0].set_xlabel("SPSA step")
        axes[0].set_ylabel(r"$\langle H\rangle$")
        axes[1].set_xlabel("SPSA step")
        axes[1].set_ylabel(r"$p(\mathrm{GS})$")
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=8, frameon=False)
        title = path.stem.replace("_", " ")
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        out = figdir / f"{path.stem}_step_traces.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        written.append(out)
        print(f"Wrote {out}", flush=True)
    return written


def write_conclusion(args: argparse.Namespace) -> Path:
    outdir = Path(args.outdir)
    lines = [
        "# Default protocol conclusion (n=7 4-SAT)",
        "",
        "Canonical noiseless + noisy-in-loop recipe for **ECD L4** and **SNAP L3**.",
        "Success = `most_likely_bitstring == ground_bitstring`. Pooled over 20 Hamiltonians × 10 trials = 200.",
        "",
        "Noise: `comprehensive` with cavity number-dephasing `κ_φ τ_app = 0.5 κτ`, plus `readout_realistic`.",
        "Mitigation: **gdr_param only**, M policy B (fit once per (H, κτ, ansatz), reuse all 200 SPSA steps).",
        "",
        "## Noiseless",
        "",
        "| ansatz | depth | k/200 | rate | mean ⟨H⟩ | mean p(GS) | wall (s) | file |",
        "|--------|------:|------:|-----:|---------:|-----------:|---------:|------|",
    ]
    noisy_rows = []
    for path, payload in _iter_protocol_json(outdir):
        proto = str(payload.get("protocol") or "")
        if proto == "default_noiseless":
            s = payload.get("summary") or {}
            lines.append(
                f"| {payload.get('ansatz')} | L{payload.get('ndepth')} | "
                f"{payload.get('k_over_n')} | {float(payload.get('success_rate') or 0):.3f} | "
                f"{float(s.get('mean_energy_physical') or float('nan')):.3f} | "
                f"{float(s.get('mean_p_gs') or payload.get('mean_p_gs') or float('nan')):.3f} | "
                f"{float(payload.get('elapsed_sec') or 0):.1f} | `{path.name}` |"
            )
        elif proto == "default_noisy_in_loop":
            noisy_rows.append((path, payload))
    lines += ["", "## Noisy-in-loop (gdr_param, policy B)", ""]
    if not noisy_rows:
        lines.append("_No noisy-in-loop JSONs written yet._")
    else:
        lines += [
            "| ansatz | depth | κτ | mit k/N | raw k/N | mean ⟨H⟩ | file |",
            "|--------|------:|---:|--------:|--------:|---------:|------|",
        ]
        for path, payload in noisy_rows:
            by_kt = payload.get("by_kappa_tau") or {}
            if by_kt:
                for kt, s in by_kt.items():
                    lines.append(
                        f"| {payload.get('ansatz')} | L{payload.get('ndepth')} | {kt} | "
                        f"{s.get('k_over_n_mit')} | {s.get('k_over_n_raw')} | "
                        f"{float(s.get('mean_energy_physical') or float('nan')):.3f} | `{path.name}` |"
                    )
            else:
                lines.append(
                    f"| {payload.get('ansatz')} | L{payload.get('ndepth')} | "
                    f"{payload.get('kappa_tau')} | {payload.get('k_over_n')} | — | "
                    f"— | `{path.name}` |"
                )
    lines += [
        "",
        "## Commands",
        "",
        "```bash",
        "export PYTHONPATH=src",
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        "python -u scripts/run_default_protocol.py noiseless --ansatz ecd --workers 4",
        "python -u scripts/run_default_protocol.py noiseless --ansatz snap --workers 4",
        "python -u scripts/run_default_protocol.py noisy --ansatz ecd --kappa-tau 0.003 --workers 4",
        "python -u scripts/run_default_protocol.py noisy --ansatz snap --kappa-tau 0.003 --workers 4",
        "python -u scripts/run_default_protocol.py noisy --ansatz ecd --kappa-tau 0.03 --workers 4",
        "python -u scripts/run_default_protocol.py noisy --ansatz snap --kappa-tau 0.03 --workers 4",
        "python -u scripts/run_default_protocol.py noisy --ansatz ecd --kappa-tau 0.1 --workers 4",
        "python -u scripts/run_default_protocol.py noisy --ansatz snap --kappa-tau 0.1 --workers 4",
        "python -u scripts/run_default_protocol.py plots",
        "python -u scripts/run_default_protocol.py conclude",
        "```",
        "",
        "In-loop cost uses the unfolded histogram after readout confusion (no extra shot noise).",
        "Twin fits for M use 8192 shots. See `docs/DEFAULT_PROTOCOL.md`.",
        "",
    ]
    out = outdir / "CONCLUSION.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return out


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ansatz", choices=("ecd", "snap"), default="ecd")
    p.add_argument("--ndepth", type=int, default=None)
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--outer-iter", type=int, default=N_STEPS)
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--seed-base", type=int, default=PROTOCOL_SEED_BASE)
    p.add_argument("--outdir", type=Path, default=OUTDIR)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max-hamiltonians", type=int, default=None)
    p.add_argument("--hamiltonian-ids", type=int, nargs="+", default=None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sm = sub.add_parser("smoke", help="Tiny end-to-end check (1 H, 1 trial, few steps).")
    _common_args(sm)
    sm.set_defaults(n_trials=1, outer_iter=2, max_hamiltonians=1, n_train=4, twin_shots=256, fit_maxiter=20)
    sm.add_argument("--kappa-tau", type=float, nargs="+", default=[0.003])
    sm.add_argument("--n-train", type=int, default=4)
    sm.add_argument("--twin-shots", type=int, default=256)
    sm.add_argument("--fit-maxiter", type=int, default=20)
    sm.add_argument("--skip-noisy", action="store_true")

    nl = sub.add_parser("noiseless", help="Canonical noiseless 20×10 run.")
    _common_args(nl)

    ny = sub.add_parser("noisy", help="GDR-in-loop noisy SPSA (policy B).")
    _common_args(ny)
    ny.add_argument("--kappa-tau", type=float, nargs="+", default=list(PROTOCOL_KAPPA_TAU))
    ny.add_argument("--n-train", type=int, default=PROTOCOL_N_TRAIN)
    ny.add_argument("--twin-shots", type=int, default=PROTOCOL_TWIN_SHOTS)
    ny.add_argument("--fit-maxiter", type=int, default=200)

    pl = sub.add_parser("plots", help="⟨H⟩ and p(GS) vs step for example trials.")
    pl.add_argument("--outdir", type=Path, default=OUTDIR)

    cc = sub.add_parser("conclude", help="Write results/protocol/CONCLUSION.md from JSONs on disk.")
    cc.add_argument("--outdir", type=Path, default=OUTDIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    if args.cmd == "noiseless":
        run_noiseless(args)
        write_conclusion(args)
        return 0
    if args.cmd == "noisy":
        run_noisy(args)
        write_conclusion(args)
        return 0
    if args.cmd == "plots":
        make_plots(args)
        return 0
    if args.cmd == "conclude":
        write_conclusion(args)
        return 0
    if args.cmd == "smoke":
        print("=== protocol smoke: noiseless then noisy ===", flush=True)
        run_noiseless(args)
        if not args.skip_noisy:
            run_noisy(args)
        make_plots(args)
        write_conclusion(args)
        return 0
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
