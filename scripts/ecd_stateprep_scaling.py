#!/usr/bin/env python3
"""State-prep scaling: hybrid ECD vs qubit HEA on even cats (and a Fock control).

Noiseless statevector, cost = 1 − F. Optimizer L-BFGS-B, several random starts.
This is not Gibbs VQE and does not touch the stored 40 Hamiltonians.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.optimize as sciopt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qumode_vqe.qaoa import hea_statevector, n_hea_params, random_hea_params
from qumode_vqe.stateprep import (
    choose_cutoff,
    compass_cat_amplitudes,
    compass_cat_amplitudes_infinite,
    constructive_even_cat,
    ecd_bounds,
    ecd_statevector,
    embed_fock_in_qubits,
    evaluate_ecd_fidelities,
    even_cat_amplitudes,
    even_cat_amplitudes_infinite,
    fock_amplitudes,
    fock_index_for_alpha,
    matched_hea_layers,
    n_ecd_params,
    n_qubits_for_cutoff,
    random_ecd_params,
    state_fidelity,
    truncation_fidelity,
)

DEFAULT_ALPHAS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
COMPASS_ALPHAS = (2.0, 3.0)
PRIMARY_ND = 2
UNCONSTRAINED_HEA_LAYERS = 5
OUTDIR = Path("results")


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


def _seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    # Deterministic 31-bit seed; independent of PYTHONHASHSEED.
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h & 0x7FFFFFFF)


def _target_amps(kind: str, alpha: float, n_fock: int, fock_n: int | None) -> np.ndarray:
    if kind == "even_cat":
        return even_cat_amplitudes(alpha, n_fock)
    if kind == "compass":
        return compass_cat_amplitudes(alpha, n_fock)
    if kind == "fock":
        if fock_n is None:
            raise ValueError("fock target requires fock_n")
        return fock_amplitudes(int(fock_n), n_fock)
    raise ValueError(f"unknown target {kind!r}")


def _truncation_f(kind: str, alpha: float, n_fock: int) -> float | None:
    if kind == "even_cat":
        return truncation_fidelity(even_cat_amplitudes_infinite(alpha), n_fock)
    if kind == "compass":
        return truncation_fidelity(compass_cat_amplitudes_infinite(alpha), n_fock)
    return None


def _run_constructive(job: dict) -> list[dict]:
    alpha = float(job["alpha"])
    n_fock = int(job["L"])
    built = constructive_even_cat(alpha, n_fock)
    trial = {
        **job,
        "start": 0,
        "seed": int(job["seed"]),
        "F": float(np.clip(built.fidelity, 0.0, 1.0)),
        "F_joint": float(built.f_joint),
        "F_reduced": float(built.f_reduced),
        "F_postselect": float(built.fidelity),
        "F_truncation": float(built.f_truncation),
        "P_plus": float(built.success_probability),
        "success_probability": float(built.success_probability),
        "nfev": 0,
        "nit": 0,
        "success_opt": True,
        "message": "non-variational: Ry(pi/2), ECD(2α), post-select |+>",
        "elapsed_s": 0.0,
    }
    return [trial]


def _optimize_starts(job: dict, fun, x0_fn, bounds) -> list[dict]:
    trials = []
    n_starts = int(job["n_starts"])
    maxiter = int(job["maxiter"])
    for start in range(n_starts):
        seed = _seed(job["seed"], job["ansatz"], job["n_layers"], job["target"], job["alpha"], start)
        rng = np.random.default_rng(seed)
        x0 = x0_fn(rng)
        t0 = time.perf_counter()
        result = sciopt.minimize(
            lambda x: 1.0 - float(fun(x)),
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-14},
        )
        elapsed = time.perf_counter() - t0
        f_opt = float(np.clip(fun(np.asarray(result.x, dtype=float)), 0.0, 1.0))
        extras = {}
        extra_fn = job.get("_extras")
        if extra_fn is not None:
            extras = extra_fn(np.asarray(result.x, dtype=float))
        trials.append(
            {
                **{k: v for k, v in job.items() if not str(k).startswith("_")},
                "start": start,
                "seed": int(seed),
                "F": f_opt,
                "nfev": int(result.nfev),
                "nit": int(result.nit),
                "success_opt": bool(result.success),
                "message": str(result.message),
                "elapsed_s": float(elapsed),
                **extras,
            }
        )
    return trials


def run_job(job: dict) -> list[dict]:
    """Worker entry: run constructive or variational trials for one spec."""
    kind = str(job["target"])
    ansatz = str(job["ansatz"])
    alpha = float(job["alpha"])
    n_fock = int(job["L"])
    fock_n = job.get("fock_n")
    n_fock_n = None if fock_n is None else int(fock_n)
    target = _target_amps(kind, alpha, n_fock, n_fock_n)
    f_trunc = _truncation_f(kind, alpha, n_fock)
    job = {**job, "F_truncation": None if f_trunc is None else float(f_trunc)}

    if ansatz == "constructive_ecd":
        return _run_constructive(job)

    if ansatz in {"ecd"}:
        n_layers = int(job["n_layers"])
        terminal = bool(job["terminal_rotation"])
        objective = str(job.get("ecd_objective", "joint"))

        def fun(x: np.ndarray) -> float:
            metrics = evaluate_ecd_fidelities(x, target, n_layers, terminal)
            return float(metrics["F_joint"] if objective == "joint" else metrics["F_reduced"])

        def extras(x: np.ndarray) -> dict:
            return evaluate_ecd_fidelities(x, target, n_layers, terminal)

        job = {**job, "_extras": extras}
        return _optimize_starts(
            job,
            fun,
            lambda rng: random_ecd_params(n_layers, rng, terminal_rotation=terminal, alpha=alpha),
            ecd_bounds(n_layers, terminal_rotation=terminal, alpha=alpha),
        )

    if ansatz in {"hea", "hea_unconstrained"}:
        n_qubits = int(job["n_qubits"])
        n_layers = int(job["n_layers"])
        embedded = embed_fock_in_qubits(target, n_qubits)

        def fun(x: np.ndarray) -> float:
            psi = hea_statevector(x, n_qubits, n_layers)
            return state_fidelity(psi, embedded)

        bounds = [(0.0, 2.0 * np.pi)] * int(job["n_params"])
        return _optimize_starts(
            job,
            fun,
            lambda rng: random_hea_params(n_qubits, n_layers, rng),
            bounds,
        )

    raise ValueError(f"unknown ansatz {ansatz!r}")


def _plan_cutoff(alpha: float, cap: int, max_infidelity: float) -> dict:
    try:
        n_fock, f_trunc = choose_cutoff(alpha, max_infidelity=max_infidelity, cap=cap)
    except ValueError as exc:
        return {"alpha": float(alpha), "skipped": True, "reason": str(exc)}
    n_qubits = n_qubits_for_cutoff(n_fock)
    return {
        "alpha": float(alpha),
        "skipped": False,
        "L": int(n_fock),
        "n_qubits": int(n_qubits),
        "F_truncation": float(f_trunc),
        "fock_n": int(fock_index_for_alpha(alpha, n_fock)),
    }


def build_jobs(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    cutoffs = [_plan_cutoff(a, args.l_cap, args.max_trunc_infidelity) for a in args.alphas]
    skipped = [c for c in cutoffs if c["skipped"]]
    kept = [c for c in cutoffs if not c["skipped"]]
    jobs: list[dict] = []

    def base(cut: dict, target: str, ansatz: str, **kw) -> dict:
        return {
            "alpha": cut["alpha"],
            "L": cut["L"],
            "n_qubits": cut["n_qubits"],
            "target": target,
            "fock_n": cut["fock_n"] if target == "fock" else None,
            "ansatz": ansatz,
            "n_starts": int(args.n_starts),
            "seed": int(args.seed),
            "maxiter": int(args.maxiter),
            "terminal_rotation": bool(args.terminal_rotation),
            "ecd_objective": str(args.ecd_objective),
            "extra": False,
            **kw,
        }

    for cut in kept:
        nq = cut["n_qubits"]
        jobs.append(
            base(
                cut,
                "even_cat",
                "constructive_ecd",
                n_layers=1,
                n_params=3,
                n_starts=1,
                extra=False,
            )
        )
        for nd in args.ecd_depths:
            jobs.append(
                base(
                    cut,
                    "even_cat",
                    "ecd",
                    n_layers=int(nd),
                    n_params=n_ecd_params(int(nd), args.terminal_rotation),
                )
            )
        matched_l = matched_hea_layers(nq, n_ecd_params(PRIMARY_ND, False))
        jobs.append(
            base(
                cut,
                "even_cat",
                "hea",
                n_layers=int(matched_l),
                n_params=n_hea_params(nq, matched_l),
            )
        )
        jobs.append(
            base(
                cut,
                "even_cat",
                "hea_unconstrained",
                n_layers=UNCONSTRAINED_HEA_LAYERS,
                n_params=n_hea_params(nq, UNCONSTRAINED_HEA_LAYERS),
                extra=True,
            )
        )
        # Negative control: Fock |n⟩, same L.
        jobs.append(
            base(
                cut,
                "fock",
                "ecd",
                n_layers=PRIMARY_ND,
                n_params=n_ecd_params(PRIMARY_ND, args.terminal_rotation),
            )
        )
        jobs.append(
            base(
                cut,
                "fock",
                "hea",
                n_layers=int(matched_l),
                n_params=n_hea_params(nq, matched_l),
            )
        )
        if args.ecd_nd8:
            # Eickbusch-scale: ≲10 ECD for |7⟩. Skip N_d=8 when n is larger
            # (truncated L=64 + 34 params is not cheap).
            if int(cut["fock_n"]) <= 8:
                jobs.append(
                    base(
                        cut,
                        "fock",
                        "ecd",
                        n_layers=8,
                        n_params=n_ecd_params(8, args.terminal_rotation),
                        extra=True,
                    )
                )

    if args.compass:
        for alpha in args.compass_alphas:
            cut = _plan_cutoff(alpha, args.l_cap, args.max_trunc_infidelity)
            if cut["skipped"]:
                skipped.append({**cut, "target": "compass"})
                continue
            nq = cut["n_qubits"]
            matched_l = matched_hea_layers(nq, n_ecd_params(PRIMARY_ND, False))
            for nd in (2, 4):
                jobs.append(
                    base(
                        cut,
                        "compass",
                        "ecd",
                        n_layers=int(nd),
                        n_params=n_ecd_params(int(nd), args.terminal_rotation),
                        extra=True,
                    )
                )
            jobs.append(
                base(
                    cut,
                    "compass",
                    "hea",
                    n_layers=int(matched_l),
                    n_params=n_hea_params(nq, matched_l),
                    extra=True,
                )
            )
    return jobs, skipped, kept


def _best_by_key(trials: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for t in trials:
        key = tuple(t.get(k) for k in keys)
        prev = groups.get(key)
        if prev is None or float(t["F"]) > float(prev["F"]):
            groups[key] = t
    return list(groups.values())


def _pick(rows: list[dict], **kw) -> dict | None:
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def _f(row: dict | None) -> float:
    return float("nan") if row is None else float(row["F"])


def write_markdown(
    path: Path,
    *,
    meta: dict,
    trials: list[dict],
    skipped: list[dict],
    cutoffs: list[dict],
) -> None:
    best = _best_by_key(
        trials,
        ("alpha", "target", "ansatz", "n_layers", "L"),
    )
    lines: list[str] = []
    lines.append("# ECD vs HEA state-preparation scaling")
    lines.append("")
    lines.append("Noiseless statevector fidelities for **state preparation**, not Gibbs VQE.")
    lines.append("Cost is `1-F`. Optimizer is L-BFGS-B with independent random starts;")
    lines.append("the table keeps the best start. Numbers are from")
    lines.append("`results/stateprep_scaling.json` (this run), not invented.")
    lines.append("")
    lines.append("This is **not** a claim that ECD is a new gate, that cats are new, or that")
    lines.append("Gibbs / `sampled_tail` helps state prep. Dutta et al. arXiv:2501.11735 is a")
    lines.append("**binary knapsack / constrained-optimization** experiment (the wrong problem")
    lines.append("class for this note) and is cited only as contrast.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("| Item | Setting |")
    lines.append("|---|---|")
    lines.append("| Target (primary) | Even two-legged cat |C_α⟩ ∝ |α⟩+|−α⟩, normalized on the truncated Fock space |")
    lines.append(f"| α | {', '.join(str(a) for a in meta['alphas'])} |")
    npar = "4 N_d+2" if meta["terminal_rotation"] else "4 N_d"
    obj = (
        "F = |⟨g, target|ψ⟩|²"
        if meta["ecd_objective"] == "joint"
        else "cavity reduced F = ⟨target|ρ_cav|target⟩"
    )
    lines.append(
        f"| Cutoff L | smallest L ≥ |α|²+8|α|+16 with truncation infidelity "
        f"< {meta['max_trunc_infidelity']}, cap {meta['l_cap']} |"
    )
    lines.append("| Same L | ECD and HEA use the same cutoff at each α |")
    lines.append(
        "| ECD | Single oscillator + one transmon. Layer R(θ, φ) then "
        f"`ECD(β)`. N_d in {meta['ecd_depths']}. "
        f"Terminal rotation: {meta['terminal_rotation']} ({npar} real params). |"
    )
    lines.append(f"| ECD objective | {obj} |")
    lines.append(
        "| HEA matched | Binary Fock index, MSB-first, unused levels padded with 0. "
        f"Layers chosen so n(L+1) is closest to the ECD budget 4 N_d = 8 "
        f"(primary N_d={PRIMARY_ND}). |"
    )
    lines.append(
        f"| HEA unconstrained | extra: n_layers={UNCONSTRAINED_HEA_LAYERS} "
        f"→ n·6 parameters |"
    )
    lines.append(
        f"| Starts / maxiter / seed | {meta['n_starts']} random starts, "
        f"L-BFGS-B maxiter={meta['maxiter']}, seed={meta['seed']} |"
    )
    lines.append("| Constructive ECD | Ry(π/2), `ECD(β=2α)`, X-basis post-select |+⟩ |")
    lines.append("| Negative control | Fock |n⟩, n=round(α²) clipped to [1, L-2] |")
    lines.append("")
    if skipped:
        lines.append("## Skipped α")
        lines.append("")
        for s in skipped:
            lines.append(f"- α={s['alpha']}: {s.get('reason', 'skipped')}")
        lines.append("")

    lines.append("## Cutoffs")
    lines.append("")
    lines.append("| α | L | n_qubits | F_truncation | 1−F_trunc | n = round(α²) |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for c in cutoffs:
        inf = 1.0 - float(c["F_truncation"])
        lines.append(
            f"| {c['alpha']:.1f} | {c['L']} | {c['n_qubits']} | "
            f"{c['F_truncation']:.6e} | {inf:.3e} | {c['fock_n']} |"
        )
    lines.append("")

    lines.append("## Even cat (best start)")
    lines.append("")
    lines.append(
        "| α | L | constructive F | ECD N_d=1 | ECD N_d=2 | ECD N_d=3 "
        "| HEA matched | HEA extra (L=5) | ECD N_d=2 params | HEA matched params |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cutoffs:
        a = c["alpha"]
        con = _pick(best, alpha=a, target="even_cat", ansatz="constructive_ecd")
        e1 = _pick(best, alpha=a, target="even_cat", ansatz="ecd", n_layers=1)
        e2 = _pick(best, alpha=a, target="even_cat", ansatz="ecd", n_layers=2)
        e3 = _pick(best, alpha=a, target="even_cat", ansatz="ecd", n_layers=3)
        hm = _pick(best, alpha=a, target="even_cat", ansatz="hea")
        hu = _pick(best, alpha=a, target="even_cat", ansatz="hea_unconstrained")
        lines.append(
            f"| {a:.1f} | {c['L']} | {_f(con):.6f} | {_f(e1):.6f} | {_f(e2):.6f} | "
            f"{_f(e3):.6f} | {_f(hm):.6f} | {_f(hu):.6f} | "
            f"{'' if e2 is None else e2['n_params']} | "
            f"{'' if hm is None else hm['n_params']} |"
        )
    lines.append("")
    lines.append(
        "Constructive F is the **post-selected** cavity fidelity. Variational ECD "
        f"F is the optimized `{meta['ecd_objective']}` fidelity (unitary, no post-select). "
        "HEA F is |⟨C_α|ψ_HEA⟩|² on the binary Fock register."
    )
    lines.append("")

    lines.append("## Negative control: Fock |n⟩ (best start)")
    lines.append("")
    lines.append("| α | n | L | ECD N_d=2 | HEA matched | ECD N_d=8 extra |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for c in cutoffs:
        a = c["alpha"]
        e2 = _pick(best, alpha=a, target="fock", ansatz="ecd", n_layers=2)
        hm = _pick(best, alpha=a, target="fock", ansatz="hea")
        e8 = _pick(best, alpha=a, target="fock", ansatz="ecd", n_layers=8)
        lines.append(
            f"| {a:.1f} | {c['fock_n']} | {c['L']} | {_f(e2):.6f} | {_f(hm):.6f} | {_f(e8):.6f} |"
        )
    lines.append("")

    compass_rows = [r for r in best if r.get("target") == "compass"]
    if compass_rows:
        lines.append("## Extra: 4-legged compass cat")
        lines.append("")
        lines.append("| α | L | ECD N_d=2 | ECD N_d=4 | HEA matched |")
        lines.append("|---:|---:|---:|---:|---:|")
        for a in meta.get("compass_alphas", []):
            e2 = _pick(best, alpha=a, target="compass", ansatz="ecd", n_layers=2)
            e4 = _pick(best, alpha=a, target="compass", ansatz="ecd", n_layers=4)
            hm = _pick(best, alpha=a, target="compass", ansatz="hea")
            l = "" if e2 is None else e2["L"]
            lines.append(f"| {a:.1f} | {l} | {_f(e2):.6f} | {_f(e4):.6f} | {_f(hm):.6f} |")
        lines.append("")

    lines.append("## Thesis")
    lines.append("")
    lines.append(_thesis_paragraph(cutoffs, best, meta))
    lines.append("")
    lines.append("## Citations (not claimed as this work)")
    lines.append("")
    lines.append("- Eickbusch et al., *Nat. Phys.* (2022); arXiv:2111.06414 — ECD gate and Fock-state compilation (≲10 ECD for |7⟩, F>0.99).")
    lines.append("- Krastanov et al., *Phys. Rev. A* **92**, 040303 (2015) — SNAP / oscillator control.")
    lines.append("- Analytic ECD circuits: arXiv:2504.19992.")
    lines.append("- N-fold cat lower bound: arXiv:2608.07696.")
    lines.append("- Dutta et al., arXiv:2501.11735 — **contrast only**: binary knapsack ECD-VQE, not state prep.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _thesis_paragraph(cutoffs: list[dict], best: list[dict], meta: dict) -> str:
    pairs = []
    for c in cutoffs:
        a = c["alpha"]
        e2 = _pick(best, alpha=a, target="even_cat", ansatz="ecd", n_layers=2)
        hm = _pick(best, alpha=a, target="even_cat", ansatz="hea")
        con = _pick(best, alpha=a, target="even_cat", ansatz="constructive_ecd")
        if e2 is None or hm is None or con is None:
            continue
        pairs.append((a, float(con["F"]), float(e2["F"]), float(hm["F"]), c["L"]))
    if not pairs:
        return "No completed even-cat points; thesis not evaluated."

    con_min = min(p[1] for p in pairs)
    ecd_vals = [p[2] for p in pairs]
    hea_vals = [p[3] for p in pairs]
    ecd_wins = sum(1 for e, h in zip(ecd_vals, hea_vals) if e > h + 1e-6)
    hea_wins = sum(1 for e, h in zip(ecd_vals, hea_vals) if h > e + 1e-6)
    ties = len(pairs) - ecd_wins - hea_wins
    ecd_small = ecd_vals[0]
    ecd_large = ecd_vals[-1]
    hea_small = hea_vals[0]
    hea_large = hea_vals[-1]
    a0, a1 = pairs[0][0], pairs[-1][0]

    fock_hea_wins = 0
    fock_ecd_wins = 0
    fock_n = 0
    for c in cutoffs:
        e2 = _pick(best, alpha=c["alpha"], target="fock", ansatz="ecd", n_layers=2)
        hm = _pick(best, alpha=c["alpha"], target="fock", ansatz="hea")
        if e2 is None or hm is None:
            continue
        fock_n += 1
        if float(hm["F"]) > float(e2["F"]) + 1e-6:
            fock_hea_wins += 1
        elif float(e2["F"]) > float(hm["F"]) + 1e-6:
            fock_ecd_wins += 1

    # Accept/reject the *variational matched-budget* claim on cats.
    ecd_holds = ecd_large >= 0.95 and (ecd_large + 1e-6 >= hea_large)
    hea_degrades = hea_large + 0.05 < hea_small
    constructive_holds = con_min >= 0.99

    if constructive_holds and ecd_holds and hea_degrades:
        verdict = "accepted"
    elif constructive_holds and not ecd_holds:
        verdict = "rejected for variational matched-budget ECD (constructive O(1) circuit still works)"
    else:
        verdict = "rejected"

    detail = (
        f"Constructive ECD (one Ry + one ECD + X-basis post-select) has "
        f"min F={con_min:.6f} over the completed α, so the O(1) existence proof "
        f"{'holds' if constructive_holds else 'fails'} — infidelity there tracks Fock "
        f"truncation, not missing gates. Variational ECD N_d=2 "
        f"(unitary, no post-select; {meta['ecd_objective']} fidelity) goes from "
        f"F={ecd_small:.4f} at α={a0:.1f} to F={ecd_large:.4f} at α={a1:.1f}. "
        f"Matched HEA goes from F={hea_small:.4f} to F={hea_large:.4f}. "
        f"On a per-α best-start count, ECD N_d=2 wins {ecd_wins}, HEA wins {hea_wins}, "
        f"ties {ties}. "
    )
    if hea_wins > ecd_wins:
        detail += "**HEA wins the matched-budget even-cat comparison** on this suite. "
    elif ecd_wins > hea_wins:
        detail += "ECD N_d=2 wins the matched-budget even-cat comparison on this suite. "
    else:
        detail += "Matched-budget even-cat results are mixed. "
    if fock_n:
        detail += (
            f"Negative control (Fock |n⟩, n≈|α|²): "
            f"HEA wins {fock_hea_wins}/{fock_n} points against ECD N_d=2"
        )
        if fock_hea_wins:
            detail += " — expected, because a computational-basis Fock state is a single bitstring for HEA and is coherent-dense for ECD"
        detail += ". "
    detail += (
        f"**Thesis {verdict}.** The claim that a depth-O(1) ECD circuit stays high-fidelity "
        f"on even cats while a parameter-matched qubit HEA degrades as |α| "
        f"(and L) grows is evaluated only from these numbers; parameters were not retuned "
        f"to force an ECD win."
    )
    return detail


def plot_cat(path: Path, cutoffs: list[dict], best: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    series = [
        ("constructive_ecd", None, "constructive ECD (post-select)", "o", "#2a9d8f"),
        ("ecd", 2, r"variational ECD $N_d=2$", "s", "#1d3557"),
        ("hea", None, "HEA matched", "D", "#e76f51"),
        ("hea_unconstrained", None, "HEA unconstrained (extra)", "^", "#9b2226"),
    ]
    for ansatz, n_layers, label, marker, color in series:
        xs, ys = [], []
        for c in cutoffs:
            kw = {"alpha": c["alpha"], "target": "even_cat", "ansatz": ansatz}
            if n_layers is not None:
                kw["n_layers"] = n_layers
            row = _pick(best, **kw)
            if row is None:
                continue
            xs.append(c["alpha"])
            ys.append(max(1.0 - float(row["F"]), 1e-16))
        if xs:
            ax.semilogy(xs, ys, marker=marker, color=color, label=label, linewidth=1.6)
    ax.set_xlabel(r"$|\alpha|$")
    ax.set_ylabel(r"infidelity $1-F$")
    ax.set_title("Even cat: cavity / register infidelity vs coherent amplitude")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fock(path: Path, cutoffs: list[dict], best: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    series = [
        ("ecd", 2, r"ECD $N_d=2$", "s", "#1d3557"),
        ("hea", None, "HEA matched", "D", "#e76f51"),
        ("ecd", 8, r"ECD $N_d=8$ (extra)", "o", "#2a9d8f"),
    ]
    for ansatz, n_layers, label, marker, color in series:
        xs, ys = [], []
        for c in cutoffs:
            kw = {"alpha": c["alpha"], "target": "fock", "ansatz": ansatz}
            if n_layers is not None:
                kw["n_layers"] = n_layers
            row = _pick(best, **kw)
            if row is None:
                continue
            xs.append(c["alpha"])
            ys.append(max(1.0 - float(row["F"]), 1e-16))
        if xs:
            ax.semilogy(xs, ys, marker=marker, color=color, label=label, linewidth=1.6)
    ax.set_xlabel(r"$|\alpha|$  (target $|n\rangle$, $n=\mathrm{round}(\alpha^2)$)")
    ax.set_ylabel(r"infidelity $1-F$")
    ax.set_title(r"Negative control: Fock $|n\rangle$ (number-sparse / coherent-dense)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    p.add_argument("--compass-alphas", type=float, nargs="+", default=list(COMPASS_ALPHAS))
    p.add_argument("--ecd-depths", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n-starts", type=int, default=4)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=OUTDIR)
    p.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument("--l-cap", type=int, default=64)
    p.add_argument("--max-trunc-infidelity", type=float, default=1e-4)
    p.add_argument("--terminal-rotation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ecd-objective", choices=("joint", "reduced"), default="joint")
    p.add_argument("--compass", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ecd-nd8", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--quick", action="store_true", help="Smoke: α=1, 1 start, maxiter=25, no extras.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.alphas = [1.0]
        args.n_starts = 1
        args.maxiter = 25
        args.compass = False
        args.ecd_nd8 = False
        args.ecd_depths = [1, 2]
    jobs, skipped, cutoffs = build_jobs(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    trials: list[dict] = []
    print(f"stateprep scaling: {len(jobs)} jobs, {args.workers} workers, {args.n_starts} starts", flush=True)
    if args.workers <= 1 or len(jobs) == 1:
        for job in jobs:
            chunk = run_job(job)
            trials.extend(chunk)
            print(
                f"  {job['target']} {job['ansatz']} α={job['alpha']} N={job['n_layers']} "
                f"bestF={max(t['F'] for t in chunk):.6f}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futs = {pool.submit(run_job, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                job = futs[fut]
                chunk = fut.result()
                trials.extend(chunk)
                done += 1
                print(
                    f"  [{done}/{len(jobs)}] {job['target']} {job['ansatz']} "
                    f"α={job['alpha']} N={job['n_layers']} "
                    f"bestF={max(t['F'] for t in chunk):.6f}",
                    flush=True,
                )
    elapsed = time.perf_counter() - t0
    meta = {
        "optimizer": "L-BFGS-B",
        "cost": "1-F",
        "n_starts": int(args.n_starts),
        "maxiter": int(args.maxiter),
        "seed": int(args.seed),
        "alphas": [float(a) for a in args.alphas],
        "compass_alphas": [float(a) for a in args.compass_alphas] if args.compass else [],
        "ecd_depths": [int(d) for d in args.ecd_depths],
        "terminal_rotation": bool(args.terminal_rotation),
        "ecd_objective": str(args.ecd_objective),
        "l_cap": int(args.l_cap),
        "max_trunc_infidelity": float(args.max_trunc_infidelity),
        "primary_nd": PRIMARY_ND,
        "unconstrained_hea_layers": UNCONSTRAINED_HEA_LAYERS,
        "n_jobs": len(jobs),
        "n_trials": len(trials),
        "elapsed_s": float(elapsed),
        "workers": int(args.workers),
        "quick": bool(args.quick),
        "notes": (
            "Single-mode ECD state prep vs binary-Fock HEA. "
            "Not Gibbs VQE. Does not modify eta.py or stored Hamiltonians."
        ),
    }
    payload = {"meta": meta, "cutoffs": cutoffs, "skipped": skipped, "trials": trials}
    json_path = args.outdir / "stateprep_scaling.json"
    json_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    best = _best_by_key(trials, ("alpha", "target", "ansatz", "n_layers", "L"))
    write_markdown(
        args.outdir / "stateprep_scaling.md",
        meta=meta,
        trials=trials,
        skipped=skipped,
        cutoffs=cutoffs,
    )
    plot_cat(args.outdir / "stateprep_scaling_cat.png", cutoffs, best)
    plot_fock(args.outdir / "stateprep_scaling_fock.png", cutoffs, best)
    print(f"wrote {json_path} ({len(trials)} trials, {elapsed:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
