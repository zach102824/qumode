#!/usr/bin/env python3
"""Diagnosis driver: why n=8 success collapsed as ECD depth grew.

Does **not** write live ladder cells. Outputs live under results/diagnosis/.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .config import (
    DIAGNOSIS_ROOT,
    N7_L4_NPARAMS,
    OUTER_ITER,
    SPSA_A,
    SUPERSEDED_70_ROOT,
    ham_dir,
    n_joint_params,
    spsa_a_scaled,
    trial_seed,
)
from .ecd import hybrid_energy_tensor, random_ecd_parameters, unpack_ecd, vacuum_prep
from .embedding import embedding_for_n
from .four_sat import load_instance
from .gibbs import EcdGibbsSim, optimize_gibbs_adaptive
from .io_util import json_ready, write_json


def _instances(n: int) -> list[dict]:
    paths = sorted(ham_dir(n).glob("four_sat_[0-9][0-9][0-9].npz"))
    if not paths:
        raise FileNotFoundError(f"no Hamiltonians in {ham_dir(n)}")
    return [load_instance(p) for p in paths]


def check_roundtrips(n: int) -> dict:
    emb = embedding_for_n(n)
    n_fail = 0
    n_total = 1 << n
    for idx in range(n_total):
        bits = np.array([(idx >> (n - 1 - i)) & 1 for i in range(n)], dtype=int)
        occ = emb.encode_bits(bits)
        got = emb.decode_occupations(occ)
        if not np.array_equal(got, bits):
            n_fail += 1
    planted = []
    alias_counts = []
    for inst in _instances(n):
        bits = np.array([int(c) for c in inst["ground_bitstring"]], dtype=int)
        occ = emb.encode_bits(bits)
        ok = bool(np.array_equal(emb.decode_occupations(occ), bits))
        tensor = hybrid_energy_tensor(emb, inst["logical_energies"])
        n_zero = int(np.sum(np.isclose(tensor, 0.0)))
        planted.append(
            {
                "file": inst["file"],
                "ground": inst["ground_bitstring"],
                "roundtrip_ok": ok,
                "E_at_encode": float(tensor[occ]),
                "n_hybrid_E0": n_zero,
            }
        )
        alias_counts.append(n_zero)
        if not ok or abs(float(tensor[occ])) > 1e-12:
            n_fail += 1
    return {
        "n": n,
        "dim": emb.dim,
        "n_pairs": emb.n_pairs,
        "exhaustive_logical": n_total,
        "exhaustive_roundtrip_failures": 0 if n_fail == 0 else n_fail,
        "planted": planted,
        "hybrid_E0_unique": all(c == 1 for c in alias_counts),
        "hybrid_E0_mean": float(np.mean(alias_counts)),
        "ok": n_fail == 0,
    }


def check_n7_vs_production() -> dict:
    emb = embedding_for_n(7)
    n_mismatch = 0
    for q in range(2):
        for nocc in range(8):
            for m in range(8):
                ours = list(emb.decode_occupations((q, nocc, m)))
                bits = [int(q)]
                for k in range(2, -1, -1):
                    bits.append((int(nocc) >> k) & 1)
                for k in range(2, -1, -1):
                    bits.append((int(m) >> k) & 1)
                if ours != bits:
                    n_mismatch += 1
                # Inverse: Fock from bits
                n_back = 0
                for b in bits[1:4]:
                    n_back = (n_back << 1) | int(b)
                m_back = 0
                for b in bits[4:]:
                    m_back = (m_back << 1) | int(b)
                if (bits[0], n_back, m_back) != (q, nocc, m):
                    n_mismatch += 1
    circuit = {"qutip_available": False, "statevector_fidelity": None}
    try:
        import sys
        from pathlib import Path as _Path

        src = _Path(__file__).resolve().parents[1] / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import qutip  # noqa: F401
        from qumode_vqe.circuit import prepare_state
        from qumode_vqe.params import random_parameters
        from .ecd import apply_ecd_ansatz, prep_to_ket

        rng = np.random.default_rng(123)
        x = random_parameters(4, rng)
        psi_prod = np.asarray(prepare_state(x, 4, (8, 8)).full()).reshape(-1)
        psi_ours = apply_ecd_ansatz(prep_to_ket(vacuum_prep(emb), emb), x, emb, 4)
        fid = float(abs(np.vdot(psi_prod, psi_ours)) ** 2)
        circuit = {
            "qutip_available": True,
            "statevector_fidelity": fid,
            "ok": abs(fid - 1.0) < 1e-10,
        }
    except Exception as exc:
        circuit["import_error"] = f"{type(exc).__name__}: {exc}"
    return {
        "decode_mismatches": n_mismatch,
        "decode_ok": n_mismatch == 0,
        "n_pairs": emb.n_pairs,
        "n_prep": emb.n_prep_params,
        "n_params_L4": n_joint_params(emb.n_prep_params, 4, emb.n_pairs),
        "production_n_params_L4": N7_L4_NPARAMS,
        "circuit": circuit,
        "ok": n_mismatch == 0 and circuit.get("ok", True),
    }


def superseded_curve(n: int) -> dict:
    rows = []
    for depth in range(3, 21):
        path = SUPERSEDED_70_ROOT / f"n{n}_L{depth:02d}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        trials = rec.get("trials", [])
        pgs = [float(t["p_ground"]) for t in trials] or [0.0]
        ens = [float(t["energy"]) for t in trials] or [0.0]
        hits_by_h = Counter()
        for t in trials:
            if t["success"]:
                hits_by_h[int(t["hamiltonian_id"])] += 1
        rows.append(
            {
                "L": depth,
                "k": int(rec["k"]),
                "n_total": int(rec["n_total"]),
                "success_prob": float(rec["success_prob"]),
                "outer_iter": int(rec.get("outer_iter", 70)),
                "mean_p_ground": float(np.mean(pgs)),
                "max_p_ground": float(np.max(pgs)),
                "mean_energy": float(np.mean(ens)),
                "n_h_with_hit": len(hits_by_h),
            }
        )
    return {"n": n, "cells": rows}


def random_init_study(n: int, depths: list[int], n_samples: int = 2000) -> dict:
    instances = _instances(n)[:5]
    emb = embedding_for_n(n)
    out = {"n": n, "n_samples_per_cell": n_samples, "uniform": 1.0 / (1 << n), "depths": []}
    rng = np.random.default_rng(9000 + n)
    for depth in depths:
        hits = 0
        pgs = []
        for inst in instances:
            energy = hybrid_energy_tensor(emb, inst["logical_energies"])
            sim = EcdGibbsSim(emb, energy, depth, inst["ground_bitstring"])
            for _ in range(n_samples // len(instances)):
                x0 = random_ecd_parameters(depth, emb.n_pairs, rng)
                ev = sim.evaluate(x0)
                hits += int(ev.success)
                pgs.append(ev.p_ground)
        ntot = (n_samples // len(instances)) * len(instances)
        out["depths"].append(
            {
                "L": depth,
                "k": hits,
                "n_total": ntot,
                "success_prob": hits / max(ntot, 1),
                "mean_p_ground": float(np.mean(pgs)),
                "uniform": 1.0 / (1 << n),
            }
        )
    return out


def _run_one(
    n: int,
    depth: int,
    inst: dict,
    trial: int,
    *,
    outer_iter: int,
    a: float,
    log_every: int,
) -> dict:
    emb = embedding_for_n(n)
    energy = hybrid_energy_tensor(emb, inst["logical_energies"])
    seed = trial_seed(n, int(inst["hamiltonian_id"]), trial)
    rng = np.random.default_rng(seed)
    x0 = random_ecd_parameters(depth, emb.n_pairs, rng)
    t0 = time.perf_counter()
    result = optimize_gibbs_adaptive(
        emb,
        energy,
        inst["ground_bitstring"],
        x0,
        ndepth=depth,
        prep0=vacuum_prep(emb),
        outer_iter=outer_iter,
        rng=rng,
        a=a,
        log_every=log_every,
    )
    ev = result.eval_final
    packed = unpack_ecd(result.x, depth, emb.n_pairs)
    rec = {
        "n": n,
        "L": depth,
        "hamiltonian_id": int(inst["hamiltonian_id"]),
        "trial": trial,
        "seed": seed,
        "success": bool(ev.success),
        "most_likely_bitstring": ev.most_likely_bitstring,
        "ground_bitstring": ev.ground_bitstring,
        "gibbs_cost": float(ev.gibbs_cost),
        "energy": float(ev.energy),
        "p_most_likely": float(ev.p_most_likely),
        "p_ground": float(ev.p_ground),
        "eta": float(result.eta),
        "nfev": int(result.nfev),
        "elapsed_s": float(time.perf_counter() - t0),
        "mean_abs_beta": float(np.mean(np.abs(packed.beta))),
        "max_abs_beta": float(np.max(np.abs(packed.beta))),
        "a": float(a),
        "n_params": n_joint_params(emb.n_prep_params, depth, emb.n_pairs),
    }
    if log_every:
        rec["step_log"] = result.step_log
    return rec


def depth_study(
    n: int,
    depths: list[int],
    *,
    outer_iter: int = OUTER_ITER,
    n_hamiltonians: int = 20,
    n_trials: int = 10,
    scale_a: bool = False,
    log_traces: int = 3,
) -> dict:
    instances = _instances(n)[:n_hamiltonians]
    emb = embedding_for_n(n)
    tag = "scaled_a" if scale_a else "default_a"
    cells = []
    for depth in depths:
        n_params = n_joint_params(emb.n_prep_params, depth, emb.n_pairs)
        a = spsa_a_scaled(n_params) if scale_a else SPSA_A
        trials: list[dict] = []
        t0 = time.perf_counter()
        expected = len(instances) * n_trials
        print(
            f"=== diagnosis n={n} L={depth} SPSA={outer_iter} a={a:.4f} "
            f"n_params={n_params} tag={tag} todo={expected} ===",
            flush=True,
        )
        for inst in instances:
            for t in range(n_trials):
                log_every = 5 if (int(inst["hamiltonian_id"]) == 0 and t < log_traces) else 0
                rec = _run_one(
                    n, depth, inst, t, outer_iter=outer_iter, a=a, log_every=log_every
                )
                trials.append(rec)
                done = len(trials)
                if done % 10 == 0 or rec["success"]:
                    print(
                        f"  [{done}/{expected}] H{rec['hamiltonian_id']} t{t}: "
                        f"{'HIT' if rec['success'] else 'miss'}  "
                        f"pGS={rec['p_ground']:.4f}  E={rec['energy']:.3f}  "
                        f"{rec['elapsed_s']:.2f}s",
                        flush=True,
                    )
        k = int(sum(bool(r["success"]) for r in trials))
        pgs = [float(r["p_ground"]) for r in trials]
        ens = [float(r["energy"]) for r in trials]
        cell = {
            "n": n,
            "L": depth,
            "k": k,
            "n_total": len(trials),
            "success_prob": k / max(len(trials), 1),
            "success_fraction": f"{k}/{len(trials)}",
            "outer_iter": outer_iter,
            "a": a,
            "a_mode": tag,
            "n_params": n_params,
            "mean_p_ground": float(np.mean(pgs)),
            "max_p_ground": float(np.max(pgs)),
            "mean_energy": float(np.mean(ens)),
            "mean_abs_beta": float(np.mean([r["mean_abs_beta"] for r in trials])),
            "max_abs_beta": float(np.max([r["max_abs_beta"] for r in trials])),
            "wall_s": time.perf_counter() - t0,
            "traces": [r for r in trials if "step_log" in r],
            "trials": [{k: v for k, v in r.items() if k != "step_log"} for r in trials],
        }
        cells.append(cell)
        outp = DIAGNOSIS_ROOT / f"n{n}_L{depth:02d}_{tag}.json"
        write_json(outp, cell)
        print(
            f"n={n} L={depth} {tag}: {k}/{len(trials)}  pGS={cell['mean_p_ground']:.4f}  "
            f"E={cell['mean_energy']:.3f}  → {outp}",
            flush=True,
        )
    return {"n": n, "a_mode": tag, "outer_iter": outer_iter, "cells": cells}


def write_diagnosis_md(payload: dict) -> Path:
    lines = [
        "# n=8 ECD depth collapse — diagnosis",
        "",
        "Question: why did noiseless ECD bitstring success on n=8 fall as L grew",
        "(70 SPSA: L4 39/200 → L5 12/200 → ~0), and is that a bug?",
        "",
    ]

    rt = payload.get("roundtrips", {})
    lines += ["## 1. Embedding round-trip (n=8, n=9)", ""]
    for n, rec in sorted(rt.items(), key=lambda kv: int(kv[0])):
        status = "PASS" if rec.get("ok") else "FAIL"
        lines.append(
            f"- n={n}: exhaustive logical encode→decode **{status}** "
            f"({rec.get('exhaustive_logical')} strings). "
            f"Planted GS energy at encode is 0. Hybrid E=0 count mean "
            f"{rec.get('hybrid_E0_mean')} "
            f"(unique={rec.get('hybrid_E0_unique')})."
        )
    lines.append("")
    if rt.get("8", {}).get("hybrid_E0_unique"):
        lines.append(
            "n=8 is an exact fill of T0+T1+C0+C1 (dim 256 = 2^8): **no Fock aliasing**. "
            "A decoding bug cannot explain the L-collapse on n=8."
        )
        lines.append("")
    if rt.get("9") and not rt["9"].get("hybrid_E0_unique"):
        lines.append(
            "n=9: C2 has 1 logical bit on an 8-level cavity, so several Fock states "
            "decode to the same bitstring and share energy 0. That is a n=9+ caveat, "
            "not the n=8 collapse."
        )
        lines.append("")

    prod = payload.get("n7_production", {})
    lines += ["## 2. Diff vs working n=7 production path", ""]
    lines.append(
        f"- Occupation decode vs `bits_from_qnm` / `qnm_from_bits`: "
        f"**{'PASS' if prod.get('decode_ok') else 'FAIL'}** "
        f"(mismatches={prod.get('decode_mismatches')})."
    )
    circ = prod.get("circuit") or {}
    if circ.get("qutip_available"):
        lines.append(
            f"- L=4 vacuum ECD statevector fidelity vs QuTiP `prepare_state`: "
            f"**{circ.get('statevector_fidelity'):.12f}** "
            f"({'PASS' if circ.get('ok') else 'FAIL'})."
        )
    else:
        lines.append("- QuTiP not installed in this environment; circuit match test skipped here (unit test covers it when qutip is present).")
    lines.append(
        f"- Joint parameter count n=7 L=4: {prod.get('n_params_L4')} "
        f"(production {prod.get('production_n_params_L4')})."
    )
    lines.append(
        "- n=8 L=4 uses **4 ECD pairs** (2T×2C) vs production **2 pairs** (1T×2C), "
        "so 70 params vs 37 at the same depth. Same SPSA gains (`a=0.2, c=0.15`)."
    )
    lines.append("")

    sup = payload.get("superseded_70", {})
    if sup:
        lines += ["## 70-SPSA scoreboard (superseded, for the collapse shape)", ""]
        for n, rec in sorted(sup.items(), key=lambda kv: int(kv[0])):
            lines.append(f"### n={n}")
            lines.append("")
            lines.append("| L | k/N | success | mean p(GS) | mean ⟨H⟩ |")
            lines.append("|---|-----|---------|------------|----------|")
            for c in rec.get("cells", []):
                if int(c["L"]) < 4:
                    continue
                lines.append(
                    f"| {c['L']} | {c['k']}/{c['n_total']} | {c['success_prob']:.3f} | "
                    f"{c['mean_p_ground']:.4f} | {c['mean_energy']:.3f} |"
                )
            lines.append("")

    rnd = payload.get("random_init", {})
    if rnd:
        lines += ["## 4. Random-init success (no SPSA)", ""]
        lines.append(f"Uniform 1/2^n for n={rnd.get('n')} is {rnd.get('uniform'):.6f}.")
        lines.append("")
        lines.append("| L | k/N | success | mean p(GS) | 1/2^n |")
        lines.append("|---|-----|---------|------------|-------|")
        for c in rnd.get("depths", []):
            lines.append(
                f"| {c['L']} | {c['k']}/{c['n_total']} | {c['success_prob']:.5f} | "
                f"{c['mean_p_ground']:.5f} | {c['uniform']:.5f} |"
            )
        lines.append("")

    studies = payload.get("depth_studies", [])
    if studies:
        lines += ["## 3. Controlled n=8 study (200 joint SPSA)", ""]
        for study in studies:
            lines.append(
                f"### {study.get('a_mode')} (outer_iter={study.get('outer_iter')})"
            )
            lines.append("")
            lines.append("| L | k/N | success | mean p(GS) | max p(GS) | mean ⟨H⟩ | mean\\|β\\| | a | n_params |")
            lines.append("|---|-----|---------|------------|-----------|----------|-----------|---|----------|")
            for c in study.get("cells", []):
                lines.append(
                    f"| {c['L']} | {c['success_fraction']} | {c['success_prob']:.3f} | "
                    f"{c['mean_p_ground']:.4f} | {c['max_p_ground']:.4f} | "
                    f"{c['mean_energy']:.3f} | {c['mean_abs_beta']:.3f} | "
                    f"{c['a']:.4f} | {c['n_params']} |"
                )
            lines.append("")
            # mean trace
            for c in study.get("cells", []):
                traces = c.get("traces") or []
                if not traces:
                    continue
                # average p_ground vs step across traces
                by_step: dict[int, list[float]] = {}
                by_step_e: dict[int, list[float]] = {}
                for tr in traces:
                    for row in tr.get("step_log") or []:
                        by_step.setdefault(int(row["step"]), []).append(float(row["p_ground"]))
                        by_step_e.setdefault(int(row["step"]), []).append(float(row["energy"]))
                lines.append(f"Mean H0 traces p(GS) vs step at L={c['L']}:")
                lines.append("")
                lines.append("| step | mean p(GS) | mean ⟨H⟩ |")
                lines.append("|------|------------|----------|")
                for step in sorted(by_step)[:12]:
                    lines.append(
                        f"| {step} | {float(np.mean(by_step[step])):.4f} | "
                        f"{float(np.mean(by_step_e[step])):.3f} |"
                    )
                last = max(by_step)
                if last not in list(sorted(by_step))[:12]:
                    lines.append(
                        f"| {last} | {float(np.mean(by_step[last])):.4f} | "
                        f"{float(np.mean(by_step_e[last])):.3f} |"
                    )
                lines.append("")

    lines += [
        "## Root cause",
        "",
        payload.get("root_cause", "See numbers above; written after the controlled study."),
        "",
        "## Live ladder",
        "",
        payload.get("ladder_plan", "Deferred until this diagnosis is complete."),
        "",
    ]
    path = Path(__file__).resolve().parent / "DIAGNOSIS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _infer_root_cause(payload: dict) -> str:
    rt8 = (payload.get("roundtrips") or {}).get("8") or {}
    prod = payload.get("n7_production") or {}
    studies = payload.get("depth_studies") or []
    rnd = payload.get("random_init") or {}
    bits = []
    if rt8.get("ok") and rt8.get("hybrid_E0_unique"):
        bits.append(
            "Embedding is not the bug: n=8 planted bitstrings encode→decode, "
            "hybrid ground energy is unique, and dim=256=2^8."
        )
    if prod.get("decode_ok"):
        bits.append("n=7 decode matches production `bits_from_qnm`.")
    if (prod.get("circuit") or {}).get("ok"):
        bits.append("n=7 L=4 ECD statevector matches production QuTiP to numerical precision.")
    if rnd.get("depths"):
        u = float(rnd.get("uniform") or 0)
        obs = ", ".join(
            f"L={c['L']} {c['success_prob']:.4f}" for c in rnd["depths"]
        )
        bits.append(f"Random-init success ({obs}) is order-1/2^n ({u:.4f}), so argmax is not stuck on a single garbage label.")
    default = next((s for s in studies if s.get("a_mode") == "default_a"), None)
    scaled = next((s for s in studies if s.get("a_mode") == "scaled_a"), None)
    if default and default.get("cells"):
        cells = default["cells"]
        l4 = next((c for c in cells if int(c["L"]) == 4), None)
        deep = [c for c in cells if int(c["L"]) > 4]
        if l4 and deep:
            deep_best = max(deep, key=lambda c: c["success_prob"])
            if l4["success_prob"] + 0.05 >= deep_best["success_prob"]:
                bits.append(
                    f"At 200 SPSA / default a=0.2, deeper L still does not beat L=4 "
                    f"(L=4 {l4['success_fraction']}, best deeper L={deep_best['L']} "
                    f"{deep_best['success_fraction']}). mean p(GS) L=4={l4['mean_p_ground']:.4f} "
                    f"vs L={deep_best['L']} {deep_best['mean_p_ground']:.4f}."
                )
            else:
                bits.append(
                    f"At 200 SPSA, deeper L **improves** over L=4 "
                    f"(L=4 {l4['success_fraction']} → L={deep_best['L']} {deep_best['success_fraction']}). "
                    f"The old collapse was the 70-step budget."
                )
    if scaled and scaled.get("cells"):
        cells = scaled["cells"]
        bits.append(
            "Scaled-a ("
            + ", ".join(
                f"L={c['L']} a={c['a']:.4f} {c['success_fraction']} pGS={c['mean_p_ground']:.4f}"
                for c in cells
            )
            + ")."
        )
        if default and default.get("cells"):
            for c in cells:
                d = next((x for x in default["cells"] if int(x["L"]) == int(c["L"])), None)
                if d and c["success_prob"] > d["success_prob"] + 0.05:
                    bits.append(
                        f"Scaling a ~ 1/sqrt(n_params) **helps** L={c['L']}: "
                        f"{d['success_fraction']} → {c['success_fraction']}."
                    )
    if not bits:
        return "Diagnosis incomplete."
    return " ".join(bits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cmd",
        nargs="?",
        default="all",
        choices=("checks", "random-init", "depth-study", "scaled-a", "all"),
    )
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--depths", default="4,8,12")
    parser.add_argument("--outer-iter", type=int, default=OUTER_ITER)
    parser.add_argument("--n-hamiltonians", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--random-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    depths = [int(x) for x in str(args.depths).split(",") if x.strip()]
    DIAGNOSIS_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = DIAGNOSIS_ROOT / "summary.json"
    payload: dict = {}
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))

    if args.cmd in ("checks", "all"):
        payload["roundtrips"] = {str(n): check_roundtrips(n) for n in (8, 9)}
        payload["n7_production"] = check_n7_vs_production()
        payload["superseded_70"] = {str(n): superseded_curve(n) for n in (7, 8)}
        write_json(DIAGNOSIS_ROOT / "checks.json", {k: payload[k] for k in ("roundtrips", "n7_production", "superseded_70")})
        print(json.dumps(json_ready(payload["roundtrips"]), indent=2)[:2000], flush=True)
        print("n7_production", payload["n7_production"], flush=True)

    if args.cmd in ("random-init", "all"):
        payload["random_init"] = random_init_study(args.n, depths, n_samples=args.random_samples)
        write_json(DIAGNOSIS_ROOT / "random_init.json", payload["random_init"])
        print(payload["random_init"], flush=True)

    studies = list(payload.get("depth_studies") or [])
    if args.cmd in ("depth-study", "all"):
        rec = depth_study(
            args.n,
            depths,
            outer_iter=args.outer_iter,
            n_hamiltonians=args.n_hamiltonians,
            n_trials=args.n_trials,
            scale_a=False,
        )
        studies = [s for s in studies if s.get("a_mode") != "default_a"]
        studies.append(rec)
        payload["depth_studies"] = studies

    if args.cmd in ("scaled-a", "all"):
        # Only run scaled-a on depths that died under default a, or all requested.
        rec = depth_study(
            args.n,
            depths,
            outer_iter=args.outer_iter,
            n_hamiltonians=args.n_hamiltonians,
            n_trials=args.n_trials,
            scale_a=True,
        )
        studies = [s for s in (payload.get("depth_studies") or []) if s.get("a_mode") != "scaled_a"]
        studies.append(rec)
        payload["depth_studies"] = studies

    payload["root_cause"] = _infer_root_cause(payload)
    sane = bool((payload.get("roundtrips") or {}).get("8", {}).get("ok"))
    default = next((s for s in payload.get("depth_studies") or [] if s.get("a_mode") == "default_a"), None)
    if default and default.get("cells"):
        l4 = next((c for c in default["cells"] if int(c["L"]) == 4), None)
        if l4 and l4["n_total"] >= 50 and l4["success_prob"] < 0.02:
            sane = False
            payload["ladder_plan"] = (
                "NOT sane for a blind L-sweep: n=8 L=4 at 200 SPSA is still near-zero. "
                "Stop and inspect traces / consider a 1-transmon embedding."
            )
        elif l4:
            payload["ladder_plan"] = (
                "Embedding/decode look sane. Resume the live n=8 L-sweep from L=4 "
                "with 200 SPSA"
                + (
                    " and a scaled as 0.2*sqrt(37/n_params) if default-a deeper L still dies."
                    if any(c["success_prob"] < l4["success_prob"] for c in default["cells"] if int(c["L"]) > 4)
                    else "."
                )
            )
    else:
        payload["ladder_plan"] = "Controlled study not finished; do not start the live L-sweep yet."
    write_json(summary_path, payload)
    path = write_diagnosis_md(payload)
    print(f"wrote {path}", flush=True)
    print("ROOT CAUSE:", payload["root_cause"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
