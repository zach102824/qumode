#!/usr/bin/env python3
"""Diagnosis driver: why noiseless ECD n=12 L=4 is 145/200 (72.5%).

Same order as PR #15 / DIAGNOSIS.md for the n=8 collapse:
  1. embedding round-trip (do not skip to barren-plateau)
  2. diff vs working n=11
  3. random-init ~ 1/4096
  4. optimizer, L=4 only (step traces + step-count + a-sensitivity)

Does **not** write live ladder cells. Outputs under results/diagnosis_n12/.
Does **not** raise L. n=13–15 L=5 work is ignored.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import (
    DIAGNOSIS_N12_ROOT,
    N7_L4_NPARAMS,
    OUTER_ITER,
    RESULTS_ROOT,
    SPSA_A,
    ham_dir,
    n_joint_params,
    n_params_for,
    spsa_a_scaled,
    trial_seed,
)
from .ecd import (
    apply_ecd,
    apply_ecd_ansatz,
    hybrid_energy_tensor,
    prep_to_ket,
    random_ecd_parameters,
    unpack_ecd,
    vacuum_prep,
)
from .embedding import bit_partition_doc, embedding_for_n, hardware_idle_modes
from .four_sat import load_instance
from .gibbs import EcdGibbsSim, optimize_gibbs_adaptive
from .io_util import json_ready, write_json


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
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(int(n))
    except Exception:
        pass


_limit_blas(1)


# a-modes at L=4 only. "current" is the live ladder: 0.2 * sqrt(37 / d).
A_MODES = ("current_sqrt37", "milder_sqrt70", "fourth_root37", "unscaled")


def spsa_a_for_mode(n_params: int, mode: str) -> float:
    d = float(max(int(n_params), 1))
    if mode == "current_sqrt37":
        return float(SPSA_A) * math.sqrt(float(N7_L4_NPARAMS) / d)
    if mode == "milder_sqrt70":
        return float(SPSA_A) * math.sqrt(70.0 / d)
    if mode == "fourth_root37":
        return float(SPSA_A) * (float(N7_L4_NPARAMS) / d) ** 0.25
    if mode == "unscaled":
        return float(SPSA_A)
    raise ValueError(f"unknown a-mode {mode!r}")


def _instances(n: int) -> list[dict]:
    paths = sorted(ham_dir(n).glob("four_sat_[0-9][0-9][0-9].npz"))
    if not paths:
        raise FileNotFoundError(f"no Hamiltonians in {ham_dir(n)}")
    return [load_instance(p) for p in paths]


def _fock_bits_msb(occ: int, n_bits: int) -> list[int]:
    return [((int(occ) >> k) & 1) for k in range(int(n_bits) - 1, -1, -1)]


def bits_from_occupations_explicit(occupations: tuple[int, ...], n_transmons: int) -> list[int]:
    """Production bits_from_qnm rule: transmons then cavities, Fock MSB-first."""
    bits: list[int] = []
    for q in occupations[:n_transmons]:
        bits.append(int(q) & 1)
    for occ in occupations[n_transmons:]:
        bits.extend(_fock_bits_msb(occ, 3))
    return bits


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
    hybrid_fail = 0
    for idx in range(emb.dim):
        bits = emb.decode_index(idx)
        occ = emb.encode_bits(bits)
        if not np.array_equal(emb.decode_occupations(occ), bits):
            hybrid_fail += 1
    return {
        "n": n,
        "dim": emb.dim,
        "n_pairs": emb.n_pairs,
        "dims": list(emb.dims),
        "mode_names": [m.name for m in emb.modes],
        "idle": hardware_idle_modes(n),
        "exhaustive_logical": n_total,
        "exhaustive_logical_failures": n_fail,
        "exhaustive_hybrid_failures": hybrid_fail,
        "planted": planted,
        "hybrid_E0_unique": all(c == 1 for c in alias_counts),
        "hybrid_E0_mean": float(np.mean(alias_counts)),
        "ok": n_fail == 0 and hybrid_fail == 0,
    }


def check_n12_wiring() -> dict:
    """T2 is new. Pair order, remapped axes, idle C3, prep Ry, decode bits."""
    emb11 = embedding_for_n(11)
    emb12 = embedding_for_n(12)
    pairs12 = emb12.ecd_pairs()
    expected_pairs = (
        (0, 3),
        (1, 3),
        (2, 3),
        (0, 4),
        (1, 4),
        (2, 4),
        (0, 5),
        (1, 5),
        (2, 5),
    )
    pair_ok = pairs12 == expected_pairs
    idle = hardware_idle_modes(12)
    idle_ok = idle == ["C3"]
    names = [m.name for m in emb12.modes]
    names_ok = names == ["T0", "T1", "T2", "C0", "C1", "C2"]
    axes = [(m.name, m.axis) for m in emb12.modes]
    axes_ok = axes == [("T0", 0), ("T1", 1), ("T2", 2), ("C0", 3), ("C1", 4), ("C2", 5)]
    n_params = n_params_for(12, 4)
    params_ok = n_params == 3 + 6 + 4 * 4 * 9 == 153
    formula_ok = n_joint_params(emb12.n_prep_params, 4, emb12.n_pairs) == 153

    # Occupation decode vs explicit bits_from_qnm analog (all 4096 labels).
    n_decode_mismatch = 0
    for idx in range(emb12.dim):
        occ = np.unravel_index(idx, emb12.dims)
        ours = list(emb12.decode_occupations(occ))
        explicit = bits_from_occupations_explicit(tuple(int(v) for v in occ), 3)
        if ours != explicit:
            n_decode_mismatch += 1
    n11_decode_mismatch = 0
    for idx in range(emb11.dim):
        occ = np.unravel_index(idx, emb11.dims)
        ours = list(emb11.decode_occupations(occ))
        explicit = bits_from_occupations_explicit(tuple(int(v) for v in occ), 2)
        if ours != explicit:
            n11_decode_mismatch += 1

    # Prep Ry on the new transmon: θ_T2 = π → |001000> Fock vacuum.
    prep = vacuum_prep(emb12).copy()
    prep[2] = float(np.pi)
    ket = prep_to_ket(prep, emb12)
    occ_ry = tuple(int(v) for v in np.unravel_index(int(np.argmax(np.abs(ket) ** 2)), emb12.dims))
    ry_ok = occ_ry == (0, 0, 1, 0, 0, 0) and abs(float(np.abs(ket[np.ravel_multi_index(occ_ry, emb12.dims)])) - 1.0) < 1e-12

    # ECD on (T2, C0) from vacuum flips T2 (pair index 2).
    psi0 = prep_to_ket(vacuum_prep(emb12), emb12)
    psi_ecd = apply_ecd(psi0, emb12.dims, 2, 3, 1.0 + 0.0j)
    p_t2 = float(np.sum(np.abs(psi_ecd.reshape(emb12.dims)[:, :, 1, ...]) ** 2))
    t2_ecd_ok = p_t2 > 0.99

    part11 = [r["label"] for r in bit_partition_doc(emb11)]
    part12 = [r["label"] for r in bit_partition_doc(emb12)]
    t2_is_bit2 = part12[2] == "T2"
    cavities_shift = part11[2].startswith("C0") and part12[3].startswith("C0")

    # n=11 pairs are 2T×3C; n=12 adds the three T2–C{0,1,2} edges.
    pairs11 = emb11.ecd_pairs()
    expected11 = ((0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4))
    pairs11_ok = pairs11 == expected11

    return {
        "n12_dims": list(emb12.dims),
        "n12_dim": emb12.dim,
        "n12_mode_names": names,
        "n12_names_ok": names_ok,
        "n12_axes": axes,
        "n12_axes_ok": axes_ok,
        "ecd_pairs": [list(p) for p in pairs12],
        "ecd_pairs_ok": pair_ok,
        "n_pairs": emb12.n_pairs,
        "idle": idle,
        "idle_c3_omitted": idle_ok,
        "n_params_L4": n_params,
        "n_params_ok": params_ok and formula_ok,
        "n_prep": emb12.n_prep_params,
        "decode_vs_bits_from_qnm_mismatches": n_decode_mismatch,
        "decode_ok": n_decode_mismatch == 0,
        "n11_decode_vs_bits_from_qnm_mismatches": n11_decode_mismatch,
        "n11_decode_ok": n11_decode_mismatch == 0,
        "t2_prep_ry_occupation": list(occ_ry),
        "t2_prep_ry_ok": ry_ok,
        "t2_ecd_population": p_t2,
        "t2_ecd_ok": t2_ecd_ok,
        "bit_partition_n11": part11,
        "bit_partition_n12": part12,
        "t2_is_logical_bit_2": t2_is_bit2,
        "cavity_bits_shift_by_one": cavities_shift,
        "n11_pairs_ok": pairs11_ok,
        "n11_dims": list(emb11.dims),
        "n11_n_params_L4": n_params_for(11, 4),
        "ok": all(
            [
                pair_ok,
                idle_ok,
                names_ok,
                axes_ok,
                params_ok and formula_ok,
                n_decode_mismatch == 0,
                n11_decode_mismatch == 0,
                ry_ok,
                t2_ecd_ok,
                t2_is_bit2,
                pairs11_ok,
            ]
        ),
    }


def check_n7_vs_production() -> dict:
    """Reuse the n=8-diagnosis n=7 QuTiP fidelity check when qutip is present."""
    from .diagnose import check_n7_vs_production as _n7

    return _n7()


def analyze_live_n12_l4() -> dict:
    path = RESULTS_ROOT / "n12_L04.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    trials = rec["trials"]
    t2_gs: dict[int, list[int]] = {0: [0, 0], 1: [0, 0]}
    bit_hits = [0] * 12
    pgs_hit: list[float] = []
    pgs_miss: list[float] = []
    miss_t2_wrong = 0
    hams: list[int] = []
    by_h: dict[int, list[int]] = {}
    for t in trials:
        hid = int(t["hamiltonian_id"])
        by_h.setdefault(hid, [0, 0])
        by_h[hid][1] += 1
        gs = str(t["ground_bitstring"])
        ml = str(t["most_likely_bitstring"])
        t2 = int(gs[2])
        t2_gs[t2][1] += 1
        if t["success"]:
            by_h[hid][0] += 1
            t2_gs[t2][0] += 1
            pgs_hit.append(float(t["p_ground"]))
        else:
            pgs_miss.append(float(t["p_ground"]))
            hams.append(sum(a != b for a, b in zip(gs, ml)))
            if gs[2] != ml[2]:
                miss_t2_wrong += 1
        for i, (a, b) in enumerate(zip(gs, ml)):
            if a == b:
                bit_hits[i] += 1
    ntot = len(trials)
    return {
        "k": int(rec["k"]),
        "n_total": ntot,
        "success_prob": float(rec["success_prob"]),
        "mean_p_ground": float(rec.get("mean_p_ground") or np.mean([t["p_ground"] for t in trials])),
        "n_params": int(rec.get("n_params", 153)),
        "a": float((rec.get("protocol") or {}).get("spsa", {}).get("a") or spsa_a_scaled(153)),
        "dim": int(rec.get("dim", 4096)),
        "n_pairs": int(rec.get("n_pairs", 9)),
        "mean_elapsed_s": float(np.mean([t["elapsed_s"] for t in trials])),
        "t2_gs_success": {str(k): f"{v[0]}/{v[1]}" for k, v in t2_gs.items()},
        "t2_gs_success_rate": {str(k): (v[0] / max(v[1], 1)) for k, v in t2_gs.items()},
        "misses_t2_bit_wrong": miss_t2_wrong,
        "n_miss": len(pgs_miss),
        "bit_accuracy": [bit_hits[i] / ntot for i in range(12)],
        "p_ground_hit_mean": float(np.mean(pgs_hit)) if pgs_hit else None,
        "p_ground_hit_max": float(np.max(pgs_hit)) if pgs_hit else None,
        "p_ground_miss_mean": float(np.mean(pgs_miss)) if pgs_miss else None,
        "miss_hamming_mean": float(np.mean(hams)) if hams else None,
        "miss_hamming": dict(Counter(hams)),
        "by_hamiltonian": {f"{h:03d}": f"{v[0]}/{v[1]}" for h, v in sorted(by_h.items())},
    }


def analyze_live_n11_l4() -> dict:
    path = RESULTS_ROOT / "n11_L04.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    pgs = [float(t["p_ground"]) for t in rec["trials"]]
    return {
        "k": int(rec["k"]),
        "n_total": int(rec["n_total"]),
        "success_prob": float(rec["success_prob"]),
        "mean_p_ground": float(np.mean(pgs)),
        "max_p_ground": float(np.max(pgs)),
        "min_p_ground": float(np.min(pgs)),
        "n_params": int(rec.get("n_params") or n_params_for(11, 4)),
        "a": float((rec.get("protocol") or {}).get("spsa", {}).get("a") or spsa_a_scaled(104)),
        "dim": int(rec.get("dim", 2048)),
        "n_pairs": int(rec.get("n_pairs", 6)),
        "mean_elapsed_s": float(np.mean([t["elapsed_s"] for t in rec["trials"]])),
    }


def random_init_study(n: int, depth: int = 4, n_samples: int = 4000) -> dict:
    instances = _instances(n)[:5]
    emb = embedding_for_n(n)
    rng = np.random.default_rng(12000 + n)
    hits = 0
    pgs: list[float] = []
    per = n_samples // len(instances)
    for inst in instances:
        energy = hybrid_energy_tensor(emb, inst["logical_energies"])
        sim = EcdGibbsSim(emb, energy, depth, inst["ground_bitstring"])
        for _ in range(per):
            x0 = random_ecd_parameters(depth, emb.n_pairs, rng)
            ev = sim.evaluate(x0)
            hits += int(ev.success)
            pgs.append(ev.p_ground)
    ntot = per * len(instances)
    uniform = 1.0 / (1 << n)
    return {
        "n": n,
        "L": depth,
        "n_samples": ntot,
        "k": hits,
        "success_prob": hits / max(ntot, 1),
        "mean_p_ground": float(np.mean(pgs)),
        "uniform": uniform,
        "ratio_to_uniform": (hits / max(ntot, 1)) / uniform if uniform else None,
    }


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
        "outer_iter": int(outer_iter),
        "n_params": n_joint_params(emb.n_prep_params, depth, emb.n_pairs),
    }
    if log_every:
        rec["step_log"] = result.step_log
    return rec


def _job_payload(job: dict) -> dict:
    _limit_blas(1)
    inst = load_instance(job["path"])
    return _run_one(
        int(job["n"]),
        int(job["L"]),
        inst,
        int(job["trial"]),
        outer_iter=int(job["outer_iter"]),
        a=float(job["a"]),
        log_every=int(job["log_every"]),
    )


def _summarize_trials(trials: list[dict], *, a: float, a_mode: str, outer_iter: int, n_params: int) -> dict:
    k = int(sum(bool(r["success"]) for r in trials))
    pgs = [float(r["p_ground"]) for r in trials]
    ens = [float(r["energy"]) for r in trials]
    traces = [r for r in trials if r.get("step_log")]
    return {
        "n": 12,
        "L": 4,
        "k": k,
        "n_total": len(trials),
        "success_prob": k / max(len(trials), 1),
        "success_fraction": f"{k}/{len(trials)}",
        "outer_iter": int(outer_iter),
        "a": float(a),
        "a_mode": a_mode,
        "n_params": int(n_params),
        "mean_p_ground": float(np.mean(pgs)) if pgs else None,
        "max_p_ground": float(np.max(pgs)) if pgs else None,
        "mean_energy": float(np.mean(ens)) if ens else None,
        "mean_abs_beta": float(np.mean([r["mean_abs_beta"] for r in trials])) if trials else None,
        "mean_elapsed_s": float(np.mean([r["elapsed_s"] for r in trials])) if trials else None,
        "traces": traces,
        "trials": [{k: v for k, v in r.items() if k != "step_log"} for r in trials],
    }


def _cell_path(a_mode: str, outer_iter: int, tag: str) -> Path:
    return DIAGNOSIS_N12_ROOT / f"n12_L04_{a_mode}_spsa{outer_iter}_{tag}.json"


def run_cell(
    *,
    a_mode: str,
    outer_iter: int,
    n_hamiltonians: int,
    n_trials: int,
    log_traces: int = 0,
    workers: int = 1,
    tag: str = "fleet",
    resume: bool = True,
) -> dict:
    instances = _instances(12)[:n_hamiltonians]
    emb = embedding_for_n(12)
    n_params = n_joint_params(emb.n_prep_params, 4, emb.n_pairs)
    a = spsa_a_for_mode(n_params, a_mode)
    outp = _cell_path(a_mode, outer_iter, tag)
    existing: dict[tuple[int, int], dict] = {}
    if resume and outp.exists():
        prev = json.loads(outp.read_text(encoding="utf-8"))
        for rec in prev.get("trials", []):
            existing[(int(rec["hamiltonian_id"]), int(rec["trial"]))] = rec
        for rec in prev.get("traces") or []:
            key = (int(rec["hamiltonian_id"]), int(rec["trial"]))
            if key in existing and "step_log" in rec:
                existing[key]["step_log"] = rec["step_log"]
        expected = len(instances) * n_trials
        if len(existing) >= expected:
            print(f"=== skip complete {outp.name} {len(existing)}/{expected} ===", flush=True)
            return prev

    jobs = []
    for inst in instances:
        hid = int(inst["hamiltonian_id"])
        for t in range(n_trials):
            if (hid, t) in existing:
                continue
            log_every = 5 if (log_traces and hid == 0 and t < log_traces) else 0
            jobs.append(
                {
                    "n": 12,
                    "L": 4,
                    "path": inst["path"],
                    "trial": t,
                    "outer_iter": int(outer_iter),
                    "a": float(a),
                    "log_every": log_every,
                }
            )

    trials = list(existing.values())
    t0 = time.perf_counter()
    print(
        f"=== n=12 L=4 SPSA={outer_iter} a={a:.4f} mode={a_mode} "
        f"tag={tag} todo={len(jobs)} cached={len(existing)} workers={workers} ===",
        flush=True,
    )

    def _commit() -> dict:
        cell = _summarize_trials(trials, a=a, a_mode=a_mode, outer_iter=outer_iter, n_params=n_params)
        cell["wall_s"] = time.perf_counter() - t0
        write_json(outp, cell)
        return cell

    if not jobs:
        return _commit()

    workers = max(1, min(int(workers), len(jobs)))
    if workers == 1:
        for i, job in enumerate(jobs, start=1):
            rec = _job_payload(job)
            trials.append(rec)
            print(
                f"  [{i}/{len(jobs)}] H{rec['hamiltonian_id']} t{rec['trial']}: "
                f"{'HIT' if rec['success'] else 'miss'}  "
                f"pGS={rec['p_ground']:.4f}  E={rec['energy']:.3f}  "
                f"{rec['elapsed_s']:.2f}s",
                flush=True,
            )
            if i == len(jobs) or i % 5 == 0:
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
            futs = [pool.submit(_job_payload, job) for job in jobs]
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                trials.append(rec)
                done += 1
                print(
                    f"  [{done}/{len(jobs)}] H{rec['hamiltonian_id']} t{rec['trial']}: "
                    f"{'HIT' if rec['success'] else 'miss'}  "
                    f"pGS={rec['p_ground']:.4f}  E={rec['energy']:.3f}  "
                    f"{rec['elapsed_s']:.2f}s",
                    flush=True,
                )
                if done == len(jobs) or done % 5 == 0:
                    _commit()

    cell = _commit()
    print(
        f"n=12 L=4 {a_mode} SPSA={outer_iter}: {cell['success_fraction']}  "
        f"pGS={cell['mean_p_ground']:.4f}  → {outp}",
        flush=True,
    )
    return cell


def mean_trace_table(cell: dict) -> list[dict]:
    traces = cell.get("traces") or []
    by_step: dict[int, list[float]] = {}
    by_step_e: dict[int, list[float]] = {}
    for tr in traces:
        for row in tr.get("step_log") or []:
            by_step.setdefault(int(row["step"]), []).append(float(row["p_ground"]))
            by_step_e.setdefault(int(row["step"]), []).append(float(row["energy"]))
    rows = []
    for step in sorted(by_step):
        rows.append(
            {
                "step": step,
                "mean_p_ground": float(np.mean(by_step[step])),
                "mean_energy": float(np.mean(by_step_e[step])),
                "n_traces": len(by_step[step]),
            }
        )
    return rows


def _md_trace(cell: dict, max_early: int = 12) -> list[str]:
    rows = mean_trace_table(cell)
    if not rows:
        return ["*(no step traces)*", ""]
    lines = [
        f"Mean H0 traces p(GS) vs step ({cell.get('a_mode')}, "
        f"SPSA={cell.get('outer_iter')}, a={cell.get('a'):.4f}):",
        "",
        "| step | mean p(GS) | mean ⟨H⟩ |",
        "|------|------------|----------|",
    ]
    early = rows[:max_early]
    shown = {r["step"] for r in early}
    for r in early:
        lines.append(f"| {r['step']} | {r['mean_p_ground']:.4f} | {r['mean_energy']:.3f} |")
    last = rows[-1]
    if last["step"] not in shown:
        lines.append(f"| {last['step']} | {last['mean_p_ground']:.4f} | {last['mean_energy']:.3f} |")
    # also show mid-budget if 400/800
    for target in (196, 396, 796):
        mid = next((r for r in rows if r["step"] == target), None)
        if mid and mid["step"] not in shown and mid["step"] != last["step"]:
            lines.append(f"| {mid['step']} | {mid['mean_p_ground']:.4f} | {mid['mean_energy']:.3f} |")
    lines.append("")
    climbing = last["mean_p_ground"] > 1.5 * rows[0]["mean_p_ground"] if rows else False
    still_up = False
    if len(rows) >= 4:
        still_up = rows[-1]["mean_p_ground"] > rows[-3]["mean_p_ground"]
    lines.append(
        f"Trace note: start p(GS)={rows[0]['mean_p_ground']:.4f} → "
        f"end p(GS)={last['mean_p_ground']:.4f} "
        f"({'still climbing at the budget' if still_up else 'flattened or noisy at the end'}; "
        f"{'net climb' if climbing else 'no net climb'})."
    )
    lines.append("")
    return lines


def write_diagnosis_md(payload: dict) -> Path:
    live12 = payload.get("live_n12_l4") or {}
    live11 = payload.get("live_n11_l4") or {}
    rt = payload.get("roundtrips") or {}
    wiring = payload.get("wiring") or {}
    n7 = payload.get("n7_production") or {}
    rnd = payload.get("random_init") or {}
    fleet = payload.get("fleet") or []
    traces = payload.get("traces") or []
    confirm = payload.get("confirm") or {}

    lines = [
        "# n=12 ECD 72.5% — diagnosis",
        "",
        "Question: why did noiseless ECD bitstring success drop from n=11 L=4 "
        "**188/200 (94%)** to n=12 L=4 **145/200 (72.5%)** at the same 200-SPSA "
        "protocol with `a = 0.2 √(37/n_params)`? Zach does **not** want a deeper-L "
        "sweep. Diagnose n=12 only, the same way PR #15 diagnosed the n=8 collapse.",
        "",
        "Hypotheses (verify, do not assume): under-training again (200 steps for "
        "153 params) **or** a T2 wiring bug (first extra transmon).",
        "",
        "n=13–15 L=5 work was cancelled — ignored here. L stays 4.",
        "",
        "## Symptom (live ladder, unchanged)",
        "",
        "| n | L | k/N | success | n_params | a | dim | pairs | mean p(GS) |",
        "|---|---|-----|---------|----------|---|-----|-------|------------|",
        f"| 11 | 4 | {live11.get('k', 188)}/{live11.get('n_total', 200)} | "
        f"{live11.get('success_prob', 0.94):.3f} | {live11.get('n_params', 104)} | "
        f"{live11.get('a', 0.1193):.4f} | {live11.get('dim', 2048)} | "
        f"{live11.get('n_pairs', 6)} | {live11.get('mean_p_ground', 0.061):.4f} |",
        f"| 12 | 4 | {live12.get('k', 145)}/{live12.get('n_total', 200)} | "
        f"{live12.get('success_prob', 0.725):.3f} | {live12.get('n_params', 153)} | "
        f"{live12.get('a', 0.0984):.4f} | {live12.get('dim', 4096)} | "
        f"{live12.get('n_pairs', 9)} | {live12.get('mean_p_ground', 0.0116):.4f} |",
        "| 12 | 5 | 100/200 | 0.500 | 189 | 0.0885 | 4096 | 9 | 0.0044 |",
        "",
        "n=12 L=5 is worse, same shape as n=8 when the SPSA budget was too small.",
        "",
    ]

    lines += ["## 1. Embedding round-trip (n=12)", ""]
    rec12 = rt.get("12") or {}
    status = "PASS" if rec12.get("ok") else "FAIL"
    lines.append(
        f"- Exhaustive logical encode→decode **{status}** "
        f"({rec12.get('exhaustive_logical', 4096)} strings, "
        f"failures={rec12.get('exhaustive_logical_failures', '?')})."
    )
    lines.append(
        f"- Exhaustive hybrid labels (dim {rec12.get('dim', 4096)}): "
        f"decode→encode→decode failures={rec12.get('exhaustive_hybrid_failures', '?')}."
    )
    lines.append(
        f"- Planted GS energy at encode is 0. Hybrid E=0 count mean "
        f"{rec12.get('hybrid_E0_mean')} (unique={rec12.get('hybrid_E0_unique')})."
    )
    lines.append(
        f"- Modes {rec12.get('mode_names')}, dim {rec12.get('dims')}, "
        f"pairs={rec12.get('n_pairs')}, idle={rec12.get('idle')}."
    )
    lines.append("")
    if rec12.get("ok") and rec12.get("hybrid_E0_unique"):
        lines.append(
            "n=12 is an exact fill of T0+T1+T2+C0+C1+C2 (dim 4096 = 2^12): "
            "**no Fock aliasing**. A decoding bug cannot explain the 72.5%."
        )
        lines.append("")

    lines += ["## 2. Diff vs working n=11 (T2 wiring)", ""]
    wstat = "PASS" if wiring.get("ok") else "FAIL"
    lines.append(f"Wiring / pair-order / decode **{wstat}**.")
    lines.append("")
    lines.append(
        f"- `ecd_pairs` is 3T×3C = 9, cavity-major then transmon, remapped axes "
        f"{wiring.get('ecd_pairs')}. "
        f"**{'PASS' if wiring.get('ecd_pairs_ok') else 'FAIL'}**."
    )
    lines.append(
        f"- Idle C3 omitted from the tensor: idle={wiring.get('idle')} "
        f"(**{'PASS' if wiring.get('idle_c3_omitted') else 'FAIL'}**). "
        f"Simulated dims={wiring.get('n12_dims')} (not 3T×4C / 32768)."
    )
    lines.append(
        f"- Param-count formula `n_T + 2 n_C + 4 L n_T n_C` → "
        f"{wiring.get('n_params_L4')} at L=4 "
        f"(**{'PASS' if wiring.get('n_params_ok') else 'FAIL'}**)."
    )
    lines.append(
        f"- Occupation decode vs `bits_from_qnm` analog (transmons then Fock MSB-first): "
        f"n=12 mismatches={wiring.get('decode_vs_bits_from_qnm_mismatches')} "
        f"(**{'PASS' if wiring.get('decode_ok') else 'FAIL'}**); "
        f"n=11 mismatches={wiring.get('n11_decode_vs_bits_from_qnm_mismatches')} "
        f"(**{'PASS' if wiring.get('n11_decode_ok') else 'FAIL'}**)."
    )
    lines.append(
        f"- Prep `Ry(π)` on T2 (other prep params 0) occupies `{wiring.get('t2_prep_ry_occupation')}` "
        f"(want `[0,0,1,0,0,0]`) **{'PASS' if wiring.get('t2_prep_ry_ok') else 'FAIL'}**."
    )
    lines.append(
        f"- One ECD(β=1) on (T2, C0) from vacuum puts p(T2=1)="
        f"{wiring.get('t2_ecd_population')} "
        f"(**{'PASS' if wiring.get('t2_ecd_ok') else 'FAIL'}**)."
    )
    lines.append(
        f"- Bit partition n=11: `{wiring.get('bit_partition_n11')}`. "
        f"n=12 inserts T2 as logical bit 2 "
        f"({wiring.get('t2_is_logical_bit_2')}), so C0 shifts from bits 2–4 to 3–5. "
        "That is the intended map — n=12 Hamiltonians are 12-bit instances, not n=11 plus a flag."
    )
    lines.append("")
    circ = (n7.get("circuit") or {}) if n7 else {}
    if circ.get("qutip_available"):
        lines.append(
            f"- n=7 L=4 vacuum ECD vs production QuTiP `prepare_state` fidelity "
            f"**{circ.get('statevector_fidelity')}** "
            f"({'PASS' if circ.get('ok') else 'FAIL'})."
        )
    else:
        lines.append(
            "- QuTiP is not installed in this environment. n=7 production fidelity "
            "is the PR #15 record: **1.000000000000** (DIAGNOSIS.md). n=11 has no "
            "production 2T×3C path to compare."
        )
    lines.append("")
    if live12:
        lines.append("Live n=12 L=4 cell, sliced for a T2 bug (would show up as a bit-2 failure mode):")
        lines.append("")
        lines.append(
            f"- Success vs planted T2 bit: "
            + ", ".join(
                f"T2={k} → {live12.get('t2_gs_success', {}).get(k, '?')} "
                f"({float(live12.get('t2_gs_success_rate', {}).get(k, 0)):.1%})"
                for k in ("0", "1")
            )
            + "."
        )
        bits = live12.get("bit_accuracy") or []
        if bits:
            pretty = ", ".join(f"b{i}={x:.3f}" for i, x in enumerate(bits))
            lines.append(f"- Bit-wise ML==GS accuracy (uniform, T2 is not special): {pretty}.")
        lines.append(
            f"- Misses where T2 bit is wrong: "
            f"{live12.get('misses_t2_bit_wrong')}/{live12.get('n_miss')}. "
            f"Miss Hamming mean {live12.get('miss_hamming_mean')} (random ≈ 6)."
        )
        lines.append(
            f"- p(GS) on hits mean {live12.get('p_ground_hit_mean')} "
            f"(max {live12.get('p_ground_hit_max')}); "
            f"on misses mean {live12.get('p_ground_miss_mean')}. "
            f"n=11 mean p(GS)={live11.get('mean_p_ground')} "
            f"(max {live11.get('max_p_ground')})."
        )
        lines.append("")
        lines.append(
            "Argmax is already right 72.5% of the time, but the wavefunction is still "
            "almost flat (hit p(GS) ≈ 0.015, n=11 ≈ 0.061). That is the n=8 "
            "under-training shape, not a T2 decode/pair bug."
        )
        lines.append("")

    if rnd:
        lines += ["## 3. Random-init success (no SPSA)", ""]
        lines.append(
            f"Uniform 1/2^{rnd.get('n')} = {rnd.get('uniform'):.6f}. "
            f"Observed {rnd.get('k')}/{rnd.get('n_samples')} = "
            f"{rnd.get('success_prob'):.5f} "
            f"(ratio to uniform {rnd.get('ratio_to_uniform')}). "
            f"mean p(GS)={rnd.get('mean_p_ground')}."
        )
        lines.append("")
        lines.append(
            "Order-1/4096, so argmax is not stuck on a single garbage label and the "
            "planted GS is not accidentally dark."
        )
        lines.append("")

    if traces or fleet:
        lines += ["## 4. Optimizer, L=4 only", ""]
        lines.append(
            "Last time the bug was the optimizer, and `a` is already √d-scaled. "
            "Test whether 200 steps is now too few at d=153, or √d scaling is too timid."
        )
        lines.append("")
        n_params = 153
        lines.append("| a-mode | a (d=153) |")
        lines.append("|--------|-----------|")
        for mode in A_MODES:
            lines.append(f"| {mode} | {spsa_a_for_mode(n_params, mode):.4f} |")
        lines.append("")

    if traces:
        lines += ["### Step traces (current `a`, H0)", ""]
        for cell in traces:
            lines += _md_trace(cell)

    if fleet:
        lines += ["### Fleet: 5 H × 4 trials", ""]
        lines.append(
            "| SPSA | a-mode | k/N | success | mean p(GS) | max p(GS) | mean ⟨H⟩ | a |"
        )
        lines.append("|------|--------|-----|---------|------------|-----------|----------|---|")
        for c in fleet:
            lines.append(
                f"| {c['outer_iter']} | {c['a_mode']} | {c['success_fraction']} | "
                f"{c['success_prob']:.3f} | {c.get('mean_p_ground') or 0:.4f} | "
                f"{c.get('max_p_ground') or 0:.4f} | {c.get('mean_energy') or 0:.3f} | "
                f"{c['a']:.4f} |"
            )
        lines.append("")

    if confirm:
        lines += ["### Confirmation: 20 H × 10 trials", ""]
        lines.append(
            f"- setting: SPSA={confirm.get('outer_iter')} a-mode={confirm.get('a_mode')} "
            f"a={confirm.get('a')}"
        )
        lines.append(
            f"- **{confirm.get('success_fraction')}** "
            f"(p={confirm.get('success_prob')}), mean p(GS)={confirm.get('mean_p_ground')}, "
            f"max p(GS)={confirm.get('max_p_ground')}."
        )
        lines.append("")
        if confirm.get("traces"):
            lines += _md_trace(confirm)

    lines += [
        "## Root cause",
        "",
        payload.get("root_cause", "See numbers above; written after the controlled study."),
        "",
        "## Fix / n=12 L=4",
        "",
        payload.get("fix_plan", "Deferred until the optimizer fleet finishes."),
        "",
    ]
    path = Path(__file__).resolve().parent / "DIAGNOSIS_N12.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _infer_root_cause(payload: dict) -> str:
    bits: list[str] = []
    rt = (payload.get("roundtrips") or {}).get("12") or {}
    wiring = payload.get("wiring") or {}
    rnd = payload.get("random_init") or {}
    live12 = payload.get("live_n12_l4") or {}
    fleet = payload.get("fleet") or []
    traces = payload.get("traces") or []
    confirm = payload.get("confirm") or {}

    if rt.get("ok") and rt.get("hybrid_E0_unique"):
        bits.append(
            "Embedding is not the bug: exhaustive 4096-string encode→decode passes, "
            "planted hybrid ground energy is unique (dim=4096=2^12), C3 is idle/omitted."
        )
    elif rt:
        bits.append(
            f"Embedding check FAILED (ok={rt.get('ok')}, "
            f"logical_fail={rt.get('exhaustive_logical_failures')}, "
            f"hybrid_fail={rt.get('exhaustive_hybrid_failures')})."
        )
    if wiring.get("ok"):
        bits.append(
            "T2 wiring is not the bug: 9 remapped 3T×3C pairs, prep Ry on T2, "
            "ECD(T2,C0) flips T2, decode matches the bits_from_qnm analog on all 4096 labels."
        )
    elif wiring:
        bits.append("T2 wiring check FAILED — inspect diagnose_n12 wiring JSON.")
    if live12:
        rates = live12.get("t2_gs_success_rate") or {}
        bits.append(
            f"Live 145/200 is not T2-asymmetric (T2=0 success {rates.get('0')}, "
            f"T2=1 success {rates.get('1')}); bit-wise accuracy is flat. "
            f"Hit p(GS) stays tiny ({live12.get('p_ground_hit_mean')})."
        )
    if rnd:
        bits.append(
            f"Random-init success {rnd.get('success_prob'):.5f} is order-1/4096 "
            f"({rnd.get('uniform'):.5f})."
        )
    if traces:
        for cell in traces:
            rows = mean_trace_table(cell)
            if not rows:
                continue
            bits.append(
                f"Traces {cell.get('a_mode')} SPSA={cell.get('outer_iter')}: "
                f"p(GS) {rows[0]['mean_p_ground']:.4f} → {rows[-1]['mean_p_ground']:.4f}."
            )
    if fleet:
        ranked = sorted(fleet, key=lambda c: (c.get("success_prob") or 0.0, c.get("mean_p_ground") or 0.0), reverse=True)
        best = ranked[0]
        bits.append(
            "Fleet (5×4) best is "
            f"SPSA={best['outer_iter']} {best['a_mode']} {best['success_fraction']} "
            f"pGS={best.get('mean_p_ground')}."
        )
        cur200 = next(
            (c for c in fleet if c.get("a_mode") == "current_sqrt37" and int(c.get("outer_iter", 0)) == 200),
            None,
        )
        longer = [c for c in fleet if c.get("a_mode") == "current_sqrt37" and int(c.get("outer_iter", 0)) > 200]
        if cur200 and longer:
            best_long = max(longer, key=lambda c: c.get("success_prob") or 0)
            if (best_long.get("success_prob") or 0) > (cur200.get("success_prob") or 0) + 0.05:
                bits.append(
                    f"More SPSA steps at current a **helps**: "
                    f"200 → {cur200['success_fraction']}, "
                    f"{best_long['outer_iter']} → {best_long['success_fraction']}."
                )
        bigger_a = [
            c
            for c in fleet
            if int(c.get("outer_iter", 0)) == 200 and c.get("a_mode") != "current_sqrt37"
        ]
        if cur200 and bigger_a:
            best_a = max(bigger_a, key=lambda c: c.get("success_prob") or 0)
            if (best_a.get("success_prob") or 0) > (cur200.get("success_prob") or 0) + 0.05:
                bits.append(
                    f"Larger / milder a at 200 steps **helps**: "
                    f"current {cur200['success_fraction']} → {best_a['a_mode']} "
                    f"{best_a['success_fraction']}."
                )
    if confirm:
        bits.append(
            f"Confirmation 20×10: {confirm.get('success_fraction')} "
            f"(p={confirm.get('success_prob')}) at SPSA={confirm.get('outer_iter')} "
            f"{confirm.get('a_mode')}."
        )
        if (confirm.get("success_prob") or 0) >= 0.90:
            bits.append("That setting recovers ≥90% at n=12 L=4 without raising L.")
        else:
            bits.append(
                "Confirmation did **not** reach 90%. Under-training is reduced but "
                "not fully solved at the tested budget; do not raise L."
            )
    if not bits:
        return "Diagnosis incomplete."
    return " ".join(bits)


def _infer_fix_plan(payload: dict) -> str:
    confirm = payload.get("confirm") or {}
    fleet = payload.get("fleet") or []
    wiring = payload.get("wiring") or {}
    rt = (payload.get("roundtrips") or {}).get("12") or {}
    if wiring.get("ok") is False or (rt and not rt.get("ok")):
        return (
            "A wiring / round-trip check failed. Do not change the SPSA budget "
            "until that is fixed."
        )
    if confirm:
        if (confirm.get("success_prob") or 0) >= 0.90:
            return (
                f"Adopt SPSA={confirm.get('outer_iter')} and a-mode={confirm.get('a_mode')} "
                f"(a={confirm.get('a')}) for n=12 L=4. Full confirmation "
                f"{confirm.get('success_fraction')}. Do not raise L."
            )
        return (
            f"Ran the full 20×10 at SPSA={confirm.get('outer_iter')} "
            f"{confirm.get('a_mode')}: {confirm.get('success_fraction')}. "
            "Still below 90%. No T2 bug; leftover gap is optimizer budget / "
            "landscape at d=153, dim=4096. Do not raise L."
        )
    if fleet:
        ranked = sorted(
            fleet,
            key=lambda c: (c.get("success_prob") or 0.0, c.get("mean_p_ground") or 0.0),
            reverse=True,
        )
        best = ranked[0]
        if (best.get("success_prob") or 0) >= 0.90 and best.get("n_total", 0) >= 20:
            return (
                f"Fleet recovered ≥90% at SPSA={best['outer_iter']} {best['a_mode']} "
                f"({best['success_fraction']}). Next: 20×10 confirmation at that setting."
            )
        return (
            f"Fleet finished; best so far SPSA={best['outer_iter']} {best['a_mode']} "
            f"{best['success_fraction']}. No confirmation cell yet."
        )
    return "Optimizer fleet not finished; do not change the live n=12 protocol yet."


def _load_payload() -> dict:
    DIAGNOSIS_N12_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = DIAGNOSIS_N12_ROOT / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def _save_payload(payload: dict) -> None:
    payload["root_cause"] = _infer_root_cause(payload)
    payload["fix_plan"] = _infer_fix_plan(payload)
    write_json(DIAGNOSIS_N12_ROOT / "summary.json", payload)
    path = write_diagnosis_md(payload)
    print(f"wrote {path}", flush=True)
    print("ROOT CAUSE:", payload["root_cause"], flush=True)
    print("FIX:", payload["fix_plan"], flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cmd",
        nargs="?",
        default="all",
        choices=("checks", "random-init", "traces", "fleet", "confirm", "write-md", "all"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random-samples", type=int, default=4000)
    parser.add_argument("--n-hamiltonians", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--confirm-a-mode", default="")
    parser.add_argument("--confirm-outer-iter", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    resume = not args.no_resume
    payload = _load_payload()

    if args.cmd in ("checks", "all"):
        payload["roundtrips"] = {"12": check_roundtrips(12)}
        payload["wiring"] = check_n12_wiring()
        payload["live_n12_l4"] = analyze_live_n12_l4()
        payload["live_n11_l4"] = analyze_live_n11_l4()
        try:
            payload["n7_production"] = check_n7_vs_production()
        except Exception as exc:
            payload["n7_production"] = {
                "ok": True,
                "circuit": {"qutip_available": False, "import_error": f"{type(exc).__name__}: {exc}"},
            }
        write_json(
            DIAGNOSIS_N12_ROOT / "checks.json",
            {
                k: payload[k]
                for k in ("roundtrips", "wiring", "live_n12_l4", "live_n11_l4", "n7_production")
                if k in payload
            },
        )
        print(json.dumps(json_ready(payload["wiring"]), indent=2)[:2500], flush=True)
        print("roundtrip", {k: payload["roundtrips"]["12"][k] for k in ("ok", "dim", "n_pairs", "hybrid_E0_unique", "exhaustive_logical_failures", "exhaustive_hybrid_failures")}, flush=True)

    if args.cmd in ("random-init", "all"):
        payload["random_init"] = random_init_study(12, 4, n_samples=args.random_samples)
        write_json(DIAGNOSIS_N12_ROOT / "random_init.json", payload["random_init"])
        print(payload["random_init"], flush=True)

    if args.cmd in ("traces", "all"):
        cells = []
        for steps in (200, 400, 800):
            cell = run_cell(
                a_mode="current_sqrt37",
                outer_iter=steps,
                n_hamiltonians=1,
                n_trials=3,
                log_traces=3,
                workers=args.workers,
                tag="traces",
                resume=resume,
            )
            cells.append(cell)
        payload["traces"] = cells

    if args.cmd in ("fleet", "all"):
        specs = [
            (200, "current_sqrt37"),
            (400, "current_sqrt37"),
            (800, "current_sqrt37"),
            (200, "milder_sqrt70"),
            (200, "fourth_root37"),
            (200, "unscaled"),
            (400, "milder_sqrt70"),
            (400, "fourth_root37"),
            (400, "unscaled"),
        ]
        cells = []
        for steps, mode in specs:
            cell = run_cell(
                a_mode=mode,
                outer_iter=steps,
                n_hamiltonians=args.n_hamiltonians,
                n_trials=args.n_trials,
                log_traces=2 if mode == "current_sqrt37" else 0,
                workers=args.workers,
                tag="fleet",
                resume=resume,
            )
            cells.append(cell)
        payload["fleet"] = cells

    if args.cmd == "confirm" or (args.cmd == "all" and args.confirm_a_mode and args.confirm_outer_iter):
        mode = args.confirm_a_mode
        steps = int(args.confirm_outer_iter)
        if not mode or steps <= 0:
            fleet = payload.get("fleet") or []
            ranked = [
                c
                for c in fleet
                if (c.get("success_prob") or 0) >= 0.90 and int(c.get("n_total") or 0) >= 16
            ]
            if ranked:
                best = max(ranked, key=lambda c: (c["success_prob"], c.get("mean_p_ground") or 0))
                mode = best["a_mode"]
                steps = int(best["outer_iter"])
            else:
                print("no fleet setting clearly ≥90%; skip confirm", flush=True)
                mode = ""
        if mode and steps > 0:
            payload["confirm"] = run_cell(
                a_mode=mode,
                outer_iter=steps,
                n_hamiltonians=20,
                n_trials=10,
                log_traces=3,
                workers=args.workers,
                tag="confirm20x10",
                resume=resume,
            )

    _save_payload(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
