#!/usr/bin/env python3
"""Cache-replay microbench for round-2 GDR ideas.

Loads existing physics caches (never ``out/`` / ``out_smoke/``). Fit-only:
no new density-matrix pass unless a cell's cache is missing.

Usage (repo root)::

    python -u Error_mitigation/run_round2.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from qumode_vqe.measurement import energy_from_histogram
from qumode_vqe.noise import noise_as_dict

from Error_mitigation.metrics import compare_histograms
from Error_mitigation.noise_models import circuit_noise, readout_as_dict, readout_spec
from Error_mitigation.run_ablation import (
    ROUND2_METHODS,
    STAGE_B_METHODS,
    STAGE_C_METHODS,
    SHOT_METHODS,
    _observe_block,
    load_cache,
    mitigate_research,
)
from Error_mitigation.run_mitigation_experiment import (
    ANSATZ_SPEC,
    DIMS,
    SEED_BASE,
    json_ready,
    load_instance,
)

HERE = Path(__file__).resolve().parent
RESEARCH = HERE / "out_research"
ROUND2_OUT = RESEARCH / "round2"
CACHE_PR8 = RESEARCH / "cache"
CACHE_MH = RESEARCH / "multi_h" / "cache"

# Adaptive-recipe cells. ``protect``: must not regress vs same-run gdr_select.
HARD_CELLS = (
    {
        "id": "ecd_rand_loss_0.1",
        "ansatz": "ecd",
        "params": "random",
        "family": "loss",
        "kappa_tau": 0.1,
        "readout": "ideal",
        "instance": 0,
        "cache": CACHE_PR8 / "ecd_random_loss_kt0.1_n40_span_nr10_lo0.25_hi1.35_x0.npz",
        "protect": False,
    },
    {
        "id": "ecd_rand_comp_0.1",
        "ansatz": "ecd",
        "params": "random",
        "family": "comprehensive",
        "kappa_tau": 0.1,
        "readout": "ideal",
        "instance": 0,
        "cache": CACHE_PR8 / "ecd_random_comprehensive_kt0.1_n40_span_nr10_lo0.25_hi1.35_x0.npz",
        "protect": False,
    },
    {
        "id": "ecd_opt_comp_0.1",
        "ansatz": "ecd",
        "params": "optimized",
        "family": "comprehensive",
        "kappa_tau": 0.1,
        "readout": "ideal",
        "instance": 0,
        "cache": CACHE_PR8 / "ecd_optimized_comprehensive_kt0.1_n40_default_nr10_lo0.25_hi1.35_x0.npz",
        "protect": True,
        "note": "PR #8 0.343 cell — never regress",
    },
    {
        "id": "snap_rand_comp_0.003",
        "ansatz": "snap",
        "params": "random",
        "family": "comprehensive",
        "kappa_tau": 0.003,
        "readout": "ideal",
        "instance": 0,
        "cache": CACHE_PR8 / "snap_random_comprehensive_kt0.003_n40_span_nr10_lo0.25_hi1.35_x0.npz",
        "protect": True,
        "note": "gated comprehensive floor",
    },
    {
        "id": "h000_opt_comp_rr_0.003",
        "ansatz": "ecd",
        "params": "optimized",
        "family": "comprehensive",
        "kappa_tau": 0.003,
        "readout": "readout_realistic",
        "instance": 0,
        "cache": CACHE_MH / "ecd_optimized_comprehensive_kt0.003_n40_default_nr10_lo0.25_hi1.35_x0.npz",
        "protect": True,
        "note": "multi-H H000 mild realistic",
    },
    {
        "id": "h004_opt_comp_rr_0.003",
        "ansatz": "ecd",
        "params": "optimized",
        "family": "comprehensive",
        "kappa_tau": 0.003,
        "readout": "readout_realistic",
        "instance": 4,
        "cache": CACHE_MH / "ecd_optimized_comprehensive_kt0.003_n40_default_nr10_lo0.25_hi1.35_x0_h004.npz",
        "protect": True,
        "note": "multi-H H004 mild realistic",
    },
    {
        "id": "h009_opt_comp_rr_0.003",
        "ansatz": "ecd",
        "params": "optimized",
        "family": "comprehensive",
        "kappa_tau": 0.003,
        "readout": "readout_realistic",
        "instance": 9,
        "cache": CACHE_MH / "ecd_optimized_comprehensive_kt0.003_n40_default_nr10_lo0.25_hi1.35_x0_h009.npz",
        "protect": True,
        "note": "multi-H H009 mild realistic",
    },
    {
        "id": "ecd_rand_comp_0.003",
        "ansatz": "ecd",
        "params": "random",
        "family": "comprehensive",
        "kappa_tau": 0.003,
        "readout": "ideal",
        "instance": 0,
        "cache": CACHE_PR8 / "ecd_random_comprehensive_kt0.003_n40_span_nr10_lo0.25_hi1.35_x0.npz",
        "protect": False,
    },
)

NEW_METHODS = (
    "gdr_anneal",
    "gdr_eta",
    "gdr_fisher",
    "gdr_select_kt",
    "gdr_mild_residual",
    "readout_then_gdr",
    "gdr_then_rtz",
)

STAGE_B_NEW = ("gdr_ensemble", "gdr_joint")
STAGE_C_NEW = ("gdr_family_eta", "gdr_shot_damp", "gdr_rl", "gdr_rl_soft", "gdr_rl_stop")
SHOT_NEW = ("gdr_shot_damp",)

SNAP_CELLS = (
    {
        "id": "snap_opt_comp_rr_0.003",
        "ansatz": "snap",
        "params": "optimized",
        "family": "comprehensive",
        "kappa_tau": 0.003,
        "readout": "readout_realistic",
        "instance": 0,
        "cache": CACHE_PR8 / "snap_optimized_comprehensive_kt0.003_n40_default_nr10_lo0.25_hi1.35_x0.npz",
        "protect": False,
        "note": "SNAP Nd=2 H000 opt comprehensive+realistic; deficit~1.93 not near-E0",
    },
    {
        "id": "snap_opt_comp_rr_0.03",
        "ansatz": "snap",
        "params": "optimized",
        "family": "comprehensive",
        "kappa_tau": 0.03,
        "readout": "readout_realistic",
        "instance": 0,
        "cache": CACHE_PR8 / "snap_optimized_comprehensive_kt0.03_n40_default_nr10_lo0.25_hi1.35_x0.npz",
        "protect": False,
        "note": "SNAP Nd=2 H000 opt comprehensive+realistic κτ=0.03",
    },
    {
        "id": "snap_opt_comp_rr_0.1",
        "ansatz": "snap",
        "params": "optimized",
        "family": "comprehensive",
        "kappa_tau": 0.1,
        "readout": "readout_realistic",
        "instance": 0,
        "cache": CACHE_PR8 / "snap_optimized_comprehensive_kt0.1_n40_default_nr10_lo0.25_hi1.35_x0.npz",
        "protect": False,
        "note": "SNAP Nd=2 H000 opt comprehensive+realistic κτ=0.1",
    },
)

# Four headline hard cells for the shot-count sweep (not a new method).
SHOT_CELLS = (
    HARD_CELLS[0],  # ecd_rand_loss_0.1
    HARD_CELLS[1],  # ecd_rand_comp_0.1
    HARD_CELLS[2],  # ecd_opt_comp_0.1  (0.343 protect)
    HARD_CELLS[3],  # snap_rand_comp_0.003 (gated floor)
)
SHOT_COUNTS = (2048, 8192, 32768)

# Extra comprehensive+readout_realistic cells at κτ=0.1 (tighter η prior).
FAMILY_EXTRA = (
    {
        "id": "h000_opt_comp_rr_0.1",
        "ansatz": "ecd",
        "params": "optimized",
        "family": "comprehensive",
        "kappa_tau": 0.1,
        "readout": "readout_realistic",
        "instance": 0,
        "cache": CACHE_MH / "ecd_optimized_comprehensive_kt0.1_n40_default_nr10_lo0.25_hi1.35_x0.npz",
        "protect": False,
        "note": "H000 opt comprehensive+realistic κτ=0.1 — tighter η prior",
    },
    {
        "id": "h004_opt_comp_rr_0.1",
        "ansatz": "ecd",
        "params": "optimized",
        "family": "comprehensive",
        "kappa_tau": 0.1,
        "readout": "readout_realistic",
        "instance": 4,
        "cache": CACHE_MH / "ecd_optimized_comprehensive_kt0.1_n40_default_nr10_lo0.25_hi1.35_x0_h004.npz",
        "protect": False,
        "note": "H004 opt comprehensive+realistic κτ=0.1",
    },
    {
        "id": "h009_opt_comp_rr_0.1",
        "ansatz": "ecd",
        "params": "optimized",
        "family": "comprehensive",
        "kappa_tau": 0.1,
        "readout": "readout_realistic",
        "instance": 9,
        "cache": CACHE_MH / "ecd_optimized_comprehensive_kt0.1_n40_default_nr10_lo0.25_hi1.35_x0_h009.npz",
        "protect": False,
        "note": "H009 opt comprehensive+realistic κτ=0.1",
    },
)

XFER_CELLS = (
    HARD_CELLS[4],  # h000 mild rr
    HARD_CELLS[5],  # h004
    HARD_CELLS[6],  # h009
)
BEAT_EPS = 0.005
REGRESS_EPS = 0.003


def score_methods(records: list[dict], new_methods: tuple[str, ...] = NEW_METHODS) -> dict:
    """Keep a method only if it beats select somewhere and never regresses protects."""
    protects = [r for r in records if r.get("protect")]
    summary = {}
    for name in new_methods:
        beats = []
        regressions = []
        deltas = []
        for rec in records:
            mets = rec["metrics"]
            sel = (mets.get("gdr_select") or {}).get("tvd")
            tvd = (mets.get(name) or {}).get("tvd")
            if sel is None or tvd is None:
                continue
            d = float(tvd) - float(sel)
            deltas.append({"id": rec["id"], "delta": d, "tvd": tvd, "select": sel})
            if d < -BEAT_EPS:
                beats.append(rec["id"])
            if rec.get("protect") and d > REGRESS_EPS:
                regressions.append({"id": rec["id"], "delta": d})
        keep = bool(beats) and not regressions
        summary[name] = {
            "keep": keep,
            "beats": beats,
            "regressions": regressions,
            "deltas": deltas,
            "mean_delta": float(np.mean([x["delta"] for x in deltas])) if deltas else None,
            "protect_max_delta": (
                None
                if not protects
                else max(
                    (
                        (mets.get(name) or {}).get("tvd", 0.0)
                        - (mets.get("gdr_select") or {}).get("tvd", 0.0)
                    )
                    for mets in (r["metrics"] for r in records if r.get("protect"))
                    if (mets.get(name) or {}).get("tvd") is not None
                    and (mets.get("gdr_select") or {}).get("tvd") is not None
                )
            ),
        }
    return summary


def write_scoreboard(
    path: Path,
    records: list[dict],
    verdict: dict,
    header: str,
    new_methods: tuple[str, ...] = NEW_METHODS,
) -> None:
    lines = ["# Round-2 microbench", "", header, ""]
    cols = ["id", "raw", "gdr_param", "gdr_select"] + list(new_methods)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    def f(x):
        return "—" if x is None else f"{float(x):.4f}"

    for rec in records:
        mets = rec["metrics"]
        row = [rec["id"]]
        for m in ("raw", "gdr_param", "gdr_select") + new_methods:
            row.append(f((mets.get(m) or {}).get("tvd")))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Keep / drop vs adaptive `gdr_select`", ""]
    lines.append("| method | keep? | beats | regressions | mean ΔTVD |")
    lines.append("|---|---|---|---|---:|")
    for name, info in verdict.items():
        beats = ", ".join(info["beats"]) or "—"
        regs = ", ".join(r["id"] for r in info["regressions"]) or "—"
        md = "—" if info["mean_delta"] is None else f"{info['mean_delta']:+.4f}"
        lines.append(f"| `{name}` | {'KEEP' if info['keep'] else 'drop'} | {beats} | {regs} | {md} |")
    path.write_text("\n".join(lines) + "\n")


def run_cell(cell: dict, *, shots: int, seed: int, fit_maxiter: int, methods: tuple[str, ...]) -> dict:
    cache_path = Path(cell["cache"])
    phys = load_cache(cache_path)
    if phys is None:
        raise FileNotFoundError(f"missing physics cache {cache_path}")
    hid = int(cell["instance"])
    inst = load_instance(hid)
    energy_tensor = np.asarray(inst["energy_tensor"], dtype=float)
    ground_qnm = tuple(int(v) for v in inst["ground_qnm"])
    ansatz = cell["ansatz"]
    pset = cell["params"]
    family = cell["family"]
    kt = float(cell["kappa_tau"])
    ndepth = int(ANSATZ_SPEC[ansatz]["ndepth"])
    spec = readout_spec(cell["readout"], shots, seed=None)
    cfg = circuit_noise(family, kt, dims=DIMS)
    q_obs, q_twins, hist_by_scale = _observe_block(
        phys, spec, ansatz, pset, family, kt, seed, f"s{shots}"
    )
    mitigated = mitigate_research(
        phys=phys,
        q_obs=q_obs,
        q_twins=q_twins,
        hist_by_scale=hist_by_scale,
        cfg=cfg,
        spec=spec,
        ndepth=ndepth,
        energy_tensor=energy_tensor,
        methods=methods,
        fit_maxiter=fit_maxiter,
        circuit_kind=pset,
        family=family,
        kappa_tau=kt,
    )
    metrics = {
        name: compare_histograms(
            blob.get("hist"),
            phys["p_ideal"],
            energy_tensor,
            ground_qnm,
            energy_mit=blob.get("energy"),
        )
        for name, blob in mitigated.items()
    }
    rec = {
        "id": cell["id"],
        "ansatz": ansatz,
        "params": pset,
        "family": family,
        "kappa_tau": kt,
        "readout": cell["readout"],
        "instance": hid,
        "protect": bool(cell.get("protect")),
        "note": cell.get("note"),
        "cache": str(cache_path),
        "n_shots": shots,
        "metrics": metrics,
        "fits": {k: v.get("fit") for k, v in mitigated.items() if v.get("fit")},
        "noise": noise_as_dict(cfg),
        "readout_spec": readout_as_dict(spec),
        "e_ideal": float(energy_from_histogram(phys["p_ideal"], energy_tensor)),
    }
    sel = (metrics.get("gdr_select") or {}).get("tvd")
    raw = (metrics.get("raw") or {}).get("tvd")
    gdr = (metrics.get("gdr_param") or {}).get("tvd")
    print(
        f"  {cell['id']:<28} raw={raw if raw is None else f'{raw:.4f}'}  "
        f"gdr={gdr if gdr is None else f'{gdr:.4f}'}  "
        f"sel={sel if sel is None else f'{sel:.4f}'}"
    )
    return rec


ACTIVE_CELLS = (
    {
        "id": "ecd_rand_loss_0.1",
        "family": "loss",
        "kappa_tau": 0.1,
        "readout": "ideal",
        "cache": CACHE_PR8 / "ecd_random_loss_kt0.1_n40_span_nr10_lo0.25_hi1.35_x0.npz",
        "protect": False,
    },
    {
        "id": "ecd_rand_comp_0.1",
        "family": "comprehensive",
        "kappa_tau": 0.1,
        "readout": "ideal",
        "cache": CACHE_PR8 / "ecd_random_comprehensive_kt0.1_n40_span_nr10_lo0.25_hi1.35_x0.npz",
        "protect": False,
    },
)


def run_active_twins(*, shots: int, seed: int, fit_maxiter: int, outdir: Path) -> dict:
    """Add 10 Fisher-greedy Gaussian twins; resim those 10 only; refit."""
    from Error_mitigation.metrics import total_variation
    from Error_mitigation.run_mitigation_experiment import (
        case_seed,
        get_or_optimize_params,
        make_sim,
        physical_probs,
    )
    from Error_mitigation.twins import (
        build_twins,
        designed_twin_plan,
        propose_active_gaussian_twins,
    )

    inst = load_instance(0)
    energy_tensor = np.asarray(inst["energy_tensor"], dtype=float)
    ground_qnm = tuple(int(v) for v in inst["ground_qnm"])
    ndepth = int(ANSATZ_SPEC["ecd"]["ndepth"])
    x_random, _, _ = get_or_optimize_params(
        ansatz="ecd",
        ndepth=ndepth,
        energy_tensor=energy_tensor,
        ground_qnm=ground_qnm,
        outdir=RESEARCH,
        hid=0,
        seed_base=int(seed),
        maxiter=200,
        n_restarts=3,
    )
    sim_ideal = make_sim("ecd", ndepth, energy_tensor, ground_qnm)
    p_check = physical_probs(sim_ideal, x_random)
    tag = "span_nr10_lo0.25_hi1.35_x0"
    rng_tw = np.random.default_rng(case_seed("twins", "ecd", "random", seed, tag))
    t_list, scales = designed_twin_plan(40, ndepth, n_rank2=10, mag_lo=0.25, mag_hi=1.35)
    twins40 = build_twins(sim_ideal, x_random, rng_tw, t_free_list=t_list, mag_scales=scales)

    records = []
    extra_cache = outdir / "cache"
    extra_cache.mkdir(parents=True, exist_ok=True)
    for cell in ACTIVE_CELLS:
        phys = load_cache(Path(cell["cache"]))
        if phys is None:
            raise FileNotFoundError(cell["cache"])
        tvd_tgt = total_variation(p_check, phys["p_ideal"])
        print(f"  active {cell['id']}  x_random vs cache p_ideal TVD={tvd_tgt:.3e}")
        if tvd_tgt > 1e-6:
            raise RuntimeError(f"random x does not match cache target ({tvd_tgt})")
        existing = [phys["twin_p_ideal"][i] for i in range(phys["twin_p_ideal"].shape[0])]
        rng_act = np.random.default_rng(case_seed("active", "ecd", cell["family"], cell["kappa_tau"], seed))
        proposed = propose_active_gaussian_twins(
            sim_ideal, x_random, existing, rng_act, n_extra=10, n_pool=80
        )
        family = cell["family"]
        kt = float(cell["kappa_tau"])
        cfg = circuit_noise(family, kt, dims=DIMS)
        extra_path = extra_cache / f"active10_{cell['id']}.npz"
        extra_json = extra_path.with_suffix(".json")
        if extra_path.is_file():
            blob = dict(np.load(extra_path, allow_pickle=False))
            extra_phys = blob["twin_phys"]
            extra_ideal = blob["twin_p_ideal"]
            print(f"    cache hit extra twins {extra_path.name}")
        else:
            sim_noisy = make_sim("ecd", ndepth, energy_tensor, ground_qnm, noise=cfg)
            extra_ideal = []
            extra_phys = []
            t0 = time.time()
            for i, cand in enumerate(proposed):
                extra_ideal.append(cand["p_ideal"])
                print(f"    sim extra twin {i+1}/10 fisher={cand['fisher']:.3f} ...", flush=True)
                extra_phys.append(physical_probs(sim_noisy, cand["x"]))
            extra_ideal = np.stack(extra_ideal, axis=0)
            extra_phys = np.stack(extra_phys, axis=0)
            np.savez_compressed(extra_path, twin_phys=extra_phys, twin_p_ideal=extra_ideal)
            extra_json.write_text(
                json.dumps(
                    {
                        "id": cell["id"],
                        "n_extra": int(extra_phys.shape[0]),
                        "fishers": [float(c["fisher"]) for c in proposed],
                        "wall_s": time.time() - t0,
                    },
                    indent=2,
                )
            )
            print(f"    wrote {extra_path.name} in {time.time() - t0:.1f}s")
        phys50 = dict(phys)
        phys50["twin_phys"] = np.concatenate([phys["twin_phys"], extra_phys], axis=0)
        phys50["twin_p_ideal"] = np.concatenate([phys["twin_p_ideal"], extra_ideal], axis=0)
        phys50["twin_t_free"] = np.concatenate(
            [phys["twin_t_free"], np.zeros(extra_phys.shape[0], dtype=int)]
        )
        if "e_twin_ideal" in phys:
            e_extra = np.array(
                [energy_from_histogram(extra_ideal[i], energy_tensor) for i in range(extra_ideal.shape[0])]
            )
            phys50["e_twin_ideal"] = np.concatenate([phys["e_twin_ideal"], e_extra])
        spec = readout_spec(cell["readout"], shots, seed=None)
        q_obs, q_twins40, hist_by_scale = _observe_block(
            phys, spec, "ecd", "random", family, kt, seed, f"s{shots}"
        )
        _, q_twins50, _ = _observe_block(
            phys50, spec, "ecd", "random", family, kt, seed, f"s{shots}"
        )
        mit40 = mitigate_research(
            phys=phys,
            q_obs=q_obs,
            q_twins=q_twins40,
            hist_by_scale=hist_by_scale,
            cfg=cfg,
            spec=spec,
            ndepth=ndepth,
            energy_tensor=energy_tensor,
            methods=("raw", "gdr_param", "gdr_damped", "gdr_select"),
            fit_maxiter=fit_maxiter,
            circuit_kind="random",
            family=family,
            kappa_tau=kt,
        )
        mit50 = mitigate_research(
            phys=phys50,
            q_obs=q_obs,
            q_twins=q_twins50,
            hist_by_scale=hist_by_scale,
            cfg=cfg,
            spec=spec,
            ndepth=ndepth,
            energy_tensor=energy_tensor,
            methods=("gdr_param", "gdr_damped", "gdr_select"),
            fit_maxiter=fit_maxiter,
            circuit_kind="random",
            family=family,
            kappa_tau=kt,
        )
        metrics40 = {
            name: compare_histograms(
                blob.get("hist"), phys["p_ideal"], energy_tensor, ground_qnm, energy_mit=blob.get("energy")
            )
            for name, blob in mit40.items()
        }
        metrics50 = {
            name: compare_histograms(
                blob.get("hist"), phys["p_ideal"], energy_tensor, ground_qnm, energy_mit=blob.get("energy")
            )
            for name, blob in mit50.items()
        }
        rec = {
            "id": cell["id"],
            "ansatz": "ecd",
            "params": "random",
            "family": family,
            "kappa_tau": kt,
            "readout": cell["readout"],
            "protect": False,
            "n_train_40": 40,
            "n_train_50": int(phys50["twin_phys"].shape[0]),
            "metrics_40": metrics40,
            "metrics_50": metrics50,
            "metrics": {
                "raw": metrics40.get("raw"),
                "gdr_select": metrics40.get("gdr_select"),
                "gdr_param": metrics40.get("gdr_param"),
                "gdr_active50": metrics50.get("gdr_param"),
                "gdr_select_50": metrics50.get("gdr_select"),
            },
            "x_random_tvd": float(tvd_tgt),
            "extra_fishers": [float(c["fisher"]) for c in proposed],
        }
        sel = (metrics40.get("gdr_select") or {}).get("tvd")
        a50 = (metrics50.get("gdr_param") or {}).get("tvd")
        print(
            f"    select40={sel if sel is None else f'{sel:.4f}'}  "
            f"gdr50={a50 if a50 is None else f'{a50:.4f}'}"
        )
        records.append(rec)
    return {"tag": "round2_active", "records": records}


def run_shots(cells, args, outdir: Path) -> dict:
    """Adaptive vs raw at 2048/8192/32768, plus optional shot-damp schedule."""
    methods = SHOT_METHODS
    t0 = time.time()
    records = []
    print(f"round2 stage=shots cells={len(cells)} counts={list(SHOT_COUNTS)} methods={','.join(methods)}")
    for n_shots in SHOT_COUNTS:
        print(f"  -- n_shots={n_shots}")
        for cell in cells:
            rec = run_cell(
                cell,
                shots=int(n_shots),
                seed=int(args.seed),
                fit_maxiter=int(args.fit_maxiter),
                methods=methods,
            )
            rec["shot_floor"] = float(
                (rec.get("fits") or {}).get("gdr_shot_damp", {}).get("shot_floor", 0.0)
            )
            records.append(rec)
    # Score shot_damp at 2048 (must beat) and 8192 (must not hurt).
    recs_2048 = [r for r in records if int(r["n_shots"]) == 2048]
    recs_8192 = [r for r in records if int(r["n_shots"]) == 8192]
    verdict = {
        "gdr_shot_damp": {
            **score_methods(recs_2048, new_methods=SHOT_NEW)["gdr_shot_damp"],
            "at_8192": score_methods(recs_8192, new_methods=SHOT_NEW)["gdr_shot_damp"],
        }
    }
    keep_2048 = bool(verdict["gdr_shot_damp"]["keep"])
    hurt_8192 = bool(verdict["gdr_shot_damp"]["at_8192"]["regressions"])
    any_keep = keep_2048 and not hurt_8192
    payload = {
        "tag": "round2_shots",
        "shot_counts": list(SHOT_COUNTS),
        "seed": int(args.seed),
        "fit_maxiter": int(args.fit_maxiter),
        "methods": list(methods),
        "beat_eps": BEAT_EPS,
        "regress_eps": REGRESS_EPS,
        "wall_s": time.time() - t0,
        "records": records,
        "verdict": verdict,
        "any_keep": any_keep,
        "keep_2048": keep_2048,
        "hurt_8192": hurt_8192,
    }
    (outdir / "shots_results.json").write_text(json.dumps(json_ready(payload), indent=2))
    lines = [
        "# Shot-count sweep (adaptive vs raw + optional shot-damp)",
        "",
        f"wall={payload['wall_s']:.1f}s  keep_2048={keep_2048}  hurt_8192={hurt_8192}  any_keep={any_keep}",
        "",
        "| id | shots | raw | gdr_param | gdr_select | gdr_shot_damp | Δ damp−select |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def f(mets, name):
        t = (mets.get(name) or {}).get("tvd")
        return "—" if t is None else f"{t:.4f}"

    for rec in records:
        mets = rec["metrics"]
        sel = (mets.get("gdr_select") or {}).get("tvd")
        sd = (mets.get("gdr_shot_damp") or {}).get("tvd")
        d = "—" if sel is None or sd is None else f"{sd - sel:+.4f}"
        lines.append(
            f"| {rec['id']} | {rec['n_shots']} | {f(mets, 'raw')} | {f(mets, 'gdr_param')} | "
            f"{f(mets, 'gdr_select')} | {f(mets, 'gdr_shot_damp')} | {d} |"
        )
    (outdir / "shots_scoreboard.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {outdir / 'shots_results.json'}")
    print(f"wrote {outdir / 'shots_scoreboard.md'}")
    return payload


def run_cross_h(*, shots: int, seed: int, fit_maxiter: int, outdir: Path) -> dict:
    """Fit M on one H's twins; unfold another H's target (same noise, κτ=0.003)."""
    from Error_mitigation.mitigation import PARAM_NAMES, params_to_kernels, unfold

    t0 = time.time()
    packed = []
    for cell in XFER_CELLS:
        phys = load_cache(Path(cell["cache"]))
        if phys is None:
            raise FileNotFoundError(cell["cache"])
        hid = int(cell["instance"])
        inst = load_instance(hid)
        energy_tensor = np.asarray(inst["energy_tensor"], dtype=float)
        ground_qnm = tuple(int(v) for v in inst["ground_qnm"])
        ansatz = cell["ansatz"]
        pset = cell["params"]
        family = cell["family"]
        kt = float(cell["kappa_tau"])
        ndepth = int(ANSATZ_SPEC[ansatz]["ndepth"])
        spec = readout_spec(cell["readout"], shots, seed=None)
        cfg = circuit_noise(family, kt, dims=DIMS)
        q_obs, q_twins, hist_by_scale = _observe_block(
            phys, spec, ansatz, pset, family, kt, seed, f"s{shots}"
        )
        mit = mitigate_research(
            phys=phys,
            q_obs=q_obs,
            q_twins=q_twins,
            hist_by_scale=hist_by_scale,
            cfg=cfg,
            spec=spec,
            ndepth=ndepth,
            energy_tensor=energy_tensor,
            methods=("raw", "gdr_param", "gdr_damped", "gdr_select"),
            fit_maxiter=fit_maxiter,
            circuit_kind=pset,
            family=family,
            kappa_tau=kt,
        )
        fitted = (mit.get("gdr_param") or {}).get("fit", {}).get("fitted") or {}
        theta = np.array([float(fitted[n]) for n in PARAM_NAMES], dtype=float)
        packed.append(
            {
                "cell": cell,
                "phys": phys,
                "q_obs": q_obs,
                "energy_tensor": energy_tensor,
                "ground_qnm": ground_qnm,
                "theta": theta,
                "kernels": params_to_kernels(theta, DIMS),
                "same": {
                    name: compare_histograms(
                        blob.get("hist"),
                        phys["p_ideal"],
                        energy_tensor,
                        ground_qnm,
                        energy_mit=blob.get("energy"),
                    )
                    for name, blob in mit.items()
                },
            }
        )
        sel = packed[-1]["same"]["gdr_select"]["tvd"]
        print(f"  xfer fit {cell['id']}  same-H select={sel:.4f}")

    pairs = []
    for src in packed:
        for tgt in packed:
            hist = unfold(tgt["q_obs"], *src["kernels"])
            mets = compare_histograms(
                hist, tgt["phys"]["p_ideal"], tgt["energy_tensor"], tgt["ground_qnm"]
            )
            same_sel = src["same"]["gdr_select"]["tvd"] if src["cell"]["id"] == tgt["cell"]["id"] else tgt["same"]["gdr_select"]["tvd"]
            same_param = tgt["same"]["gdr_param"]["tvd"]
            rec = {
                "source": src["cell"]["id"],
                "target": tgt["cell"]["id"],
                "cross": src["cell"]["id"] != tgt["cell"]["id"],
                "tvd_xfer": float(mets["tvd"]),
                "tvd_same_param": float(same_param),
                "tvd_same_select": float(same_sel),
                "delta_vs_same_param": float(mets["tvd"]) - float(same_param),
                "delta_vs_same_select": float(mets["tvd"]) - float(same_sel),
            }
            pairs.append(rec)
            mark = "same" if not rec["cross"] else "XFER"
            print(
                f"    {mark} {rec['source']} -> {rec['target']}  "
                f"xfer={rec['tvd_xfer']:.4f}  same_param={rec['tvd_same_param']:.4f}  "
                f"Δ={rec['delta_vs_same_param']:+.4f}"
            )

    cross = [p for p in pairs if p["cross"]]
    beats = [p for p in cross if p["delta_vs_same_select"] < -BEAT_EPS]
    regressions = [p for p in cross if p["delta_vs_same_select"] > REGRESS_EPS]
    # Transfer "keep" if it matches or beats same-H fit (mean Δ ≤ 0) without
    # a >0.003 regression vs same-H select on any pair.
    mean_d = float(np.mean([p["delta_vs_same_param"] for p in cross])) if cross else None
    keep = bool(cross) and mean_d is not None and mean_d <= 0.0 and not regressions
    payload = {
        "tag": "round2_xfer",
        "shots": int(shots),
        "seed": int(seed),
        "fit_maxiter": int(fit_maxiter),
        "wall_s": time.time() - t0,
        "pairs": pairs,
        "mean_delta_vs_same_param": mean_d,
        "beats": [f"{p['source']}->{p['target']}" for p in beats],
        "regressions": [f"{p['source']}->{p['target']}" for p in regressions],
        "any_keep": keep,
    }
    (outdir / "xfer_results.json").write_text(json.dumps(json_ready(payload), indent=2))
    lines = [
        "# Cross-H twin bank (fit M on source twins, unfold target)",
        "",
        f"wall={payload['wall_s']:.1f}s  mean Δ vs same-H param={mean_d if mean_d is None else f'{mean_d:+.4f}'}  keep={keep}",
        "",
        "| source | target | xfer | same_param | same_select | Δ vs param | Δ vs select |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in pairs:
        lines.append(
            f"| {p['source']} | {p['target']} | {p['tvd_xfer']:.4f} | {p['tvd_same_param']:.4f} | "
            f"{p['tvd_same_select']:.4f} | {p['delta_vs_same_param']:+.4f} | {p['delta_vs_same_select']:+.4f} |"
        )
    (outdir / "xfer_scoreboard.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {outdir / 'xfer_results.json'}")
    print(f"wrote {outdir / 'xfer_scoreboard.md'}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=ROUND2_OUT)
    p.add_argument("--shots", type=int, default=8192)
    p.add_argument("--seed", type=int, default=SEED_BASE)
    p.add_argument("--fit-maxiter", type=int, default=120)
    p.add_argument("--methods", default="")
    p.add_argument(
        "--stage",
        choices=("micro", "b", "active", "c", "shots", "family", "xfer", "all"),
        default="b",
        help="micro: pass 1; b: pass 2; c: pass 3 (shots+family+xfer+rl); "
        "shots/family/xfer: pass-3 pieces.",
    )
    p.add_argument(
        "--cells",
        default="all",
        help="Comma-separated cell ids, or 'all'.",
    )
    return p.parse_args(argv)


def _run_batch(cells, methods, new_methods, args, outdir, tag, json_name, md_name) -> dict:
    print(
        f"round2 stage={tag} cells={len(cells)} shots={args.shots} "
        f"fit_maxiter={args.fit_maxiter} methods={','.join(methods)}"
    )
    t0 = time.time()
    records = []
    for cell in cells:
        records.append(
            run_cell(
                cell,
                shots=int(args.shots),
                seed=int(args.seed),
                fit_maxiter=int(args.fit_maxiter),
                methods=methods,
            )
        )
    verdict = score_methods(records, new_methods=new_methods)
    payload = {
        "tag": tag,
        "shots": int(args.shots),
        "seed": int(args.seed),
        "fit_maxiter": int(args.fit_maxiter),
        "methods": list(methods),
        "beat_eps": BEAT_EPS,
        "regress_eps": REGRESS_EPS,
        "wall_s": time.time() - t0,
        "records": records,
        "verdict": verdict,
        "any_keep": any(v["keep"] for v in verdict.values()),
    }
    (outdir / json_name).write_text(json.dumps(json_ready(payload), indent=2))
    header = (
        f"stage={tag} shots={args.shots} seed={args.seed} fit_maxiter={args.fit_maxiter} "
        f"wall={payload['wall_s']:.1f}s  any_keep={payload['any_keep']}"
    )
    write_scoreboard(outdir / md_name, records, verdict, header, new_methods=new_methods)
    print(f"\nwrote {outdir / json_name}")
    print(f"wrote {outdir / md_name}")
    print(f"any KEEP: {payload['any_keep']}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stage = args.stage
    if stage == "micro":
        methods = tuple(x.strip() for x in (args.methods or ",".join(ROUND2_METHODS)).split(",") if x.strip())
        cells = list(HARD_CELLS) if args.cells == "all" else [c for c in HARD_CELLS if c["id"] in args.cells.split(",")]
        _run_batch(cells, methods, NEW_METHODS, args, outdir, "round2_micro", "micro_results.json", "micro_scoreboard.md")
        return 0
    if stage in ("b", "all"):
        methods = tuple(x.strip() for x in (args.methods or ",".join(STAGE_B_METHODS)).split(",") if x.strip())
        cells = list(HARD_CELLS) + list(SNAP_CELLS)
        _run_batch(
            cells,
            methods,
            STAGE_B_NEW,
            args,
            outdir,
            "round2_stage_b",
            "stage_b_results.json",
            "stage_b_scoreboard.md",
        )
    if stage in ("active", "all"):
        t0 = time.time()
        payload = run_active_twins(
            shots=int(args.shots), seed=int(args.seed), fit_maxiter=int(args.fit_maxiter), outdir=outdir
        )
        payload["wall_s"] = time.time() - t0
        (outdir / "active_results.json").write_text(json.dumps(json_ready(payload), indent=2))
        lines = ["# Active twin add (40 span + 10 Fisher-greedy)", "", f"wall={payload['wall_s']:.1f}s", ""]
        lines.append("| id | raw | select40 | gdr40 | gdr50 | select50 | Δ gdr50−select40 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for rec in payload["records"]:
            m = rec["metrics"]

            def f(name):
                t = (m.get(name) or {}).get("tvd")
                return "—" if t is None else f"{t:.4f}"

            sel = (m.get("gdr_select") or {}).get("tvd")
            g50 = (m.get("gdr_active50") or {}).get("tvd")
            d = "—" if sel is None or g50 is None else f"{g50 - sel:+.4f}"
            lines.append(
                f"| {rec['id']} | {f('raw')} | {f('gdr_select')} | {f('gdr_param')} | "
                f"{f('gdr_active50')} | {f('gdr_select_50')} | {d} |"
            )
        (outdir / "active_scoreboard.md").write_text("\n".join(lines) + "\n")
        print(f"wrote {outdir / 'active_results.json'}")
        print(f"wrote {outdir / 'active_scoreboard.md'}")
    if stage in ("shots", "c"):
        run_shots(list(SHOT_CELLS), args, outdir)
    if stage in ("family", "c"):
        methods = tuple(
            x.strip() for x in (args.methods or ",".join(STAGE_C_METHODS)).split(",") if x.strip()
        )
        cells = list(HARD_CELLS) + list(FAMILY_EXTRA) + [SNAP_CELLS[0], SNAP_CELLS[2]]
        _run_batch(
            cells,
            methods,
            STAGE_C_NEW,
            args,
            outdir,
            "round2_stage_c",
            "stage_c_results.json",
            "stage_c_scoreboard.md",
        )
    if stage in ("xfer", "c"):
        run_cross_h(
            shots=int(args.shots),
            seed=int(args.seed),
            fit_maxiter=int(args.fit_maxiter),
            outdir=outdir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
