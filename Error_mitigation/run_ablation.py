#!/usr/bin/env python3
"""Cheap ablation driver for GDR research methods.

Writes only under ``Error_mitigation/out_research/`` (never ``out/`` or
``out_smoke/``). Physical histograms are cached so fit-only method changes
can be replayed without a second density-matrix pass.

Usage (from repo root)::

    python -u Error_mitigation/run_ablation.py --tag micro_ab --ansatz ecd \\
        --params both --families loss --kappa-tau 0.003,0.1 --shots 4096 --n-train 20
    python -u Error_mitigation/run_ablation.py --preset research_smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
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

from Error_mitigation.metrics import compare_histograms, total_variation
from Error_mitigation.mitigation import (
    apply_scalar_cdr,
    choose_damp_alpha,
    damp_histogram,
    energy_weights,
    holdout_indices,
    score_unfold_tvd,
    fit_gdr_afterburn,
    fit_gdr_holdout,
    fit_gdr_interleave,
    fit_gdr_mid,
    fit_gdr_param,
    fit_gdr_residual,
    fit_gdr_ridge,
    fit_gdr_split,
    fit_gdr_band,
    fit_gdr_tfree,
    choose_mix_alpha,
    classify_opt_quality,
    select_research_method,
    tfree_indices,
    fit_scalar_cdr,
    observe_histogram,
    oracle_kernels,
    oracle_residual,
    params_to_kernels,
    readout_then_zne,
    run_readout_only,
    safe_histogram,
    unfold,
    zne_histogram,
    zne_then_readout,
)
from Error_mitigation.noise_models import (
    CIRCUIT_FAMILIES,
    READOUT_SPECS,
    circuit_noise,
    readout_as_dict,
    readout_spec,
    scale_noise,
)
from Error_mitigation.run_mitigation_experiment import (
    ANSATZ_SPEC,
    DIMS,
    SEED_BASE,
    case_seed,
    get_or_optimize_params,
    json_ready,
    load_instance,
    make_sim,
    physical_probs,
    write_summary_txt,
)
from Error_mitigation.twins import build_twins, designed_twin_plan

HERE = Path(__file__).resolve().parent
BASELINE_OUT = HERE / "out"
DEFAULT_OUT = HERE / "out_research"
PARAM_FILES = (
    "optimized_params_ecd_h000_nd5.json",
    "optimized_params_snap_h000_nd2.json",
)

ALL_METHODS = (
    "raw",
    "readout_only",
    "oracle_binomial",
    "gdr_param",
    "gdr_ridge",
    "gdr_holdout",
    "gdr_damped",
    "gdr_floor",
    "gdr_reg",
    "gdr_mid",
    "gdr_tfree",
    "gdr_residual",
    "gdr_afterburn",
    "gdr_interleave",
    "gdr_blend",
    "gdr_energy",
    "gdr_select",
    "gdr_split",
    "gdr_band",
    "scalar_cdr",
    "zne_idle",
    "readout_then_zne",
    "zne_then_readout",
)

CHEAP_METHODS = (
    "raw",
    "readout_only",
    "oracle_binomial",
    "gdr_param",
    "gdr_ridge",
    "gdr_holdout",
    "gdr_damped",
    "gdr_floor",
    "gdr_reg",
    "gdr_mid",
    "gdr_tfree",
    "gdr_residual",
    "gdr_afterburn",
    "gdr_interleave",
    "gdr_blend",
    "gdr_select",
    "gdr_split",
    "gdr_band",
    "zne_idle",
    "readout_then_zne",
    "zne_then_readout",
)


def copy_optimized_params(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name in PARAM_FILES:
        src = BASELINE_OUT / name
        dst = outdir / name
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
            print(f"  copied {name} -> {outdir.name}/")


def _parse_csv_floats(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not text:
        return default
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _parse_csv_str(text: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not text:
        return default
    return tuple(x.strip() for x in text.split(",") if x.strip())


def twin_tag(args: argparse.Namespace) -> str:
    return (
        f"{args.twin_design}_nr{args.n_rank2 if args.n_rank2 is not None else 'auto'}"
        f"_lo{args.mag_lo:g}_hi{args.mag_hi:g}_x{args.extra_t_free}"
    )


def cache_key(
    ansatz: str,
    pset: str,
    family: str,
    kt: float,
    n_train: int,
    tag: str,
    hid: int = 0,
) -> str:
    key = f"{ansatz}_{pset}_{family}_kt{kt:g}_n{n_train}_{tag}"
    if int(hid) != 0:
        key += f"_h{int(hid):03d}"
    return key


def save_cache(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in blob.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in blob.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(json_ready(meta), indent=2))


def load_cache(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = dict(np.load(path, allow_pickle=False))
    meta_path = path.with_suffix(".json")
    if meta_path.is_file():
        data["_meta"] = json.loads(meta_path.read_text())
    return data


def build_or_load_physics(
    *,
    ansatz: str,
    pset: str,
    xvec: np.ndarray,
    family: str,
    kt: float,
    n_train: int,
    args: argparse.Namespace,
    energy_tensor: np.ndarray,
    ground_qnm: tuple[int, int, int],
    cache_dir: Path,
) -> dict:
    tag = twin_tag(args)
    hid = int(getattr(args, "instance", 0) or 0)
    key = cache_key(ansatz, pset, family, kt, n_train, tag, hid=hid)
    path = cache_dir / f"{key}.npz"
    cached = None if args.no_cache else load_cache(path)
    if cached is not None:
        print(f"    cache hit {path.name}")
        return cached

    ndepth = int(ANSATZ_SPEC[ansatz]["ndepth"])
    sim_ideal = make_sim(ansatz, ndepth, energy_tensor, ground_qnm)
    rng_tw = np.random.default_rng(case_seed("twins", ansatz, pset, args.seed, tag))
    if args.twin_design == "span":
        t_list, scales = designed_twin_plan(
            n_train,
            ndepth,
            n_rank2=args.n_rank2,
            mag_lo=args.mag_lo,
            mag_hi=args.mag_hi,
            extra_t_free=args.extra_t_free,
        )
        twins = build_twins(sim_ideal, xvec, rng_tw, t_free_list=t_list, mag_scales=scales)
    elif args.twin_design == "more_tfree":
        n_rank2 = args.n_rank2 if args.n_rank2 is not None else max(n_train // 2, 1)
        t_list, scales = designed_twin_plan(
            n_train,
            ndepth,
            n_rank2=n_rank2,
            mag_lo=args.mag_lo,
            mag_hi=args.mag_hi,
            extra_t_free=args.extra_t_free,
        )
        twins = build_twins(sim_ideal, xvec, rng_tw, t_free_list=t_list, mag_scales=scales)
    else:
        twins = build_twins(
            sim_ideal,
            xvec,
            rng_tw,
            n_train=n_train,
            n_rank2=args.n_rank2,
            mag_scale_range=(args.mag_lo, args.mag_hi) if args.twin_design == "wide" else (0.5, 1.0),
        )

    p_ideal = physical_probs(sim_ideal, xvec)
    cfg = circuit_noise(family, kt, dims=DIMS)
    sim_noisy = make_sim(ansatz, ndepth, energy_tensor, ground_qnm, noise=cfg)
    print(f"    sim {family} κτ={kt} {pset}  target+{len(twins)} twins ...")
    t0 = time.time()
    p_phys = physical_probs(sim_noisy, xvec)
    twin_phys = np.stack([physical_probs(sim_noisy, tw.x) for tw in twins], axis=0)
    scales = {1: p_phys}
    for s in (2, 3):
        cfg_s = scale_noise(cfg, float(s))
        sim_s = make_sim(ansatz, ndepth, energy_tensor, ground_qnm, noise=cfg_s)
        scales[s] = physical_probs(sim_s, xvec)
    print(f"      physical histograms in {time.time() - t0:.1f}s")

    twin_p_ideal = np.stack([t.p_ideal for t in twins], axis=0)
    twin_t_free = np.array([t.t_free for t in twins], dtype=int)
    product_tvds = [t.product_tvd for t in twins if t.product_tvd is not None]
    blob = {
        "p_ideal": p_ideal,
        "p_phys": p_phys,
        "twin_phys": twin_phys,
        "twin_p_ideal": twin_p_ideal,
        "twin_t_free": twin_t_free,
        "scale_1": scales[1],
        "scale_2": scales[2],
        "scale_3": scales[3],
        "e_ideal": np.array(energy_from_histogram(p_ideal, energy_tensor)),
        "e_twin_ideal": np.array([energy_from_histogram(t.p_ideal, energy_tensor) for t in twins]),
        "product_tvd_max": np.array(0.0 if not product_tvds else max(product_tvds)),
        "ansatz": ansatz,
        "params": pset,
        "family": family,
        "kappa_tau": float(kt),
        "n_train": int(n_train),
        "twin_tag": tag,
    }
    if not args.no_cache:
        save_cache(path, blob)
        print(f"      wrote cache {path.name}")
    return blob


def _observe_block(phys: dict, spec, ansatz, pset, family, kt, seed, shots_key: str):
    p_phys = phys["p_phys"]
    twin_phys = phys["twin_phys"]
    seed_t = case_seed("obs", ansatz, pset, family, kt, spec.level, seed, "target", shots_key)
    q_obs = observe_histogram(p_phys, spec, DIMS, seed_t)
    q_twins = [
        observe_histogram(
            twin_phys[i],
            spec,
            DIMS,
            case_seed("obs", ansatz, pset, family, kt, spec.level, seed, "twin", i, shots_key),
        )
        for i in range(twin_phys.shape[0])
    ]
    hist_by_scale = {}
    for s, key in ((1, "scale_1"), (2, "scale_2"), (3, "scale_3")):
        hist_by_scale[s] = observe_histogram(
            phys[key],
            spec,
            DIMS,
            case_seed("obs", ansatz, pset, family, kt, spec.level, seed, "zne", s, shots_key),
        )
    return q_obs, q_twins, hist_by_scale


def _kernels_from_fit(theta, dims):
    return params_to_kernels(theta, dims)


def slice_twin_indices(n_total: int, n_keep: int) -> np.ndarray:
    """Evenly spaced subset of a cached twin roster (fit-only n_train sweep)."""
    n_total = int(n_total)
    n_keep = int(n_keep)
    if n_keep <= 0 or n_keep >= n_total:
        return np.arange(n_total, dtype=int)
    return np.unique(np.round(np.linspace(0, n_total - 1, num=n_keep)).astype(int))


def slice_twin_phys(phys: dict, n_keep: int | None) -> dict:
    if n_keep is None:
        return phys
    t = np.asarray(phys["twin_t_free"])
    keep = slice_twin_indices(int(t.size), int(n_keep))
    if keep.size == t.size:
        return phys
    out = dict(phys)
    for key in ("twin_phys", "twin_p_ideal"):
        out[key] = np.asarray(phys[key])[keep]
    out["twin_t_free"] = np.asarray(phys["twin_t_free"])[keep]
    if "e_twin_ideal" in phys:
        out["e_twin_ideal"] = np.asarray(phys["e_twin_ideal"])[keep]
    out["n_train"] = int(keep.size)
    out["fit_twin_index"] = keep
    return out


def mitigate_research(
    *,
    phys: dict,
    q_obs: np.ndarray,
    q_twins: list[np.ndarray],
    hist_by_scale: dict[int, np.ndarray],
    cfg,
    spec,
    ndepth: int,
    energy_tensor: np.ndarray,
    methods: tuple[str, ...],
    fit_maxiter: int,
    circuit_kind: str | None = None,
    family: str | None = None,
    kappa_tau: float | None = None,
) -> dict:
    p_ideal = phys["p_ideal"]
    p_twin = [phys["twin_p_ideal"][i] for i in range(phys["twin_p_ideal"].shape[0])]
    t_free = [int(t) for t in phys["twin_t_free"]]
    e_twin_ideal = phys["e_twin_ideal"]
    e_twin_noisy = np.array([energy_from_histogram(q, energy_tensor) for q in q_twins])
    e_obs = energy_from_histogram(q_obs, energy_tensor)
    p_safe_target = safe_histogram(q_obs, spec, DIMS)
    p_safe_twins = [safe_histogram(q, spec, DIMS) for q in q_twins]
    out: dict = {}
    kernels: dict[str, tuple] = {}

    if "raw" in methods:
        out["raw"] = {"hist": q_obs, "energy": e_obs}

    if "readout_only" in methods:
        ro = run_readout_only(q_obs, spec, DIMS)
        if ro is not None:
            out["readout_only"] = {
                "hist": ro.histogram,
                "energy": energy_from_histogram(ro.histogram, energy_tensor),
            }

    if "oracle_binomial" in methods or "gdr_select" in methods:
        cq_o, c1_o, c2_o = oracle_kernels(cfg, spec, ndepth, DIMS)
        kernels["oracle_binomial"] = (cq_o, c1_o, c2_o)
        p_oracle = unfold(q_obs, cq_o, c1_o, c2_o)
        if "oracle_binomial" in methods:
            out["oracle_binomial"] = {
                "hist": p_oracle,
                "energy": energy_from_histogram(p_oracle, energy_tensor),
                "residual_tvd": oracle_residual(p_ideal, q_obs, cq_o, c1_o, c2_o),
            }

    theta_base = fit_info_base = None
    if any(
        m in methods
        for m in (
            "gdr_param",
            "gdr_damped",
            "gdr_floor",
            "gdr_reg",
            "gdr_full",
            "gdr_select",
            "gdr_split",
            "gdr_band",
        )
    ):
        theta_base, fit_info_base = fit_gdr_param(
            p_twin, q_twins, cfg, spec, ndepth, DIMS, maxiter=fit_maxiter
        )
        cq, c1, c2 = _kernels_from_fit(theta_base, DIMS)
        kernels["gdr_param"] = (cq, c1, c2)
        p_gdr = unfold(q_obs, cq, c1, c2)
        if "gdr_param" in methods:
            out["gdr_param"] = {
                "hist": p_gdr,
                "energy": energy_from_histogram(p_gdr, energy_tensor),
                "fit": fit_info_base,
                "residual_tvd": oracle_residual(p_ideal, q_obs, cq, c1, c2),
            }
        if "gdr_damped" in methods:
            slack, gap = 0.0, None
            if (
                str(circuit_kind or "").lower() == "random"
                and str(family or "").lower() == "comprehensive"
                and kappa_tau is not None
                and float(kappa_tau) <= 0.003 + 1e-12
            ):
                slack, gap = 0.003, 0.01
            alpha, ainfo = choose_damp_alpha(
                p_twin, q_twins, cq, c1, c2, p_safe_twins, slack=slack, safe_gap=gap
            )
            p_d = damp_histogram(p_gdr, p_safe_target, alpha)
            out["gdr_damped"] = {
                "hist": p_d,
                "energy": energy_from_histogram(p_d, energy_tensor),
                "fit": {**(fit_info_base or {}), **ainfo, "kind": "gdr_damped"},
            }
        if "gdr_floor" in methods:
            alpha_f, finfo = choose_damp_alpha(
                p_twin, q_twins, cq, c1, c2, p_safe_twins, slack=0.003, safe_gap=0.01
            )
            p_f = damp_histogram(p_gdr, p_safe_target, alpha_f)
            out["gdr_floor"] = {
                "hist": p_f,
                "energy": energy_from_histogram(p_f, energy_tensor),
                "fit": {**(fit_info_base or {}), **finfo, "kind": "gdr_floor"},
            }

    if "gdr_ridge" in methods:
        theta, info = fit_gdr_ridge(p_twin, q_twins, cfg, spec, ndepth, DIMS, maxiter=fit_maxiter, lam=1e-3)
        cq, c1, c2 = _kernels_from_fit(theta, DIMS)
        kernels["gdr_ridge"] = (cq, c1, c2)
        p = unfold(q_obs, cq, c1, c2)
        out["gdr_ridge"] = {
            "hist": p,
            "energy": energy_from_histogram(p, energy_tensor),
            "fit": info,
        }

    if "gdr_holdout" in methods or "gdr_reg" in methods:
        theta_h, info_h = fit_gdr_holdout(
            p_twin, q_twins, cfg, spec, ndepth, DIMS, maxiter=fit_maxiter
        )
        cq, c1, c2 = _kernels_from_fit(theta_h, DIMS)
        kernels["gdr_holdout"] = (cq, c1, c2)
        p_h = unfold(q_obs, cq, c1, c2)
        if "gdr_holdout" in methods:
            out["gdr_holdout"] = {
                "hist": p_h,
                "energy": energy_from_histogram(p_h, energy_tensor),
                "fit": info_h,
            }
        if "gdr_reg" in methods:
            alpha, ainfo = choose_damp_alpha(p_twin, q_twins, cq, c1, c2, p_safe_twins)
            p_r = damp_histogram(p_h, p_safe_target, alpha)
            out["gdr_reg"] = {
                "hist": p_r,
                "energy": energy_from_histogram(p_r, energy_tensor),
                "fit": {**info_h, **ainfo, "kind": "gdr_reg"},
            }

    if "gdr_mid" in methods:
        theta, info = fit_gdr_mid(p_twin, q_twins, cfg, spec, ndepth, DIMS, maxiter=fit_maxiter)
        cq, c1, c2 = _kernels_from_fit(theta, DIMS)
        kernels["gdr_mid"] = (cq, c1, c2)
        p = unfold(q_obs, cq, c1, c2)
        out["gdr_mid"] = {"hist": p, "energy": energy_from_histogram(p, energy_tensor), "fit": info}

    if "gdr_tfree" in methods:
        theta, info = fit_gdr_tfree(
            p_twin, q_twins, t_free, cfg, spec, ndepth, DIMS, maxiter=fit_maxiter
        )
        cq, c1, c2 = _kernels_from_fit(theta, DIMS)
        p = unfold(q_obs, cq, c1, c2)
        out["gdr_tfree"] = {"hist": p, "energy": energy_from_histogram(p, energy_tensor), "fit": info}

    info_res: dict | None = None
    if "gdr_residual" in methods or "gdr_select" in methods:
        (cq, c1, c2), info_res = fit_gdr_residual(
            p_twin, q_twins, cfg, spec, ndepth, DIMS, maxiter=min(fit_maxiter, 120), t_free=t_free
        )
        kernels["gdr_residual"] = (cq, c1, c2)
        p = unfold(q_obs, cq, c1, c2)
        if "gdr_residual" in methods:
            out["gdr_residual"] = {
                "hist": p,
                "energy": energy_from_histogram(p, energy_tensor),
                "fit": info_res,
                "residual_tvd": oracle_residual(p_ideal, q_obs, cq, c1, c2),
            }

    if "gdr_interleave" in methods:
        (cq, c1, c2), info_il = fit_gdr_interleave(
            p_twin, q_twins, cfg, spec, ndepth, DIMS, maxiter=min(fit_maxiter, 160), t_free=t_free
        )
        kernels["gdr_interleave"] = (cq, c1, c2)
        p = unfold(q_obs, cq, c1, c2)
        out["gdr_interleave"] = {
            "hist": p,
            "energy": energy_from_histogram(p, energy_tensor),
            "fit": info_il,
            "residual_tvd": oracle_residual(p_ideal, q_obs, cq, c1, c2),
        }

    if "gdr_afterburn" in methods or "gdr_select" in methods:
        (cq, c1, c2), info_ab = fit_gdr_afterburn(
            p_twin, q_twins, cfg, spec, ndepth, DIMS, maxiter=min(fit_maxiter, 120), t_free=t_free
        )
        kernels["gdr_afterburn"] = (cq, c1, c2)
        p = unfold(q_obs, cq, c1, c2)
        if "gdr_afterburn" in methods:
            out["gdr_afterburn"] = {
                "hist": p,
                "energy": energy_from_histogram(p, energy_tensor),
                "fit": info_ab,
                "residual_tvd": oracle_residual(p_ideal, q_obs, cq, c1, c2),
            }

    if "gdr_blend" in methods and "gdr_param" in kernels and "oracle_binomial" in kernels:
        gdr_u = [unfold(q, *kernels["gdr_param"]) for q in q_twins]
        ora_u = [unfold(q, *kernels["oracle_binomial"]) for q in q_twins]
        beta, binfo = choose_mix_alpha(p_twin, gdr_u, ora_u)
        p_g = unfold(q_obs, *kernels["gdr_param"])
        p_o = unfold(q_obs, *kernels["oracle_binomial"])
        p_b = damp_histogram(p_g, p_o, beta)
        out["gdr_blend"] = {
            "hist": p_b,
            "energy": energy_from_histogram(p_b, energy_tensor),
            "fit": {**(fit_info_base or {}), **binfo, "kind": "gdr_blend"},
        }

    if "gdr_split" in methods and "gdr_param" in kernels:
        (cq, c1, c2), info_sp = fit_gdr_split(
            p_twin,
            q_twins,
            *kernels["gdr_param"],
            spec.n_shots,
            DIMS,
            maxiter=min(fit_maxiter, 80),
        )
        kernels["gdr_split"] = (cq, c1, c2)
        p = unfold(q_obs, cq, c1, c2)
        out["gdr_split"] = {
            "hist": p,
            "energy": energy_from_histogram(p, energy_tensor),
            "fit": info_sp,
            "residual_tvd": oracle_residual(p_ideal, q_obs, cq, c1, c2),
        }

    if "gdr_band" in methods and "gdr_param" in kernels:
        (cq, c1, c2), info_bd = fit_gdr_band(
            p_twin,
            q_twins,
            *kernels["gdr_param"],
            spec.n_shots,
            DIMS,
            maxiter=min(fit_maxiter, 80),
        )
        kernels["gdr_band"] = (cq, c1, c2)
        p = unfold(q_obs, cq, c1, c2)
        out["gdr_band"] = {
            "hist": p,
            "energy": energy_from_histogram(p, energy_tensor),
            "fit": info_bd,
            "residual_tvd": oracle_residual(p_ideal, q_obs, cq, c1, c2),
        }

    if "gdr_energy" in methods:
        w = energy_weights(e_twin_ideal, "absE")
        theta, info = fit_gdr_ridge(
            p_twin, q_twins, cfg, spec, ndepth, DIMS, maxiter=fit_maxiter, lam=1e-3, weights=w
        )
        cq, c1, c2 = _kernels_from_fit(theta, DIMS)
        p = unfold(q_obs, cq, c1, c2)
        info["kind"] = "gdr_energy"
        out["gdr_energy"] = {"hist": p, "energy": energy_from_histogram(p, energy_tensor), "fit": info}

    if "scalar_cdr" in methods:
        a1, a0 = fit_scalar_cdr(e_twin_ideal, e_twin_noisy)
        out["scalar_cdr"] = {"hist": None, "energy": apply_scalar_cdr(e_obs, a1, a0), "a1": a1, "a0": a0}

    if "zne_idle" in methods and 1 in hist_by_scale and 2 in hist_by_scale:
        deg = 2 if 3 in hist_by_scale else 1
        p = zne_histogram(hist_by_scale, degree=deg)
        out["zne_idle"] = {"hist": p, "energy": energy_from_histogram(p, energy_tensor)}

    if "readout_then_zne" in methods and 1 in hist_by_scale and 2 in hist_by_scale:
        p = readout_then_zne(hist_by_scale, spec, DIMS)
        out["readout_then_zne"] = {"hist": p, "energy": energy_from_histogram(p, energy_tensor)}

    if "zne_then_readout" in methods and 1 in hist_by_scale and 2 in hist_by_scale:
        p = zne_then_readout(hist_by_scale, spec, DIMS)
        out["zne_then_readout"] = {"hist": p, "energy": energy_from_histogram(p, energy_tensor)}

    if "gdr_select" in methods and kernels:
        train_i, hold_i = holdout_indices(len(p_twin), 0.25)
        if hold_i.size == 0:
            hold_i = train_i
        tf_i = tfree_indices(t_free)
        if tf_i.size == 0:
            tf_i = hold_i
        cand_scores: list[tuple[str, float]] = []
        safe_hold = float(
            np.mean([total_variation(p_safe_twins[int(i)], p_twin[int(i)]) for i in hold_i])
        )
        cand_scores.append(("safe", safe_hold))
        for name, (cq, c1, c2) in kernels.items():
            cand_scores.append((name, score_unfold_tvd(p_twin, q_twins, cq, c1, c2, hold_i)))
        damp_alpha = 0.0
        if "gdr_param" in kernels:
            cq, c1, c2 = kernels["gdr_param"]
            damp_alpha, _ = choose_damp_alpha(
                [p_twin[int(i)] for i in hold_i],
                [q_twins[int(i)] for i in hold_i],
                cq,
                c1,
                c2,
                [p_safe_twins[int(i)] for i in hold_i],
            )
            d_tvds = []
            for i in hold_i:
                p_u = unfold(q_twins[int(i)], cq, c1, c2)
                mix = damp_histogram(p_u, p_safe_twins[int(i)], damp_alpha)
                d_tvds.append(total_variation(mix, p_twin[int(i)]))
            cand_scores.append(("gdr_damped", float(np.mean(d_tvds))))
        res_hops = None if info_res is None else float(info_res.get("hops", 0.0))
        res_tf = (
            score_unfold_tvd(p_twin, q_twins, *kernels["gdr_residual"], tf_i)
            if "gdr_residual" in kernels
            else None
        )
        ab_tf = (
            score_unfold_tvd(p_twin, q_twins, *kernels["gdr_afterburn"], tf_i)
            if "gdr_afterburn" in kernels
            else None
        )
        gdr_tf = (
            score_unfold_tvd(p_twin, q_twins, *kernels["gdr_param"], tf_i)
            if "gdr_param" in kernels
            else None
        )
        ora_tf = (
            score_unfold_tvd(p_twin, q_twins, *kernels["oracle_binomial"], tf_i)
            if "oracle_binomial" in kernels
            else None
        )
        chosen, extra = select_research_method(
            cand_scores,
            residual_hops=res_hops,
            residual_tfree=res_tf,
            afterburn_tfree=ab_tf,
            gdr_tfree=gdr_tf,
            oracle_tfree=ora_tf,
            circuit_kind=circuit_kind,
        )
        if chosen == "safe":
            hist = p_safe_target
        elif chosen == "gdr_damped":
            cq, c1, c2 = kernels["gdr_param"]
            hist = damp_histogram(unfold(q_obs, cq, c1, c2), p_safe_target, damp_alpha)
        else:
            cq, c1, c2 = kernels[chosen]
            hist = unfold(q_obs, cq, c1, c2)
        out["gdr_select"] = {
            "hist": hist,
            "energy": energy_from_histogram(hist, energy_tensor),
            "fit": {
                "kind": "gdr_select",
                "chosen": chosen,
                "damp_alpha": float(damp_alpha),
                **extra,
            },
        }

    return out


def load_baseline() -> list[dict]:
    path = BASELINE_OUT / "results.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text()).get("records", [])


def baseline_match(rec: dict, baseline: list[dict]) -> dict | None:
    for b in baseline:
        if (
            b.get("ansatz") == rec["ansatz"]
            and b.get("params") == rec["params"]
            and b.get("family") == rec["family"]
            and abs(float(b.get("kappa_tau", -1)) - float(rec["kappa_tau"])) < 1e-12
            and b.get("readout") == rec["readout"]
        ):
            return b
    return None


def write_ablation_md(path: Path, records: list[dict], baseline: list[dict], header: str) -> None:
    lines = [
        "# Ablation summary",
        "",
        header,
        "",
        "TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).",
        "Same-run `gdr_param` is the controlled baseline for method changes.",
        "",
        "| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |",
        "|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    new_names = [
        m
        for m in ALL_METHODS
        if m
        not in (
            "raw",
            "readout_only",
            "oracle_binomial",
            "gdr_param",
            "gdr_full",
            "scalar_cdr",
            "zne_idle",
        )
    ]
    for rec in records:
        mets = rec["metrics"]
        raw = (mets.get("raw") or {}).get("tvd")
        gdr = (mets.get("gdr_param") or {}).get("tvd")
        best_name, best_tvd = None, None
        for name in new_names:
            tvd = (mets.get(name) or {}).get("tvd")
            if tvd is None:
                continue
            if best_tvd is None or tvd < best_tvd:
                best_tvd, best_name = tvd, name
        b = baseline_match(rec, baseline)
        b_raw = None if b is None else (b.get("metrics", {}).get("raw") or {}).get("tvd")
        b_gdr = None if b is None else (b.get("metrics", {}).get("gdr_param") or {}).get("tvd")
        delta = None if (best_tvd is None or gdr is None) else best_tvd - gdr

        def f(x):
            return "—" if x is None else f"{float(x):.4f}"

        lines.append(
            f"| {rec['ansatz']} | {rec['params']} | {rec['family']} | {rec['kappa_tau']:g} | "
            f"{rec['readout']} | {f(raw)} | {f(gdr)} | {f(best_tvd)} | {best_name or '—'} | "
            f"{f(b_raw)} | {f(b_gdr)} | {f(delta)} |"
        )
    lines += ["", "## Per-method TVD", ""]
    header2 = (
        f"{'ansatz':<5} {'params':<9} {'family':<22} {'kt':>5} {'readout':<18} "
        f"{'method':<18} {'TVD':>8} {'dE':>8}"
    )
    lines.append("```")
    lines.append(header2)
    lines.append("-" * len(header2))
    for rec in records:
        for method, met in rec["metrics"].items():
            tvd = met.get("tvd")
            de = met.get("dE")
            tvd_s = "   —   " if tvd is None else f"{tvd:8.4f}"
            de_s = "   —   " if de is None else f"{de:8.4f}"
            lines.append(
                f"{rec['ansatz']:<5} {rec['params']:<9} {rec['family']:<22} "
                f"{rec['kappa_tau']:5.3f} {rec['readout']:<18} {method:<18} {tvd_s} {de_s}"
            )
    lines.append("```")
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    copy_optimized_params(outdir)

    shots = int(args.shots)
    n_train = int(args.n_train)
    families = _parse_csv_str(args.families, CIRCUIT_FAMILIES)
    kappas = _parse_csv_floats(args.kappa_tau, (0.003, 0.1))
    readout_levels = (
        tuple(READOUT_SPECS)
        if args.readout == "all"
        else _parse_csv_str(args.readout, ("ideal", "readout_realistic", "readout_strong"))
    )
    ansätze = ("ecd", "snap") if args.ansatz == "both" else (args.ansatz,)
    param_names = ("random", "optimized") if args.params == "both" else (args.params,)
    if args.params == "auto":
        param_names = ("auto",)
    methods = _parse_csv_str(args.methods, CHEAP_METHODS)
    for m in methods:
        if m not in ALL_METHODS:
            raise ValueError(f"unknown method {m!r}; expected one of {ALL_METHODS}")

    inst = load_instance(int(args.instance))
    energy_tensor = np.asarray(inst["energy_tensor"], dtype=float)
    ground_qnm = tuple(int(v) for v in inst["ground_qnm"])
    hid = int(inst["hamiltonian_id"])
    fit_n_train = args.fit_n_train
    print(
        f"ablation tag={args.tag}  H{hid:03d}  shots={shots} n_train={n_train} "
        f"fit_n_train={fit_n_train or n_train}  twins={args.twin_design}  "
        f"methods={','.join(methods)}"
    )

    records: list[dict] = []
    for ansatz in ansätze:
        ndepth = int(ANSATZ_SPEC[ansatz]["ndepth"])
        print(f"\n=== {ansatz} N_d={ndepth} ===")
        x_random, x_opt, _meta = get_or_optimize_params(
            ansatz=ansatz,
            ndepth=ndepth,
            energy_tensor=energy_tensor,
            ground_qnm=ground_qnm,
            outdir=outdir,
            hid=hid,
            seed_base=int(args.seed),
            maxiter=200,
            n_restarts=3,
        )
        sim_e = make_sim(ansatz, ndepth, energy_tensor, ground_qnm)
        e_rand = float(energy_from_histogram(physical_probs(sim_e, x_random), energy_tensor))
        e_opt = float(energy_from_histogram(physical_probs(sim_e, x_opt), energy_tensor))
        print(
            f"  noiseless E_random={e_rand:.4f}  E_opt={e_opt:.4f}  "
            f"E0={float(inst.get('energy_min', float('nan'))):.4f}"
        )
        param_sets = {"random": x_random, "optimized": x_opt}
        e0 = float(inst.get("energy_min", float("nan")))
        auto_info = None
        jobs: list[tuple[str, object, str, dict | None]] = []
        if args.params == "auto":
            auto_info = classify_opt_quality(
                e_opt,
                e0,
                gap=inst.get("gap"),
                abs_tol=float(args.auto_abs_tol),
                rel_gap=float(args.auto_rel_gap),
            )
            recipe = str(auto_info["recipe"])
            print(
                f"  params=auto  deficit={auto_info['deficit']:.3f}  "
                f"thresh={auto_info['thresh']:.3f}  recipe={recipe}"
            )
            jobs.append(("optimized", x_opt, recipe, auto_info))
        else:
            for name in param_names:
                jobs.append((name, param_sets[name], name, None))
        for pset, xvec, circuit_kind, gate_info in jobs:
            args_phys = argparse.Namespace(**vars(args))
            if args.params == "auto":
                args_phys.twin_design = "span" if circuit_kind == "random" else "default"
            elif args.twin_design == "adaptive":
                args_phys.twin_design = "span" if circuit_kind == "random" else "default"
            for family in families:
                for kt in kappas:
                    phys = build_or_load_physics(
                        ansatz=ansatz,
                        pset=pset,
                        xvec=xvec,
                        family=family,
                        kt=float(kt),
                        n_train=n_train,
                        args=args_phys,
                        energy_tensor=energy_tensor,
                        ground_qnm=ground_qnm,
                        cache_dir=cache_dir,
                    )
                    phys = slice_twin_phys(phys, fit_n_train)
                    cfg = circuit_noise(family, float(kt), dims=DIMS)
                    for ro in readout_levels:
                        spec = readout_spec(ro, shots, seed=None)
                        q_obs, q_twins, hist_by_scale = _observe_block(
                            phys, spec, ansatz, pset, family, kt, args.seed, f"s{shots}"
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
                            fit_maxiter=int(args.fit_maxiter),
                            circuit_kind=circuit_kind,
                            family=family,
                            kappa_tau=float(kt),
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
                            "tag": args.tag,
                            "ansatz": ansatz,
                            "ndepth": ndepth,
                            "params": pset,
                            "circuit_kind": circuit_kind,
                            "auto": gate_info,
                            "family": family,
                            "kappa_tau": float(kt),
                            "readout": ro,
                            "n_shots": shots,
                            "n_train": int(phys.get("n_train", n_train)),
                            "n_train_phys": n_train,
                            "twin_design": args_phys.twin_design,
                            "twin_tag": twin_tag(args_phys),
                            "noise": noise_as_dict(cfg),
                            "readout_spec": readout_as_dict(spec),
                            "metrics": metrics,
                            "fits": {k: v.get("fit") for k, v in mitigated.items() if v.get("fit")},
                        }
                        records.append(rec)
                        gdr = (metrics.get("gdr_param") or {}).get("tvd")
                        raw = (metrics.get("raw") or {}).get("tvd")
                        sel = (metrics.get("gdr_select") or {}).get("tvd")
                        print(
                            f"    {pset:<9} kind={circuit_kind:<9} {family:<22} "
                            f"kt={kt:g} {ro:<20} "
                            f"raw={raw if raw is None else f'{raw:.4f}'}  "
                            f"gdr={gdr if gdr is None else f'{gdr:.4f}'}  "
                            f"sel={sel if sel is None else f'{sel:.4f}'}"
                        )

    run_dir = outdir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline()
    header = (
        f"tag=`{args.tag}` shots={shots} n_train={n_train} twin={args.twin_design} "
        f"ansatz={args.ansatz} params={args.params} families={','.join(families)} "
        f"kappa={','.join(str(k) for k in kappas)}"
    )
    result = {
        "tag": args.tag,
        "shots": shots,
        "n_train": n_train,
        "fit_n_train": fit_n_train,
        "seed": int(args.seed),
        "twin_design": args.twin_design,
        "twin_tag": twin_tag(args),
        "families": list(families),
        "kappa_tau": list(kappas),
        "readout_levels": list(readout_levels),
        "methods": list(methods),
        "hamiltonian_id": hid,
        "records": records,
    }
    (run_dir / "results.json").write_text(json.dumps(json_ready(result), indent=2))
    write_summary_txt(run_dir / "summary.txt", records, float(kappas[0]))
    write_ablation_md(run_dir / "ablation_summary.md", records, baseline, header)
    print(f"\nwrote {run_dir / 'results.json'}")
    print(f"wrote {run_dir / 'ablation_summary.md'}")
    return result


RESEARCH_SMOKE = {
    "tag": "research_smoke",
    "ansatz": "ecd",
    "params": "optimized",
    "families": "loss",
    "kappa_tau": "0.003",
    "shots": 2048,
    "n_train": 40,
    "n_rank2": 10,
    "twin_design": "adaptive",
    "readout": "ideal,readout_realistic",
    "methods": "raw,gdr_param,gdr_damped,gdr_select",
}


def apply_research_smoke(args: argparse.Namespace) -> argparse.Namespace:
    """Tiny adaptive-recipe slice; writes only under out_research/."""
    for key, val in RESEARCH_SMOKE.items():
        setattr(args, key, val)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset",
        choices=("none", "research_smoke"),
        default="none",
        help="research_smoke: cached ECD opt loss κτ=0.003 adaptive slice.",
    )
    p.add_argument("--tag", default="micro", help="Subdirectory under out_research/")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ansatz", choices=("ecd", "snap", "both"), default="ecd")
    p.add_argument(
        "--params",
        choices=("random", "optimized", "both", "auto"),
        default="both",
        help="auto: keep the optimized circuit but apply the random recipe "
        "if E_opt is worse than E0 by more than max(abs_tol, rel_gap*gap).",
    )
    p.add_argument("--auto-abs-tol", type=float, default=0.5)
    p.add_argument("--auto-rel-gap", type=float, default=0.2)
    p.add_argument("--instance", type=int, default=0)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument(
        "--fit-n-train",
        type=int,
        default=None,
        help="Fit on an evenly spaced subset of cached twins (does not resimulate).",
    )
    p.add_argument("--seed", type=int, default=SEED_BASE)
    p.add_argument("--families", default="loss")
    p.add_argument("--kappa-tau", default="0.003,0.1")
    p.add_argument("--readout", default="all")
    p.add_argument("--methods", default=",".join(CHEAP_METHODS))
    p.add_argument("--fit-maxiter", type=int, default=120)
    p.add_argument(
        "--twin-design",
        choices=("default", "wide", "span", "more_tfree", "adaptive"),
        default="default",
    )
    p.add_argument("--n-rank2", type=int, default=None)
    p.add_argument("--mag-lo", type=float, default=0.25)
    p.add_argument("--mag-hi", type=float, default=1.35)
    p.add_argument("--extra-t-free", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preset == "research_smoke":
        args = apply_research_smoke(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
