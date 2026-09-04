#!/usr/bin/env python3
"""Run Gaussian Data Regression mitigation on one mixed p-spin instance.

Circuit noise (loss / thermal-dephasing / comprehensive) is applied inside
the hybrid simulator. Readout confusion is applied only to the final
|q, n, m> histogram. Gaussian twins of the target ECD/SNAP circuit are
used to fit a histogram transfer map, which is then inverted.

Usage
-----
    python Error_mitigation/run_mitigation_experiment.py --preset smoke
    python Error_mitigation/run_mitigation_experiment.py --preset full --ansatz both
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from qumode_vqe.hamiltonian import (
    DEFAULT_NFOCKS,
    DEFAULT_MIXED_P_SPIN_DIR,
    diagonal_hybrid_hamiltonian,
    load_mixed_p_spin_instances,
)
from qumode_vqe.measurement import energy_from_histogram, joint_probabilities, probabilities_from_ket
from qumode_vqe.noise import LossModel, NoiseConfig, noise_as_dict
from qumode_vqe.params import random_parameters, random_snap_parameters
from qumode_vqe.vqe import HybridSimulator, optimize_vqe

from Error_mitigation.metrics import compare_histograms, total_variation
from Error_mitigation.mitigation import (
    apply_scalar_cdr,
    choose_damp_alpha,
    choose_mix_alpha,
    damp_histogram,
    fit_gdr_afterburn,
    fit_gdr_full,
    fit_gdr_holdout,
    fit_gdr_mid,
    fit_gdr_param,
    fit_gdr_residual,
    fit_scalar_cdr,
    holdout_indices,
    moment_ratios,
    observe_histogram,
    oracle_kernels,
    oracle_residual,
    params_to_kernels,
    readout_then_zne,
    run_readout_only,
    safe_histogram,
    score_unfold_tvd,
    select_research_method,
    tfree_indices,
    unfold,
    zne_histogram,
)
from Error_mitigation.noise_models import (
    CIRCUIT_FAMILIES,
    READOUT_SPECS,
    circuit_noise,
    family_description,
    readout_as_dict,
    readout_spec,
    scale_noise,
)
from Error_mitigation.twins import build_twins, designed_twin_plan

HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = HERE / "out"
HAM_DIR = ROOT / DEFAULT_MIXED_P_SPIN_DIR
SEED_BASE = 2026
NFOCKS = DEFAULT_NFOCKS
DIMS = (2, int(NFOCKS[0]), int(NFOCKS[1]))

ANSATZ_SPEC = {
    "ecd": {"ndepth": 5},
    "snap": {"ndepth": 2},
}

PRESETS = {
    "smoke": {
        "shots": 4000,
        "n_train": 12,
        "kappa_tau": (0.003,),
        "readout": ("ideal", "readout_realistic"),
        "families": CIRCUIT_FAMILIES,
        "opt_maxiter": 5,
        "opt_restarts": 1,
        "fit_maxiter": 40,
        "zne_scales": (1, 2, 3),
    },
    "full": {
        "shots": 20000,
        "n_train": 40,
        "kappa_tau": (0.003, 0.03, 0.1),
        "readout": ("ideal", "readout_realistic", "readout_strong"),
        "families": CIRCUIT_FAMILIES,
        "opt_maxiter": 200,
        "opt_restarts": 3,
        "fit_maxiter": 200,
        "zne_scales": (1, 2, 3),
    },
}

HIST_METHODS = (
    "raw",
    "readout_only",
    "oracle_binomial",
    "gdr_param",
    "gdr_damped",
    "gdr_mid",
    "gdr_residual",
    "gdr_afterburn",
    "gdr_blend",
    "gdr_select",
    "gdr_full",
    "zne_idle",
    "readout_then_zne",
)
BAR_METHODS = ("raw", "readout_only", "gdr_param", "gdr_select", "oracle_binomial", "readout_then_zne")
METHOD_COLORS = {
    "ideal": "black",
    "raw": "0.55",
    "readout_only": "#e67e22",
    "oracle_binomial": "#27ae60",
    "gdr_param": "#2980b9",
    "gdr_damped": "#1abc9c",
    "gdr_mid": "#16a085",
    "gdr_residual": "#8e44ad",
    "gdr_afterburn": "#6c3483",
    "gdr_blend": "#5dade2",
    "gdr_select": "#2c3e50",
    "gdr_full": "#9b59b6",
    "zne_idle": "#c0392b",
    "readout_then_zne": "#d35400",
    "scalar_cdr": "#7f8c8d",
}
READOUT_STYLES = {
    "ideal": "-",
    "readout_realistic": "--",
    "readout_strong": ":",
}


def json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def case_seed(*parts) -> int:
    payload = "|".join(str(p) for p in parts).encode()
    return int(hashlib.md5(payload).hexdigest()[:8], 16)


def make_sim(
    ansatz: str,
    ndepth: int,
    energy_tensor: np.ndarray,
    ground_qnm: tuple[int, int, int],
    *,
    noise: NoiseConfig | None = None,
) -> HybridSimulator:
    tensor = np.asarray(energy_tensor, dtype=float)
    return HybridSimulator(
        ndepth=int(ndepth),
        nfocks=NFOCKS,
        noise=noise if noise is not None else NoiseConfig(loss_model=LossModel.NONE, dims=DIMS),
        measurement=None,
        energy_tensor=tensor,
        hamiltonian=diagonal_hybrid_hamiltonian(tensor),
        target_qnm=tuple(int(v) for v in ground_qnm),
        ansatz=str(ansatz),
        cost_kind="energy",
    )


def physical_probs(sim: HybridSimulator, xvec: np.ndarray) -> np.ndarray:
    x = np.asarray(xvec, dtype=float)
    if sim.noise.is_identity():
        psi = sim.statevector(x)
        return probabilities_from_ket(np.asarray(psi.full()).reshape(-1), sim.dims)
    rho = sim.density_matrix(x)
    return joint_probabilities(rho, sim.dims)


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def get_or_optimize_params(
    *,
    ansatz: str,
    ndepth: int,
    energy_tensor: np.ndarray,
    ground_qnm: tuple[int, int, int],
    outdir: Path,
    hid: int,
    seed_base: int,
    maxiter: int,
    n_restarts: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (x_random, x_optimized, meta). Cache the optimized vector."""
    rng = np.random.default_rng(int(seed_base) + 17 * (0 if ansatz == "ecd" else 1) + hid)
    if ansatz == "snap":
        x_random = random_snap_parameters(ndepth, NFOCKS, rng)
    else:
        x_random = random_parameters(ndepth, rng)

    cache_path = outdir / f"optimized_params_{ansatz}_h{hid:03d}_nd{ndepth}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if (
            int(cached.get("maxiter", -1)) >= int(maxiter)
            and int(cached.get("n_restarts", -1)) >= int(n_restarts)
            and int(cached.get("seed_base", -1)) == int(seed_base)
            and str(cached.get("ansatz")) == ansatz
            and int(cached.get("ndepth", -1)) == int(ndepth)
            and int(cached.get("hamiltonian_id", -1)) == int(hid)
        ):
            x_opt = np.asarray(cached["x"], dtype=float)
            print(f"  loaded optimized {ansatz} params from {cache_path.name}")
            return x_random, x_opt, cached

    sim = make_sim(ansatz, ndepth, energy_tensor, ground_qnm)
    best = None
    history = []
    t0 = time.time()
    for r in range(int(n_restarts)):
        rng_r = np.random.default_rng(int(seed_base) + 1009 * r + (0 if ansatz == "ecd" else 53))
        if ansatz == "snap":
            x0 = random_snap_parameters(ndepth, NFOCKS, rng_r)
        else:
            x0 = random_parameters(ndepth, rng_r)
        opt = optimize_vqe(
            sim,
            x0,
            method="L-BFGS-B",
            maxiter=int(maxiter),
            record_every=0,
            verbose=False,
        )
        rec = {"restart": r, "fun": float(opt.fun), "success": bool(opt.success), "nfev": int(opt.nfev)}
        history.append(rec)
        print(f"  {ansatz} restart {r + 1}/{n_restarts}: E={opt.fun:.6f} nfev={opt.nfev}")
        if best is None or float(opt.fun) < float(best.fun):
            best = opt
    elapsed = time.time() - t0
    meta = {
        "ansatz": ansatz,
        "ndepth": int(ndepth),
        "hamiltonian_id": int(hid),
        "seed_base": int(seed_base),
        "maxiter": int(maxiter),
        "n_restarts": int(n_restarts),
        "fun": float(best.fun),
        "nfev": int(best.nfev),
        "elapsed_s": elapsed,
        "restarts": history,
        "x": np.asarray(best.x, dtype=float).tolist(),
        "x_random": np.asarray(x_random, dtype=float).tolist(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(json_ready(meta), indent=2))
    print(f"  wrote {cache_path.name}  E={best.fun:.6f}  ({elapsed:.1f}s)")
    return x_random, np.asarray(best.x, dtype=float), meta


def _top_bins(p_ideal: np.ndarray, k: int = 12) -> list[tuple[int, int, int]]:
    order = np.argsort(np.asarray(p_ideal).ravel())[::-1][:k]
    return [tuple(int(v) for v in np.unravel_index(int(i), p_ideal.shape)) for i in order]


def plot_case_histograms(path: Path, title: str, rows: list[dict], energy_tensor: np.ndarray) -> None:
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11.5, 3.1 * n_rows), squeeze=False)
    for r, rec in enumerate(rows):
        p_id = np.asarray(rec["p_ideal"])
        bins = _top_bins(p_id, 12)
        labels = [f"|{q}{n}{m}⟩" for q, n, m in bins]
        methods = ["ideal"] + [m for m in BAR_METHODS if rec["hists"].get(m) is not None]
        x = np.arange(len(bins))
        width = 0.8 / max(len(methods), 1)
        ax = axes[r, 0]
        for j, name in enumerate(methods):
            hist = p_id if name == "ideal" else rec["hists"][name]
            vals = [float(hist[q, n, m]) for q, n, m in bins]
            ax.bar(
                x + (j - 0.5 * (len(methods) - 1)) * width,
                vals,
                width=width,
                label=name,
                color=METHOD_COLORS.get(name, f"C{j}"),
                edgecolor="none",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("probability")
        ax.set_title(f"κτ = {rec['kappa_tau']}" + (f"   TVD raw={rec['tvd_raw']:.3f}" if rec.get("tvd_raw") is not None else ""))
        ax.legend(frameon=False, fontsize=7, ncol=2)
        _style(ax)

        for col, mode, ylab in ((1, 1, "P(n₁)"), (2, 2, "P(n₂)")):
            axm = axes[r, col]
            for name in methods:
                hist = p_id if name == "ideal" else rec["hists"][name]
                marg = hist.sum(axis=(0, 2)) if mode == 1 else hist.sum(axis=(0, 1))
                axm.plot(np.arange(marg.size), marg, label=name, color=METHOD_COLORS.get(name, "C0"), lw=1.4)
            axm.set_xlabel(f"n_{mode}")
            axm.set_ylabel(ylab)
            _style(axm)
            if r == 0:
                axm.legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_summary(path: Path, records: list[dict], ansatz: str) -> None:
    families = list(dict.fromkeys(r["family"] for r in records if r["ansatz"] == ansatz))
    param_sets = ("random", "optimized")
    fig, axes = plt.subplots(len(families), 4, figsize=(13.5, 3.0 * max(len(families), 1)), squeeze=False)
    methods = (
        "raw",
        "readout_only",
        "oracle_binomial",
        "gdr_param",
        "gdr_damped",
        "gdr_mid",
        "gdr_residual",
        "gdr_afterburn",
        "gdr_blend",
        "gdr_select",
        "gdr_full",
        "zne_idle",
        "readout_then_zne",
        "scalar_cdr",
    )
    for i, fam in enumerate(families):
        for j, pset in enumerate(param_sets):
            for k, metric in enumerate(("tvd", "dE")):
                ax = axes[i, 2 * j + k]
                for method in methods:
                    if metric == "tvd" and method == "scalar_cdr":
                        continue
                    for ro, ls in READOUT_STYLES.items():
                        xs, ys = [], []
                        for rec in records:
                            if rec["ansatz"] != ansatz or rec["family"] != fam:
                                continue
                            if rec["params"] != pset or rec["readout"] != ro:
                                continue
                            m = rec["metrics"].get(method)
                            if not m or m.get(metric) is None:
                                continue
                            xs.append(float(rec["kappa_tau"]))
                            ys.append(float(m[metric]))
                        if not xs:
                            continue
                        order = np.argsort(xs)
                        ax.plot(
                            np.asarray(xs)[order],
                            np.asarray(ys)[order],
                            ls=ls,
                            color=METHOD_COLORS.get(method, "C0"),
                            marker="o",
                            ms=3.5,
                            label=f"{method}" if ro == "ideal" else None,
                        )
                ax.set_xscale("log")
                ax.set_xlabel("κτ per application")
                ax.set_ylabel(metric)
                ax.set_title(f"{fam}  {pset}  {metric}")
                _style(ax)
                if i == 0 and j == 0 and k == 0:
                    ax.legend(frameon=False, fontsize=6)
    fig.suptitle(f"{ansatz} mitigation vs κτ  (solid=ideal readout, dashed=realistic, dotted=strong)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _fmt(val, digits=4):
    if val is None:
        return "   —   "
    return f"{float(val):.{digits}f}"


def write_summary_txt(path: Path, records: list[dict], headline_kt: float) -> None:
    lines = [
        "Gaussian Data Regression on mixed p-spin (hybrid ECD / SNAP)",
        "Methods: raw, readout_only, oracle_binomial, gdr_param, gdr_damped, gdr_mid, "
        "gdr_residual, gdr_afterburn, gdr_blend, gdr_select, gdr_full, scalar_cdr, "
        "zne_idle, readout_then_zne",
        "",
        "Headline at κτ = "
        + str(headline_kt)
        + "  (readout_only vs gdr_param vs gdr_select vs oracle_binomial)",
        "",
    ]
    head_methods = ("raw", "readout_only", "oracle_binomial", "gdr_param", "gdr_select")
    for rec in records:
        if abs(float(rec["kappa_tau"]) - float(headline_kt)) > 1e-12:
            continue
        if rec["params"] != "optimized":
            continue
        tag = f"{rec['ansatz']}  {rec['family']:<24}  {rec['readout']:<20}"
        bits = []
        for m in head_methods:
            met = rec["metrics"].get(m) or {}
            bits.append(f"{m}.TVD={_fmt(met.get('tvd'))} dE={_fmt(met.get('dE'))}")
        lines.append(tag + "  " + "  ".join(bits))
    lines += ["", "Full table (all cases)", ""]
    header = (
        f"{'ansatz':<6} {'params':<10} {'family':<24} {'kt':>6} {'readout':<18} "
        f"{'method':<18} {'TVD':>8} {'dE':>8} {'dPgs':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for rec in records:
        for method, met in rec["metrics"].items():
            lines.append(
                f"{rec['ansatz']:<6} {rec['params']:<10} {rec['family']:<24} "
                f"{rec['kappa_tau']:6.3f} {rec['readout']:<18} {method:<18} "
                f"{_fmt(met.get('tvd')):>8} {_fmt(met.get('dE')):>8} {_fmt(met.get('dPgs')):>8}"
            )
    path.write_text("\n".join(lines) + "\n")


def mitigate_target(
    *,
    p_ideal: np.ndarray,
    q_obs: np.ndarray,
    p_twin_ideal: list[np.ndarray],
    q_twin_obs: list[np.ndarray],
    e_twin_ideal: np.ndarray,
    e_twin_noisy: np.ndarray,
    cfg: NoiseConfig,
    spec,
    ndepth: int,
    energy_tensor: np.ndarray,
    hist_by_scale: dict[int, np.ndarray],
    fit_maxiter: int,
    t_free: list[int] | None = None,
    circuit_kind: str | None = None,
) -> dict:
    """Run every mitigation method on one (target, readout) histogram."""
    out: dict = {}
    e_obs = energy_from_histogram(q_obs, energy_tensor)
    out["raw"] = {"hist": q_obs, "energy": e_obs}
    p_safe = safe_histogram(q_obs, spec, DIMS)
    p_safe_twins = [safe_histogram(q, spec, DIMS) for q in q_twin_obs]
    kernels: dict[str, tuple] = {}

    ro = run_readout_only(q_obs, spec, DIMS)
    if ro is not None:
        out["readout_only"] = {"hist": ro.histogram, "energy": energy_from_histogram(ro.histogram, energy_tensor)}

    cq_o, c1_o, c2_o = oracle_kernels(cfg, spec, ndepth, DIMS)
    kernels["oracle_binomial"] = (cq_o, c1_o, c2_o)
    p_oracle = unfold(q_obs, cq_o, c1_o, c2_o)
    out["oracle_binomial"] = {
        "hist": p_oracle,
        "energy": energy_from_histogram(p_oracle, energy_tensor),
        "residual_tvd": oracle_residual(p_ideal, q_obs, cq_o, c1_o, c2_o),
        "true_eta": float(np.exp(-cfg.cumulative_kappa_t(ndepth))),
    }

    theta, fit_info = fit_gdr_param(
        p_twin_ideal, q_twin_obs, cfg, spec, ndepth, DIMS, maxiter=fit_maxiter
    )
    cq, c1, c2 = params_to_kernels(theta, DIMS)
    kernels["gdr_param"] = (cq, c1, c2)
    p_gdr = unfold(q_obs, cq, c1, c2)
    p_gdr_nnls = unfold(q_obs, cq, c1, c2, method="nnls")
    out["gdr_param"] = {
        "hist": p_gdr,
        "hist_nnls": p_gdr_nnls,
        "energy": energy_from_histogram(p_gdr, energy_tensor),
        "fit": fit_info,
        "residual_tvd": oracle_residual(p_ideal, q_obs, cq, c1, c2),
    }

    alpha, ainfo = choose_damp_alpha(p_twin_ideal, q_twin_obs, cq, c1, c2, p_safe_twins)
    p_damped = damp_histogram(p_gdr, p_safe, alpha)
    out["gdr_damped"] = {
        "hist": p_damped,
        "energy": energy_from_histogram(p_damped, energy_tensor),
        "fit": {**fit_info, **ainfo, "kind": "gdr_damped"},
    }

    theta_m, info_m = fit_gdr_mid(p_twin_ideal, q_twin_obs, cfg, spec, ndepth, DIMS, maxiter=fit_maxiter)
    cq_m, c1_m, c2_m = params_to_kernels(theta_m, DIMS)
    kernels["gdr_mid"] = (cq_m, c1_m, c2_m)
    p_mid = unfold(q_obs, cq_m, c1_m, c2_m)
    out["gdr_mid"] = {"hist": p_mid, "energy": energy_from_histogram(p_mid, energy_tensor), "fit": info_m}

    (cq_r, c1_r, c2_r), info_r = fit_gdr_residual(
        p_twin_ideal, q_twin_obs, cfg, spec, ndepth, DIMS, maxiter=min(int(fit_maxiter), 120), t_free=t_free
    )
    kernels["gdr_residual"] = (cq_r, c1_r, c2_r)
    p_res = unfold(q_obs, cq_r, c1_r, c2_r)
    out["gdr_residual"] = {
        "hist": p_res,
        "energy": energy_from_histogram(p_res, energy_tensor),
        "fit": info_r,
        "residual_tvd": oracle_residual(p_ideal, q_obs, cq_r, c1_r, c2_r),
    }

    (cq_ab, c1_ab, c2_ab), info_ab = fit_gdr_afterburn(
        p_twin_ideal, q_twin_obs, cfg, spec, ndepth, DIMS, maxiter=min(int(fit_maxiter), 120), t_free=t_free
    )
    kernels["gdr_afterburn"] = (cq_ab, c1_ab, c2_ab)
    p_ab = unfold(q_obs, cq_ab, c1_ab, c2_ab)
    out["gdr_afterburn"] = {
        "hist": p_ab,
        "energy": energy_from_histogram(p_ab, energy_tensor),
        "fit": info_ab,
        "residual_tvd": oracle_residual(p_ideal, q_obs, cq_ab, c1_ab, c2_ab),
    }

    gdr_u = [unfold(q, cq, c1, c2) for q in q_twin_obs]
    ora_u = [unfold(q, cq_o, c1_o, c2_o) for q in q_twin_obs]
    beta, binfo = choose_mix_alpha(p_twin_ideal, gdr_u, ora_u)
    p_blend = damp_histogram(p_gdr, p_oracle, beta)
    out["gdr_blend"] = {
        "hist": p_blend,
        "energy": energy_from_histogram(p_blend, energy_tensor),
        "fit": {**fit_info, **binfo, "kind": "gdr_blend"},
    }

    theta_h, info_h = fit_gdr_holdout(p_twin_ideal, q_twin_obs, cfg, spec, ndepth, DIMS, maxiter=fit_maxiter)
    kernels["gdr_holdout"] = params_to_kernels(theta_h, DIMS)
    train_i, hold_i = holdout_indices(len(p_twin_ideal), 0.25)
    if hold_i.size == 0:
        hold_i = train_i
    tf_i = tfree_indices(t_free)
    if tf_i.size == 0:
        tf_i = hold_i
    cand = [("safe", float(np.mean([total_variation(p_safe_twins[int(i)], p_twin_ideal[int(i)]) for i in hold_i])))]
    for name, (kcq, kc1, kc2) in kernels.items():
        cand.append((name, score_unfold_tvd(p_twin_ideal, q_twin_obs, kcq, kc1, kc2, hold_i)))
    # damp is an explicit candidate (do not apply it after another pick)
    alpha_sel, _ = choose_damp_alpha(
        [p_twin_ideal[int(i)] for i in hold_i],
        [q_twin_obs[int(i)] for i in hold_i],
        cq,
        c1,
        c2,
        [p_safe_twins[int(i)] for i in hold_i],
    )
    d_tvds = []
    for i in hold_i:
        mix = damp_histogram(unfold(q_twin_obs[int(i)], cq, c1, c2), p_safe_twins[int(i)], alpha_sel)
        d_tvds.append(total_variation(mix, p_twin_ideal[int(i)]))
    cand.append(("gdr_damped", float(np.mean(d_tvds))))
    chosen, extra = select_research_method(
        cand,
        residual_hops=float(info_r.get("hops", 0.0)),
        residual_tfree=score_unfold_tvd(p_twin_ideal, q_twin_obs, cq_r, c1_r, c2_r, tf_i),
        afterburn_tfree=score_unfold_tvd(p_twin_ideal, q_twin_obs, cq_ab, c1_ab, c2_ab, tf_i),
        gdr_tfree=score_unfold_tvd(p_twin_ideal, q_twin_obs, cq, c1, c2, tf_i),
        oracle_tfree=score_unfold_tvd(p_twin_ideal, q_twin_obs, cq_o, c1_o, c2_o, tf_i),
        circuit_kind=circuit_kind,
    )
    if chosen == "safe":
        hist_sel = p_safe
    elif chosen == "gdr_damped":
        hist_sel = damp_histogram(p_gdr, p_safe, alpha_sel)
    else:
        hist_sel = unfold(q_obs, *kernels[chosen])
    out["gdr_select"] = {
        "hist": hist_sel,
        "energy": energy_from_histogram(hist_sel, energy_tensor),
        "fit": {"kind": "gdr_select", "chosen": chosen, "damp_alpha": float(alpha_sel), **extra, **info_h},
    }

    cq_f, c1_f, c2_f = fit_gdr_full(p_twin_ideal, q_twin_obs, cq, c1, c2)
    p_full = unfold(q_obs, cq_f, c1_f, c2_f)
    out["gdr_full"] = {
        "hist": p_full,
        "energy": energy_from_histogram(p_full, energy_tensor),
        "residual_tvd": oracle_residual(p_ideal, q_obs, cq_f, c1_f, c2_f),
    }

    a1, a0 = fit_scalar_cdr(e_twin_ideal, e_twin_noisy)
    out["scalar_cdr"] = {"hist": None, "energy": apply_scalar_cdr(e_obs, a1, a0), "a1": a1, "a0": a0}

    if hist_by_scale and 1 in hist_by_scale and 2 in hist_by_scale:
        p_lin = zne_histogram(hist_by_scale, degree=1)
        extra = {"hist_linear": p_lin}
        if 3 in hist_by_scale:
            p_quad = zne_histogram(hist_by_scale, degree=2)
            extra["hist"] = p_quad
            extra["hist_quadratic"] = p_quad
        else:
            extra["hist"] = p_lin
        extra["energy"] = energy_from_histogram(extra["hist"], energy_tensor)
        out["zne_idle"] = extra
        p_hyb = readout_then_zne(hist_by_scale, spec, DIMS)
        out["readout_then_zne"] = {"hist": p_hyb, "energy": energy_from_histogram(p_hyb, energy_tensor)}
    return out


def load_instance(hid: int) -> dict:
    ham_dir = HAM_DIR if HAM_DIR.is_dir() else DEFAULT_MIXED_P_SPIN_DIR
    instances = load_mixed_p_spin_instances(ham_dir, nfocks=NFOCKS)
    for inst in instances:
        if int(inst["hamiltonian_id"]) == int(hid):
            return inst
    raise FileNotFoundError(f"mixed p-spin hamiltonian_id={hid} not found in {ham_dir}")


def run(args: argparse.Namespace) -> dict:
    preset = PRESETS[args.preset]
    shots = int(args.shots if args.shots is not None else preset["shots"])
    n_train = int(args.n_train if args.n_train is not None else preset["n_train"])
    readout_levels = (
        tuple(READOUT_SPECS)
        if args.readout == "all"
        else ((args.readout,) if args.readout else preset["readout"])
    )
    families = tuple(preset["families"])
    if getattr(args, "families", None):
        families = tuple(x.strip() for x in str(args.families).split(",") if x.strip())
    kappas = tuple(preset["kappa_tau"])
    if getattr(args, "kappa_tau", None):
        kappas = tuple(float(x.strip()) for x in str(args.kappa_tau).split(",") if x.strip())
    param_filter = getattr(args, "params", None) or "both"
    ansätze = ("ecd", "snap") if args.ansatz == "both" else (args.ansatz,)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inst = load_instance(int(args.instance))
    energy_tensor = np.asarray(inst["energy_tensor"], dtype=float)
    ground_qnm = tuple(int(v) for v in inst["ground_qnm"])
    hid = int(inst["hamiltonian_id"])
    print(
        f"instance H{hid:03d}  file={inst.get('file')}  "
        f"E0={inst['energy_min']:.4f}  ground={inst.get('ground_bitstring')} {ground_qnm}"
    )

    records: list[dict] = []
    payload_cases: list[dict] = []
    poisson_tvds: list[float] = []
    product_tvds: list[float] = []

    for ansatz in ansätze:
        ndepth = int(ANSATZ_SPEC[ansatz]["ndepth"])
        print(f"\n=== {ansatz}  N_d={ndepth} ===")
        x_random, x_opt, opt_meta = get_or_optimize_params(
            ansatz=ansatz,
            ndepth=ndepth,
            energy_tensor=energy_tensor,
            ground_qnm=ground_qnm,
            outdir=outdir,
            hid=hid,
            seed_base=int(args.seed),
            maxiter=int(preset["opt_maxiter"]),
            n_restarts=int(preset["opt_restarts"]),
        )
        param_sets = {"random": x_random, "optimized": x_opt}
        if param_filter != "both":
            param_sets = {param_filter: param_sets[param_filter]}
        sim_ideal = make_sim(ansatz, ndepth, energy_tensor, ground_qnm)

        for pset, xvec in param_sets.items():
            rng_tw = np.random.default_rng(case_seed("twins", ansatz, pset, args.seed))
            print(f"  building {n_train} Gaussian twins for {pset} ...")
            twin_design = getattr(args, "twin_design", "span") or "span"
            if twin_design == "adaptive":
                twin_design = "span" if pset == "random" else "default"
            if twin_design == "span":
                t_list, scales = designed_twin_plan(
                    n_train,
                    ndepth,
                    n_rank2=getattr(args, "n_rank2", None),
                    mag_lo=float(getattr(args, "mag_scale_min", None) or 0.25),
                    mag_hi=float(getattr(args, "mag_scale_max", None) or 1.35),
                )
                twins = build_twins(sim_ideal, xvec, rng_tw, t_free_list=t_list, mag_scales=scales)
            else:
                twins = build_twins(
                    sim_ideal,
                    xvec,
                    rng_tw,
                    n_train=n_train,
                    n_rank2=getattr(args, "n_rank2", None),
                    mag_scale_range=(
                        float(getattr(args, "mag_scale_min", 0.5) or 0.5),
                        float(getattr(args, "mag_scale_max", 1.0) or 1.0),
                    ),
                )
            poisson_tvds.extend(t.poisson_tvd for t in twins if t.poisson_tvd is not None)
            product_tvds.extend(t.product_tvd for t in twins if t.product_tvd is not None)
            p_ideal = physical_probs(sim_ideal, xvec)
            e_ideal = energy_from_histogram(p_ideal, energy_tensor)
            e_twin_ideal = np.array([energy_from_histogram(t.p_ideal, energy_tensor) for t in twins])

            plot_rows: dict[tuple[str, str], list[dict]] = {}
            for family in families:
                for kt in kappas:
                    cfg = circuit_noise(family, kt, dims=DIMS)
                    sim_noisy = make_sim(ansatz, ndepth, energy_tensor, ground_qnm, noise=cfg)
                    print(f"  sim {family}  κτ={kt}  {pset}  (target + {len(twins)} twins) ...")
                    t_sim = time.time()
                    p_phys = physical_probs(sim_noisy, xvec)
                    twin_phys = [physical_probs(sim_noisy, tw.x) for tw in twins]
                    hist_phys_scale = {1: p_phys}
                    for s in preset["zne_scales"]:
                        if int(s) == 1:
                            continue
                        cfg_s = scale_noise(cfg, float(s))
                        sim_s = make_sim(ansatz, ndepth, energy_tensor, ground_qnm, noise=cfg_s)
                        hist_phys_scale[int(s)] = physical_probs(sim_s, xvec)
                    print(f"    physical histograms in {time.time() - t_sim:.1f}s")

                    eta_true = float(np.exp(-cfg.cumulative_kappa_t(ndepth)))
                    moments = moment_ratios(p_phys, p_ideal, eta_true)

                    for ro in readout_levels:
                        spec = readout_spec(ro, shots, seed=None)
                        seed_t = case_seed("obs", ansatz, pset, family, kt, ro, args.seed, "target")
                        q_obs = observe_histogram(p_phys, spec, DIMS, seed_t)
                        q_twins = [
                            observe_histogram(
                                tp,
                                spec,
                                DIMS,
                                case_seed("obs", ansatz, pset, family, kt, ro, args.seed, "twin", i),
                            )
                            for i, tp in enumerate(twin_phys)
                        ]
                        e_twin_noisy = np.array([energy_from_histogram(q, energy_tensor) for q in q_twins])
                        hist_by_scale = {
                            s: observe_histogram(
                                hist_phys_scale[s],
                                spec,
                                DIMS,
                                case_seed("obs", ansatz, pset, family, kt, ro, args.seed, "zne", s),
                            )
                            for s in hist_phys_scale
                        }
                        mitigated = mitigate_target(
                            p_ideal=p_ideal,
                            q_obs=q_obs,
                            p_twin_ideal=[t.p_ideal for t in twins],
                            q_twin_obs=q_twins,
                            e_twin_ideal=e_twin_ideal,
                            e_twin_noisy=e_twin_noisy,
                            cfg=cfg,
                            spec=spec,
                            ndepth=ndepth,
                            energy_tensor=energy_tensor,
                            hist_by_scale=hist_by_scale,
                            fit_maxiter=int(preset["fit_maxiter"]),
                            t_free=[int(t.t_free) for t in twins],
                            circuit_kind=pset,
                        )
                        metrics = {}
                        for name, blob in mitigated.items():
                            metrics[name] = compare_histograms(
                                blob.get("hist"),
                                p_ideal,
                                energy_tensor,
                                ground_qnm,
                                energy_mit=blob.get("energy"),
                            )
                        rec = {
                            "ansatz": ansatz,
                            "ndepth": ndepth,
                            "params": pset,
                            "family": family,
                            "kappa_tau": float(kt),
                            "readout": ro,
                            "n_shots": shots,
                            "n_train": n_train,
                            "noise": noise_as_dict(cfg),
                            "readout_spec": readout_as_dict(spec),
                            "metrics": metrics,
                            "oracle_residual_tvd": mitigated.get("oracle_binomial", {}).get("residual_tvd"),
                            "gdr_fit": mitigated.get("gdr_param", {}).get("fit"),
                            "scalar_cdr": {
                                "a1": mitigated.get("scalar_cdr", {}).get("a1"),
                                "a0": mitigated.get("scalar_cdr", {}).get("a0"),
                            },
                            "moments": moments,
                            "energy_ideal": e_ideal,
                        }
                        records.append(rec)
                        tvd_raw = metrics["raw"]["tvd"]
                        tvd_gdr = (metrics.get("gdr_param") or {}).get("tvd")
                        print(
                            f"    {ro:<20}  raw TVD={tvd_raw:.4f}  "
                            f"gdr={_fmt(tvd_gdr)}  oracle={_fmt((metrics.get('oracle_binomial') or {}).get('tvd'))}"
                        )
                        hists = {name: blob["hist"] for name, blob in mitigated.items() if blob.get("hist") is not None}
                        plot_rows.setdefault((family, ro), []).append(
                            {
                                "kappa_tau": float(kt),
                                "p_ideal": p_ideal,
                                "hists": hists,
                                "tvd_raw": tvd_raw,
                            }
                        )
                        payload_cases.append(
                            {
                                **rec,
                                "p_ideal": p_ideal.tolist(),
                                "q_obs": q_obs.tolist(),
                                "hists": {k: v.tolist() for k, v in hists.items()},
                            }
                        )

            for (family, ro), rows in plot_rows.items():
                fig_path = outdir / f"hist_{ansatz}_{pset}_{family}_{ro}.png"
                plot_case_histograms(
                    fig_path,
                    f"{ansatz} {pset}  {family}  {ro}  H{hid:03d}",
                    rows,
                    energy_tensor,
                )

        plot_summary(outdir / f"summary_tvd_{ansatz}.png", records, ansatz)

    headline_kt = float(kappas[0])
    write_summary_txt(outdir / "summary.txt", records, headline_kt)
    result = {
        "preset": args.preset,
        "hamiltonian_id": hid,
        "file": str(inst.get("file")),
        "ground_qnm": list(ground_qnm),
        "ground_bitstring": inst.get("ground_bitstring"),
        "energy_min": inst.get("energy_min"),
        "nfocks": list(NFOCKS),
        "seed": int(args.seed),
        "shots": shots,
        "n_train": n_train,
        "families": list(families),
        "family_descriptions": {f: family_description(f) for f in families},
        "kappa_tau": list(kappas),
        "readout_levels": list(readout_levels),
        "methods": list(HIST_METHODS) + ["scalar_cdr"],
        "twin_design": getattr(args, "twin_design", "span"),
        "poisson_tvd_max": None if not poisson_tvds else float(max(poisson_tvds)),
        "poisson_tvd_mean": None if not poisson_tvds else float(np.mean(poisson_tvds)),
        "product_tvd_max": None if not product_tvds else float(max(product_tvds)),
        "product_tvd_mean": None if not product_tvds else float(np.mean(product_tvds)),
        "records": records,
        "cases": payload_cases,
    }
    (outdir / "results.json").write_text(json.dumps(json_ready(result), indent=2))
    print(f"\nwrote {outdir / 'results.json'}")
    print(f"wrote {outdir / 'summary.txt'}")
    if product_tvds:
        print(f"Gaussian-twin product-state TVD: mean={np.mean(product_tvds):.3e}  max={max(product_tvds):.3e}")
    if poisson_tvds:
        print(f"Gaussian-twin Poisson TVD (diagnostic): mean={np.mean(poisson_tvds):.3e}  max={max(poisson_tvds):.3e}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    p.add_argument("--ansatz", choices=("ecd", "snap", "both"), default="both")
    p.add_argument("--instance", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED_BASE)
    p.add_argument(
        "--readout",
        choices=("ideal", "readout_realistic", "readout_strong", "all"),
        default=None,
        help="Override the preset readout levels.",
    )
    p.add_argument("--families", default=None, help="Comma-separated circuit families.")
    p.add_argument("--kappa-tau", default=None, dest="kappa_tau", help="Comma-separated κτ values.")
    p.add_argument("--params", choices=("random", "optimized", "both"), default="both")
    p.add_argument("--n-rank2", type=int, default=None, dest="n_rank2")
    p.add_argument("--mag-scale-min", type=float, default=None, dest="mag_scale_min")
    p.add_argument("--mag-scale-max", type=float, default=None, dest="mag_scale_max")
    p.add_argument(
        "--twin-design",
        choices=("default", "span", "adaptive"),
        default="span",
        dest="twin_design",
        help="span: log-spaced |α|. default: U(0.5,1) PR #6 mix. adaptive: span on random, default on optimized.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
