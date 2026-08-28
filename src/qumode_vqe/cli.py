"""Command-line experiment driver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_reference
from .hamiltonian import EXACT_GROUND_ENERGY, TARGET_QNM
from .measurement import MeasurementConfig, nearest_neighbor_fock_confusion, qubit_bitflip_confusion
from .noise import (
    LossModel,
    NoiseConfig,
    TimingMode,
    paper_loss_config,
    realistic_lindblad_config,
    comprehensive_config,
)
from .params import ParamLayout, random_parameters
from .plotting import plot_energy_history, plot_histogram, plot_kappa_overlay
from .vqe import HybridSimulator, evaluate_fixed_parameters, optimize_vqe


def _save_json(path: Path, payload: dict) -> None:
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    path.write_text(json.dumps(_convert(payload), indent=2), encoding="utf-8")


def _eval_to_dict(ev) -> dict:
    d = ev.as_dict()
    d["physicality"] = ev.physicality
    return d


def run_noiseless(outdir: Path, maxiter: int, seed: int, skip_optimize: bool) -> dict:
    ref = load_reference()
    sim = HybridSimulator(ndepth=5, layout=ParamLayout.PAPER)
    fixed = sim.evaluate(ref["xvec"], include_ideal=True)
    payload = {
        "reference_eval": _eval_to_dict(fixed),
        "reference_energy_published": ref["energy_after_200_bfgs"],
        "exact_ground_energy": EXACT_GROUND_ENERGY,
        "target_qnm": list(TARGET_QNM),
    }
    plot_histogram(
        fixed.measurement.physical_probs,
        outdir / "noiseless_reference_histogram.png",
        title="Reference parameters (noiseless)",
    )
    if not skip_optimize:
        rng = np.random.default_rng(seed)
        x0 = random_parameters(5, rng, ParamLayout.PAPER)
        opt = optimize_vqe(sim, x0, method="BFGS", maxiter=maxiter, record_every=max(1, maxiter // 20), verbose=True)
        payload["optimize"] = {
            "fun": opt.fun,
            "nit": opt.nit,
            "nfev": opt.nfev,
            "success": opt.success,
            "message": opt.message,
            "final": opt.history[-1] if opt.history else None,
        }
        np.savez(outdir / "noiseless_optimize.npz", x=opt.x, fun=np.array([opt.fun]))
        if opt.history:
            plot_energy_history(opt.history, outdir / "noiseless_energy.png")
            last = sim.evaluate(opt.x)
            plot_histogram(
                last.measurement.physical_probs,
                outdir / "noiseless_final_histogram.png",
                title=f"Noiseless BFGS ({opt.nit} iterations)",
            )
    _save_json(outdir / "noiseless.json", payload)
    return payload


def run_paper_loss(
    outdir: Path,
    maxiter: int,
    seed: int,
    skip_optimize: bool,
    kappas: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1),
) -> dict:
    ref = load_reference()
    x_ref = ref["xvec"]
    series = {}
    payload: dict = {"fixed": {}, "reoptimized": {}}
    for kt in kappas:
        noise = paper_loss_config(kt, TimingMode.PER_UER_LAYER)
        ev = evaluate_fixed_parameters(x_ref, noise)
        payload["fixed"][str(kt)] = _eval_to_dict(ev)
        series[rf"κτ={kt:g}"] = ev.measurement.physical_probs
        plot_histogram(
            ev.measurement.physical_probs,
            outdir / f"paper_loss_fixed_kt{kt:g}.png",
            title=f"Fixed parameters, paper Kraus κτ={kt:g}",
        )
        if not skip_optimize and kt > 0.0:
            sim = HybridSimulator(noise=noise)
            rng = np.random.default_rng(seed)
            # Warm start from the noiseless reference (noise-aware VQE).
            opt = optimize_vqe(
                sim,
                x_ref if kt <= 0.01 else random_parameters(5, rng),
                method="BFGS",
                maxiter=maxiter,
                record_every=max(1, maxiter // 10),
                verbose=True,
            )
            final = sim.evaluate(opt.x)
            payload["reoptimized"][str(kt)] = {
                "fun": opt.fun,
                "nit": opt.nit,
                "nfev": opt.nfev,
                "final": _eval_to_dict(final),
            }
            plot_histogram(
                final.measurement.physical_probs,
                outdir / f"paper_loss_reopt_kt{kt:g}.png",
                title=f"Reoptimized, paper Kraus κτ={kt:g}",
            )
    plot_kappa_overlay(series, outdir / "paper_loss_fixed_overlay.png")
    _save_json(outdir / "paper_loss.json", payload)
    return payload


def run_comprehensive(outdir: Path, seed: int) -> dict:
    ref = load_reference()
    x_ref = ref["xvec"]
    payload: dict = {"ablations": {}}

    cases = {
        "lindblad_cav_only": realistic_lindblad_config(
            TimingMode.PER_ECD_PAIR, enable_transmon=False, nth_cav=0.01, kappa_phi=2e3
        ),
        "transmon_t1t2": NoiseConfig(
            timing=TimingMode.PER_ECD_PAIR,
            loss_model=LossModel.NONE,
            enable_transmon=True,
        ),
        "coherent_control": NoiseConfig(
            loss_model=LossModel.NONE,
            rotation_rel_error=0.02,
            ecd_amp_rel_error=0.02,
            ecd_phase_error=0.05,
        ),
        "kerr_crosstalk": NoiseConfig(
            timing=TimingMode.PER_ECD_PAIR,
            loss_model=LossModel.NONE,
            kerr=2 * np.pi * 1e3,
            cross_kerr=2 * np.pi * 200.0,
            chi_dispersive=2 * np.pi * 50.0,
        ),
        "combined_lindblad": realistic_lindblad_config(TimingMode.PER_ECD_PAIR),
    }

    meas_readout = MeasurementConfig(
        qubit_c=qubit_bitflip_confusion(0.02, 0.03),
        fock1_c=nearest_neighbor_fock_confusion(8, 0.04),
        fock2_c=nearest_neighbor_fock_confusion(8, 0.04),
    )

    for name, noise in cases.items():
        ev = evaluate_fixed_parameters(x_ref, noise)
        payload["ablations"][name] = _eval_to_dict(ev)
        plot_histogram(
            ev.measurement.physical_probs,
            outdir / f"ablation_{name}.png",
            title=name,
        )

    ev_meas = evaluate_fixed_parameters(
        x_ref, NoiseConfig(loss_model=LossModel.NONE), measurement=meas_readout
    )
    payload["ablations"]["readout_confusion"] = _eval_to_dict(ev_meas)
    plot_histogram(
        ev_meas.measurement.observed_probs,
        outdir / "ablation_readout_confusion.png",
        title="Readout confusion (observed)",
    )

    combined_noise = comprehensive_config()
    ev_all = evaluate_fixed_parameters(x_ref, combined_noise, measurement=meas_readout)
    payload["ablations"]["combined_plus_readout"] = _eval_to_dict(ev_all)
    payload["timing"] = {
        "mode": combined_noise.timing.value,
        "tau_application_s": combined_noise.tau_application,
        "kappa_tau_per_application": combined_noise.kappa_tau_used(),
        "cumulative_kappa_t_nd5": combined_noise.cumulative_kappa_t(5),
        "t1_cav_s": combined_noise.t1_cav,
        "tau_ecd_s": combined_noise.tau_ecd,
        "t1_q_s": combined_noise.t1_q,
        "t2_q_s": combined_noise.t2_q,
    }
    _save_json(outdir / "comprehensive.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ECD-VQE experiment driver")
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument(
        "--mode",
        choices=("all", "noiseless", "paper-loss", "comprehensive"),
        default="all",
    )
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--full", action="store_true", help="Use 200 BFGS iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-optimize", action="store_true")
    args = parser.parse_args(argv)

    maxiter = 200 if args.full else args.maxiter
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {"maxiter": maxiter, "seed": args.seed, "mode": args.mode}
    if args.mode in ("all", "noiseless"):
        summary["noiseless"] = run_noiseless(outdir, maxiter, args.seed, args.skip_optimize)
    if args.mode in ("all", "paper-loss"):
        reopt_iter = maxiter if args.full else min(maxiter, 40)
        summary["paper_loss"] = run_paper_loss(
            outdir, reopt_iter, args.seed, args.skip_optimize and not args.full
        )
    if args.mode in ("all", "comprehensive"):
        summary["comprehensive"] = run_comprehensive(outdir, args.seed)
    _save_json(outdir / "summary.json", summary)
    print(f"Wrote results to {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
