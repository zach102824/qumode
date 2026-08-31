#!/usr/bin/env python3
"""Alternating even/odd-CZ hardware-efficient ansatz on mixed p-spin instances.

One HEA layer is a single column of Ry gates, then two unparameterized CZ
groups, with no Rz:

    Ry on all 7 qubits
    even CZ on (0,1), (2,3), (4,5)
    odd  CZ on (1,2), (3,4), (5,6)

The circuit repeats that layer 6 times (42 Ry parameters; 7 does not divide
the earlier 40-parameter target). The script writes a diagram to ``circuit/``,
prints the circuit, then SPSA-optimizes the Gibbs (or energy) cost on all 20
saved mixed p-spin Hamiltonians.
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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Shared helpers from the product-R_y baseline
# ---------------------------------------------------------------------------
_RY_PATH = Path(__file__).resolve().parent / "mixed_p_spin_ry_spsa.py"
_SPEC = importlib.util.spec_from_file_location("mixed_p_spin_ry_spsa_lib", _RY_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"could not load {_RY_PATH}")
ry = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ry
_SPEC.loader.exec_module(ry)

# ---------------------------------------------------------------------------
# HEA layout: one layer = Ry × 7, then even CZ, then odd CZ. No Rz.
# 6 layers × 7 Ry = 42 parameters (nearest to 40 that 7 qubits allow).
# ---------------------------------------------------------------------------
N_QUBITS = 7
N_LAYERS = 6
PARAMS_PER_LAYER = N_QUBITS
N_PARAMS = N_LAYERS * PARAMS_PER_LAYER
EVEN_PAIRS = ((0, 1), (2, 3), (4, 5))
ODD_PAIRS = ((1, 2), (3, 4), (5, 6))

CIRCUIT_DIR = Path("circuit")
OUTDIR = Path("results")
HAM_DIR = Path("Hamiltonians") / "mixed_p_spin"
OUTPUT_JSON = "mixed_p_spin_hea_spsa.json"
OUTPUT_JSON_ENERGY = "mixed_p_spin_hea_spsa_energy.json"
SEED_BASE = 3000
N_TRIALS = 5
MAXITER = 200
WORKERS = 7
SPSA_A = 0.2
SPSA_C = 0.15
SPSA_A_STAB = 10.0
SPSA_ALPHA = 0.602
SPSA_GAMMA = 0.101


def unpack_layer(params: np.ndarray, layer: int) -> np.ndarray:
    base = layer * PARAMS_PER_LAYER
    ry_angles = np.asarray(params[base : base + N_QUBITS], dtype=float)
    if ry_angles.size != N_QUBITS:
        raise ValueError("layer parameter slice does not match 7 Ry per layer.")
    return ry_angles


def _bit_mask(qubit: int, n: int = N_QUBITS) -> int:
    return 1 << (n - 1 - int(qubit))


def apply_ry(psi: np.ndarray, qubit: int, theta: float) -> None:
    c = math.cos(0.5 * float(theta))
    s = math.sin(0.5 * float(theta))
    mask = _bit_mask(qubit)
    i0 = np.arange(psi.size)
    i0 = i0[(i0 & mask) == 0]
    i1 = i0 | mask
    a = psi[i0].copy()
    b = psi[i1].copy()
    psi[i0] = c * a - s * b
    psi[i1] = s * a + c * b


def apply_cz(psi: np.ndarray, q1: int, q2: int) -> None:
    m1 = _bit_mask(q1)
    m2 = _bit_mask(q2)
    idx = np.arange(psi.size)
    psi[((idx & m1) != 0) & ((idx & m2) != 0)] *= -1.0


def hea_statevector(params: np.ndarray) -> np.ndarray:
    """|0...0⟩ after repeated (Ry → even CZ → odd CZ) layers."""
    x = ry.wrap_angles(np.asarray(params, dtype=float).reshape(-1))
    if x.size != N_PARAMS:
        raise ValueError(f"expected {N_PARAMS} parameters, got {x.size}.")
    psi = np.zeros(1 << N_QUBITS, dtype=complex)
    psi[0] = 1.0
    for layer in range(N_LAYERS):
        for q, theta in enumerate(unpack_layer(x, layer)):
            apply_ry(psi, q, theta)
        for a, b in EVEN_PAIRS:
            apply_cz(psi, a, b)
        for a, b in ODD_PAIRS:
            apply_cz(psi, a, b)
    return psi


def hea_probs(params: np.ndarray) -> np.ndarray:
    psi = hea_statevector(params)
    p = np.abs(psi) ** 2
    total = float(p.sum())
    if total <= 0.0:
        return np.full(p.shape, 1.0 / p.size)
    return p.real / total


def ascii_circuit() -> str:
    lines = [
        f"Hardware-efficient ansatz, {N_QUBITS} qubits, {N_PARAMS} parameters",
        "One layer: Ry on all qubits, then even CZ, then odd CZ. No Rz.",
        "Even CZ: (0,1) (2,3) (4,5)     Odd CZ: (1,2) (3,4) (5,6)",
        "",
    ]
    wires = [f"q{q}: |" for q in range(N_QUBITS)]
    even_map = {q: p for p in EVEN_PAIRS for q in p}
    odd_map = {q: p for p in ODD_PAIRS for q in p}
    for layer in range(N_LAYERS):
        base = layer * PARAMS_PER_LAYER
        for q in range(N_QUBITS):
            wires[q] += f" Ry{base + q:02d}"
            wires[q] += " ●e" if q in even_map else "   "
            wires[q] += " ●o" if q in odd_map else "   "
        for q in range(N_QUBITS):
            wires[q] += " —"
    lines.extend(w + " |" for w in wires)
    lines.append("")
    lines.append(f"parameter count: {N_LAYERS} × 7 Ry = {N_PARAMS}")
    return "\n".join(lines) + "\n"


def save_circuit_diagram(circuit_dir: Path) -> tuple[Path, Path]:
    """Draw the Ry + even/odd CZ HEA and write PNG + ASCII into ``circuit_dir``."""
    circuit_dir = Path(circuit_dir)
    circuit_dir.mkdir(parents=True, exist_ok=True)
    txt_path = circuit_dir / "hea_even_odd_cz.txt"
    png_path = circuit_dir / "hea_even_odd_cz.png"
    txt_path.write_text(ascii_circuit(), encoding="utf-8")

    n = N_QUBITS
    col_w = 0.95
    fig_w = 1.4 + N_LAYERS * 3 * col_w + 0.6
    fig_h = 0.85 * n + 1.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ys = {q: n - 1 - q for q in range(n)}
    x_left = 0.4
    x_right = x_left + N_LAYERS * 3 * col_w + 0.25
    for q in range(n):
        y = ys[q]
        ax.plot([x_left, x_right], [y, y], color="0.25", lw=1.2, zorder=1)
        ax.text(x_left - 0.12, y, f"q{q}", ha="right", va="center", fontsize=11)

    even_cz = "#1f6aa5"
    odd_cz = "#c45c26"
    ry_face = "#d9ecff"

    def box(x, y, label, face):
        w, h = 0.72, 0.50
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=face,
                edgecolor="0.15",
                lw=1.0,
                zorder=3,
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=7, zorder=4)

    def cz_column(x, pairs, color):
        for a, b in pairs:
            y1, y2 = ys[a], ys[b]
            ax.plot([x, x], [y1, y2], color=color, lw=1.8, zorder=2)
            for y in (y1, y2):
                ax.add_patch(Circle((x, y), 0.10, facecolor=color, edgecolor=color, zorder=4))

    x = x_left + 0.65
    for layer in range(N_LAYERS):
        base = layer * PARAMS_PER_LAYER
        x_ry = x
        x_even = x + col_w
        x_odd = x + 2 * col_w
        mid = 0.5 * (x_ry + x_odd)
        ax.add_patch(
            Rectangle(
                (x_ry - 0.42, -0.40),
                2 * col_w + 0.84,
                n + 0.45,
                facecolor="#6b7c8a",
                edgecolor="none",
                alpha=0.05,
                zorder=0,
            )
        )
        ax.text(mid, n - 0.18, f"layer {layer}", ha="center", va="bottom", fontsize=8, color="0.25")
        for q in range(n):
            box(x_ry, ys[q], rf"$R_y$ {base + q}", ry_face)
        cz_column(x_even, EVEN_PAIRS, even_cz)
        cz_column(x_odd, ODD_PAIRS, odd_cz)
        x += 3 * col_w

    ax.set_xlim(0.0, x_right + 0.15)
    ax.set_ylim(-0.65, n + 0.5)
    ax.axis("off")
    ax.set_title(
        f"Hardware-efficient ansatz  ·  {N_QUBITS} qubits  ·  {N_LAYERS} layers  ·  {N_PARAMS} $R_y$ parameters\n"
        r"one layer = $R_y$ then even CZ (0,1)(2,3)(4,5) then odd CZ (1,2)(3,4)(5,6)    no $R_z$",
        fontsize=11,
        pad=8,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, txt_path


def run_trial(job: dict) -> dict:
    t0 = time.perf_counter()
    n_spins = int(job["num_spins"])
    energies = np.asarray(job["energies"], dtype=float)
    maxiter = int(job["maxiter"])
    objective = str(job.get("objective", "gibbs"))
    rng = np.random.default_rng(int(job["seed"]))
    theta0 = ry.wrap_angles(rng.uniform(0.0, 2.0 * np.pi, size=N_PARAMS))
    policy = None
    eta0 = None
    if objective == "gibbs":
        policy = ry.SampledTailEta()
        eta0 = float(policy.initialize(energies, hea_probs(theta0)).eta)

        def fun(theta: np.ndarray) -> float:
            return ry.gibbs_objective(hea_probs(theta), energies, float(policy.eta))

        def before_step(k: int, theta: np.ndarray) -> None:
            policy.maybe_update(k, maxiter, energies, hea_probs(theta))

    elif objective == "energy":

        def fun(theta: np.ndarray) -> float:
            p = hea_probs(theta)
            return float(np.dot(p, energies))

        def before_step(k: int, theta: np.ndarray) -> None:
            del k, theta

    else:
        raise ValueError(f"unknown objective {objective!r}")

    opt = ry.run_spsa(
        fun,
        theta0,
        maxiter=maxiter,
        rng=rng,
        project=ry.wrap_angles,
        a=float(job["spsa_a"]),
        c=float(job["spsa_c"]),
        A=float(job["spsa_A"]),
        alpha=float(job["spsa_alpha"]),
        gamma=float(job["spsa_gamma"]),
        on_before_step=before_step,
    )
    theta = ry.wrap_angles(opt.x)
    eta_final = None if policy is None else float(policy.eta)
    init = ry.histogram_stats(hea_probs(theta0), energies, n_spins, eta=eta0)
    final = ry.histogram_stats(hea_probs(theta), energies, n_spins, eta=eta_final)
    if objective == "energy":
        init["cost"] = float(init["energy_physical"])
        final["cost"] = float(final["energy_physical"])
        method = "hea_even_odd_cz_energy_spsa"
        eta_policy = None
        snap = {"n_clamps": 0, "n_fallbacks": 0}
    else:
        init["cost"] = float(init["gibbs_cost"])
        final["cost"] = float(final["gibbs_cost"])
        method = "hea_even_odd_cz_gibbs_spsa"
        eta_policy = "sampled_tail"
        snap = policy.snapshot()
    return {
        "trial": int(job["trial"]),
        "hamiltonian_id": int(job["hamiltonian_id"]),
        "file": str(job["file"]),
        "family": "mixed_p_spin",
        "kind": "mixed_p_spin",
        "method": method,
        "objective": objective,
        "ansatz": "hea_even_odd_cz",
        "n_params": N_PARAMS,
        "num_spins": n_spins,
        "seed": int(job["seed"]),
        "theta0": theta0,
        "theta": theta,
        "init": init,
        "nit": int(opt.nit),
        "nfev": int(opt.nfev),
        "eta_policy": eta_policy,
        "eta0": eta0,
        "n_eta_clamps": int(snap["n_clamps"]),
        "n_eta_fallbacks": int(snap["n_fallbacks"]),
        "elapsed_s": float(time.perf_counter() - t0),
        **final,
    }


def _print_job_line(i: int, n: int, rec: dict) -> None:
    eta = rec.get("eta")
    eta_s = "" if eta is None else f"  η={float(eta):.3f}"
    label = "⟨H⟩" if rec.get("objective") == "energy" else "Gibbs"
    print(
        f"[{i}/{n}] H{rec['hamiltonian_id']:03d} t{rec['trial']}  {label}={rec['cost']:.4f}"
        f"{eta_s}  ⟨H⟩={rec['energy_physical']:.3f}  "
        f"E={rec['energy_diag']:.3f} (min {rec['energy_min']:.3f}, "
        f"rel={rec['rel_gap']:.3f})  gs={rec['success']}  "
        f"bits={rec['most_likely_bitstring']}",
        flush=True,
    )


def _run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    records: list[dict] = []
    if workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs, 1):
            rec = run_trial(job)
            records.append(rec)
            _print_job_line(i, len(jobs), rec)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = {pool.submit(run_trial, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                done += 1
                _print_job_line(done, len(jobs), rec)
    records.sort(key=lambda r: (int(r["hamiltonian_id"]), int(r["trial"])))
    return records


def run_experiment(
    *,
    ham_dir: Path,
    outdir: Path,
    output: Path | None,
    circuit_dir: Path,
    seed_base: int,
    n_trials: int,
    maxiter: int,
    workers: int,
    objective: str,
    spsa_a: float,
    spsa_c: float,
    spsa_A: float,
    spsa_alpha: float,
    spsa_gamma: float,
    skip_run: bool = False,
) -> dict | None:
    print("=== Hardware-efficient ansatz (Ry then even/odd CZ, no Rz) ===", flush=True)
    print(ascii_circuit(), end="", flush=True)
    png_path, txt_path = save_circuit_diagram(circuit_dir)
    print(f"Saved circuit diagram: {png_path}", flush=True)
    print(f"Saved circuit ASCII:   {txt_path}", flush=True)
    if skip_run:
        return None

    instances = ry.load_instances(ham_dir)
    n_spins = int(instances[0]["num_spins"])
    if n_spins != N_QUBITS:
        raise ValueError(f"HEA is built for {N_QUBITS} qubits, dataset has {n_spins}.")
    print("=== Mixed p-spin instances (landscape, before VQE) ===", flush=True)
    print(f"{'H':>3}  {'file':<28}  {'Emin':>8}  {'spread':>8}  {'gap':>6}", flush=True)
    jobs: list[dict] = []
    ham_meta: list[dict] = []
    n_t = max(int(n_trials), 1)
    for inst in instances:
        meta = {k: v for k, v in inst.items() if k != "energies"}
        ham_meta.append(meta)
        print(
            f"{inst['hamiltonian_id']:3d}  {inst['file']:<28}  "
            f"{inst['energy_min']:8.3f}  {inst['spread']:8.3f}  {inst['gap']:6.3f}",
            flush=True,
        )
        for trial in range(n_t):
            jobs.append(
                {
                    "trial": trial,
                    "hamiltonian_id": int(inst["hamiltonian_id"]),
                    "file": inst["file"],
                    "num_spins": n_spins,
                    "energies": inst["energies"],
                    "seed": int(seed_base) + 100 * int(inst["hamiltonian_id"]) + trial,
                    "objective": objective,
                    "maxiter": int(maxiter),
                    "spsa_a": spsa_a,
                    "spsa_c": spsa_c,
                    "spsa_A": spsa_A,
                    "spsa_alpha": spsa_alpha,
                    "spsa_gamma": spsa_gamma,
                }
            )
    objective = str(objective)
    if objective not in ("gibbs", "energy"):
        raise ValueError(f"unknown objective {objective!r}")
    cost_label = "energy ⟨H⟩" if objective == "energy" else "Gibbs"
    print(
        f"=== HEA even/odd-CZ {cost_label} SPSA, {len(ham_meta)} Hamiltonians × {n_t} trial(s), "
        f"{N_PARAMS} params, maxiter={maxiter}, workers={workers} ===",
        flush=True,
    )
    records = _run_jobs(jobs, workers)
    finite = all(math.isfinite(float(r["cost"])) for r in records)
    steps_ok = all(int(r["nit"]) == int(maxiter) for r in records)
    meta_by_id = {int(m["hamiltonian_id"]): m for m in ham_meta}
    gs_ok = all(
        np.isclose(
            float(r["energy_min"]),
            float(meta_by_id[int(r["hamiltonian_id"])]["energy_min"]),
            atol=1e-10,
            rtol=0.0,
        )
        for r in records
    )
    if not finite:
        raise RuntimeError("non-finite cost in at least one trial")
    if not steps_ok:
        raise RuntimeError("a trial did not use the requested SPSA step count")
    if not gs_ok:
        raise RuntimeError("saved ground energies do not match exact enumeration")
    summary = ry._summarize(records)
    repeats = ry._repeat_summary(records)
    payload = {
        "method": "hea_even_odd_cz_energy_spsa" if objective == "energy" else "hea_even_odd_cz_gibbs_spsa",
        "ansatz": "hea_even_odd_cz",
        "n_params": N_PARAMS,
        "n_layers": N_LAYERS,
        "params_per_layer": PARAMS_PER_LAYER,
        "has_rz": False,
        "layer": "Ry then even CZ then odd CZ",
        "even_cz_pairs": [list(p) for p in EVEN_PAIRS],
        "odd_cz_pairs": [list(p) for p in ODD_PAIRS],
        "circuit_png": str(png_path),
        "circuit_txt": str(txt_path),
        "objective": objective,
        "eta_policy": None if objective == "energy" else "sampled_tail",
        "initial_state": "|0>^n",
        "n_hamiltonians": len(ham_meta),
        "n_trials_per_hamiltonian": n_t,
        "num_spins": n_spins,
        "maxiter": int(maxiter),
        "seed_base": int(seed_base),
        "workers": int(workers),
        "ham_dir": str(Path(ham_dir)),
        "n_success": summary["n_success"],
        "success_rate": summary["success_rate"],
        "mean_success_rate": repeats["mean_success_rate"],
        "std_success_rate": repeats["std_success_rate"],
        "mean_rel_gap": summary["mean_rel_gap"],
        "mean_energy_physical": summary["mean_energy_physical"],
        "mean_eta": summary["mean_eta"],
        "summary": summary,
        "repeats": repeats,
        "hamiltonians": ham_meta,
        "trials": records,
        "spsa": {
            "a": spsa_a,
            "c": spsa_c,
            "A": spsa_A,
            "alpha": spsa_alpha,
            "gamma": spsa_gamma,
        },
        "checks": {
            "finite_costs": finite,
            "requested_steps": steps_ok,
            "ground_energy_match": gs_ok,
        },
    }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = Path(output) if output is not None else outdir / (
        OUTPUT_JSON_ENERGY if objective == "energy" else OUTPUT_JSON
    )
    if not path.is_absolute():
        path = outdir / path.name if path.parent == Path(".") else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ry._json_ready(payload), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    print(
        f"  HEA {cost_label} exact-GS hits {summary['n_success']}/{summary['n']}  "
        f"pooled rate {summary['success_rate']:.3f}  "
        f"mean of {repeats['n_repeats']} runs {repeats['mean_success_rate']:.3f}"
        + (
            f" ± {repeats['std_success_rate']:.3f}"
            if repeats["n_repeats"] > 1
            else ""
        )
        + f"  mean ⟨H⟩={summary['mean_energy_physical']:.3f}"
        + f"  mean rel-gap {summary['mean_rel_gap']:.3f}",
        flush=True,
    )
    for rep in repeats["repeats"]:
        print(
            f"    run {rep['trial']}: {rep['n_success']}/{rep['n']}  "
            f"success_rate={rep['success_rate']:.3f}  mean rel-gap={rep['mean_rel_gap']:.3f}",
            flush=True,
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ham-dir", type=Path, default=HAM_DIR)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--circuit-dir", type=Path, default=CIRCUIT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--objective", choices=("gibbs", "energy"), default="gibbs")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--spsa-a", type=float, default=SPSA_A)
    parser.add_argument("--spsa-c", type=float, default=SPSA_C)
    parser.add_argument("--spsa-A", type=float, default=SPSA_A_STAB)
    parser.add_argument("--spsa-alpha", type=float, default=SPSA_ALPHA)
    parser.add_argument("--spsa-gamma", type=float, default=SPSA_GAMMA)
    parser.add_argument(
        "--diagram-only",
        action="store_true",
        help="Write the circuit diagram and exit without SPSA.",
    )
    args = parser.parse_args(argv)
    run_experiment(
        ham_dir=args.ham_dir,
        outdir=args.outdir,
        output=args.output,
        circuit_dir=args.circuit_dir,
        seed_base=args.seed_base,
        n_trials=args.n_trials,
        maxiter=args.maxiter,
        workers=args.workers,
        objective=args.objective,
        spsa_a=args.spsa_a,
        spsa_c=args.spsa_c,
        spsa_A=args.spsa_A,
        spsa_alpha=args.spsa_alpha,
        spsa_gamma=args.spsa_gamma,
        skip_run=args.diagram_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
