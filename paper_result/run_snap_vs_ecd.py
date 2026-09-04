#!/usr/bin/env python3
"""SNAP vs ECD on the paper knapsack: who recovers ``0110000``.

Same Fig. 5 protocol except the hybrid ansatz. Parameter count is matched
by SNAP depth, not layer count:

* ECD–rotation, N_d = 5 (40 parameters)
* SNAP+displacement, N_d = 1 (18 parameters) and N_d = 2 (36 parameters)

N_d = 2 is the matched-budget arm (36 vs 40). Shared: vacuum |0,0,0⟩,
notebook-style Uniform init, BFGS maxiter=200, 50 starts, energy ⟨H⟩ and
Gibbs −ln⟨e^{−ηE}⟩ with sampled_tail η. ECD trials are reused from
fig5.json when the settings match.
"""

from __future__ import annotations

import argparse
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
    EXACT_GROUND_ENERGY,
    TARGET_BITSTRING,
    TARGET_QNM,
)
from qumode_vqe.params import n_parameters, n_snap_parameters

from paper_result.ecd import notebook_x0, run_ecd_trial
from paper_result.run_fig5_fig7 import (
    _compare_pstar,
    _final_is_optimum,
    _paired_success,
    _run_jobs,
    _summarize_trials,
    _trial_out,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = HERE / "out"
SEED_BASE = 2026
ECD_NDEPTH = 5
SNAP_NDEPTHS = (1, 2)
JSON_NAME = "snap_vs_ecd.json"
PLOT_NAME = "snap_vs_ecd.png"
TXT_NAME = "snap_vs_ecd.txt"

PRESETS = {
    "smoke": {"maxiter": 2, "trials": 1, "record_every": 1},
    "quick": {"maxiter": 40, "trials": 8, "record_every": 10},
    "paper": {"maxiter": 200, "trials": 50, "record_every": 10},
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


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def _n_params(ansatz: str, ndepth: int, nfocks: tuple[int, int]) -> int:
    if ansatz == "snap":
        return n_snap_parameters(ndepth, nfocks)
    return n_parameters(ndepth)


def _success_rate(block: dict) -> tuple[float, int]:
    trials = block.get("trials") or []
    if trials:
        if "success_rate" in block:
            return float(block["success_rate"]), int(block.get("n") or len(trials))
        hits = sum(_final_is_optimum(t) for t in trials)
        return hits / max(len(trials), 1), len(trials)
    return (1.0 if _final_is_optimum(block) else 0.0), 1


def _fig5_reusable(path: Path, *, n_trials: int, maxiter: int, seed_base: int, ndepth: int) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("ndepth", -1)) != int(ndepth):
        return None
    if int(payload.get("maxiter", -1)) != int(maxiter):
        return None
    if int(payload.get("seed_base", -1)) != int(seed_base):
        return None
    energy = list((payload.get("energy") or {}).get("trials") or [])
    gibbs = list((payload.get("gibbs") or {}).get("trials") or [])
    if len(energy) < int(n_trials) or len(gibbs) < int(n_trials):
        return None
    print(f"reusing ECD from {path}  (n={n_trials})", flush=True)
    return payload


def _payload_from_records(
    *,
    ansatz: str,
    ndepth: int,
    n_params: int,
    n_trials: int,
    maxiter: int,
    seed_base: int,
    elapsed_s: float,
    records: list[dict],
    reused: bool,
) -> dict:
    by_obj = {"energy": [], "gibbs": []}
    for rec in records:
        by_obj[str(rec["objective"])].append(rec)
    for rows in by_obj.values():
        rows.sort(key=lambda r: int(r["trial"]))
        del rows[int(n_trials) :]
    summaries = {k: _summarize_trials(v) for k, v in by_obj.items()}
    paired = _paired_success(by_obj["energy"], by_obj["gibbs"])
    label = "ECD" if ansatz == "ecd" else "SNAP"
    return {
        "ansatz": f"{ansatz}_nd{int(ndepth)}",
        "n_trials": int(n_trials),
        "maxiter": int(maxiter),
        "seed_base": int(seed_base),
        "ndepth": int(ndepth),
        "n_parameters": int(n_params),
        "init": "notebook Uniform: mag~U(0,3), angles~U(0,π)",
        "vacuum": "|0⟩⊗|0⟩⊗|0⟩",
        "target_qnm": list(TARGET_QNM),
        "target_bitstring": TARGET_BITSTRING,
        "exact_ground_energy": EXACT_GROUND_ENERGY,
        "elapsed_s": float(elapsed_s),
        "reused": bool(reused),
        "paired": paired,
        "energy": {
            **summaries["energy"],
            "trials": [_trial_out(r) for r in by_obj["energy"]],
        },
        "gibbs": {
            **summaries["gibbs"],
            "trials": [_trial_out(r) for r in by_obj["gibbs"]],
        },
        "comparison": {
            "success_rate": _compare_pstar(
                summaries["energy"]["success_rate"],
                summaries["gibbs"]["success_rate"],
                f"{label} fraction with most-likely = {TARGET_BITSTRING}",
            ),
            "best_pstar": _compare_pstar(
                summaries["energy"]["best_pstar"],
                summaries["gibbs"]["best_pstar"],
                f"{label} best P(|0,6,0⟩)",
            ),
            "mean_pstar": _compare_pstar(
                summaries["energy"]["mean_pstar"],
                summaries["gibbs"]["mean_pstar"],
                f"{label} mean P(|0,6,0⟩)",
            ),
        },
    }


def run_ansatz(
    *,
    ansatz: str,
    outdir: Path,
    n_trials: int,
    maxiter: int,
    seed_base: int,
    ndepth: int,
    nfocks: tuple[int, int],
    workers: int,
    record_every: int,
    adaptive_eta: bool,
    partial_name: str,
    already: list[dict],
) -> tuple[dict, float]:
    n_params = _n_params(ansatz, ndepth, nfocks)
    starts = []
    for t in range(int(n_trials)):
        rng = np.random.default_rng(int(seed_base) + t)
        starts.append(notebook_x0(ansatz, ndepth, rng, nfocks))

    jobs = []
    for t, x0 in enumerate(starts):
        for objective in ("energy", "gibbs"):
            jobs.append(
                {
                    "trial": t,
                    "seed": int(seed_base) + t,
                    "ansatz": ansatz,
                    "ndepth": int(ndepth),
                    "x0": x0,
                    "objective": objective,
                    "maxiter": int(maxiter),
                    "record_every": int(record_every),
                    "adaptive_eta": bool(adaptive_eta),
                }
            )

    done_keys = {(int(r["trial"]), str(r["objective"])) for r in already}
    n_before = len(jobs)
    jobs = [j for j in jobs if (int(j["trial"]), str(j["objective"])) not in done_keys]
    if already:
        print(
            f"resume {ansatz}: {len(already)} trials from {partial_name}, "
            f"{len(jobs)} remaining of {n_before}",
            flush=True,
        )

    print(
        f"=== {ansatz.upper()}-VQE  N_d={ndepth}  params={n_params}  "
        f"trials={n_trials}  maxiter={maxiter}  workers={workers} ===",
        flush=True,
    )
    partial = outdir / partial_name

    def _ckpt(recs: list[dict]) -> None:
        combined = already + recs
        if len(recs) % 5 != 0 and len(recs) != len(jobs):
            return
        _save_json(
            partial,
            {
                "ansatz": ansatz,
                "n_done": len(combined),
                "n_jobs": len(already) + len(jobs),
                "records": [_trial_out(r) for r in combined],
            },
        )

    t0 = time.perf_counter()
    fresh = _run_jobs(run_ecd_trial, jobs, workers, label=ansatz, checkpoint=_ckpt) if jobs else []
    elapsed = time.perf_counter() - t0
    records = already + fresh
    records.sort(key=lambda r: (str(r["objective"]), int(r["trial"])))
    payload = _payload_from_records(
        ansatz=ansatz,
        ndepth=ndepth,
        n_params=n_params,
        n_trials=n_trials,
        maxiter=maxiter,
        seed_base=seed_base,
        elapsed_s=elapsed,
        records=records,
        reused=False,
    )
    for name in ("energy", "gibbs"):
        s = payload[name]
        print(
            f"  {ansatz} {name}: {s['n_success']}/{s['n']} hit GS  "
            f"best P*={s['best_pstar']:.4f}  mean P*={s['mean_pstar']:.4f}",
            flush=True,
        )
    p = payload["paired"]
    print(
        f"  paired: Gibbs-only {p['gibbs_only']}  energy-only {p['energy_only']}  "
        f"both {p['both']}  neither {p['neither']}",
        flush=True,
    )
    print(f"  wall {elapsed:.1f}s", flush=True)
    if partial.exists():
        partial.unlink()
    return payload, elapsed


def _cross_ansatz_paired(ecd: dict, snap: dict, objective: str) -> dict:
    e_ok = {int(t["trial"]) for t in (ecd[objective].get("trials") or []) if _final_is_optimum(t)}
    s_ok = {int(t["trial"]) for t in (snap[objective].get("trials") or []) if _final_is_optimum(t)}
    n = max(len(ecd[objective].get("trials") or []), len(snap[objective].get("trials") or []))
    return {
        "n": n,
        "both": len(e_ok & s_ok),
        "ecd_only": len(e_ok - s_ok),
        "snap_only": len(s_ok - e_ok),
        "neither": n - len(e_ok | s_ok),
    }


def plot_comparison(arms: list[dict], path: Path) -> None:
    labels = []
    energy_vals = []
    gibbs_vals = []
    energy_ns = []
    gibbs_ns = []
    for block in arms:
        kind = "ECD" if str(block.get("ansatz", "")).startswith("ecd") else "SNAP+D"
        labels.append(
            f"{kind}\n$N_d={block.get('ndepth')}$, {block.get('n_parameters')} params"
        )
        e, ne = _success_rate(block["energy"])
        g, ng = _success_rate(block["gibbs"])
        energy_vals.append(e)
        gibbs_vals.append(g)
        energy_ns.append(ne)
        gibbs_ns.append(ng)

    x = np.arange(len(labels), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(3.2 + 2.0 * max(len(labels), 1), 4.2))
    b1 = ax.bar(x - width / 2, energy_vals, width, label="energy ⟨H⟩", color="#2ca02c", zorder=2)
    b2 = ax.bar(x + width / 2, gibbs_vals, width, label="Gibbs", color="#d62728", zorder=2)
    ax.set_xticks(x, labels, fontsize=11)
    ax.set_ylabel("Most-likely bitstring is optimal", fontsize=12)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars, ns in ((b1, energy_ns), (b2, gibbs_ns)):
        for bar, n in zip(bars, ns):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.03,
                f"{100.0 * h:.0f}%" + (f"\n$n={n}$" if n else ""),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_title("SNAP vs ECD on recovering 0110000", fontsize=12, pad=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _arm_heading(block: dict) -> str:
    kind = "ECD-VQE" if str(block.get("ansatz", "")).startswith("ecd") else "SNAP-VQE"
    return (
        f"## {kind}  N_d={block.get('ndepth')}  params={block.get('n_parameters')}  "
        f"n={block.get('n_trials')}"
    )


def write_text(outdir: Path, ecd: dict, snaps: list[dict], cross: dict) -> str:
    lines = [
        "# SNAP vs ECD: most-likely bitstring is the knapsack optimum (0110000)",
        "",
        "Same Fig. 5 protocol (vacuum, notebook Uniform init, BFGS).",
        "SNAP depths 1–2 keep the parameter count at or under ECD N_d=5 (40 params).",
        "",
    ]
    for block in [ecd, *snaps]:
        e = block["energy"]
        g = block["gibbs"]
        p = block.get("paired") or {}
        lines += [
            _arm_heading(block),
            f"- energy: {100.0 * e['success_rate']:.0f}%  ({e['n_success']}/{e['n']})  "
            f"mean P*={e['mean_pstar']:.3f}",
            f"- Gibbs:  {100.0 * g['success_rate']:.0f}%  ({g['n_success']}/{g['n']})  "
            f"mean P*={g['mean_pstar']:.3f}",
            f"- paired: Gibbs-only {p.get('gibbs_only')}, energy-only {p.get('energy_only')}, "
            f"both {p.get('both')}, neither {p.get('neither')}",
            "",
        ]
    for snap in snaps:
        key = str(snap["ndepth"])
        c = cross[key]
        lines += [
            f"## Same-seed ECD N_d={ecd.get('ndepth')} vs SNAP N_d={snap['ndepth']}",
            f"- energy: both {c['energy']['both']}, ECD-only {c['energy']['ecd_only']}, "
            f"SNAP-only {c['energy']['snap_only']}, neither {c['energy']['neither']}",
            f"- Gibbs:  both {c['gibbs']['both']}, ECD-only {c['gibbs']['ecd_only']}, "
            f"SNAP-only {c['gibbs']['snap_only']}, neither {c['gibbs']['neither']}",
            "",
        ]
    text = "\n".join(lines)
    (outdir / TXT_NAME).write_text(text + "\n", encoding="utf-8")
    return text


def _load_snap_partial(path: Path, ndepth: int) -> list[dict]:
    if not path.exists():
        return []
    saved = json.loads(path.read_text(encoding="utf-8"))
    rows = list(saved.get("records") or [])
    return [r for r in rows if int(r.get("ndepth", ndepth)) == int(ndepth)]


def _arms_from_payload(payload: dict) -> tuple[dict, list[dict], dict]:
    ecd = payload["ecd"]
    snap_by_depth = payload.get("snap_by_depth") or {}
    if not snap_by_depth and payload.get("snap"):
        snap_by_depth = {str(payload["snap"]["ndepth"]): payload["snap"]}
    snaps = [snap_by_depth[k] for k in sorted(snap_by_depth, key=int)]
    return ecd, snaps, payload.get("cross") or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="paper")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--ecd-ndepth", type=int, default=ECD_NDEPTH)
    parser.add_argument(
        "--snap-ndepths",
        type=int,
        nargs="+",
        default=None,
        help="SNAP+D layer counts (default: 1 2, i.e. 18 and 36 parameters).",
    )
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--record-every", type=int, default=None)
    parser.add_argument("--skip-ecd", action="store_true")
    parser.add_argument("--skip-snap", action="store_true")
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Do not reuse ECD trials from fig5.json.",
    )
    parser.add_argument(
        "--fig5",
        type=Path,
        default=None,
        help="fig5.json to reuse ECD from (default: <outdir>/fig5.json).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore snap_nd*.partial.json and rerun SNAP jobs.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild the figure from saved snap_vs_ecd.json.",
    )
    args = parser.parse_args(argv)

    preset = PRESETS[args.preset]
    n_trials = preset["trials"] if args.trials is None else args.trials
    maxiter = preset["maxiter"] if args.maxiter is None else args.maxiter
    record_every = preset["record_every"] if args.record_every is None else args.record_every
    ecd_ndepth = int(args.ecd_ndepth)
    snap_ndepths = tuple(args.snap_ndepths) if args.snap_ndepths else SNAP_NDEPTHS
    nfocks = (int(DEFAULT_NFOCKS[0]), int(DEFAULT_NFOCKS[1]))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / JSON_NAME
    plot_path = outdir / PLOT_NAME

    if args.plot_only:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        ecd, snaps, cross = _arms_from_payload(payload)
        plot_comparison([ecd, *snaps], plot_path)
        print(write_text(outdir, ecd, snaps, cross), end="")
        print(f"Wrote {plot_path}", flush=True)
        return 0

    print(
        f"SNAP vs ECD  preset={args.preset}  ECD N_d={ecd_ndepth}  "
        f"SNAP N_d={list(snap_ndepths)}  trials={n_trials}  "
        f"maxiter={maxiter}  workers={args.workers}",
        flush=True,
    )

    ecd = None
    snap_by_depth: dict[str, dict] = {}
    if not args.skip_ecd:
        fig5_path = args.fig5 if args.fig5 is not None else outdir / "fig5.json"
        reused = None if args.no_reuse else _fig5_reusable(
            fig5_path,
            n_trials=n_trials,
            maxiter=maxiter,
            seed_base=args.seed_base,
            ndepth=ecd_ndepth,
        )
        if reused is not None:
            energy = list(reused["energy"]["trials"])[:n_trials]
            gibbs = list(reused["gibbs"]["trials"])[:n_trials]
            records = energy + gibbs
            ecd = _payload_from_records(
                ansatz="ecd",
                ndepth=ecd_ndepth,
                n_params=_n_params("ecd", ecd_ndepth, nfocks),
                n_trials=n_trials,
                maxiter=maxiter,
                seed_base=args.seed_base,
                elapsed_s=float(reused.get("elapsed_s") or 0.0),
                records=records,
                reused=True,
            )
        else:
            ecd, _ = run_ansatz(
                ansatz="ecd",
                outdir=outdir,
                n_trials=n_trials,
                maxiter=maxiter,
                seed_base=args.seed_base,
                ndepth=ecd_ndepth,
                nfocks=nfocks,
                workers=args.workers,
                record_every=record_every,
                adaptive_eta=True,
                partial_name="fig5.partial.json",
                already=[],
            )
            _save_json(outdir / "fig5.json", ecd)

    if not args.skip_snap:
        for nd in snap_ndepths:
            already: list[dict] = []
            partial_name = f"snap_nd{int(nd)}.partial.json"
            if not args.fresh:
                already = _load_snap_partial(outdir / partial_name, nd)
            snap, _ = run_ansatz(
                ansatz="snap",
                outdir=outdir,
                n_trials=n_trials,
                maxiter=maxiter,
                seed_base=args.seed_base,
                ndepth=int(nd),
                nfocks=nfocks,
                workers=args.workers,
                record_every=record_every,
                adaptive_eta=True,
                partial_name=partial_name,
                already=already,
            )
            snap_by_depth[str(int(nd))] = snap
            _save_json(outdir / f"snap_nd{int(nd)}.json", snap)

    if json_path.exists() and (ecd is None or not snap_by_depth):
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        if ecd is None:
            ecd = saved.get("ecd")
        if not snap_by_depth:
            _, snaps, _ = _arms_from_payload(saved)
            snap_by_depth = {str(s["ndepth"]): s for s in snaps}
    if ecd is None or not snap_by_depth:
        raise FileNotFoundError("need ECD and at least one SNAP depth to write the comparison")

    snaps = [snap_by_depth[k] for k in sorted(snap_by_depth, key=int)]
    cross = {
        str(snap["ndepth"]): {
            "energy": _cross_ansatz_paired(ecd, snap, "energy"),
            "gibbs": _cross_ansatz_paired(ecd, snap, "gibbs"),
        }
        for snap in snaps
    }
    payload = {
        "protocol": "fig5: vacuum, notebook Uniform init, BFGS, energy vs Gibbs",
        "param_match": "SNAP N_d=1/2 (18/36 params) vs ECD N_d=5 (40 params)",
        "ecd_ndepth": ecd_ndepth,
        "snap_ndepths": [int(s["ndepth"]) for s in snaps],
        "n_trials": n_trials,
        "maxiter": maxiter,
        "seed_base": int(args.seed_base),
        "target_bitstring": TARGET_BITSTRING,
        "ecd": ecd,
        "snap_by_depth": snap_by_depth,
        "cross": cross,
        "plot": str(plot_path),
    }
    _save_json(json_path, payload)
    plot_comparison([ecd, *snaps], plot_path)
    print(write_text(outdir, ecd, snaps, cross), end="")
    print(f"Wrote {plot_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
