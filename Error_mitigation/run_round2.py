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

# Shot-noise scale from PR #8 bootstrap (~0.004–0.013). Require a clear beat.
BEAT_EPS = 0.005
REGRESS_EPS = 0.003


def score_methods(records: list[dict]) -> dict:
    """Keep a method only if it beats select somewhere and never regresses protects."""
    protects = [r for r in records if r.get("protect")]
    summary = {}
    for name in NEW_METHODS:
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


def write_scoreboard(path: Path, records: list[dict], verdict: dict, header: str) -> None:
    lines = ["# Round-2 microbench", "", header, ""]
    cols = ["id", "raw", "gdr_param", "gdr_select"] + list(NEW_METHODS)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    def f(x):
        return "—" if x is None else f"{float(x):.4f}"

    for rec in records:
        mets = rec["metrics"]
        row = [rec["id"]]
        for m in ("raw", "gdr_param", "gdr_select") + NEW_METHODS:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=ROUND2_OUT)
    p.add_argument("--shots", type=int, default=8192)
    p.add_argument("--seed", type=int, default=SEED_BASE)
    p.add_argument("--fit-maxiter", type=int, default=120)
    p.add_argument("--methods", default=",".join(ROUND2_METHODS))
    p.add_argument(
        "--cells",
        default="all",
        help="Comma-separated cell ids, or 'all'.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    if args.cells == "all":
        cells = list(HARD_CELLS)
    else:
        want = {x.strip() for x in args.cells.split(",") if x.strip()}
        cells = [c for c in HARD_CELLS if c["id"] in want]
        missing = want - {c["id"] for c in cells}
        if missing:
            raise SystemExit(f"unknown cell ids: {sorted(missing)}")
    print(
        f"round2 cells={len(cells)} shots={args.shots} fit_maxiter={args.fit_maxiter} "
        f"methods={','.join(methods)}"
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
    verdict = score_methods(records)
    payload = {
        "tag": "round2_micro",
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
    (outdir / "micro_results.json").write_text(json.dumps(json_ready(payload), indent=2))
    header = (
        f"shots={args.shots} seed={args.seed} fit_maxiter={args.fit_maxiter} "
        f"wall={payload['wall_s']:.1f}s  any_keep={payload['any_keep']}"
    )
    write_scoreboard(outdir / "micro_scoreboard.md", records, verdict, header)
    print(f"\nwrote {outdir / 'micro_results.json'}")
    print(f"wrote {outdir / 'micro_scoreboard.md'}")
    print(f"any KEEP: {payload['any_keep']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
