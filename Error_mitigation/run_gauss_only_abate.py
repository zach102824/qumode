#!/usr/bin/env python3
"""Default-mix vs fully-Gaussian twins for gdr_param (adaptive recipe).

Arm A: n_rank2=10 (= n_train//4 auto for n_train=40; hits existing nr10 caches).
Arm B: n_rank2=0 (100% t_free=0; new cache key).

Writes only under Error_mitigation/out_research/gauss_only/.
Does not touch src/, Error_mitigation/out/, or out_smoke/.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Error_mitigation.run_ablation import parse_args, run  # noqa: E402
from Error_mitigation.run_mitigation_experiment import SEED_BASE  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_RESEARCH = HERE / "out_research"
OUT = OUT_RESEARCH / "gauss_only"
TIE_EPS = 0.002

ARMS = (
    {
        "name": "default",
        "n_rank2": 10,
        "label": "default mix (n_rank2=10 = n_train//4 auto)",
    },
    {
        "name": "nr0",
        "n_rank2": 0,
        "label": "fully Gaussian (n_rank2=0, all t_free=0)",
    },
)

STAGES = (
    {
        "name": "smoke",
        "ansatz": "ecd",
        "params": "random",
        "families": "loss",
        "kappa_tau": "0.003",
        "readout": "ideal",
    },
    {
        "name": "ecd_priority",
        "ansatz": "ecd",
        "params": "both",
        "families": "loss,comprehensive",
        "kappa_tau": "0.003,0.03,0.1",
        "readout": "ideal,readout_realistic",
    },
    {
        "name": "snap_priority",
        "ansatz": "snap",
        "params": "both",
        "families": "loss,comprehensive",
        "kappa_tau": "0.003,0.03,0.1",
        "readout": "ideal,readout_realistic",
    },
    {
        "name": "thermal",
        "ansatz": "both",
        "params": "both",
        "families": "loss_thermal_dephasing",
        "kappa_tau": "0.003,0.03,0.1",
        "readout": "ideal,readout_realistic",
    },
)


def log(path: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(line + "\n")


def cell_key(rec: dict) -> tuple:
    return (
        rec["ansatz"],
        rec["params"],
        rec["family"],
        float(rec["kappa_tau"]),
        rec["readout"],
    )


def tvd_of(rec: dict, method: str) -> float | None:
    met = (rec.get("metrics") or {}).get(method) or {}
    val = met.get("tvd")
    return None if val is None else float(val)


def load_arm_records(arm: str) -> dict[tuple, dict]:
    """Latest record per cell for one arm (later stages overwrite earlier)."""
    out: dict[tuple, dict] = {}
    for path in sorted(OUT.glob(f"{arm}_*/results.json")):
        blob = json.loads(path.read_text())
        for rec in blob.get("records") or []:
            out[cell_key(rec)] = rec
    return out


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "median": None}
    return {
        "n": len(xs),
        "mean": float(statistics.fmean(xs)),
        "median": float(statistics.median(xs)),
    }


def _wtl(deltas: list[float]) -> tuple[int, int, int]:
    w = t = l = 0
    for d in deltas:
        if abs(d) < TIE_EPS:
            t += 1
        elif d > 0:
            w += 1  # fully Gaussian better
        else:
            l += 1
    return w, t, l


def pair_cells() -> list[dict]:
    default = load_arm_records("default")
    nr0 = load_arm_records("nr0")
    keys = sorted(set(default) & set(nr0), key=lambda k: (k[0], k[1], k[2], k[3], k[4]))
    rows = []
    for key in keys:
        a, b = default[key], nr0[key]
        raw_a, raw_b = tvd_of(a, "raw"), tvd_of(b, "raw")
        gdr_a, gdr_b = tvd_of(a, "gdr_param"), tvd_of(b, "gdr_param")
        delta = None if (gdr_a is None or gdr_b is None) else gdr_a - gdr_b
        if delta is None:
            verdict = "missing"
        elif abs(delta) < TIE_EPS:
            verdict = "tie"
        elif delta > 0:
            verdict = "gauss_better"
        else:
            verdict = "mix_better"
        rows.append(
            {
                "ansatz": key[0],
                "params": key[1],
                "family": key[2],
                "kappa_tau": key[3],
                "readout": key[4],
                "raw_default": raw_a,
                "raw_nr0": raw_b,
                "gdr_default": gdr_a,
                "gdr_nr0": gdr_b,
                "delta": delta,
                "verdict": verdict,
                "twin_tag_default": a.get("twin_tag"),
                "twin_tag_nr0": b.get("twin_tag"),
            }
        )
    return rows


def write_comparison(rows: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paired_cells.json").write_text(json.dumps(rows, indent=2))

    deltas = [r["delta"] for r in rows if r["delta"] is not None]
    overall = _stats(deltas)
    w, t, l = _wtl(deltas)

    groups = {
        "ansatz": defaultdict(list),
        "params": defaultdict(list),
        "kappa_tau": defaultdict(list),
        "family": defaultdict(list),
    }
    for r in rows:
        if r["delta"] is None:
            continue
        groups["ansatz"][r["ansatz"]].append(r["delta"])
        groups["params"][r["params"]].append(r["delta"])
        groups["kappa_tau"][r["kappa_tau"]].append(r["delta"])
        groups["family"][r["family"]].append(r["delta"])

    n = len(deltas)
    # Need the ECD priority matrix (24 cells) before proposing a default change.
    min_for_call = 20
    nearly_all_ok = (
        n >= min_for_call
        and (w + t) >= math.ceil(0.9 * n)
        and (overall["mean"] or 0.0) >= -TIE_EPS
        and l <= max(1, n // 10)
    )
    mix_helps = n > 0 and (l > w) and (overall["mean"] or 0.0) < 0.0
    if n == 0:
        rec_line = "No paired cells yet."
    elif n < min_for_call:
        rec_line = (
            f"**Recommendation:** pending — {n} paired cell(s); need the ECD priority "
            f"matrix (≥{min_for_call} cells) before a keep-vs-switch call."
        )
    elif nearly_all_ok:
        rec_line = (
            "**Recommendation:** fully Gaussian is OK as the `gdr_param` default "
            "(better or tied on nearly all paired cells). Propose flipping "
            "`n_rank2` auto→0 in a follow-up — do **not** silently change production defaults here."
        )
    elif mix_helps:
        rec_line = (
            "**Recommendation:** keep the ~25% `t_free=2` mix (`n_rank2 = n_train//4`). "
            "Fully Gaussian is not strictly better; the mix still helps enough on `gdr_param`."
        )
    else:
        rec_line = (
            "**Recommendation:** keep the ~25% `t_free=2` mix (`n_rank2 = n_train//4`) "
            "until a larger or clearer win for fully Gaussian. Mixed / near-tie result."
        )

    lines = [
        "# Fully Gaussian twins vs default mix (`gdr_param` only)",
        "",
        "H000 / instance 0, shots=8192, n_train=40, seed=`SEED_BASE=2026`, "
        "`--twin-design adaptive` (span on random, U(0.5,1) on optimized).",
        "",
        "- **Arm A (default mix):** `--n-rank2 10`. Equal to omitted/auto for "
        "`n_train=40` (`n_train//4`). Explicit 10 reuses existing "
        "`out_research/cache/*_nr10_*` keys (`nrauto` would miss them).",
        "- **Arm B (fully Gaussian):** `--n-rank2 0` → new `*_nr0_*` cache keys, all `t_free=0`.",
        "- Methods: `raw`, `gdr_param`. Readout: `ideal` + `readout_realistic` "
        "(no `readout_strong`).",
        "",
        f"Δ = TVD(`gdr_param` | default mix) − TVD(`gdr_param` | fully Gaussian). "
        f"Positive ⇒ fully Gaussian better. Tie if |Δ| < {TIE_EPS}.",
        "",
        rec_line,
        "",
        "## Headline",
        "",
        f"| metric | value |",
        f"|---|---:|",
        f"| paired cells | {n} |",
        f"| mean Δ | {_fmt(overall['mean'])} |",
        f"| median Δ | {_fmt(overall['median'])} |",
        f"| gauss better / tie / mix better | {w} / {t} / {l} |",
        "",
        "## By slice",
        "",
        "| slice | n | mean Δ | median Δ | gauss better | tie | mix better |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in ("ansatz", "params", "kappa_tau", "family"):
        for label, xs in sorted(groups[kind].items(), key=lambda kv: str(kv[0])):
            st = _stats(xs)
            ww, tt, ll = _wtl(xs)
            lines.append(
                f"| {kind}={label} | {st['n']} | {_fmt(st['mean'])} | {_fmt(st['median'])} "
                f"| {ww} | {tt} | {ll} |"
            )

    lines += [
        "",
        "## Per-cell TVD",
        "",
        "| ansatz | params | family | κτ | readout | raw mix | raw gauss | "
        "gdr mix | gdr gauss | Δ | verdict |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['ansatz']} | {r['params']} | {r['family']} | {r['kappa_tau']:g} | "
            f"{r['readout']} | {_fmt(r['raw_default'])} | {_fmt(r['raw_nr0'])} | "
            f"{_fmt(r['gdr_default'])} | {_fmt(r['gdr_nr0'])} | {_fmt(r['delta'])} | "
            f"{r['verdict']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Physical histograms for arm A reuse `Error_mitigation/out_research/cache/` "
        "`nr10` adaptive keys (span on random, default on optimized).",
        "- Arm B writes new `nr0` keys into that same shared cache.",
        "- Production adaptive defaults are **not** changed in this PR.",
        "",
    ]
    (OUT / "COMPARISON.md").write_text("\n".join(lines))
    summary = {
        "n": n,
        "mean_delta": overall["mean"],
        "median_delta": overall["median"],
        "gauss_better": w,
        "tie": t,
        "mix_better": l,
        "recommendation": rec_line,
    }
    (OUT / "headline.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_stage(stage: dict, arm: dict, progress: Path) -> Path:
    tag = f"gauss_only/{arm['name']}_{stage['name']}"
    argv = [
        "--tag",
        tag,
        "--outdir",
        str(OUT_RESEARCH),
        "--ansatz",
        stage["ansatz"],
        "--params",
        stage["params"],
        "--families",
        stage["families"],
        "--kappa-tau",
        stage["kappa_tau"],
        "--readout",
        stage["readout"],
        "--shots",
        "8192",
        "--n-train",
        "40",
        "--seed",
        str(SEED_BASE),
        "--instance",
        "0",
        "--twin-design",
        "adaptive",
        "--n-rank2",
        str(arm["n_rank2"]),
        "--methods",
        "raw,gdr_param",
    ]
    log(progress, f"START arm={arm['name']} stage={stage['name']} tag={tag} n_rank2={arm['n_rank2']}")
    t0 = time.time()
    args = parse_args(argv)
    run(args)
    elapsed = time.time() - t0
    dest = OUT_RESEARCH / tag / "results.json"
    log(progress, f"DONE  arm={arm['name']} stage={stage['name']} wall={elapsed:.1f}s -> {dest}")
    return dest


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stages",
        default="all",
        help="Comma list of smoke,ecd_priority,snap_priority,thermal or 'all'.",
    )
    p.add_argument("--arms", default="both", help="default,nr0, or both.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    cli = parse_cli(argv)
    wanted = (
        [s["name"] for s in STAGES]
        if cli.stages.strip() == "all"
        else [x.strip() for x in cli.stages.split(",") if x.strip()]
    )
    known = {s["name"]: s for s in STAGES}
    unknown = [n for n in wanted if n not in known]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; expected {list(known)}")
    arm_names = ("default", "nr0") if cli.arms == "both" else tuple(x.strip() for x in cli.arms.split(","))
    arms = [a for a in ARMS if a["name"] in arm_names]
    if not arms:
        raise SystemExit(f"no arms selected from {cli.arms}")

    progress = OUT / "PROGRESS.md"
    if not progress.is_file():
        progress.write_text(
            "# gauss_only progress\n\n"
            "Adaptive twin design. Arm A = n_rank2=10 (auto). Arm B = n_rank2=0.\n\n"
        )
    log(progress, f"driver stages={wanted} arms={[a['name'] for a in arms]}")

    for stage_name in wanted:
        stage = known[stage_name]
        for arm in arms:
            run_stage(stage, arm, progress)
            rows = pair_cells()
            summary = write_comparison(rows)
            log(
                progress,
                f"PAIRED n={summary['n']} meanΔ={summary['mean_delta']} "
                f"W/T/L={summary['gauss_better']}/{summary['tie']}/{summary['mix_better']}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
