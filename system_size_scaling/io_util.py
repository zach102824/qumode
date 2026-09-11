"""JSON helpers and SCALE_CONCLUSION.md writer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import L_START, LADDER_NS, PRIOR_N7, RESULTS_ROOT, ROOT, SUCCESS_THRESHOLD, results_path, summary_path


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


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_depth_result(n: int, depth: int) -> dict | None:
    path = results_path(n, depth)
    if not path.exists():
        return None
    return read_json(path)


def curve_from_disk(n: int, l_min: int = L_START, l_max: int = 20) -> list[dict]:
    curve = []
    for depth in range(int(l_min), int(l_max) + 1):
        rec = load_depth_result(n, depth)
        if rec is None:
            continue
        curve.append(
            {
                "L": int(depth),
                "k": int(rec["k"]),
                "n_total": int(rec["n_total"]),
                "success_prob": float(rec["success_prob"]),
                "wall_s": float(rec.get("wall_s", 0.0)),
            }
        )
    return curve


def row_from_curve(n: int, curve: list[dict]) -> dict:
    if not curve:
        return {
            "n": n,
            "L_star": None,
            "k": 0,
            "n_total": 0,
            "success_prob": 0.0,
            "wall_s": 0.0,
            "curve": [],
            "status": "not_started",
        }
    best = max(curve, key=lambda r: (r["success_prob"], -r["L"]))
    hit = next((c for c in curve if c["success_prob"] >= SUCCESS_THRESHOLD), None)
    last_l = int(curve[-1]["L"])
    if hit is not None:
        status = "hit_threshold"
    elif last_l >= 20:
        status = "capped_L20_below_threshold"
    else:
        status = "in_progress"
    chosen = hit or best
    return {
        "n": n,
        "L_star": None if hit is None else int(hit["L"]),
        "k": int(chosen["k"]),
        "n_total": int(chosen["n_total"]),
        "success_prob": float(chosen["success_prob"]),
        "wall_s": float(sum(c["wall_s"] for c in curve)),
        "curve": curve,
        "status": status,
    }


def write_conclusion(status_note: str = "") -> Path:
    """Rebuild SCALE_CONCLUSION.md. n=7 is prior PR #14 data, not this sweep."""
    rows = [dict(PRIOR_N7)]
    for n in LADDER_NS:
        sp = summary_path(n)
        if sp.exists():
            rec = read_json(sp)
            rec["curve"] = [c for c in rec.get("curve", []) if int(c["L"]) >= L_START]
            if rec["curve"]:
                rebuilt = row_from_curve(n, rec["curve"])
                rec["L_star"] = rebuilt["L_star"]
                rec["k"] = rebuilt["k"]
                rec["n_total"] = rebuilt["n_total"]
                rec["success_prob"] = rebuilt["success_prob"]
                rec["status"] = rec.get("status") or rebuilt["status"]
            rows.append(rec)
            continue
        rows.append(row_from_curve(n, curve_from_disk(n)))

    lines = [
        "# ECD system-size scaling — conclusion",
        "",
        "Noiseless Gibbs **ECD only** (no SNAP, no GDR / noise).",
        "Success = `most_likely_bitstring == ground_bitstring`.",
        "Each live cell is **20 Hamiltonians × 10 trials = 200**.",
        "Cost = Gibbs `-ln⟨e^{-ηE}⟩` with `sampled_tail` η.",
        "Hardware target: **2 transmons × 3 cavities × 8 levels = dim 2048** (n=11 exact fill).",
        "Live ladder is **n=8…11 starting at L=4** (increment until ≥90% or L=20).",
        "",
        "## Summary table",
        "",
        "| n | L* | k/200 | success | wall (s) | status |",
        "|---|----|-------|---------|----------|--------|",
    ]
    for row in rows:
        n = int(row["n"])
        lstar = row.get("L_star")
        l_s = "—" if lstar is None else str(int(lstar))
        k = int(row.get("k", 0))
        ntot = int(row.get("n_total", 0))
        status = str(row.get("status", ""))
        wall = row.get("wall_s")
        if n == 7:
            lines.append(
                f"| {n} | {l_s} | {k}/{ntot} | {float(row['success_prob']):.3f} | "
                f"— | {status} |"
            )
            continue
        if ntot == 0:
            lines.append(f"| {n} | — | — | — | — | {status} |")
        else:
            wall_s = 0.0 if wall is None else float(wall)
            lines.append(
                f"| {n} | {l_s} | {k}/{ntot} | {float(row['success_prob']):.3f} | "
                f"{wall_s:.1f} | {status} |"
            )

    lines.extend(
        [
            "",
            "## n=7 prior data (not re-run here)",
            "",
            "Official L* comes from [PR #14](https://github.com/zach102824/qumode/pull/14) "
            "noiseless ECD **L4: 186/200 = 93%** (20 H × 10 trials, **200** joint SPSA, "
            "vacuum, `sampled_tail` η, production 1q+2cav / original `four_sat` fleet).",
            "",
            "This folder’s n=7 L=3 / 70-SPSA smoke was **158/200 = 79%** and is **not** "
            "the scoreboard L*. n=7 L=4…20 cells in `results/` are leftover from an "
            "earlier mis-scoped sweep and are not used in the table.",
            "",
            "## Depth curves (live ladder, L≥4)",
            "",
        ]
    )
    for row in rows:
        n = int(row["n"])
        if n == 7:
            continue
        lines.append(f"### n={n}")
        lines.append("")
        curve = [c for c in row.get("curve", []) if int(c["L"]) >= L_START]
        if not curve:
            lines.append("Not started.")
            lines.append("")
            continue
        lines.append("| L | k/N | success | wall (s) |")
        lines.append("|---|-----|---------|----------|")
        for c in curve:
            lines.append(
                f"| {int(c['L'])} | {int(c['k'])}/{int(c['n_total'])} | "
                f"{float(c['success_prob']):.3f} | {float(c.get('wall_s', 0.0)):.1f} |"
            )
        lines.append("")

    if status_note:
        lines.extend(["## Notes", "", status_note.rstrip(), ""])

    path = ROOT / "SCALE_CONCLUSION.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# Keep RESULTS_ROOT referenced so editors see it as used.
_ = RESULTS_ROOT
