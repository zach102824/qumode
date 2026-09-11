"""JSON helpers and SCALE_CONCLUSION.md writer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import LADDER_NS, RESULTS_ROOT, ROOT, SUCCESS_THRESHOLD, results_path, summary_path


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


def write_conclusion(status_note: str = "") -> Path:
    """Rebuild SCALE_CONCLUSION.md from whatever result JSONs exist."""
    rows: list[dict] = []
    for n in LADDER_NS:
        sp = summary_path(n)
        if sp.exists():
            rows.append(read_json(sp))
            continue
        curve = []
        for depth in range(3, 21):
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
        if not curve:
            rows.append(
                {
                    "n": n,
                    "L_star": None,
                    "k": 0,
                    "n_total": 0,
                    "success_prob": 0.0,
                    "wall_s": 0.0,
                    "curve": [],
                    "status": "not_started",
                }
            )
            continue
        best = max(curve, key=lambda r: (r["success_prob"], -r["L"]))
        hit = next((c for c in curve if c["success_prob"] >= SUCCESS_THRESHOLD), None)
        rows.append(
            {
                "n": n,
                "L_star": None if hit is None else int(hit["L"]),
                "k": int((hit or best)["k"]),
                "n_total": int((hit or best)["n_total"]),
                "success_prob": float((hit or best)["success_prob"]),
                "wall_s": float(sum(c["wall_s"] for c in curve)),
                "curve": curve,
                "status": "partial" if hit is None else "complete",
            }
        )

    lines = [
        "# ECD system-size scaling — conclusion",
        "",
        "Noiseless Gibbs **ECD only** (no SNAP, no GDR / noise).",
        "Success = `most_likely_bitstring == ground_bitstring`.",
        "Each cell is **20 Hamiltonians × 10 trials = 200**.",
        "Cost = Gibbs `-ln⟨e^{-ηE}⟩` with `sampled_tail` η.",
        "Hardware target: **2 transmons × 3 cavities × 8 levels = dim 2048** (n=11 exact fill).",
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
        frac = "—" if ntot == 0 else f"{k}/{ntot}"
        status = str(row.get("status", ""))
        if ntot == 0:
            lines.append(f"| {n} | — | — | — | — | {status} |")
        else:
            prob = float(row.get("success_prob", 0.0))
            wall = float(row.get("wall_s", 0.0))
            lines.append(f"| {n} | {l_s} | {frac} | {prob:.3f} | {wall:.1f} | {status} |")

    lines.extend(
        [
            "",
            "## Depth curves",
            "",
        ]
    )
    for row in rows:
        n = int(row["n"])
        lines.append(f"### n={n}")
        lines.append("")
        if not row.get("curve"):
            lines.append("Not started.")
            lines.append("")
            continue
        lines.append("| L | k/N | success | wall (s) |")
        lines.append("|---|-----|---------|----------|")
        for c in row.get("curve", []):
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
