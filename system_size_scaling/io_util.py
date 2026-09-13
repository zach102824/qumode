"""JSON helpers and SCALE_CONCLUSION.md writer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import (
    FINISHED_LADDER_NS,
    HIGHER_NS,
    L_MAX,
    L_MAX_HIGHER,
    L_START,
    OUTER_ITER,
    PRIOR_N7,
    PROTOCOL_TAG,
    RESULTS_ROOT,
    ROOT,
    SCOREBOARD_NS,
    SUCCESS_THRESHOLD,
    SUPERSEDED_70_SPSA,
    is_canonical_cell,
    n_params_for,
    results_path,
    soft_cap_for_n,
    spsa_a_scaled,
    summary_path,
)


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
    rec = read_json(path)
    if not is_canonical_cell(rec):
        return None
    return rec


def curve_from_disk(n: int, l_min: int = L_START, l_max: int = L_MAX) -> list[dict]:
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
                "outer_iter": int(rec.get("outer_iter", OUTER_ITER)),
            }
        )
    return curve


def row_from_curve(n: int, curve: list[dict]) -> dict:
    cap = soft_cap_for_n(n)
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
            "outer_iter": OUTER_ITER,
        }
    best = max(curve, key=lambda r: (r["success_prob"], -r["L"]))
    hit = next((c for c in curve if c["success_prob"] >= SUCCESS_THRESHOLD), None)
    last_l = int(curve[-1]["L"])
    if hit is not None:
        status = "hit_threshold"
    elif last_l >= cap:
        status = f"capped_L{cap}_below_threshold"
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
        "outer_iter": OUTER_ITER,
    }


def _scoreboard_rows() -> list[dict]:
    """n=7 prior data + finished n=8…11 + higher-n cells (from disk)."""
    rows = [dict(PRIOR_N7)]
    for n in SCOREBOARD_NS:
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
    return rows


def cell_stats(n: int, depth: int | None = None) -> dict | None:
    """k/N, n_params, a, dim, pairs, mean p(GS), wall from a canonical cell."""
    if depth is None:
        sp = summary_path(n)
        if not sp.exists():
            return None
        summary = read_json(sp)
        depth = summary.get("L_star") or (summary.get("curve") or [{}])[-1].get("L")
        if depth is None:
            return None
        rec = load_depth_result(n, int(depth)) or summary
    else:
        rec = load_depth_result(n, int(depth))
        if rec is None:
            return None
    trials = rec.get("trials") or []
    pgs = [float(t["p_ground"]) for t in trials if "p_ground" in t]
    proto = rec.get("protocol") or {}
    spsa = proto.get("spsa") or {}
    emb = rec.get("embedding") or {}
    l_used = int(rec.get("L") or rec.get("L_star") or depth)
    n_params = spsa.get("n_params")
    if n_params is None:
        n_params = n_params_for(n, l_used)
    a = spsa.get("a")
    if a is None:
        a = spsa_a_scaled(int(n_params))
    return {
        "n": int(n),
        "L": l_used,
        "k": int(rec.get("k", 0)),
        "n_total": int(rec.get("n_total", 0)),
        "success_prob": float(rec.get("success_prob", 0.0)),
        "success_fraction": rec.get("success_fraction")
        or f"{int(rec.get('k', 0))}/{int(rec.get('n_total', 0))}",
        "wall_s": rec.get("wall_s"),
        "n_params": int(n_params),
        "a": float(a),
        "dim": emb.get("dim"),
        "pairs": emb.get("n_pairs"),
        "n_transmons": emb.get("n_transmons"),
        "n_cavities": emb.get("n_cavities"),
        "mean_p_ground": float(np.mean(pgs)) if pgs else None,
        "n_hamiltonians": rec.get("n_hamiltonians"),
        "n_trials_per_h": rec.get("n_trials_per_h"),
        "status": rec.get("status"),
        "scout": bool(rec.get("scout"))
        or (
            int(rec.get("n_total", 0)) > 0
            and int(rec.get("n_total", 0)) < 200
        ),
    }


def _fmt_wall(wall) -> str:
    if wall is None:
        return "—"
    return f"{float(wall):.1f}"


def _fmt_mean_p(val) -> str:
    if val is None:
        return "—"
    return f"{float(val):.4f}"


def write_higher_n(status_note: str = "") -> Path:
    """Write HIGHER_N.md — n=12+ scoreboard with n_params / a / dim / pairs / p(GS)."""
    lines = [
        "# ECD system-size scaling — n=12+ (this PR)",
        "",
        "Noiseless Gibbs **ECD only** (no SNAP, no GDR / noise).",
        "Success = `most_likely_bitstring == ground_bitstring`.",
        "Full cell = **20 Hamiltonians × 10 trials = 200**.",
        "A cell that is too slow may first report a **5 H × 4 trial scout** (20 trials); that is labelled.",
        "Protocol: 200 joint SPSA, `a = 0.2 × √(37 / n_params)`, `c=0.15`, vacuum start.",
        f"Start L={L_START}; if <90% raise L (soft cap **L={L_MAX_HIGHER}** — deeper L at this",
        "budget already collapsed on n=8). n=7…11 are **not** re-run (PR #15 / #14).",
        "",
        "## Hardware growth",
        "",
        "Each transmon = 1 logical bit. Each cavity at `FOCK_CUTOFF=8` = 3 Fock bits (MSB first).",
        "When n exceeds capacity, append one transmon **and** one cavity.",
        "Idle modes (0 assigned bits) stay vacuum and are omitted from the simulated tensor.",
        "",
        "| n | hardware plan | simulated | dim | pairs | idle |",
        "|---|---------------|-----------|-----|-------|------|",
        "| 7 | 2T×3C parent (special case) | 1T×2C | 128 | 2 | T1, C2 |",
        "| 8 | 2T×3C | 2T×2C | 256 | 4 | C2 |",
        "| 9–11 | 2T×3C | 2T×3C | 2048 | 6 | C2 bits 1…3 |",
        "| 12 | 3T×4C | 3T×3C | 4096 | 9 | C3 |",
        "| 13–15 | 3T×4C | 3T×4C | 32768 | 12 | — |",
        "| 16 | 4T×5C | 4T×4C | 65536 | 16 | C4 |",
        "| 17–19 | 4T×5C | 4T×5C | 524288 | 20 | — |",
        "",
        "`n_params = n_T + 2 n_C + 4 L n_T n_C`.",
        "",
        "## Scoreboard (this extension)",
        "",
        "| n | L* | k/N | success | n_params | a | dim | pairs | mean p(GS) | wall (s) | status |",
        "|---|----|-----|---------|----------|---|-----|-------|------------|----------|--------|",
    ]
    for n in HIGHER_NS:
        stats = cell_stats(n)
        row = row_from_curve(n, curve_from_disk(n))
        if stats is None or int(row.get("n_total", 0)) == 0:
            lines.append(f"| {n} | — | — | — | {n_params_for(n, L_START)} | {spsa_a_scaled(n_params_for(n, L_START)):.4f} | — | — | — | — | not_started |")
            continue
        lstar = stats["L"]
        k = stats["k"]
        ntot = stats["n_total"]
        wall = _fmt_wall(stats["wall_s"])
        mean_p = _fmt_mean_p(stats["mean_p_ground"])
        dim = "—" if stats["dim"] is None else str(int(stats["dim"]))
        pairs = "—" if stats["pairs"] is None else str(int(stats["pairs"]))
        status = str(row.get("status") or stats.get("status") or "")
        if stats.get("scout"):
            status = (status + " scout").strip()
        lines.append(
            f"| {n} | {lstar} | {k}/{ntot} | {stats['success_prob']:.3f} | "
            f"{stats['n_params']} | {stats['a']:.4f} | {dim} | {pairs} | "
            f"{mean_p} | {wall} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Depth curves",
            "",
        ]
    )
    for n in HIGHER_NS:
        lines.append(f"### n={n}")
        lines.append("")
        curve = curve_from_disk(n)
        if not curve:
            lines.append("Not started.")
            lines.append("")
            continue
        lines.append("| L | k/N | success | n_params | a | dim | pairs | mean p(GS) | wall (s) | notes |")
        lines.append("|---|-----|---------|----------|---|-----|-------|------------|----------|-------|")
        for c in curve:
            st = cell_stats(n, int(c["L"]))
            if st is None:
                lines.append(
                    f"| {int(c['L'])} | {int(c['k'])}/{int(c['n_total'])} | "
                    f"{float(c['success_prob']):.3f} | {n_params_for(n, int(c['L']))} | "
                    f"{spsa_a_scaled(n_params_for(n, int(c['L']))):.4f} | — | — | — | "
                    f"{float(c.get('wall_s', 0.0)):.1f} | |"
                )
                continue
            note = "scout" if st.get("scout") else ""
            lines.append(
                f"| {st['L']} | {st['k']}/{st['n_total']} | {st['success_prob']:.3f} | "
                f"{st['n_params']} | {st['a']:.4f} | {st['dim']} | {st['pairs']} | "
                f"{_fmt_mean_p(st['mean_p_ground'])} | {_fmt_wall(st['wall_s'])} | {note} |"
            )
        lines.append("")

    lines.extend(
        [
            "## n=7…11 (not re-run; PR #15 / #14)",
            "",
            "| n | L* | k/200 | success | n_params | a | dim | pairs | mean p(GS) | wall (s) |",
            "|---|----|-------|---------|----------|---|-----|-------|------------|----------|",
            "| 7 | 4 | 186/200 | 0.930 | 37 | 0.2000 | 128 | 2 | — | — |",
        ]
    )
    for n in FINISHED_LADDER_NS:
        st = cell_stats(n)
        if st is None:
            lines.append(f"| {n} | — | — | — | {n_params_for(n, 4)} | {spsa_a_scaled(n_params_for(n, 4)):.4f} | — | — | — | — |")
            continue
        lines.append(
            f"| {n} | {st['L']} | {st['k']}/{st['n_total']} | {st['success_prob']:.3f} | "
            f"{st['n_params']} | {st['a']:.4f} | {st['dim']} | {st['pairs']} | "
            f"{_fmt_mean_p(st['mean_p_ground'])} | {_fmt_wall(st['wall_s'])} |"
        )

    note = status_note.rstrip() if status_note else (
        f"Higher-n extension of PR #15. Soft cap L={L_MAX_HIGHER}. "
        "Do not rerun n=7…11. Noisy/GDR is out of scope."
    )
    lines.extend(["", "## Notes", "", note, ""])
    path = ROOT / "HIGHER_N.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_conclusion(status_note: str = "") -> Path:
    """Rebuild SCALE_CONCLUSION.md. n=7 is prior PR #14 data, not this sweep."""
    rows = _scoreboard_rows()
    write_higher_n(status_note)

    lines = [
        "# ECD system-size scaling — conclusion",
        "",
        "Noiseless Gibbs **ECD only** (no SNAP, no GDR / noise).",
        "Success = `most_likely_bitstring == ground_bitstring`.",
        "Each live cell is **20 Hamiltonians × 10 trials = 200**.",
        "Cost = Gibbs `-ln⟨e^{-ηE}⟩` with `sampled_tail` η.",
        "Hardware: n=7…11 stay on the locked 2T×3C map (n=11 exact fill, dim 2048).",
        "For n>11, append one transmon+cavity when the register is full "
        "(3T+4C capacity 15, dim 32768 when C3 is live). Idle 0-bit modes are omitted.",
        f"PR #15 live ladder **n=8…11** finished at L*=4. This PR continues **n=12…15** "
        f"at L={L_START}, **{OUTER_ITER} joint SPSA**, `a = 0.2 × √(37 / n_params)`, "
        f"soft cap **L={L_MAX_HIGHER}** (do not blindly go to L={L_MAX}).",
        f"Protocol tag: `{PROTOCOL_TAG}`. Full n=12+ table: `HIGHER_N.md`.",
        "",
        "## Summary table (canonical, 200 joint SPSA)",
        "",
        "| n | L* | k/N | success | wall (s) | status |",
        "|---|----|-----|---------|----------|--------|",
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
            "the scoreboard L*.",
            "",
            "## Depth curves (live ladder, 200 joint SPSA, L≥4)",
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
            lines.append("Not started under the 200-SPSA protocol.")
            lines.append("")
            continue
        lines.append("| L | k/N | success | wall (s) | SPSA |")
        lines.append("|---|-----|---------|----------|------|")
        for c in curve:
            spsa = int(c.get("outer_iter", OUTER_ITER))
            lines.append(
                f"| {int(c['L'])} | {int(c['k'])}/{int(c['n_total'])} | "
                f"{float(c['success_prob']):.3f} | {float(c.get('wall_s', 0.0)):.1f} | {spsa} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Superseded: 70-SPSA L=3…20 (not canonical)",
            "",
            "Previous PR #15 cells used **70** joint SPSA and a hard L=20 cap. "
            "They never hit 90% for n=8–10; deeper L was systematically worse. "
            "Those JSON files are kept under `results_70spsa_superseded/` and "
            "**must not** be mixed into the live scoreboard.",
            "",
            "| n | best L≥4 (70 SPSA) | k/200 | note |",
            "|---|--------------------|-------|------|",
        ]
    )
    for n, rec in SUPERSEDED_70_SPSA.items():
        lines.append(
            f"| {n} | {rec['best_L']} | {rec['k']}/{rec['n_total']} | {rec['note']} |"
        )
    lines.extend(
        [
            "",
            "Why deeper L looked worse: not a unitarity/decoding bug (n=7 ECD "
            "matches production QuTiP; gates stay norm-preserving at L=20/40). "
            "At fixed 70 SPSA, extra layers add parameters that the budget cannot "
            "train — both ⟨H⟩ and p_ground degrade. n=7 L=4 was 168/200 at 70 SPSA "
            "vs **186/200 at 200 SPSA** in PR #14. This restart tests whether 200 "
            "joint SPSA plus uncapped L recovers ≥90% for n=8…11.",
            "",
            "Noisy GDR-in-loop / comprehensive κ_φ τ = 0.5 κτ is **deferred** to the "
            "n=7 default-redo agent. This ladder is noiseless mode-finding; κ_φ does "
            "not enter the cost.",
            "",
        ]
    )

    note = status_note.rstrip() if status_note else (
        f"Protocol: n=7 is PR #14 prior data (L*=4, 186/200 = 93%). "
        f"n=8…11 is the finished PR #15 ladder (all L*=4, ≥90%). "
        f"This PR extends n=12…15 at {OUTER_ITER} joint SPSA, soft cap L={L_MAX_HIGHER}. "
        f"See HIGHER_N.md for n_params / a / dim / pairs / mean p(GS)."
    )
    lines.extend(["## Notes", "", note, ""])

    path = ROOT / "SCALE_CONCLUSION.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# Keep RESULTS_ROOT referenced so editors see it as used.
_ = RESULTS_ROOT
