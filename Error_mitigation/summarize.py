#!/usr/bin/env python3
"""Summarize mitigation ``results.json`` into a markdown scoreboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HEAD = (
    "raw",
    "readout_only",
    "oracle_binomial",
    "gdr_param",
    "gdr_param_reg",
    "gdr_eta",
    "gdr_eta_nth",
    "gdr_two_stage",
    "gdr_indep",
    "gdr_energy",
    "gdr_full",
    "zne_idle",
    "readout_then_zne",
    "zne_then_readout",
    "scalar_cdr",
)


def _fmt(val, digits=4):
    if val is None:
        return "—"
    return f"{float(val):.{digits}f}"


def _beat(challenger, baseline) -> str:
    if challenger is None or baseline is None:
        return ""
    if float(challenger) < float(baseline) - 1e-12:
        return "WIN"
    if float(challenger) > float(baseline) + 1e-12:
        return "lose"
    return "tie"


def records_from(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict) and "records" in payload:
        return list(payload["records"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"unrecognised results file {path}")


def scoreboard(records: list[dict]) -> str:
    lines = [
        "| ansatz | params | family | κτ | readout | method | TVD | vs raw | vs ro | \|ΔE\| |",
        "|--------|--------|--------|----|---------|--------|----:|:------:|:-----:|------:|",
    ]
    for rec in records:
        raw = (rec.get("metrics") or {}).get("raw") or {}
        ro = (rec.get("metrics") or {}).get("readout_only") or {}
        raw_tvd, ro_tvd = raw.get("tvd"), ro.get("tvd")
        for method in HEAD:
            met = (rec.get("metrics") or {}).get(method)
            if not met:
                continue
            tvd, de = met.get("tvd"), met.get("dE")
            vs_raw = _beat(tvd, raw_tvd) if method != "raw" else ""
            vs_ro = _beat(tvd, ro_tvd) if method not in ("raw", "readout_only") else ""
            lines.append(
                f"| {rec.get('ansatz')} | {rec.get('params')} | {rec.get('family')} | "
                f"{float(rec.get('kappa_tau', 0)):.3f} | {rec.get('readout')} | {method} | "
                f"{_fmt(tvd)} | {vs_raw} | {vs_ro} | {_fmt(de)} |"
            )
    return "\n".join(lines) + "\n"


def wins_summary(records: list[dict]) -> str:
    lines = ["## Headline wins vs `raw` (TVD)", ""]
    focus = ("gdr_param", "gdr_param_reg", "gdr_eta", "gdr_eta_nth", "readout_then_zne", "zne_idle", "oracle_binomial")
    for rec in records:
        raw = ((rec.get("metrics") or {}).get("raw") or {}).get("tvd")
        bits = []
        for m in focus:
            tvd = ((rec.get("metrics") or {}).get(m) or {}).get("tvd")
            tag = _beat(tvd, raw)
            if tag:
                bits.append(f"{m}:{tag}({_fmt(tvd)})")
        lines.append(
            f"- {rec.get('ansatz')} {rec.get('params')} {rec.get('family')} "
            f"κτ={float(rec.get('kappa_tau', 0)):.3f} {rec.get('readout')}  raw={_fmt(raw)}  "
            + "  ".join(bits)
        )
    return "\n".join(lines) + "\n"


def write_markdown(records: list[dict], path: Path, title: str = "GDR research scoreboard") -> None:
    body = f"# {title}\n\n{wins_summary(records)}\n{scoreboard(records)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    recs = []
    for path in args.results:
        recs.extend(records_from(path))
    text = "# GDR research scoreboard\n\n" + wins_summary(recs) + "\n" + scoreboard(recs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
