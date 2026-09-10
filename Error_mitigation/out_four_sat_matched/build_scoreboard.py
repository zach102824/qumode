#!/usr/bin/env python3
"""Aggregate matched 4-SAT noiseless success % and GDR raw vs gdr_select TVD."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = Path(__file__).resolve().parent
KT = (0.003, 0.03, 0.1)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def noiseless_table(payload: dict) -> dict:
    trials = payload["trials"]
    n = len(trials)
    n_success = int(sum(bool(t.get("success")) for t in trials))
    by_h = payload.get("by_hamiltonian") or {}
    per_h = []
    for hid in payload.get("hamiltonian_ids", range(20)):
        rec = by_h.get(str(int(hid)), {})
        group = [t for t in trials if int(t["hamiltonian_id"]) == int(hid)]
        n_h = int(rec.get("n") or len(group))
        n_s = int(rec.get("n_success") or sum(bool(t.get("success")) for t in group))
        energies = [float(t["energy_physical"]) for t in group]
        per_h.append(
            {
                "hamiltonian_id": int(hid),
                "n_success": n_s,
                "n": n_h,
                "success_rate": n_s / max(n_h, 1),
                "pick_trial": rec.get("pick_trial"),
                "pick_success": rec.get("pick_success"),
                "pick_cost": rec.get("pick_cost"),
                "pick_energy_physical": rec.get("pick_energy_physical"),
                "mean_energy_physical": float(sum(energies) / max(len(energies), 1)),
                "best_energy_physical": min(energies) if energies else None,
            }
        )
    return {
        "ansatz": payload.get("ansatz"),
        "ndepths": payload.get("ndepths"),
        "n_trials_per_hamiltonian": payload.get("n_trials_per_hamiltonian"),
        "seed_base": payload.get("seed_base"),
        "n_trials": n,
        "n_success": n_success,
        "success_pct": 100.0 * n_success / max(n, 1),
        "success_metric": payload.get("success_metric"),
        "mean_energy_physical": payload.get("mean_energy_physical"),
        "by_hamiltonian": per_h,
    }


def _cell(records: list[dict], *, params: str, kt: float, method: str) -> dict | None:
    hits = [
        r
        for r in records
        if r.get("params") == params
        and r.get("family") == "comprehensive"
        and r.get("readout") == "readout_realistic"
        and abs(float(r["kappa_tau"]) - float(kt)) < 1e-12
    ]
    if not hits:
        return None
    rec = hits[0]
    met = (rec.get("metrics") or {}).get(method) or {}
    return {
        "tvd": met.get("tvd"),
        "success_gs": met.get("success_gs"),
        "p_gs_mit": met.get("p_gs_mit"),
        "ideal_success_gs": rec.get("ideal_success_gs", met.get("success_gs_ideal")),
    }


def gdr_table(ansatz: str, ndepth: int) -> dict:
    rows = []
    missing = []
    for hid in range(20):
        path = OUTDIR / f"{ansatz}_h{hid:03d}_s8192" / "results.json"
        if not path.is_file():
            missing.append(hid)
            continue
        blob = _load(path)
        recs = blob.get("records") or []
        meta_path = OUTDIR / f"{ansatz}_h{hid:03d}_s8192" / f"optimized_params_{ansatz}_h{hid:03d}_nd{ndepth}.json"
        pick = _load(meta_path) if meta_path.is_file() else {}
        row = {
            "hamiltonian_id": hid,
            "gibbs_trial": pick.get("gibbs_trial"),
            "gibbs_pick": pick.get("gibbs_pick"),
            "pick_success": pick.get("success"),
            "pick_energy_physical": pick.get("energy_physical"),
            "ideal_success_gs": None,
            "by_kt": {},
        }
        for kt in KT:
            raw = _cell(recs, params="optimized", kt=kt, method="raw")
            sel = _cell(recs, params="optimized", kt=kt, method="gdr_select")
            if raw and row["ideal_success_gs"] is None:
                row["ideal_success_gs"] = raw.get("ideal_success_gs")
            win = None
            if raw and sel and raw.get("tvd") is not None and sel.get("tvd") is not None:
                win = float(sel["tvd"]) < float(raw["tvd"]) - 1e-12
            row["by_kt"][str(kt)] = {"raw": raw, "gdr_select": sel, "select_beats_raw": win}
        rows.append(row)
    summary = {}
    for kt in KT:
        opt_wins = 0
        opt_n = 0
        raw_gs = 0
        sel_gs = 0
        ideal_gs = 0
        raw_tvds = []
        sel_tvds = []
        for row in rows:
            cell = row["by_kt"].get(str(kt)) or {}
            raw = cell.get("raw") or {}
            sel = cell.get("gdr_select") or {}
            if raw.get("tvd") is None or sel.get("tvd") is None:
                continue
            opt_n += 1
            opt_wins += int(bool(cell.get("select_beats_raw")))
            raw_tvds.append(float(raw["tvd"]))
            sel_tvds.append(float(sel["tvd"]))
            raw_gs += int(bool(raw.get("success_gs")))
            sel_gs += int(bool(sel.get("success_gs")))
            ideal_gs += int(bool(row.get("ideal_success_gs")))
        summary[str(kt)] = {
            "n": opt_n,
            "select_beats_raw": opt_wins,
            "mean_raw_tvd": sum(raw_tvds) / max(len(raw_tvds), 1) if raw_tvds else None,
            "mean_select_tvd": sum(sel_tvds) / max(len(sel_tvds), 1) if sel_tvds else None,
            "raw_gs_mode": raw_gs,
            "select_gs_mode": sel_gs,
            "ideal_gs_mode": ideal_gs,
        }
    return {"ansatz": ansatz, "ndepth": ndepth, "missing": missing, "hamiltonians": rows, "summary": summary}


def main() -> int:
    snap_json = ROOT / "results" / "gibbs_four_sat_snap_matched_n10.json"
    ecd_json = ROOT / "results" / "gibbs_four_sat_ecd_matched_n10.json"
    out = {
        "success_metric": "most_likely_bitstring == ground_bitstring (JSON success flag)",
        "energy_bar_used": False,
        "n_hamiltonians": 20,
        "n_trials_per_hamiltonian": 10,
        "seed_base": 4000,
        "outer_iter": 200,
        "eta_policy": "sampled_tail",
        "initial_state": "vacuum",
        "noiseless": {},
        "gdr": {},
    }
    if snap_json.is_file():
        out["noiseless"]["snap"] = noiseless_table(_load(snap_json))
    if ecd_json.is_file():
        out["noiseless"]["ecd"] = noiseless_table(_load(ecd_json))
    out["gdr"]["snap"] = gdr_table("snap", 3)
    out["gdr"]["ecd"] = gdr_table("ecd", 4)
    path = OUTDIR / "scoreboard.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")
    for name, blob in out["noiseless"].items():
        print(
            f"  noiseless {name}: {blob['n_success']}/{blob['n_trials']} "
            f"= {blob['success_pct']:.1f}%"
        )
    for name, blob in out["gdr"].items():
        print(f"  GDR {name} missing={blob['missing']} summary={blob['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
