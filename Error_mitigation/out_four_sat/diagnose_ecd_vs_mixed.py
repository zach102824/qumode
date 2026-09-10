#!/usr/bin/env python3
"""Cheap diagnostics: ECD Gibbs on mixed p-spin vs 4-SAT (no new optimization).

Reads saved NPZs + Gibbs JSONs, checks the hybrid encoding / H load, and
reconstructs Born histograms for a few stored (prep, x) endpoints.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qumode_vqe.circuit import prep_params_to_ket
from qumode_vqe.hamiltonian import (
    DEFAULT_NFOCKS,
    N_QUBITS,
    bits_from_qnm,
    diagonal_hybrid_hamiltonian,
    energy_from_z_terms,
    energy_tensor_from_z_terms,
    load_four_sat_instances,
    load_mixed_p_spin_instances,
    qnm_from_bits,
    z_terms_from_four_sat_npz,
    z_terms_from_mixed_p_spin_npz,
)
from qumode_vqe.params import ansatz_inventory
from qumode_vqe.vqe import HybridSimulator

OUTDIR = Path(__file__).resolve().parent
GATE = 0.5
NFOCKS = DEFAULT_NFOCKS
PARTITION = (1, 3, 3)


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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hamming_neighbors(bits: np.ndarray) -> list[np.ndarray]:
    out = []
    for i in range(bits.size):
        nbr = bits.copy()
        nbr[i] = 1 - nbr[i]
        out.append(nbr)
    return out


def _energy_of_bits(bits: np.ndarray, terms, identity: float) -> float:
    return float(energy_from_z_terms(bits, terms, identity))


def landscape_record(inst: dict, terms, identity: float) -> dict:
    tensor = np.asarray(inst["energy_tensor"], dtype=float)
    flat = tensor.reshape(-1)
    emin = float(np.min(flat))
    gs_bits = bits_from_qnm(*inst["ground_qnm"], PARTITION)
    gs_e = _energy_of_bits(gs_bits, terms, identity)
    nbr_e = [_energy_of_bits(nb, terms, identity) for nb in _hamming_neighbors(gs_bits)]
    dlt = np.asarray(nbr_e, dtype=float) - float(gs_e)
    rounded = np.round(flat, decimals=8)
    uniq, counts = np.unique(rounded, return_counts=True)
    level_hist = {
        f"{float(u):.4f}": int(c) for u, c in zip(uniq[:8], counts[:8])
    }
    return {
        "hamiltonian_id": int(inst["hamiltonian_id"]),
        "file": inst["file"],
        "family": inst["family"],
        "dim": int(flat.size),
        "nfocks": list(NFOCKS),
        "tensor_shape": list(tensor.shape),
        "num_spins": int(inst["num_spins"]),
        "n_terms": int(inst["n_terms"]),
        "identity": float(identity),
        "max_pauli_weight": int(inst["max_pauli_weight"]),
        "terms_by_order": inst.get("terms_by_order"),
        "energy_min": emin,
        "energy_max": float(np.max(flat)),
        "spread": float(inst["spread"]),
        "gap": float(inst["gap"]),
        "n_ground": int(inst["n_ground"]),
        "ground_bitstring": inst["ground_bitstring"],
        "ground_qnm": list(inst["ground_qnm"]),
        "hamming1_delta_mean": float(np.mean(dlt)),
        "hamming1_delta_min": float(np.min(dlt)),
        "hamming1_delta_max": float(np.max(dlt)),
        "n_hamming1_at_least_one": int(np.sum(dlt >= 1.0 - 1e-6)),
        "low_energy_occupancy": level_hist,
        "n_unique_energies": int(uniq.size),
    }


def family_landscapes(name: str) -> tuple[list[dict], list[dict]]:
    if name == "four_sat":
        instances = load_four_sat_instances(ROOT / "Hamiltonians" / "four_sat")
        recs = []
        for inst in instances:
            terms, meta = z_terms_from_four_sat_npz(ROOT / "Hamiltonians" / "four_sat" / inst["file"])
            recs.append(landscape_record(inst, terms, float(meta["identity"])))
        return instances, recs
    instances = load_mixed_p_spin_instances(ROOT / "Hamiltonians" / "mixed_p_spin")
    recs = []
    for inst in instances:
        terms, meta = z_terms_from_mixed_p_spin_npz(
            ROOT / "Hamiltonians" / "mixed_p_spin" / inst["file"]
        )
        recs.append(landscape_record(inst, terms, 0.0))
    return instances, recs


def summarize_landscapes(recs: list[dict]) -> dict:
    def col(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in recs], dtype=float)

    return {
        "n": len(recs),
        "dim": recs[0]["dim"] if recs else None,
        "mean_energy_min": float(np.mean(col("energy_min"))),
        "mean_spread": float(np.mean(col("spread"))),
        "mean_gap": float(np.mean(col("gap"))),
        "min_gap": float(np.min(col("gap"))),
        "max_gap": float(np.max(col("gap"))),
        "mean_n_ground": float(np.mean(col("n_ground"))),
        "mean_n_terms": float(np.mean(col("n_terms"))),
        "mean_hamming1_delta": float(np.mean(col("hamming1_delta_mean"))),
        "min_hamming1_delta": float(np.min(col("hamming1_delta_min"))),
        "all_hamming1_rigid": bool(all(int(r["n_hamming1_at_least_one"]) == 7 for r in recs)),
        "mean_n_unique_energies": float(np.mean(col("n_unique_energies"))),
        "mean_identity": float(np.mean(col("identity"))),
        "max_pauli_weight": int(max(int(r["max_pauli_weight"]) for r in recs)),
    }


def trial_lite(rec: dict) -> dict:
    emin = rec.get("energy_min")
    ephys = rec.get("energy_physical")
    deficit = None
    if emin is not None and ephys is not None:
        deficit = float(ephys) - float(emin)
    return {
        "hamiltonian_id": rec.get("hamiltonian_id"),
        "ndepth": rec.get("ndepth"),
        "trial": rec.get("trial"),
        "success": bool(rec.get("success")),
        "energy_physical": None if ephys is None else float(ephys),
        "energy_min": None if emin is None else float(emin),
        "energy_diag": rec.get("energy_diag"),
        "deficit": deficit,
        "near_e0": None if deficit is None else bool(deficit <= GATE),
        "cost": rec.get("cost"),
        "eta": rec.get("eta"),
        "eta0": rec.get("eta0"),
        "most_likely_bitstring": rec.get("most_likely_bitstring"),
        "ground_bitstring": rec.get("ground_bitstring"),
    }


def summarize_trials(trials: list[dict], *, ndepth: int | None = None) -> dict:
    recs = [trial_lite(t) for t in trials]
    if ndepth is not None:
        recs = [r for r in recs if int(r["ndepth"]) == int(ndepth)]
    n = len(recs)
    if not n:
        return {"n": 0}
    deficits = [float(r["deficit"]) for r in recs if r["deficit"] is not None]
    etas = [float(r["eta"]) for r in recs if r["eta"] is not None]
    eta0s = [float(r["eta0"]) for r in recs if r["eta0"] is not None]
    ephys = [float(r["energy_physical"]) for r in recs if r["energy_physical"] is not None]
    n_gs = int(sum(bool(r["success"]) for r in recs))
    n_gate = int(sum(bool(r["near_e0"]) for r in recs))
    mismatch = int(
        sum(bool(r["success"]) and (r["deficit"] is not None) and float(r["deficit"]) > GATE for r in recs)
    )
    best = min(recs, key=lambda r: float("inf") if r["deficit"] is None else float(r["deficit"]))
    by_h: dict[int, list[dict]] = {}
    for r in recs:
        hid = int(r["hamiltonian_id"])
        by_h.setdefault(hid, []).append(r)
    best_by_h = []
    for hid, group in sorted(by_h.items()):
        b = min(group, key=lambda r: float("inf") if r["deficit"] is None else float(r["deficit"]))
        best_by_h.append(b)
    h_deficits = [float(r["deficit"]) for r in best_by_h if r["deficit"] is not None]
    return {
        "n": n,
        "n_hamiltonians": len(by_h),
        "n_success_gs": n_gs,
        "success_gs_rate": n_gs / n,
        "n_near_e0": n_gate,
        "near_e0_rate": n_gate / n,
        "n_success_gs_but_deficit_gt_gate": mismatch,
        "mean_deficit": float(np.mean(deficits)) if deficits else None,
        "median_deficit": float(np.median(deficits)) if deficits else None,
        "best_deficit": float(np.min(deficits)) if deficits else None,
        "mean_energy_physical": float(np.mean(ephys)) if ephys else None,
        "mean_eta": float(np.mean(etas)) if etas else None,
        "mean_eta0": float(np.mean(eta0s)) if eta0s else None,
        "n_h_with_a_near_e0_trial": int(sum(bool(r["near_e0"]) for r in best_by_h)),
        "mean_best_per_h_deficit": float(np.mean(h_deficits)) if h_deficits else None,
        "best_trial": best,
        "best_per_h": best_by_h,
    }


def reconstruct_endpoint(trial: dict, inst: dict) -> dict:
    nfocks = tuple(int(v) for v in (trial.get("nfocks") or NFOCKS))
    prep = np.asarray(trial["prep"], dtype=float)
    x = np.asarray(trial["x"], dtype=float)
    tensor = np.asarray(inst["energy_tensor"], dtype=float)
    sim = HybridSimulator(
        ndepth=int(trial["ndepth"]),
        nfocks=nfocks,
        hamiltonian=diagonal_hybrid_hamiltonian(tensor),
        energy_tensor=tensor,
        target_qnm=None,
        ansatz=str(trial.get("ansatz", "ecd")),
        initial_state=prep_params_to_ket(prep, nfocks),
    )
    ev = sim.evaluate(x)
    probs = np.asarray(ev.measurement.physical_probs, dtype=float)
    gs = tuple(int(v) for v in inst["ground_qnm"])
    p_gs = float(probs[gs])
    e = tensor
    p_e1 = float(np.sum(probs[np.isclose(e, 1.0, atol=0.05)]))
    p_gt1 = float(np.sum(probs[e > 1.05]))
    bits_gs = bits_from_qnm(*gs, PARTITION)
    p_hamming1 = 0.0
    for nb in _hamming_neighbors(bits_gs):
        q, n, m = qnm_from_bits(nb, PARTITION)
        p_hamming1 += float(probs[q, n, m])
    json_e = float(trial["energy_physical"])
    return {
        "hamiltonian_id": int(trial["hamiltonian_id"]),
        "ansatz": trial.get("ansatz"),
        "ndepth": int(trial["ndepth"]),
        "trial": int(trial["trial"]),
        "json_energy_physical": json_e,
        "replay_energy_physical": float(ev.energy_physical),
        "energy_match": bool(abs(float(ev.energy_physical) - json_e) < 1e-8),
        "json_bits": trial.get("most_likely_bitstring"),
        "replay_bits": ev.most_likely_bitstring,
        "p_gs": p_gs,
        "p_e1_band": p_e1,
        "p_gt1": p_gt1,
        "p_hamming1_of_gs": p_hamming1,
        "implied_e_if_rest_on_e1": float(1.0 - p_gs),
    }


def pick_best(trials: list[dict], hid: int, ndepth: int) -> dict | None:
    cand = [
        t
        for t in trials
        if int(t.get("hamiltonian_id", -1)) == int(hid) and int(t.get("ndepth", -1)) == int(ndepth)
    ]
    if not cand:
        return None
    return min(cand, key=lambda t: float(t["energy_physical"]))


def wireup_checks(sat_inst, spin_inst) -> dict:
    sat_terms, sat_meta = z_terms_from_four_sat_npz(
        ROOT / "Hamiltonians" / "four_sat" / sat_inst["file"]
    )
    untilted = energy_tensor_from_z_terms(
        sat_terms, NFOCKS, PARTITION, float(sat_meta["identity"]), tilt=0.0
    )
    tilted = np.asarray(sat_inst["energy_tensor"], dtype=float)
    spin_terms, _ = z_terms_from_mixed_p_spin_npz(
        ROOT / "Hamiltonians" / "mixed_p_spin" / spin_inst["file"]
    )
    spin_untilted = energy_tensor_from_z_terms(spin_terms, NFOCKS, PARTITION, 0.0, tilt=0.0)
    return {
        "same_hybrid_dim": int(np.asarray(sat_inst["energy_tensor"]).size)
        == int(np.asarray(spin_inst["energy_tensor"]).size)
        == 128,
        "same_nfocks": list(NFOCKS),
        "n_qubits": N_QUBITS,
        "four_sat_untilted_gs_energy": float(untilted[tuple(sat_inst["ground_qnm"])]),
        "four_sat_tilted_gs_energy": float(tilted[tuple(sat_inst["ground_qnm"])]),
        "four_sat_identity": float(sat_meta["identity"]),
        "four_sat_untilted_is_zero": bool(abs(float(np.min(untilted))) < 1e-12),
        "mixed_untilted_gs_energy": float(spin_untilted[tuple(spin_inst["ground_qnm"])]),
        "ecd_l5_params": ansatz_inventory("ecd", 5, NFOCKS),
        "ecd_l4_params": ansatz_inventory("ecd", 4, NFOCKS),
        "ecd_l8_params": ansatz_inventory("ecd", 8, NFOCKS),
        "snap_l3_params": ansatz_inventory("snap", 3, NFOCKS),
        "snap_l2_params": ansatz_inventory("snap", 2, NFOCKS),
        "shared_loader_path": "run_saved_hamiltonian_suite + energy_tensor_from_z_terms",
        "tilt": 1e-10,
    }


def main() -> int:
    sat_inst, sat_land = family_landscapes("four_sat")
    spin_inst, spin_land = family_landscapes("mixed_p_spin")
    sat_by_id = {int(i["hamiltonian_id"]): i for i in sat_inst}
    spin_by_id = {int(i["hamiltonian_id"]): i for i in spin_inst}

    files = {
        "mixed_ecd": ROOT / "results" / "gibbs_mixed_p_spin_ecd.json",
        "mixed_snap": ROOT / "results" / "gibbs_mixed_p_spin_snap.json",
        "sat_ecd": ROOT / "results" / "gibbs_four_sat_ecd.json",
        "sat_snap": ROOT / "results" / "gibbs_four_sat_snap.json",
        "sat_ecd_scout_n8": ROOT / "results" / "gibbs_four_sat_ecd_scout_n8.json",
        "sat_snap_scout_n5": ROOT / "results" / "gibbs_four_sat_snap_scout_n5.json",
        "sat_ecd_l6_400": ROOT / "results" / "gibbs_four_sat_ecd_scout_L6_400.json",
    }
    payloads = {k: _load_json(p) for k, p in files.items() if p.is_file()}

    gibbs = {}
    for key, payload in payloads.items():
        trials = payload.get("trials") or []
        depths = payload.get("ndepths") or sorted({int(t["ndepth"]) for t in trials})
        gibbs[key] = {
            "file": str(files[key].relative_to(ROOT)),
            "family": payload.get("family"),
            "ansatz": payload.get("ansatz"),
            "n_hamiltonians": payload.get("n_hamiltonians"),
            "n_trials_per_hamiltonian": payload.get("n_trials_per_hamiltonian"),
            "ndepths": depths,
            "outer_iterations": payload.get("outer_iterations"),
            "eta_policy": payload.get("eta_policy"),
            "nfocks": payload.get("nfocks"),
            "initial_state": payload.get("initial_state"),
            "by_ndepth": {str(d): summarize_trials(trials, ndepth=int(d)) for d in depths},
            "pooled": summarize_trials(trials),
        }

    reconstruct = []
    sat_ecd_trials = payloads.get("sat_ecd", {}).get("trials") or []
    sat_snap_trials = payloads.get("sat_snap", {}).get("trials") or []
    sat_ecd_scout = payloads.get("sat_ecd_scout_n8", {}).get("trials") or []
    mixed_ecd_trials = payloads.get("mixed_ecd", {}).get("trials") or []
    mixed_snap_trials = payloads.get("mixed_snap", {}).get("trials") or []
    picks = [
        ("four_sat", sat_ecd_scout, 0, 5, "ecd H000 L5 scout"),
        ("four_sat", sat_ecd_scout, 0, 6, "ecd H000 L6 scout"),
        ("four_sat", sat_ecd_scout, 0, 8, "ecd H000 L8 scout"),
        ("four_sat", sat_ecd_trials, 0, 4, "ecd H000 L4 fleet"),
        ("four_sat", sat_ecd_trials, 16, 4, "ecd H016 L4 passer"),
        ("four_sat", sat_snap_trials, 0, 3, "snap H000 L3 fleet"),
        ("four_sat", sat_snap_trials, 10, 3, "snap H010 L3 best"),
        ("mixed_p_spin", mixed_ecd_trials, 0, 5, "ecd mixed H000 L5"),
        ("mixed_p_spin", mixed_ecd_trials, 4, 5, "ecd mixed H004 L5"),
        ("mixed_p_spin", mixed_snap_trials, 0, 2, "snap mixed H000 L2"),
    ]
    inst_map = {"four_sat": sat_by_id, "mixed_p_spin": spin_by_id}
    for family, trials, hid, depth, label in picks:
        t = pick_best(trials, hid, depth)
        if t is None:
            continue
        rec = reconstruct_endpoint(t, inst_map[family][hid])
        rec["label"] = label
        reconstruct.append(rec)

    payload = {
        "gate": GATE,
        "wireup": wireup_checks(sat_inst[0], spin_inst[0]),
        "landscapes": {
            "four_sat": summarize_landscapes(sat_land),
            "mixed_p_spin": summarize_landscapes(spin_land),
            "four_sat_h000": sat_land[0],
            "mixed_p_spin_h000": spin_land[0],
            "four_sat_all": sat_land,
            "mixed_p_spin_all": spin_land,
        },
        "gibbs": gibbs,
        "reconstructed_endpoints": reconstruct,
    }
    out_json = OUTDIR / "ecd_vs_mixed_diag.json"
    out_json.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")

    lines = ["# ECD vs mixed p-spin / 4-SAT diagnostic dump", ""]
    w = payload["wireup"]
    lines.append(
        f"Encoding: both families dim={w['same_hybrid_dim'] and 128}, "
        f"nfocks={w['same_nfocks']}, n_qubits={w['n_qubits']}. "
        f"4-SAT untilted GS energy={w['four_sat_untilted_gs_energy']:.3e} "
        f"(identity={w['four_sat_identity']})."
    )
    lines.append(
        f"Ansatz params: ECD L4={w['ecd_l4_params']['n_ansatz_params']}, "
        f"L5={w['ecd_l5_params']['n_ansatz_params']}, "
        f"L8={w['ecd_l8_params']['n_ansatz_params']}; "
        f"SNAP L2={w['snap_l2_params']['n_ansatz_params']}, "
        f"L3={w['snap_l3_params']['n_ansatz_params']}."
    )
    for fam in ("four_sat", "mixed_p_spin"):
        s = payload["landscapes"][fam]
        lines.append(
            f"{fam}: Emin={s['mean_energy_min']:.3f}, spread={s['mean_spread']:.3f}, "
            f"gap={s['mean_gap']:.3f} (min {s['min_gap']:.3f}), "
            f"Hamming-1 ΔE mean={s['mean_hamming1_delta']:.3f} "
            f"(min {s['min_hamming1_delta']:.3f}), "
            f"all 7 neighbors rigid≥1: {s['all_hamming1_rigid']}, "
            f"unique energies={s['mean_n_unique_energies']:.1f}, "
            f"n_terms={s['mean_n_terms']:.1f}."
        )
    for key, block in gibbs.items():
        lines.append(f"## {key}")
        for d, s in block["by_ndepth"].items():
            lines.append(
                f"  L{d}: n={s['n']}  GS-hits={s['n_success_gs']}/{s['n']}  "
                f"near-E0={s['n_near_e0']}/{s['n']}  "
                f"GS-but-highE={s['n_success_gs_but_deficit_gt_gate']}  "
                f"mean deficit={s['mean_deficit']:.3f}  best={s['best_deficit']:.3f}  "
                f"mean η={s['mean_eta']:.2f} (η0={s['mean_eta0']:.2f})  "
                f"H with a passer={s['n_h_with_a_near_e0_trial']}/{s['n_hamiltonians']}"
            )
    lines.append("## Reconstructed endpoints")
    for rec in reconstruct:
        lines.append(
            f"  {rec['label']}: ⟨H⟩ json={rec['json_energy_physical']:.4f} "
            f"replay={rec['replay_energy_physical']:.4f} match={rec['energy_match']}  "
            f"p_GS={rec['p_gs']:.3f} p_Hamming1={rec['p_hamming1_of_gs']:.3f} "
            f"1-p_GS={rec['implied_e_if_rest_on_e1']:.3f} bits={rec['replay_bits']}"
        )
    text = "\n".join(lines) + "\n"
    (OUTDIR / "ecd_vs_mixed_diag.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
