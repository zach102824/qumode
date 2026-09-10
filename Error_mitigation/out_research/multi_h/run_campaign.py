#!/usr/bin/env python3
"""12h multi-H / multi-seed validation of the frozen PR #8 adaptive GDR recipe.

Writes only under Error_mitigation/out_research/multi_h/. Never touches src/,
Error_mitigation/out/, or out_smoke/. One heavy process; resumable.

Official recipe (unchanged kernels):
  optimized → U(0.5,1) twins + gdr_param (gdr_select == gdr_param)
  random    → span twins + gdr_select (gated damped floor on
              comprehensive κτ≤0.003)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from Error_mitigation.metrics import compare_histograms  # noqa: E402
from Error_mitigation.noise_models import circuit_noise, readout_as_dict, readout_spec  # noqa: E402
from Error_mitigation.run_ablation import (  # noqa: E402
    build_or_load_physics,
    mitigate_research,
    twin_tag,
)
from Error_mitigation.run_mitigation_experiment import (  # noqa: E402
    ANSATZ_SPEC,
    DIMS,
    SEED_BASE,
    json_ready,
    load_instance,
    make_sim,
    physical_probs,
)
from qumode_vqe.measurement import energy_from_histogram  # noqa: E402
from qumode_vqe.noise import noise_as_dict  # noqa: E402
from qumode_vqe.params import random_parameters, random_snap_parameters  # noqa: E402
from qumode_vqe.vqe import optimize_vqe  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_OFFICIAL = ROOT / "Error_mitigation" / "out"
OUT_RESEARCH = ROOT / "Error_mitigation" / "out_research"
CACHE_SRC = OUT_RESEARCH / "cache"

FAMILY = "comprehensive"
READOUT = "readout_realistic"
KAPPAS = (0.003, 0.03, 0.1)
SHOTS = 8192
N_TRAIN = 40
N_RANK2 = 10
FIT_MAXITER = 200
STRICT_GATE = 0.5
# Frozen PR #8 H000 ECD deficit vs energy_min. Used only as a documented
# reference; other H must still pass STRICT_GATE to enter optimized xfer.
H000_REF_DEFICIT = 0.880890480747827
HIDS = tuple(range(10))
MAX_NEW_RESTARTS = 5
TARGET_PASSERS = 8
METHODS_OPT = ("raw", "gdr_param", "gdr_select")
METHODS_RAND = ("raw", "gdr_param", "gdr_damped", "gdr_mid", "gdr_select")

INCUMBENT_PATHS = {
    0: [
        OUT_OFFICIAL / "optimized_params_ecd_h000_nd5.json",
        OUT_RESEARCH / "optimized_params_ecd_h000_nd5.json",
    ],
    1: [
        OUT_RESEARCH / "leftover_xfer_h001_opt2" / "optimized_params_ecd_h001_nd5.json",
        OUT_RESEARCH / "optimized_params_ecd_h001_nd5.json",
    ],
    2: [
        OUT_RESEARCH / "optimized_params_ecd_h002_nd5.json",
    ],
}

H000_OPT_CACHE = (
    "ecd_optimized_comprehensive_kt0.003_n40_default_nr10_lo0.25_hi1.35_x0.npz",
    "ecd_optimized_comprehensive_kt0.03_n40_default_nr10_lo0.25_hi1.35_x0.npz",
    "ecd_optimized_comprehensive_kt0.1_n40_default_nr10_lo0.25_hi1.35_x0.npz",
)
H000_RAND_CACHE = (
    "ecd_random_comprehensive_kt0.003_n40_span_nr10_lo0.25_hi1.35_x0.npz",
    "ecd_random_comprehensive_kt0.03_n40_span_nr10_lo0.25_hi1.35_x0.npz",
    "ecd_random_comprehensive_kt0.1_n40_span_nr10_lo0.25_hi1.35_x0.npz",
)


def now() -> float:
    return time.time()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n")


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def log(msg: str) -> None:
    print(msg, flush=True)


def remaining(deadline: float) -> float:
    return deadline - now()


def timed_out(deadline: float, slack: float = 30.0) -> bool:
    return remaining(deadline) < slack


class Paths:
    def __init__(self, root: Path = HERE):
        self.root = root
        self.phase_a = root / "phase_a"
        self.phase_b = root / "phase_b"
        self.phase_c = root / "phase_c"
        self.phase_d = root / "phase_d"
        self.phase_e = root / "phase_e"
        self.cache = root / "cache"
        self.ledger = self.phase_a / "opt_ledger.json"
        self.status = root / "STATUS.md"
        self.summary = root / "MULTI_H_SUMMARY.md"
        self.xfer = root / "transfer_records.json"
        for p in (self.phase_a, self.phase_b, self.phase_c, self.phase_d, self.phase_e, self.cache):
            p.mkdir(parents=True, exist_ok=True)

    def opt_path(self, hid: int, ansatz: str = "ecd") -> Path:
        nd = ANSATZ_SPEC[ansatz]["ndepth"]
        return self.phase_a / f"optimized_params_{ansatz}_h{hid:03d}_nd{nd}.json"


def empty_ledger() -> dict:
    return {
        "started_unix": now(),
        "strict_gate": STRICT_GATE,
        "h000_ref_deficit": H000_REF_DEFICIT,
        "hamiltonians": {},
        "restarts": [],
    }


def load_ledger(P: Paths) -> dict:
    return read_json(P.ledger, empty_ledger())


def save_ledger(P: Paths, ledger: dict) -> None:
    write_json(P.ledger, ledger)


def instance_meta(hid: int) -> dict:
    inst = load_instance(hid)
    return {
        "hamiltonian_id": hid,
        "file": str(inst.get("file")),
        "energy_min": float(inst["energy_min"]),
        "gap": float(inst["gap"]),
        "spread": float(inst["spread"]),
        "ground_qnm": [int(v) for v in inst["ground_qnm"]],
        "ground_bitstring": inst.get("ground_bitstring"),
        "energy_tensor": np.asarray(inst["energy_tensor"], float),
        "raw": inst,
    }


def seed_incumbents(P: Paths, ledger: dict) -> None:
    for hid in HIDS:
        rec = ledger["hamiltonians"].setdefault(str(hid), {})
        rec.setdefault("hamiltonian_id", hid)
        meta = instance_meta(hid)
        rec["energy_min"] = meta["energy_min"]
        rec["gap"] = meta["gap"]
        rec["ground_qnm"] = meta["ground_qnm"]
        rec["file"] = meta["file"]
        dest = P.opt_path(hid)
        if dest.is_file() and rec.get("fun") is not None:
            continue
        src = next((p for p in INCUMBENT_PATHS.get(hid, []) if p.is_file()), None)
        if src is None:
            rec.setdefault("status", "pending")
            rec.setdefault("new_restarts", 0)
            continue
        blob = json.loads(src.read_text())
        e0 = float(meta["energy_min"])
        fun = float(blob["fun"])
        payload = {
            **blob,
            "hamiltonian_id": hid,
            "seed_base": int(blob.get("seed_base", SEED_BASE)),
            "maxiter": max(int(blob.get("maxiter", 200)), 200),
            "n_restarts": max(int(blob.get("n_restarts", 3)), 3),
            "energy_min": e0,
            "gap": float(meta["gap"]),
            "deficit": fun - e0,
            "near_e0": (fun - e0) <= STRICT_GATE,
            "source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
            "note": "incumbent copied into multi_h/; not overwritten in out/ or out_smoke/",
        }
        write_json(dest, payload)
        rec.update(
            {
                "fun": fun,
                "deficit": fun - e0,
                "near_e0": (fun - e0) <= STRICT_GATE,
                "source": payload["source"],
                "status": "pass" if (fun - e0) <= STRICT_GATE else "opt-failed-incumbent",
                "new_restarts": rec.get("new_restarts", 0),
                "h000_quality": (fun - e0) <= H000_REF_DEFICIT + 1e-9,
                "x_path": str(dest),
            }
        )
        log(
            f"incumbent H{hid:03d} E={fun:.6f} E0={e0:.6f} "
            f"deficit={fun - e0:.6f} src={src.name}"
        )
    save_ledger(P, ledger)


def load_x(P: Paths, hid: int) -> np.ndarray | None:
    path = P.opt_path(hid)
    if not path.is_file():
        return None
    return np.asarray(json.loads(path.read_text())["x"], float)


def write_opt(P: Paths, hid: int, meta: dict, x: np.ndarray, history: list, e0: float, gap: float) -> None:
    dest = P.opt_path(hid)
    old = read_json(dest, {}) or {}
    payload = {
        "ansatz": "ecd",
        "ndepth": 5,
        "hamiltonian_id": hid,
        "seed_base": SEED_BASE,
        "maxiter": 200,
        "n_restarts": max(int(meta.get("n_restarts", 0)), len(history), 3),
        "fun": float(meta["fun"]),
        "nfev": int(meta.get("nfev", 0)),
        "elapsed_s": float(meta.get("elapsed_s", 0.0)),
        "restarts": history,
        "x": np.asarray(x, float).tolist(),
        "x_random": old.get("x_random"),
        "energy_min": e0,
        "gap": gap,
        "deficit": float(meta["fun"]) - e0,
        "near_e0": (float(meta["fun"]) - e0) <= STRICT_GATE,
        "note": "multi_h Phase A ECD Nd=5 L-BFGS-B maxiter=200",
    }
    write_json(dest, payload)


def one_restart(sim, x0: np.ndarray):
    return optimize_vqe(sim, x0, method="L-BFGS-B", maxiter=200, record_every=0, verbose=False)


def choose_next_restart(ledger: dict) -> tuple[int, str, int] | None:
    """Return (hid, kind, local_index) or None.

    kind: continue_best | random | warm_h000
    """
    states = {int(k): v for k, v in ledger["hamiltonians"].items()}
    n_pass = sum(1 for v in states.values() if v.get("near_e0"))
    if n_pass >= TARGET_PASSERS:
        return None

    def nnew(hid: int) -> int:
        return int(states.get(hid, {}).get("new_restarts", 0))

    h0 = states.get(0, {})
    # 1) One H000 continuation first (closest incumbent to the 0.5 gate).
    if h0.get("fun") is not None and not h0.get("near_e0") and nnew(0) == 0:
        return 0, "continue_best", 0

    # 2) First random on unseen H003–H009.
    for hid in range(3, 10):
        if nnew(hid) == 0:
            return hid, "random", 0

    # 3) More H000 attempts if it still misses the gate.
    if h0.get("fun") is not None and not h0.get("near_e0") and nnew(0) < MAX_NEW_RESTARTS:
        kind = "continue_best" if nnew(0) % 2 == 1 else "random"
        return 0, kind, nnew(0)

    # 4) Second look at the closest new H (deficit < 1.6).
    ranked = sorted(
        (
            (hid, states[hid])
            for hid in range(3, 10)
            if hid in states and states[hid].get("fun") is not None
        ),
        key=lambda kv: float(kv[1].get("deficit", 9e9)),
    )
    for hid, rec in ranked:
        if rec.get("near_e0"):
            continue
        if nnew(hid) < min(3, MAX_NEW_RESTARTS) and float(rec.get("deficit", 9e9)) < 1.6:
            kind = "continue_best" if nnew(hid) % 2 == 1 else "random"
            return hid, kind, nnew(hid)

    # 5) Fill remaining budget on the globally closest non-passers (incl. H001/H002).
    ranked_all = sorted(
        ((hid, rec) for hid, rec in states.items() if rec.get("fun") is not None),
        key=lambda kv: float(kv[1].get("deficit", 9e9)),
    )
    for hid, rec in ranked_all:
        if rec.get("near_e0"):
            continue
        if nnew(hid) < MAX_NEW_RESTARTS:
            kind = "continue_best" if nnew(hid) % 2 == 0 else "random"
            return hid, kind, nnew(hid)

    # 6) A couple of extra H001/H002 starts only after new H have been tried.
    for hid in (1, 2):
        if nnew(hid) < 2:
            return hid, "continue_best", nnew(hid)
    return None


def make_x0(kind: str, hid: int, local_index: int, P: Paths, ndepth: int) -> np.ndarray:
    if kind == "continue_best":
        x = load_x(P, hid)
        if x is not None:
            return x
        kind = "random"
    if kind == "warm_h000":
        x = load_x(P, 0)
        if x is not None:
            return x
    rng = np.random.default_rng(int(SEED_BASE) + 1009 * (10 + local_index) + 17 * hid + 333)
    return random_parameters(ndepth, rng)


def update_status(P: Paths, ledger: dict, extra: str = "") -> None:
    lines = [
        "# multi_h campaign status",
        "",
        f"updated_unix: {now():.0f}",
        f"strict_gate: `{STRICT_GATE}` (E_opt − E0)",
        "",
        "| H | E0 | E_opt | deficit | new_restarts | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for hid in HIDS:
        rec = ledger["hamiltonians"].get(str(hid), {})
        e0 = rec.get("energy_min")
        fun = rec.get("fun")
        defc = rec.get("deficit")
        lines.append(
            f"| H{hid:03d} | "
            f"{'' if e0 is None else f'{e0:.4f}'} | "
            f"{'' if fun is None else f'{fun:.4f}'} | "
            f"{'' if defc is None else f'{defc:.4f}'} | "
            f"{rec.get('new_restarts', 0)} | "
            f"{rec.get('status', 'pending')} |"
        )
    if extra:
        lines += ["", extra]
    P.status.write_text("\n".join(lines) + "\n")


def phase_a(P: Paths, deadline: float) -> dict:
    ledger = load_ledger(P)
    seed_incumbents(P, ledger)
    update_status(P, ledger, extra="Phase A running.")
    ndepth = int(ANSATZ_SPEC["ecd"]["ndepth"])
    sims: dict[int, object] = {}

    def get_sim(hid: int):
        if hid not in sims:
            meta = instance_meta(hid)
            sims[hid] = (
                make_sim("ecd", ndepth, meta["energy_tensor"], tuple(meta["ground_qnm"])),
                meta,
            )
        return sims[hid]

    while not timed_out(deadline, slack=90.0):
        nxt = choose_next_restart(ledger)
        if nxt is None:
            log("Phase A: no more scheduled restarts (passers or caps).")
            break
        hid, kind, local_index = nxt
        sim, meta = get_sim(hid)
        e0 = float(meta["energy_min"])
        gap = float(meta["gap"])
        x0 = make_x0(kind, hid, local_index, P, ndepth)
        log(
            f"Phase A H{hid:03d} {kind} #{local_index + 1}  "
            f"E0={e0:.4f} remaining={remaining(deadline)/60:.1f} min"
        )
        t0 = now()
        opt = one_restart(sim, x0)
        elapsed = now() - t0
        fun = float(opt.fun)
        deficit = fun - e0
        rec = {
            "hid": hid,
            "kind": kind,
            "local_index": local_index,
            "fun": fun,
            "deficit": deficit,
            "success": bool(opt.success),
            "nfev": int(opt.nfev),
            "elapsed_s": elapsed,
            "unix": now(),
        }
        ledger["restarts"].append(rec)
        hrec = ledger["hamiltonians"].setdefault(str(hid), {})
        hrec["new_restarts"] = int(hrec.get("new_restarts", 0)) + 1
        hrec["energy_min"] = e0
        hrec["gap"] = gap
        history = list(hrec.get("restart_history") or [])
        history.append(rec)
        hrec["restart_history"] = history
        best = hrec.get("fun")
        improved = best is None or fun < float(best) - 1e-12
        if improved:
            hrec["fun"] = fun
            hrec["deficit"] = deficit
            hrec["nfev"] = int(opt.nfev)
            hrec["elapsed_s"] = float(hrec.get("elapsed_s", 0.0)) + elapsed
            write_opt(P, hid, hrec, np.asarray(opt.x, float), history, e0, gap)
            if hid == 0:
                invalidate_h000_opt_cache(P)
        else:
            hrec["elapsed_s"] = float(hrec.get("elapsed_s", 0.0)) + elapsed
            if P.opt_path(hid).is_file():
                blob = json.loads(P.opt_path(hid).read_text())
                blob["restarts"] = history
                blob["n_restarts"] = max(int(blob.get("n_restarts", 3)), len(history), 3)
                write_json(P.opt_path(hid), blob)
        hrec["near_e0"] = float(hrec.get("deficit", 9e9)) <= STRICT_GATE
        hrec["h000_quality"] = float(hrec.get("deficit", 9e9)) <= H000_REF_DEFICIT + 1e-9
        if hrec["near_e0"]:
            hrec["status"] = "pass"
        elif hrec.get("fun") is not None:
            hrec["status"] = "opt-failed"
        log(
            f"  → E={fun:.6f} deficit={deficit:.6f} nfev={opt.nfev} "
            f"{elapsed:.1f}s  best={hrec.get('fun'):.6f} "
            f"near_e0={hrec['near_e0']}"
        )
        save_ledger(P, ledger)
        update_status(P, ledger, extra="Phase A running.")

    for hid in HIDS:
        hrec = ledger["hamiltonians"].setdefault(str(hid), {})
        if hrec.get("fun") is None:
            hrec["status"] = "opt-failed-no-start"
            hrec["near_e0"] = False
        elif hrec.get("near_e0"):
            hrec["status"] = "pass"
        else:
            hrec["status"] = "opt-failed"
    save_ledger(P, ledger)
    update_status(P, ledger, extra="Phase A finished.")
    n_pass = sum(1 for hid in HIDS if ledger["hamiltonians"][str(hid)].get("near_e0"))
    log(f"Phase A done. strict passers={n_pass}")
    return ledger


def passers(ledger: dict) -> list[int]:
    out = []
    for hid in HIDS:
        rec = ledger["hamiltonians"].get(str(hid), {})
        if rec.get("near_e0") and rec.get("fun") is not None:
            out.append(hid)
    # Frozen H000 is always eligible for optimized transfer (PR #8 reference
    # + Phase C). Documented in the summary if it misses STRICT_GATE.
    if 0 not in out and ledger["hamiltonians"].get("0", {}).get("fun") is not None:
        out.insert(0, 0)
    return out


def phys_args(circuit_kind: str, hid: int, seed: int = SEED_BASE) -> argparse.Namespace:
    design = "span" if circuit_kind == "random" else "default"
    return argparse.Namespace(
        twin_design=design,
        n_rank2=N_RANK2,
        mag_lo=0.25,
        mag_hi=1.35,
        extra_t_free=0,
        seed=int(seed),
        no_cache=False,
        instance=int(hid),
    )


def h000_x_matches_official(P: Paths) -> bool:
    path = P.opt_path(0)
    src_off = INCUMBENT_PATHS[0][0]
    if not path.is_file() or not src_off.is_file():
        return False
    x = np.asarray(json.loads(path.read_text())["x"], float)
    x0 = np.asarray(json.loads(src_off.read_text())["x"], float)
    return x.shape == x0.shape and float(np.max(np.abs(x - x0))) <= 1e-12


def invalidate_h000_opt_cache(P: Paths) -> None:
    for name in H000_OPT_CACHE:
        for p in (P.cache / name, (P.cache / name).with_suffix(".json")):
            if p.is_file():
                p.unlink()
                log(f"removed stale H000 cache {p.name}")


def maybe_copy_h000_opt_cache(P: Paths, ledger: dict) -> None:
    """Reuse official H000 optimized comprehensive caches if x is unchanged."""
    if not h000_x_matches_official(P):
        invalidate_h000_opt_cache(P)
        log("H000 x changed vs out/; will resimulate optimized cells.")
        return
    for name in H000_OPT_CACHE:
        src = CACHE_SRC / name
        dst = P.cache / name
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
            meta = src.with_suffix(".json")
            if meta.is_file():
                shutil.copy2(meta, dst.with_suffix(".json"))
            log(f"copied official physics cache {name}")


def observe_only(phys: dict, spec, ansatz, pset, family, kt, seed, shots_key: str):
    from Error_mitigation.mitigation import observe_histogram
    from Error_mitigation.run_mitigation_experiment import case_seed

    p_phys = phys["p_phys"]
    twin_phys = phys["twin_phys"]
    seed_t = case_seed("obs", ansatz, pset, family, kt, spec.level, seed, "target", shots_key)
    q_obs = observe_histogram(p_phys, spec, DIMS, seed_t)
    q_twins = [
        observe_histogram(
            twin_phys[i],
            spec,
            DIMS,
            case_seed("obs", ansatz, pset, family, kt, spec.level, seed, "twin", i, shots_key),
        )
        for i in range(twin_phys.shape[0])
    ]
    hist_by_scale = {1: q_obs}
    if "scale_2" in phys:
        hist_by_scale[2] = observe_histogram(
            phys["scale_2"],
            spec,
            DIMS,
            case_seed("obs", ansatz, pset, family, kt, spec.level, seed, "zne", 2, shots_key),
        )
    if "scale_3" in phys:
        hist_by_scale[3] = observe_histogram(
            phys["scale_3"],
            spec,
            DIMS,
            case_seed("obs", ansatz, pset, family, kt, spec.level, seed, "zne", 3, shots_key),
        )
    return q_obs, q_twins, hist_by_scale


def run_cell(
    *,
    P: Paths,
    hid: int,
    energy_tensor: np.ndarray,
    ground_qnm: tuple[int, int, int],
    xvec: np.ndarray,
    circuit_kind: str,
    kt: float,
    seed: int,
    methods: tuple[str, ...],
    pset_label: str,
    ansatz: str = "ecd",
) -> dict:
    args = phys_args(circuit_kind, hid, seed=SEED_BASE)
    ndepth = int(ANSATZ_SPEC[ansatz]["ndepth"])
    phys = build_or_load_physics(
        ansatz=ansatz,
        pset=pset_label,
        xvec=xvec,
        family=FAMILY,
        kt=float(kt),
        n_train=N_TRAIN,
        args=args,
        energy_tensor=energy_tensor,
        ground_qnm=ground_qnm,
        cache_dir=P.cache,
    )
    spec = readout_spec(READOUT, SHOTS, seed=None)
    q_obs, q_twins, hist_by_scale = observe_only(
        phys, spec, ansatz, pset_label, FAMILY, kt, seed, f"s{SHOTS}"
    )
    cfg = circuit_noise(FAMILY, float(kt), dims=DIMS)
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
        fit_maxiter=FIT_MAXITER,
        circuit_kind=circuit_kind,
        family=FAMILY,
        kappa_tau=float(kt),
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
        "hamiltonian_id": hid,
        "ansatz": ansatz,
        "ndepth": ndepth,
        "params": pset_label,
        "circuit_kind": circuit_kind,
        "family": FAMILY,
        "kappa_tau": float(kt),
        "readout": READOUT,
        "n_shots": SHOTS,
        "n_train": N_TRAIN,
        "seed": int(seed),
        "twin_design": args.twin_design,
        "twin_tag": twin_tag(args),
        "noise": noise_as_dict(cfg),
        "readout_spec": readout_as_dict(spec),
        "metrics": metrics,
        "select_fit": (mitigated.get("gdr_select") or {}).get("fit"),
        "energy_ideal": float(energy_from_histogram(phys["p_ideal"], energy_tensor)),
    }
    raw = (metrics.get("raw") or {}).get("tvd")
    gdr = (metrics.get("gdr_param") or {}).get("tvd")
    sel = (metrics.get("gdr_select") or {}).get("tvd")
    log(
        f"    H{hid:03d} {pset_label} {circuit_kind} κτ={kt:g} seed={seed}  "
        f"raw={raw if raw is None else f'{raw:.4f}'}  "
        f"gdr={gdr if gdr is None else f'{gdr:.4f}'}  "
        f"sel={sel if sel is None else f'{sel:.4f}'}"
    )
    return rec


def phase_b(P: Paths, ledger: dict, deadline: float) -> list[dict]:
    maybe_copy_h000_opt_cache(P, ledger)
    records = read_json(P.phase_b / "results.json", {"records": []}).get("records", [])
    done = {
        (int(r["hamiltonian_id"]), float(r["kappa_tau"]), int(r.get("seed", SEED_BASE)))
        for r in records
    }
    ids = passers(ledger)
    log(f"Phase B optimized transfer on H={ids}")
    for hid in ids:
        if timed_out(deadline, slack=60.0):
            log("Phase B: wall clock; stop expanding H.")
            break
        x = load_x(P, hid)
        if x is None:
            log(f"Phase B skip H{hid:03d}: no x")
            continue
        meta = instance_meta(hid)
        hrec = ledger["hamiltonians"][str(hid)]
        if hid != 0 and not hrec.get("near_e0"):
            log(f"Phase B skip H{hid:03d}: opt-failed deficit={hrec.get('deficit')}")
            continue
        for kt in KAPPAS:
            key = (hid, float(kt), SEED_BASE)
            if key in done:
                continue
            if timed_out(deadline, slack=60.0):
                break
            rec = run_cell(
                P=P,
                hid=hid,
                energy_tensor=meta["energy_tensor"],
                ground_qnm=tuple(meta["ground_qnm"]),
                xvec=x,
                circuit_kind="optimized",
                kt=float(kt),
                seed=SEED_BASE,
                methods=METHODS_OPT,
                pset_label="optimized",
            )
            rec["opt_deficit"] = hrec.get("deficit")
            rec["opt_fun"] = hrec.get("fun")
            rec["opt_gate"] = "strict" if hrec.get("near_e0") else "h000_reference"
            records.append(rec)
            write_json(P.phase_b / "results.json", {"records": records})
            done.add(key)
    write_json(P.xfer, {"phase_b": records})
    return records


def phase_c(P: Paths, ledger: dict, deadline: float) -> list[dict]:
    """10 shot seeds on H000 optimized, same 3 κτ cells. Physics cached."""
    maybe_copy_h000_opt_cache(P, ledger)
    x = load_x(P, 0)
    if x is None:
        log("Phase C skipped: no H000 x")
        return []
    meta = instance_meta(0)
    records = read_json(P.phase_c / "results.json", {"records": []}).get("records", [])
    done = {(float(r["kappa_tau"]), int(r["seed"])) for r in records}
    seeds = [SEED_BASE + 10007 * k for k in range(10)]
    log(f"Phase C H000 × 10 seeds {seeds}")
    for seed in seeds:
        for kt in KAPPAS:
            if (float(kt), int(seed)) in done:
                continue
            if timed_out(deadline, slack=45.0):
                log("Phase C wall clock stop")
                break
            rec = run_cell(
                P=P,
                hid=0,
                energy_tensor=meta["energy_tensor"],
                ground_qnm=tuple(meta["ground_qnm"]),
                xvec=x,
                circuit_kind="optimized",
                kt=float(kt),
                seed=int(seed),
                methods=METHODS_OPT,
                pset_label="optimized",
            )
            records.append(rec)
            write_json(P.phase_c / "results.json", {"records": records, "seeds": seeds})
            done.add((float(kt), int(seed)))
    return records


def official_random_x(hid: int, ansatz: str = "ecd") -> np.ndarray:
    nd = int(ANSATZ_SPEC[ansatz]["ndepth"])
    rng = np.random.default_rng(int(SEED_BASE) + 17 * (0 if ansatz == "ecd" else 1) + hid)
    if ansatz == "snap":
        return random_snap_parameters(nd, (8, 8), rng)
    return random_parameters(nd, rng)


def phase_d(P: Paths, ledger: dict, deadline: float, kappas: tuple[float, ...] | None = None) -> list[dict]:
    kappas = kappas or KAPPAS
    records = read_json(P.phase_d / "results.json", {"records": []}).get("records", [])
    done = {
        (int(r["hamiltonian_id"]), int(r.get("random_id", -1)), float(r["kappa_tau"]))
        for r in records
    }
    # Prefer a second H that passed the strict gate; else the next-best passer.
    second = next((h for h in passers(ledger) if h != 0 and ledger["hamiltonians"][str(h)].get("near_e0")), None)
    if second is None:
        ranked = sorted(
            (
                (int(k), v)
                for k, v in ledger["hamiltonians"].items()
                if int(k) != 0 and v.get("fun") is not None
            ),
            key=lambda kv: float(kv[1].get("deficit", 9e9)),
        )
        second = ranked[0][0] if ranked else None
    jobs = [(0, rid) for rid in range(8)]
    if second is not None:
        jobs += [(second, rid) for rid in range(4)]
    log(f"Phase D random ECD jobs={jobs} second_H={second} kappas={kappas}")

    # Official H000 random (rid=0) can reuse span caches if present.
    for name in H000_RAND_CACHE:
        src = CACHE_SRC / name
        dst = P.cache / name
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
            meta = src.with_suffix(".json")
            if meta.is_file():
                shutil.copy2(meta, dst.with_suffix(".json"))
            log(f"copied official random cache {name}")

    for hid, rid in jobs:
        meta = instance_meta(hid)
        if hid == 0 and rid == 0:
            xvec = official_random_x(0)
            pset_label = "random"
        else:
            rng = np.random.default_rng(int(SEED_BASE) + 90011 * hid + 17 * rid + 4242)
            xvec = random_parameters(int(ANSATZ_SPEC["ecd"]["ndepth"]), rng)
            pset_label = f"random_{rid}"
        for kt in kappas:
            key = (hid, rid, float(kt))
            if key in done:
                continue
            if timed_out(deadline, slack=60.0):
                if set(kappas) == set(KAPPAS) and remaining(deadline) > 180:
                    # Fall back to the two bookend κτ values.
                    return phase_d(P, ledger, deadline, kappas=(0.003, 0.1))
                log("Phase D wall clock stop")
                write_json(P.phase_d / "results.json", {"records": records, "second_h": second})
                return records
            rec = run_cell(
                P=P,
                hid=hid,
                energy_tensor=meta["energy_tensor"],
                ground_qnm=tuple(meta["ground_qnm"]),
                xvec=xvec,
                circuit_kind="random",
                kt=float(kt),
                seed=SEED_BASE + rid,
                methods=METHODS_RAND,
                pset_label=pset_label,
            )
            rec["random_id"] = rid
            records.append(rec)
            write_json(P.phase_d / "results.json", {"records": records, "second_h": second})
            done.add(key)
    return records


def phase_e_snap(P: Paths, ledger: dict, deadline: float) -> list[dict]:
    """SNAP Nd=2 only for H that already passed the ECD gate, if ≥2 such H."""
    ecd_pass = [h for h in passers(ledger) if h == 0 or ledger["hamiltonians"][str(h)].get("near_e0")]
    if len(ecd_pass) < 2:
        log("Phase E skipped: <2 ECD-pass H")
        return []
    if remaining(deadline) < 20 * 60:
        log("Phase E skipped: <20 min left")
        return []
    records = read_json(P.phase_e / "results.json", {"records": []}).get("records", [])
    snap_pass = []
    nd = int(ANSATZ_SPEC["snap"]["ndepth"])
    for hid in ecd_pass:
        if timed_out(deadline, slack=90.0):
            break
        dest = P.opt_path(hid, "snap")
        meta = instance_meta(hid)
        e0 = float(meta["energy_min"])
        if dest.is_file():
            blob = json.loads(dest.read_text())
            if float(blob["fun"]) - e0 <= STRICT_GATE:
                snap_pass.append(hid)
            continue
        sim = make_sim("snap", nd, meta["energy_tensor"], tuple(meta["ground_qnm"]))
        best_x, best_e, history = None, float("inf"), []
        for r in range(3):
            if timed_out(deadline, slack=90.0):
                break
            rng = np.random.default_rng(int(SEED_BASE) + 53 + 1009 * r + hid)
            x0 = random_snap_parameters(nd, (8, 8), rng)
            log(f"Phase E SNAP H{hid:03d} restart {r + 1}/3")
            opt = optimize_vqe(sim, x0, method="L-BFGS-B", maxiter=200, record_every=0, verbose=False)
            history.append({"restart": r, "fun": float(opt.fun), "nfev": int(opt.nfev)})
            if float(opt.fun) < best_e:
                best_e = float(opt.fun)
                best_x = np.asarray(opt.x, float)
        if best_x is None:
            continue
        payload = {
            "ansatz": "snap",
            "ndepth": nd,
            "hamiltonian_id": hid,
            "seed_base": SEED_BASE,
            "maxiter": 200,
            "n_restarts": max(len(history), 3),
            "fun": best_e,
            "restarts": history,
            "x": best_x.tolist(),
            "energy_min": e0,
            "deficit": best_e - e0,
            "near_e0": (best_e - e0) <= STRICT_GATE,
        }
        write_json(dest, payload)
        if payload["near_e0"]:
            snap_pass.append(hid)
            log(f"  SNAP H{hid:03d} PASS deficit={payload['deficit']:.4f}")
        else:
            log(f"  SNAP H{hid:03d} fail deficit={payload['deficit']:.4f}")
    if len(snap_pass) < 2:
        log(f"Phase E: only {len(snap_pass)} near-E0 SNAP opts; skip xfer")
        return records
    for hid in snap_pass:
        dest = P.opt_path(hid, "snap")
        x = np.asarray(json.loads(dest.read_text())["x"], float)
        meta = instance_meta(hid)
        for kt in KAPPAS:
            if timed_out(deadline, slack=45.0):
                break
            rec = run_cell(
                P=P,
                hid=hid,
                energy_tensor=meta["energy_tensor"],
                ground_qnm=tuple(meta["ground_qnm"]),
                xvec=x,
                circuit_kind="optimized",
                kt=float(kt),
                seed=SEED_BASE,
                methods=METHODS_OPT,
                pset_label="optimized",
                ansatz="snap",
            )
            records.append(rec)
            write_json(P.phase_e / "results.json", {"records": records, "snap_pass": snap_pass})
    return records


def _tvd(rec: dict, method: str):
    m = (rec.get("metrics") or {}).get(method) or {}
    return m.get("tvd"), m.get("dE")


def _meanstd(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)


def write_summary(P: Paths, ledger: dict) -> None:
    b = read_json(P.phase_b / "results.json", {"records": []}).get("records", [])
    c = read_json(P.phase_c / "results.json", {"records": []}).get("records", [])
    d = read_json(P.phase_d / "results.json", {"records": []}).get("records", [])
    e = read_json(P.phase_e / "results.json", {"records": []}).get("records", [])
    lines = [
        "# Multi-H realistic comprehensive validation (PR #8 adaptive GDR)",
        "",
        "Frozen recipe, no new mitigation kernels. Noise locked to "
        "`comprehensive` + `readout_realistic`, 8192 shots, 40 twins, "
        "adaptive design (span on random, U(0.5,1) on optimized).",
        "",
        f"Keep gate: `E_opt - E0 <= {STRICT_GATE}`. "
        "H000 frozen ECD from `Error_mitigation/out/optimized_params_ecd_h000_nd5.json` "
        "is the PR #8 reference circuit and is always included in optimized "
        "transfer / 10-seed even if it misses the strict gate; that exception "
        "is labeled `h000_reference`. Every other H that misses the gate is "
        "**opt-failed** and has no optimized transfer matrix.",
        "",
        "## Phase A — which H passed / failed the E0 gate",
        "",
        "| H | file | E0 | E_opt | deficit | new restarts | gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    n_strict = 0
    for hid in HIDS:
        rec = ledger["hamiltonians"].get(str(hid), {})
        e0 = rec.get("energy_min")
        fun = rec.get("fun")
        defc = rec.get("deficit")
        gate = "PASS" if rec.get("near_e0") else ("h000_reference" if hid == 0 and fun is not None else "FAIL")
        if rec.get("near_e0"):
            n_strict += 1
        lines.append(
            f"| H{hid:03d} | `{rec.get('file', '')}` | "
            f"{'' if e0 is None else f'{e0:.4f}'} | "
            f"{'' if fun is None else f'{fun:.4f}'} | "
            f"{'' if defc is None else f'{defc:.4f}'} | "
            f"{rec.get('new_restarts', 0)} | {gate} |"
        )
    failed = [
        f"H{hid:03d}"
        for hid in HIDS
        if not ledger["hamiltonians"].get(str(hid), {}).get("near_e0")
    ]
    lines += [
        "",
        f"Strict passers (deficit ≤ {STRICT_GATE}): **{n_strict}**.",
        f"Honest opt-fail list: {', '.join(failed) if failed else '(none)'}.",
        "",
        "## Phase B — optimized ECD transfer (adaptive vs raw)",
        "",
        "Each cell: comprehensive + readout_realistic. Adaptive select on "
        "optimized **is** `gdr_param`.",
        "",
        "| H | κτ | raw TVD | adaptive select TVD | gdr_param TVD | raw \\|ΔE\\| | select \\|ΔE\\| | beat raw? |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    mild_wins = []
    for rec in b:
        raw_t, raw_e = _tvd(rec, "raw")
        sel_t, sel_e = _tvd(rec, "gdr_select")
        gdr_t, _ = _tvd(rec, "gdr_param")
        beat = None
        if raw_t is not None and sel_t is not None:
            beat = sel_t < raw_t - 1e-12
            if abs(float(rec["kappa_tau"]) - 0.003) < 1e-12:
                mild_wins.append((int(rec["hamiltonian_id"]), beat, raw_t, sel_t))
        lines.append(
            f"| H{int(rec['hamiltonian_id']):03d} | {rec['kappa_tau']:g} | "
            f"{'' if raw_t is None else f'{raw_t:.4f}'} | "
            f"{'' if sel_t is None else f'{sel_t:.4f}'} | "
            f"{'' if gdr_t is None else f'{gdr_t:.4f}'} | "
            f"{'' if raw_e is None else f'{raw_e:.4f}'} | "
            f"{'' if sel_e is None else f'{sel_e:.4f}'} | "
            f"{'' if beat is None else ('yes' if beat else 'no')} |"
        )
    n_mild_yes = len({h for h, beat, *_ in mild_wins if beat})
    lines += [
        "",
        f"Hamiltonians that beat raw at κτ=0.003 (comprehensive+readout_realistic): "
        f"**{n_mild_yes}** ({', '.join(f'H{h:03d}' for h, beat, *_ in mild_wins if beat) or 'none'}).",
        "",
        f"**Success bar 1 (≥4 H beat raw at κτ=0.003): {'YES' if n_mild_yes >= 4 else 'NO'}.**",
        "",
        "κτ=0.03 and 0.1 are reported in the table above without filtering. "
        "Wins and losses both stand.",
        "",
        "## Phase C — H000 10-seed mean ± std",
        "",
        "| κτ | raw TVD | adaptive select TVD | gdr_param TVD | seed flips (select loses to raw) |",
        "|---:|---:|---:|---:|---|",
    ]
    mild_mean_beat = None
    for kt in KAPPAS:
        rows = [r for r in c if abs(float(r["kappa_tau"]) - kt) < 1e-12]
        raws = [ _tvd(r, "raw")[0] for r in rows if _tvd(r, "raw")[0] is not None ]
        sels = [ _tvd(r, "gdr_select")[0] for r in rows if _tvd(r, "gdr_select")[0] is not None ]
        gdrs = [ _tvd(r, "gdr_param")[0] for r in rows if _tvd(r, "gdr_param")[0] is not None ]
        flips = []
        for r in rows:
            rt, _ = _tvd(r, "raw")
            st, _ = _tvd(r, "gdr_select")
            if rt is not None and st is not None and st >= rt - 1e-12:
                flips.append(str(r.get("seed")))
        rm, rs = _meanstd(raws)
        sm, ss = _meanstd(sels)
        gm, gs = _meanstd(gdrs)
        if abs(kt - 0.003) < 1e-12 and rm is not None and sm is not None:
            mild_mean_beat = sm < rm
        lines.append(
            f"| {kt:g} | "
            f"{'' if rm is None else f'{rm:.4f} ± {rs:.4f}'} | "
            f"{'' if sm is None else f'{sm:.4f} ± {ss:.4f}'} | "
            f"{'' if gm is None else f'{gm:.4f} ± {gs:.4f}'} | "
            f"{', '.join(flips) if flips else 'none'} |"
        )
    lines += [
        "",
        f"**Success bar 3 (H000 10-seed mean beats raw on the mild cell): "
        f"{'YES' if mild_mean_beat else 'NO' if mild_mean_beat is False else 'incomplete'}.**",
        "",
        "## Phase D — random ECD targets",
        "",
    ]
    if not d:
        lines.append("No random-target cells finished.")
    else:
        lines += [
            "| H | random_id | κτ | raw TVD | adaptive select TVD | gdr_param TVD | beat raw? |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
        n_win = n_tot = 0
        for rec in d:
            raw_t, _ = _tvd(rec, "raw")
            sel_t, _ = _tvd(rec, "gdr_select")
            gdr_t, _ = _tvd(rec, "gdr_param")
            beat = raw_t is not None and sel_t is not None and sel_t < raw_t - 1e-12
            n_tot += 1
            n_win += int(beat)
            lines.append(
                f"| H{int(rec['hamiltonian_id']):03d} | {rec.get('random_id', '')} | "
                f"{rec['kappa_tau']:g} | "
                f"{'' if raw_t is None else f'{raw_t:.4f}'} | "
                f"{'' if sel_t is None else f'{sel_t:.4f}'} | "
                f"{'' if gdr_t is None else f'{gdr_t:.4f}'} | "
                f"{'yes' if beat else 'no'} |"
            )
        lines += ["", f"Random cells where adaptive select beats raw: **{n_win} / {n_tot}**.", ""]
    if e:
        lines += [
            "## Phase E — SNAP (only if ≥2 near-E0 SNAP opts)",
            "",
            "| H | κτ | raw TVD | adaptive select TVD | beat raw? |",
            "|---|---:|---:|---:|:---:|",
        ]
        for rec in e:
            raw_t, _ = _tvd(rec, "raw")
            sel_t, _ = _tvd(rec, "gdr_select")
            beat = raw_t is not None and sel_t is not None and sel_t < raw_t - 1e-12
            lines.append(
                f"| H{int(rec['hamiltonian_id']):03d} | {rec['kappa_tau']:g} | "
                f"{'' if raw_t is None else f'{raw_t:.4f}'} | "
                f"{'' if sel_t is None else f'{sel_t:.4f}'} | "
                f"{'yes' if beat else 'no'} |"
            )
        lines.append("")
    else:
        lines += ["## Phase E — SNAP", "", "Not run or no ≥2 near-E0 SNAP opts.", ""]
    lines += [
        "## Headline",
        "",
        f"≥4 H beat raw at κτ=0.003 under realistic device noise "
        f"(comprehensive + readout_realistic): **{'YES' if n_mild_yes >= 4 else 'NO'}** "
        f"({n_mild_yes} H).",
        "",
        "Do not treat mid-quality VQE losses as a recipe bug. Official adaptive "
        "defaults were not changed.",
        "",
    ]
    P.summary.write_text("\n".join(lines) + "\n")
    log(f"wrote {P.summary}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", default="all", choices=("a", "b", "c", "d", "e", "f", "all"))
    p.add_argument("--deadline-h", type=float, default=11.5, help="Wall hours from now.")
    p.add_argument("--phase-a-h", type=float, default=3.5)
    p.add_argument("--phase-b-h", type=float, default=3.0)
    p.add_argument("--phase-c-h", type=float, default=2.0)
    p.add_argument("--phase-d-h", type=float, default=1.5)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    P = Paths()
    t_start = now()
    hard = t_start + float(args.deadline_h) * 3600.0
    log(
        f"multi_h campaign phase={args.phase} deadline={args.deadline_h}h "
        f"root={P.root}"
    )
    ledger = load_ledger(P)
    seed_incumbents(P, ledger)

    if args.phase in ("a", "all"):
        a_dead = min(hard, now() + float(args.phase_a_h) * 3600.0)
        ledger = phase_a(P, a_dead)
        write_summary(P, ledger)

    ledger = load_ledger(P)

    if args.phase in ("b", "all"):
        b_dead = min(hard, now() + float(args.phase_b_h) * 3600.0)
        if args.phase == "all":
            # Prefer finishing B+C+F: keep at least ~2.2h after B for C+F,
            # but do not starve B if Phase A ran long.
            b_dead = min(b_dead, hard - 2.2 * 3600.0)
            b_dead = max(b_dead, now() + 20 * 60)
        phase_b(P, ledger, b_dead)
        write_summary(P, ledger)

    if args.phase in ("c", "all"):
        c_dead = min(hard, now() + float(args.phase_c_h) * 3600.0)
        phase_c(P, ledger, c_dead)
        write_summary(P, ledger)

    if args.phase in ("d", "all"):
        d_dead = min(hard, now() + float(args.phase_d_h) * 3600.0)
        if remaining(hard) > 25 * 60:
            phase_d(P, ledger, d_dead)
            write_summary(P, ledger)
        else:
            log("Skip Phase D to protect F summary / remaining wall.")

    if args.phase in ("e", "all"):
        if remaining(hard) > 25 * 60:
            phase_e_snap(P, ledger, hard - 10 * 60)
            write_summary(P, ledger)
        else:
            log("Skip Phase E (SNAP).")

    if args.phase in ("f", "all"):
        write_summary(P, load_ledger(P))

    log(f"campaign slice done in {(now() - t_start)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
