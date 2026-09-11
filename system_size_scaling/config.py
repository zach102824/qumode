"""Protocol constants for the isolated ECD system-size ladder."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
HAM_ROOT = ROOT / "Hamiltonians"
RESULTS_ROOT = ROOT / "results"
SUPERSEDED_70_ROOT = ROOT / "results_70spsa_superseded"

# Production n=7 4-SAT used ~18 clauses / 7 variables.
CLAUSE_DENSITY = 18 / 7
CLAUSE_WIDTH = 4
N_HAMILTONIANS = 20
N_TRIALS = 10
SEARCH_TRIALS = 4000

# Depth sweep for the live ladder (n=8…11): start L=4, increment until
# success ≥ 90%. No hard cap at L=20. Soft safety stop at L=40.
L_START = 4
L_MAX = 40
SUCCESS_THRESHOLD = 0.90

# Canonical optimizer budget — PR #14 / new default (not the superseded 70).
OUTER_ITER = 200
PROTOCOL_TAG = "200_joint_spsa_noiseless"

# Official n=7 scoreboard row — PR #14 noiseless ECD L4 (200 joint SPSA),
# not this folder's n=7 L-sweep. Do not regenerate n=7 for the table.
PRIOR_N7 = {
    "n": 7,
    "L_star": 4,
    "k": 186,
    "n_total": 200,
    "success_prob": 0.93,
    "success_fraction": "186/200",
    "wall_s": None,
    "status": "prior_data_pr14",
    "source": "https://github.com/zach102824/qumode/pull/14",
    "protocol_note": (
        "PR #14 matched 4-SAT fleet, noiseless ECD L4, 20 H × 10 trials, "
        "200 joint SPSA, vacuum, sampled_tail η. Not re-run in this folder."
    ),
    "this_folder_l3_smoke": {
        "k": 158,
        "n_total": 200,
        "success_prob": 0.79,
        "note": "this folder n=7 L=3 / 70 SPSA; not the official L*",
    },
}

# Hardware target: 2 transmons × 3 cavities × 8 levels = 2048.
N_TRANSMONS = 2
N_CAVITIES = 3
FOCK_CUTOFF = 8
HARDWARE_DIM = 2 * 2 * 8 * 8 * 8  # 2048
MAX_LOGICAL_BITS = N_TRANSMONS + N_CAVITIES * 3  # 11

# Distinct from production four_sat.py (seed 11000) and size-sweep ham seeds.
HAM_SEED_BASE = 27700

# Joint SPSA — same gains as production optimize_gibbs_adaptive / ECD script.
SPSA_A = 0.2
SPSA_C = 0.15
SPSA_A_STAB = 10.0
SPSA_ALPHA = 0.602
SPSA_GAMMA = 0.101
PREP_STEP_SCALE = 1.0
SEED_BASE = 41000

# Clause-count band around density × n (uniqueness loop, not the old 12–20 lock).
CLAUSE_BAND_BELOW = 6
CLAUSE_BAND_ABOVE = 8

LADDER_NS = (8, 9, 10, 11)
ALL_NS = (7, 8, 9, 10, 11)

# Frozen 70-SPSA L=4…20 scoreboard. Not canonical. Deeper L collapsed because
# a fixed 70-step SPSA budget under-trains the extra parameters (p_ground and
# ⟨H⟩ both degrade). See results_70spsa_superseded/.
SUPERSEDED_70_SPSA = {
    8: {"best_L": 4, "k": 39, "n_total": 200, "note": "L=4…20 done; never ≥90%"},
    9: {"best_L": 4, "k": 51, "n_total": 200, "note": "L=4…20 done; never ≥90%"},
    10: {"best_L": 4, "k": 11, "n_total": 200, "note": "L=4…20 done; never ≥90%"},
    11: {"best_L": 4, "k": 2, "n_total": 200, "note": "L=4…10 on disk; cancelled"},
}


def clause_window(n: int) -> tuple[int, int, int]:
    """Return (min_clauses, target_clauses, max_clauses) at density 18/7."""
    n = int(n)
    target = int(round(n * CLAUSE_DENSITY))
    lo = max(n, target - CLAUSE_BAND_BELOW)
    hi = max(target + CLAUSE_BAND_ABOVE, lo + 1)
    return lo, target, hi


def ham_seed(n: int) -> int:
    return int(HAM_SEED_BASE) + 100 * int(n)


def trial_seed(n: int, hid: int, trial: int) -> int:
    return int(SEED_BASE) + 1000 * int(n) + 10 * int(hid) + int(trial)


def ham_dir(n: int) -> Path:
    return HAM_ROOT / f"n{int(n)}"


def results_path(n: int, depth: int) -> Path:
    return RESULTS_ROOT / f"n{int(n)}_L{int(depth):02d}.json"


def summary_path(n: int) -> Path:
    return RESULTS_ROOT / f"n{int(n)}_summary.json"


def is_canonical_cell(payload: dict, outer_iter: int = OUTER_ITER) -> bool:
    """True if a results JSON was produced under the live 200-SPSA protocol."""
    if not payload:
        return False
    got = payload.get("outer_iter")
    if got is None:
        got = (payload.get("protocol") or {}).get("spsa", {}).get("outer_iter")
    if int(got or 0) != int(outer_iter):
        return False
    tag = (payload.get("protocol") or {}).get("tag")
    if tag is not None and str(tag) != PROTOCOL_TAG:
        return False
    return True
