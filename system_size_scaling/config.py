"""Protocol constants for the isolated ECD system-size ladder."""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HAM_ROOT = ROOT / "Hamiltonians"
RESULTS_ROOT = ROOT / "results"
SUPERSEDED_70_ROOT = ROOT / "results_70spsa_superseded"
DIAGNOSIS_ROOT = ROOT / "results" / "diagnosis"

# Production n=7 4-SAT used ~18 clauses / 7 variables.
CLAUSE_DENSITY = 18 / 7
CLAUSE_WIDTH = 4
N_HAMILTONIANS = 20
N_TRIALS = 10
SEARCH_TRIALS = 4000

# Depth sweep: start L=4, increment until success ≥ 90%.
# PR #15 n=8…11 used a soft cap of L=40 (all hit at L=4).
# This extension (n≥12) uses a soft cap of L=12 — deeper L at 200 SPSA
# already collapsed on n=8 (L=8 is 84/200 even with scaled a).
L_START = 4
L_MAX = 40
L_MAX_HIGHER = 12
SUCCESS_THRESHOLD = 0.90

# Canonical optimizer budget — PR #14 / new default (not the superseded 70).
OUTER_ITER = 200
PROTOCOL_TAG = "200_joint_spsa_noiseless_a_scaled"

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

# Hardware: start at 2T×3C (n=8…11, dim 2048). When the register is full,
# append one transmon + one cavity (FOCK_CUTOFF stays 8 = 3 logical bits).
# n=7 stays the production 1T×2C special case.
N_TRANSMONS = 2
N_CAVITIES = 3
FOCK_CUTOFF = 8
FOCK_BITS = 3  # log2(FOCK_CUTOFF); Fock binary, MSB first
HARDWARE_DIM = 2 * 2 * 8 * 8 * 8  # 2048 = 2T×3C
# Encoding is implemented through 4T+5C (capacity 19). n=16+ is optional.
MAX_LOGICAL_BITS = 19

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

# PR #15 finished n=8…11 at L*=4. Do not rerun those cells.
FINISHED_LADDER_NS = (8, 9, 10, 11)
HIGHER_NS = (12, 13, 14, 15)
# Default run_ladder range (this PR).
LADDER_NS = HIGHER_NS
SCOREBOARD_NS = FINISHED_LADDER_NS + HIGHER_NS
ALL_NS = (7,) + SCOREBOARD_NS

# Frozen 70-SPSA L=4…20 scoreboard. Not canonical. Deeper L collapsed because
# a fixed 70-step SPSA budget under-trains the extra parameters (p_ground and
# ⟨H⟩ both degrade). See results_70spsa_superseded/.
SUPERSEDED_70_SPSA = {
    8: {"best_L": 4, "k": 39, "n_total": 200, "note": "L=4…20 done; never ≥90%"},
    9: {"best_L": 4, "k": 51, "n_total": 200, "note": "L=4…20 done; never ≥90%"},
    10: {"best_L": 4, "k": 11, "n_total": 200, "note": "L=4…20 done; never ≥90%"},
    11: {"best_L": 4, "k": 2, "n_total": 200, "note": "L=4…10 on disk; cancelled"},
}


def plan_capacity(n_transmons: int, n_cavities: int, fock_bits: int = FOCK_BITS) -> int:
    return int(n_transmons) + int(n_cavities) * int(fock_bits)


def plan_hardware_dim(n_transmons: int, n_cavities: int, fock_cutoff: int = FOCK_CUTOFF) -> int:
    return (2 ** int(n_transmons)) * (int(fock_cutoff) ** int(n_cavities))


def hardware_plan(n: int) -> tuple[int, int]:
    """Return (n_transmons, n_cavities) in the hardware plan for logical n.

    n=7 is the production 1T×2C special case. n≥8 starts at 2T+3C
    (capacity 11) and appends one T+C whenever n exceeds capacity.
    """
    n = int(n)
    if n < 7:
        raise ValueError(f"n={n} is below the n=7 production floor")
    if n == 7:
        return 1, 2
    n_t, n_c = int(N_TRANSMONS), int(N_CAVITIES)
    while n > plan_capacity(n_t, n_c):
        n_t += 1
        n_c += 1
    return n_t, n_c


def soft_cap_for_n(n: int) -> int:
    """L soft-cap: 40 for the finished n=8…11 ladder, 12 for n≥12."""
    return int(L_MAX_HIGHER) if int(n) >= 12 else int(L_MAX)


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


# Production n=7 L=4 joint dimension: 5 prep + 4*4*2 ECD = 37.
N7_L4_NPARAMS = 37


def live_register(n: int) -> tuple[int, int]:
    """Simulated (n_transmons, n_cavities) after dropping idle 0-bit modes."""
    n = int(n)
    if n == 7:
        return 1, 2
    n_t, n_c = hardware_plan(n)
    leftover = n - n_t
    n_c_live = 0
    for _ in range(n_c):
        take = min(FOCK_BITS, leftover)
        leftover -= take
        if take > 0:
            n_c_live += 1
    return n_t, n_c_live


def n_joint_params(n_prep: int, ndepth: int, n_pairs: int) -> int:
    return int(n_prep) + 4 * int(ndepth) * int(n_pairs)


def n_params_for(n: int, ndepth: int) -> int:
    """n_params = n_T + 2 n_C + 4 L n_T n_C on the simulated (live) register."""
    n_t, n_c = live_register(n)
    return n_joint_params(n_t + 2 * n_c, ndepth, n_t * n_c)


def spsa_a_scaled(n_params: int, a0: float = SPSA_A, n_ref: int = N7_L4_NPARAMS) -> float:
    """Keep SPSA RMS step similar to the working n=7 L=4 budget."""
    return float(a0) * math.sqrt(float(n_ref) / float(max(int(n_params), 1)))


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
