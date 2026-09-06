# Multi-H realistic comprehensive validation (PR #8 adaptive GDR)

Frozen recipe, no new mitigation kernels. Noise locked to `comprehensive` + `readout_realistic`, 8192 shots, 40 twins, adaptive design (span on random, U(0.5,1) on optimized).

Keep gate: `E_opt - E0 <= 0.5`. H000 frozen ECD from `Error_mitigation/out/optimized_params_ecd_h000_nd5.json` is the PR #8 reference circuit and is always included in optimized transfer / 10-seed even if it misses the strict gate; that exception is labeled `h000_reference`. Every other H that misses the gate is **opt-failed** and has no optimized transfer matrix.

## Phase A — which H passed / failed the E0 gate

| H | file | E0 | E_opt | deficit | new restarts | gate |
|---|---|---:|---:|---:|---:|---|
| H000 | `mixed_p_spin_p2-4_000.npz` | -7.1107 | -6.3547 | 0.7560 | 5 | h000_reference |
| H001 | `mixed_p_spin_p2-4_001.npz` | -6.0317 | -4.8195 | 1.2123 | 0 | FAIL |
| H002 | `mixed_p_spin_p2-4_002.npz` | -9.3715 | -8.0205 | 1.3510 | 0 | FAIL |
| H003 | `mixed_p_spin_p2-4_003.npz` | -6.0743 | -3.6784 | 2.3959 | 1 | FAIL |
| H004 | `mixed_p_spin_p2-4_004.npz` | -5.3462 | -5.0200 | 0.3262 | 1 | PASS |
| H005 | `mixed_p_spin_p2-4_005.npz` | -7.0504 | -4.6734 | 2.3770 | 1 | FAIL |
| H006 | `mixed_p_spin_p2-4_006.npz` | -6.6912 | -5.0891 | 1.6022 | 1 | FAIL |
| H007 | `mixed_p_spin_p2-4_007.npz` | -6.6168 | -4.6344 | 1.9823 | 1 | FAIL |
| H008 | `mixed_p_spin_p2-4_008.npz` | -6.1975 | -3.8496 | 2.3479 | 1 | FAIL |
| H009 | `mixed_p_spin_p2-4_009.npz` | -5.2544 | -4.8056 | 0.4488 | 5 | PASS |

Strict passers (deficit ≤ 0.5): **2**.
Honest opt-fail list: H000, H001, H002, H003, H005, H006, H007, H008.

## Phase B — optimized ECD transfer (adaptive vs raw)

Each cell: comprehensive + readout_realistic. Adaptive select on optimized **is** `gdr_param`.

| H | κτ | raw TVD | adaptive select TVD | gdr_param TVD | raw \|ΔE\| | select \|ΔE\| | beat raw? |
|---|---:|---:|---:|---:|---:|---:|:---:|

Hamiltonians that beat raw at κτ=0.003 (comprehensive+readout_realistic): **0** (none).

**Success bar 1 (≥4 H beat raw at κτ=0.003): NO.**

κτ=0.03 and 0.1 are reported in the table above without filtering. Wins and losses both stand.

## Phase C — H000 10-seed mean ± std

| κτ | raw TVD | adaptive select TVD | gdr_param TVD | seed flips (select loses to raw) |
|---:|---:|---:|---:|---|
| 0.003 |  |  |  | none |
| 0.03 |  |  |  | none |
| 0.1 |  |  |  | none |

**Success bar 3 (H000 10-seed mean beats raw on the mild cell): incomplete.**

## Phase D — random ECD targets

No random-target cells finished.
## Phase E — SNAP

Not run or no ≥2 near-E0 SNAP opts.

## Headline

≥4 H beat raw at κτ=0.003 under realistic device noise (comprehensive + readout_realistic): **NO** (0 H).

Do not treat mid-quality VQE losses as a recipe bug. Official adaptive defaults were not changed.

