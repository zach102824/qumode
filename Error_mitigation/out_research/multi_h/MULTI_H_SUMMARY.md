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
| H000 | 0.003 | 0.3247 | 0.1012 | 0.1012 | 2.2885 | 0.6732 | yes |
| H000 | 0.03 | 0.7809 | 0.4101 | 0.4101 | 5.5299 | 3.1220 | yes |
| H000 | 0.1 | 0.9527 | 0.8688 | 0.8688 | 6.7282 | 7.4337 | yes |
| H004 | 0.003 | 0.2007 | 0.0763 | 0.0763 | 0.9172 | 0.1865 | yes |
| H004 | 0.03 | 0.6704 | 0.4460 | 0.4460 | 2.9691 | 1.8301 | yes |
| H004 | 0.1 | 0.8527 | 0.7612 | 0.7612 | 4.1315 | 3.4286 | yes |
| H009 | 0.003 | 0.2764 | 0.0932 | 0.0932 | 1.1754 | 0.3406 | yes |
| H009 | 0.03 | 0.7398 | 0.4423 | 0.4423 | 3.5983 | 1.2283 | yes |
| H009 | 0.1 | 0.9132 | 0.9670 | 0.9670 | 4.6140 | 2.9698 | no |

Hamiltonians that beat raw at κτ=0.003 (comprehensive+readout_realistic): **3** (H000, H004, H009).

**Success bar 1 (≥4 H beat raw at κτ=0.003): NO.**

κτ=0.03 and 0.1 are reported in the table above without filtering. Wins and losses both stand.

## Phase C — H000 10-seed mean ± std

| κτ | raw TVD | adaptive select TVD | gdr_param TVD | seed flips (select loses to raw) |
|---:|---:|---:|---:|---|
| 0.003 | 0.3169 ± 0.0057 | 0.0925 ± 0.0071 | 0.0925 ± 0.0071 | none |
| 0.03 | 0.7911 ± 0.0051 | 0.4397 ± 0.0174 | 0.4397 ± 0.0174 | none |
| 0.1 | 0.9523 ± 0.0012 | 0.8588 ± 0.0158 | 0.8588 ± 0.0158 | none |

**Success bar 3 (H000 10-seed mean beats raw on the mild cell): YES.**

## Phase D — random ECD targets

No random-target cells finished.
## Phase E — SNAP

Not run or no ≥2 near-E0 SNAP opts.

## Headline

≥4 H beat raw at κτ=0.003 under realistic device noise (comprehensive + readout_realistic): **NO** (3 H).

Do not treat mid-quality VQE losses as a recipe bug. Official adaptive defaults were not changed.

