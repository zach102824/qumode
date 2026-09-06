# Multi-H realistic comprehensive validation (PR #8 adaptive GDR)

Frozen recipe. **No new mitigation kernels. No `src/` edits.** Noise locked to
`comprehensive` + `readout_realistic`, 8192 shots, `n_train=40`, official
adaptive twins (span on random, \(U(0.5,1)\) on optimized). Everything lives
under `Error_mitigation/out_research/multi_h/`. `out/` and `out_smoke/` were
not overwritten.

Keep gate: `E_opt - E0 <= 0.5`. Failed opts get **no** optimized transfer
matrix. H000 is the PR #8 reference circuit: an extra L-BFGS-B start improved
it from deficit 0.881 → **0.756**, still above the gate. It is included in
Phases B/C and labeled `h000_reference`.

## Headline for Zach

**≥4 H beat raw at κτ=0.003: NO.** Only **2** Hamiltonians cleared the 0.5
near-\(E_0\) gate (H004, H009). Adding the H000 reference gives **3** optimized
transfer Hamiltonians, not 4.

On those 3, adaptive still beats raw at mild realistic-device noise, on the
H000 10-seed mean, and on 33/36 random cells. The recipe was not changed.

| success bar | result |
|---|---|
| ≥4 H beat raw at κτ=0.003 comprehensive+readout_realistic | **NO** (3 H: H000, H004, H009) |
| Report κτ=0.03 and 0.1 honestly | **yes** — all 3 win at 0.03; H009 **loses** at 0.1 |
| H000 10-seed mean beats raw on the mild cell | **YES** (0.0925 ± 0.0071 vs 0.3169 ± 0.0057; no seed flips) |
| Honest opt-fail list | **yes** (below) |

## Phase A — which H passed / failed the E0 gate

ECD \(N_d=5\), L-BFGS-B maxiter 200. Numbers from `phase_a/opt_ledger.json`.

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

Strict passers (deficit ≤ 0.5): **H004, H009**.

**Honest opt-fail list (no optimized transfer):** H001, H002, H003, H005,
H006, H007, H008. H000 missed the 0.5 gate (deficit 0.756) but is the PR #8
reference and **was** transferred.

## Phase B — optimized ECD transfer (adaptive vs raw)

Each cell: comprehensive + readout_realistic, official optimized recipe
(`gdr_select` **is** `gdr_param`). Tables match `phase_b/results.json`.

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

- κτ=0.003: **3/3 win** (H000, H004, H009).
- κτ=0.03: **3/3 win**.
- κτ=0.1: H000 and H004 win on TVD; **H009 loses** (0.967 vs 0.913). H000’s
  κτ=0.1 TVD win comes with a **worse** \|ΔE\| (7.43 vs 6.73). Both facts stand.

**Success bar 1 (≥4 H beat raw at κτ=0.003): NO.**

## Phase C — H000 10-seed mean ± std

Same 3 cells, 10 independent shot seeds. Physics cached; only shots+fit change.
Matches `phase_c/results.json`.

| κτ | raw TVD | adaptive select TVD | gdr_param TVD | seed flips (select loses to raw) |
|---:|---:|---:|---:|---|
| 0.003 | 0.3169 ± 0.0057 | 0.0925 ± 0.0071 | 0.0925 ± 0.0071 | none |
| 0.03 | 0.7911 ± 0.0051 | 0.4397 ± 0.0174 | 0.4397 ± 0.0174 | none |
| 0.1 | 0.9523 ± 0.0012 | 0.8588 ± 0.0158 | 0.8588 ± 0.0158 | none |

**Success bar 3 (H000 10-seed mean beats raw on the mild cell): YES.** No seed
flips at any of the three κτ.

## Phase D — random ECD targets

8 random ECD circuits on H000 + 4 on H004 (a strict passer). Official
span/adaptive-random path. Matches `phase_d/results.json`.

| H | random_id | κτ | raw TVD | adaptive select TVD | gdr_param TVD | beat raw? |
|---|---:|---:|---:|---:|---:|:---:|
| H000 | 0 | 0.003 | 0.1016 | 0.0776 | 0.0776 | yes |
| H000 | 0 | 0.03 | 0.2237 | 0.1530 | 0.1731 | yes |
| H000 | 0 | 0.1 | 0.3980 | 0.3027 | 0.3410 | yes |
| H000 | 1 | 0.003 | 0.0858 | 0.0772 | 0.0791 | yes |
| H000 | 1 | 0.03 | 0.2126 | 0.1681 | 0.1824 | yes |
| H000 | 1 | 0.1 | 0.3827 | 0.2697 | 0.2772 | yes |
| H000 | 2 | 0.003 | 0.1170 | 0.0866 | 0.0866 | yes |
| H000 | 2 | 0.03 | 0.2501 | 0.1415 | 0.1458 | yes |
| H000 | 2 | 0.1 | 0.4318 | 0.3022 | 0.2990 | yes |
| H000 | 3 | 0.003 | 0.1154 | 0.0813 | 0.0813 | yes |
| H000 | 3 | 0.03 | 0.2745 | 0.2360 | 0.2460 | yes |
| H000 | 3 | 0.1 | 0.4589 | 0.3967 | 0.4042 | yes |
| H000 | 4 | 0.003 | 0.0828 | 0.0683 | 0.0683 | yes |
| H000 | 4 | 0.03 | 0.1874 | 0.1403 | 0.1331 | yes |
| H000 | 4 | 0.1 | 0.3127 | 0.2680 | 0.2672 | yes |
| H000 | 5 | 0.003 | 0.1253 | 0.0880 | 0.0867 | yes |
| H000 | 5 | 0.03 | 0.2652 | 0.2190 | 0.2254 | yes |
| H000 | 5 | 0.1 | 0.4152 | 0.3674 | 0.3726 | yes |
| H000 | 6 | 0.003 | 0.1344 | 0.0869 | 0.0834 | yes |
| H000 | 6 | 0.03 | 0.3532 | 0.2773 | 0.2676 | yes |
| H000 | 6 | 0.1 | 0.4363 | 0.4784 | 0.5321 | no |
| H000 | 7 | 0.003 | 0.0996 | 0.0729 | 0.0729 | yes |
| H000 | 7 | 0.03 | 0.2099 | 0.1553 | 0.1553 | yes |
| H000 | 7 | 0.1 | 0.3522 | 0.3218 | 0.3218 | yes |
| H004 | 0 | 0.003 | 0.0838 | 0.0678 | 0.0681 | yes |
| H004 | 0 | 0.03 | 0.1903 | 0.1280 | 0.1299 | yes |
| H004 | 0 | 0.1 | 0.3608 | 0.2606 | 0.2628 | yes |
| H004 | 1 | 0.003 | 0.0991 | 0.1093 | 0.1093 | no |
| H004 | 1 | 0.03 | 0.2318 | 0.2136 | 0.2301 | yes |
| H004 | 1 | 0.1 | 0.3306 | 0.3037 | 0.3871 | yes |
| H004 | 2 | 0.003 | 0.1137 | 0.0809 | 0.0829 | yes |
| H004 | 2 | 0.03 | 0.2835 | 0.2632 | 0.2822 | yes |
| H004 | 2 | 0.1 | 0.4343 | 0.4383 | 0.4793 | no |
| H004 | 3 | 0.003 | 0.1311 | 0.1027 | 0.0967 | yes |
| H004 | 3 | 0.03 | 0.3097 | 0.2384 | 0.2344 | yes |
| H004 | 3 | 0.1 | 0.5200 | 0.3981 | 0.4040 | yes |

Random cells where adaptive select beats raw: **33 / 36**.

Honest losses:

| cell | raw | select |
|---|---:|---:|
| H000 random 6, κτ=0.1 | 0.4363 | 0.4784 |
| H004 random 1, κτ=0.003 | 0.0991 | 0.1093 |
| H004 random 2, κτ=0.1 | 0.4343 | 0.4383 |

## Phase E — SNAP

**Not run** (stopped on purpose). A–D already answer the success bars.

Partial SNAP \(N_d=2\) opts that finished before the stop both **failed** the
0.5 gate (H000 deficit 1.93, H004 deficit 1.41). H009 SNAP was killed mid-restart.
No SNAP transfer matrix. Official ECD/SNAP adaptive defaults unchanged.

## Notes

- Prefer A–C–F over expanding to 20 H: done. Phase D finished as well.
- Mid-quality VQE (H001/H002-class deficits ≥1.2) were not transferred as
  “optimized.” That is the Phase 8 caveat, not a recipe bug.
- Official adaptive defaults were not changed.
