# ECD system-size scaling — conclusion

Noiseless Gibbs **ECD only** (no SNAP, no GDR / noise).
Success = `most_likely_bitstring == ground_bitstring`.
Each live cell is **20 Hamiltonians × 10 trials = 200**.
Cost = Gibbs `-ln⟨e^{-ηE}⟩` with `sampled_tail` η.
Hardware target: **2 transmons × 3 cavities × 8 levels = dim 2048** (n=11 exact fill).
Live ladder is **n=8…11 starting at L=4**, **200 joint SPSA**, `a = 0.2 × √(37 / n_params)`, increment L until ≥90% or soft cap **L=40** (not a hard stop at 20).
Protocol tag: `200_joint_spsa_noiseless_a_scaled`.

## Summary table (canonical, 200 joint SPSA)

| n | L* | k/200 | success | wall (s) | status |
|---|----|-------|---------|----------|--------|
| 7 | 4 | 186/200 | 0.930 | — | prior_data_pr14 |
| 8 | 4 | 196/200 | 0.980 | 85.4 | hit_threshold |
| 9 | 4 | 198/200 | 0.990 | 155.1 | hit_threshold |
| 10 | 4 | 196/200 | 0.980 | 157.1 | hit_threshold |
| 11 | 4 | 188/200 | 0.940 | 158.8 | hit_threshold |

## n=7 prior data (not re-run here)

Official L* comes from [PR #14](https://github.com/zach102824/qumode/pull/14) noiseless ECD **L4: 186/200 = 93%** (20 H × 10 trials, **200** joint SPSA, vacuum, `sampled_tail` η, production 1q+2cav / original `four_sat` fleet).

This folder’s n=7 L=3 / 70-SPSA smoke was **158/200 = 79%** and is **not** the scoreboard L*.

## Depth curves (live ladder, 200 joint SPSA, L≥4)

### n=8

| L | k/N | success | wall (s) | SPSA |
|---|-----|---------|----------|------|
| 4 | 196/200 | 0.980 | 85.4 | 200 |

### n=9

| L | k/N | success | wall (s) | SPSA |
|---|-----|---------|----------|------|
| 4 | 198/200 | 0.990 | 155.1 | 200 |

### n=10

| L | k/N | success | wall (s) | SPSA |
|---|-----|---------|----------|------|
| 4 | 196/200 | 0.980 | 157.1 | 200 |

### n=11

| L | k/N | success | wall (s) | SPSA |
|---|-----|---------|----------|------|
| 4 | 188/200 | 0.940 | 158.8 | 200 |

## Superseded: 70-SPSA L=3…20 (not canonical)

Previous PR #15 cells used **70** joint SPSA and a hard L=20 cap. They never hit 90% for n=8–10; deeper L was systematically worse. Those JSON files are kept under `results_70spsa_superseded/` and **must not** be mixed into the live scoreboard.

| n | best L≥4 (70 SPSA) | k/200 | note |
|---|--------------------|-------|------|
| 8 | 4 | 39/200 | L=4…20 done; never ≥90% |
| 9 | 4 | 51/200 | L=4…20 done; never ≥90% |
| 10 | 4 | 11/200 | L=4…20 done; never ≥90% |
| 11 | 4 | 2/200 | L=4…10 on disk; cancelled |

Why deeper L looked worse: **not** a unitarity/decoding bug. Full write-up: `DIAGNOSIS.md`. Short version: production `a=0.2` is sized for n=7 L=4 (37 params). n=8 L=4 has 70 params; 70 SPSA + unscaled `a` cannot train them. 200 SPSA with `a = 0.2√(37/n_params)` hits ≥90% at **L=4** for n=8…11. Raising L at a fixed 200-step budget still collapses (n=8 L=8 is 84/200 even with scaled `a`).

Noisy GDR-in-loop / comprehensive κ_φ τ = 0.5 κτ is **deferred** to the n=7 default-redo agent. This ladder is noiseless mode-finding; κ_φ does not enter the cost.

## Notes

Live ladder **complete** (48h extension received after this already finished; not used for an L=5…40 sweep). Protocol: 200 joint SPSA, `a=0.2*sqrt(37/n_params)`. Every n=8…11 hits ≥90% at **L=4** (n=8 196/200, n=9 198/200, n=10 196/200, n=11 188/200). The original “increase L until ≥90%” rule therefore **stops at L=4**. Diagnosis (`DIAGNOSIS.md`) showed raising L at this budget collapses success (n=8 L=8 is 84/200 even with scaled `a`; 8/200 at unscaled `a=0.2`).

