# ECD system-size scaling — conclusion

Noiseless Gibbs **ECD only** (no SNAP, no GDR / noise).
Success = `most_likely_bitstring == ground_bitstring`.
Each live cell is **20 Hamiltonians × 10 trials = 200**.
Cost = Gibbs `-ln⟨e^{-ηE}⟩` with `sampled_tail` η.
Hardware target: **2 transmons × 3 cavities × 8 levels = dim 2048** (n=11 exact fill).
Live ladder is **n=8…11 starting at L=4** (increment until ≥90% or L=20).

## Summary table

| n | L* | k/200 | success | wall (s) | status |
|---|----|-------|---------|----------|--------|
| 7 | 4 | 186/200 | 0.930 | — | prior_data_pr14 |
| 8 | — | 39/200 | 0.195 | 2390.6 | capped_L20_below_threshold |
| 9 | — | 51/200 | 0.255 | 3244.9 | in_progress |
| 10 | — | — | — | — | not_started |
| 11 | — | — | — | — | not_started |

## n=7 prior data (not re-run here)

Official L* comes from [PR #14](https://github.com/zach102824/qumode/pull/14) noiseless ECD **L4: 186/200 = 93%** (20 H × 10 trials, **200** joint SPSA, vacuum, `sampled_tail` η, production 1q+2cav / original `four_sat` fleet).

This folder’s n=7 L=3 / 70-SPSA smoke was **158/200 = 79%** and is **not** the scoreboard L*. n=7 L=4…20 cells in `results/` are leftover from an earlier mis-scoped sweep and are not used in the table.

## Depth curves (live ladder, L≥4)

### n=8

| L | k/N | success | wall (s) |
|---|-----|---------|----------|
| 4 | 39/200 | 0.195 | 49.1 |
| 5 | 12/200 | 0.060 | 59.8 |
| 6 | 3/200 | 0.015 | 70.5 |
| 7 | 3/200 | 0.015 | 81.0 |
| 8 | 1/200 | 0.005 | 96.4 |
| 9 | 1/200 | 0.005 | 109.2 |
| 10 | 2/200 | 0.010 | 116.1 |
| 11 | 2/200 | 0.010 | 125.8 |
| 12 | 3/200 | 0.015 | 136.8 |
| 13 | 2/200 | 0.010 | 151.6 |
| 14 | 0/200 | 0.000 | 166.1 |
| 15 | 1/200 | 0.005 | 176.8 |
| 16 | 0/200 | 0.000 | 188.8 |
| 17 | 1/200 | 0.005 | 200.5 |
| 18 | 1/200 | 0.005 | 208.6 |
| 19 | 0/200 | 0.000 | 219.5 |
| 20 | 1/200 | 0.005 | 234.0 |

### n=9

| L | k/N | success | wall (s) |
|---|-----|---------|----------|
| 4 | 51/200 | 0.255 | 82.4 |
| 5 | 24/200 | 0.120 | 100.9 |
| 6 | 8/200 | 0.040 | 123.2 |
| 7 | 2/200 | 0.010 | 143.5 |
| 8 | 5/200 | 0.025 | 163.9 |
| 9 | 1/200 | 0.005 | 182.9 |
| 10 | 1/200 | 0.005 | 201.5 |
| 11 | 0/200 | 0.000 | 218.5 |
| 12 | 3/200 | 0.015 | 240.5 |
| 13 | 0/200 | 0.000 | 247.6 |
| 14 | 2/200 | 0.010 | 268.4 |
| 15 | 0/200 | 0.000 | 288.8 |
| 16 | 0/200 | 0.000 | 307.7 |
| 17 | 0/200 | 0.000 | 326.1 |
| 18 | 1/200 | 0.005 | 349.1 |

### n=10

Not started.

### n=11

Not started.

## Notes

Protocol retargeted: n=7 is PR #14 prior data (L*=4, 186/200). Live ladder is n=8…11 starting at L=4. n=8 L=4…20 is complete and never hit 90% (best L=4 at 39/200). n=9 L≥4 through L=18 on disk (best L=4 at 51/200); L=19–20 finishing. n=10–11 not started.

