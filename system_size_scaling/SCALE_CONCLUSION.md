# ECD system-size scaling — conclusion

Noiseless Gibbs **ECD only** (no SNAP, no GDR / noise).
Success = `most_likely_bitstring == ground_bitstring`.
Each cell is **20 Hamiltonians × 10 trials = 200**.
Cost = Gibbs `-ln⟨e^{-ηE}⟩` with `sampled_tail` η.
Hardware target: **2 transmons × 3 cavities × 8 levels = dim 2048** (n=11 exact fill).

## Summary table

| n | L* | k/200 | success | wall (s) | status |
|---|----|-------|---------|----------|--------|
| 7 | — | 168/200 | 0.840 | 2848.6 | capped_L20_below_threshold |
| 8 | — | — | — | — | not_started |
| 9 | — | — | — | — | not_started |
| 10 | — | — | — | — | not_started |
| 11 | — | — | — | — | not_started |

## Depth curves

### n=7

| L | k/N | success | wall (s) |
|---|-----|---------|----------|
| 3 | 158/200 | 0.790 | 1689.2 |
| 4 | 168/200 | 0.840 | 27.2 |
| 5 | 156/200 | 0.780 | 31.3 |
| 6 | 100/200 | 0.500 | 36.6 |
| 7 | 76/200 | 0.380 | 42.0 |
| 8 | 53/200 | 0.265 | 47.3 |
| 9 | 28/200 | 0.140 | 52.2 |
| 10 | 16/200 | 0.080 | 57.3 |
| 11 | 10/200 | 0.050 | 63.0 |
| 12 | 6/200 | 0.030 | 67.9 |
| 13 | 7/200 | 0.035 | 72.7 |
| 14 | 4/200 | 0.020 | 77.8 |
| 15 | 3/200 | 0.015 | 83.4 |
| 16 | 5/200 | 0.025 | 88.0 |
| 17 | 2/200 | 0.010 | 93.1 |
| 18 | 2/200 | 0.010 | 99.1 |
| 19 | 1/200 | 0.005 | 107.0 |
| 20 | 3/200 | 0.015 | 113.6 |

### n=8

Not started.

### n=9

Not started.

### n=10

Not started.

### n=11

Not started.

## Notes

n=7 finished the L=3…20 protocol and never hit 90%. Best cell is L=4 at 168/200 (84%). With a fixed 70-step joint SPSA budget, deeper random ECD ansatze collapse (L≥10 is ≤8%). L* is undefined (cap).

