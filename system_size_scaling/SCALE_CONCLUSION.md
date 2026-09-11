# ECD system-size scaling — conclusion

Noiseless Gibbs **ECD only** (no SNAP, no GDR / noise).
Success = `most_likely_bitstring == ground_bitstring`.
Each cell is **20 Hamiltonians × 10 trials = 200**.
Cost = Gibbs `-ln⟨e^{-ηE}⟩` with `sampled_tail` η.
Hardware target: **2 transmons × 3 cavities × 8 levels = dim 2048** (n=11 exact fill).

## Summary table

| n | L* | k/200 | success | wall (s) | status |
|---|----|-------|---------|----------|--------|
| 7 | — | 158/200 | 0.790 | 1689.2 | in_progress |
| 8 | — | — | — | — | not_started |
| 9 | — | — | — | — | not_started |
| 10 | — | — | — | — | not_started |
| 11 | — | — | — | — | not_started |

## Depth curves

### n=7

| L | k/N | success | wall (s) |
|---|-----|---------|----------|
| 3 | 158/200 | 0.790 | 1689.2 |

### n=8

Not started.

### n=9

Not started.

### n=10

Not started.

### n=11

Not started.

