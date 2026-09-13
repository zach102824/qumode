# ECD system-size scaling — n=12+ (this PR)

Noiseless Gibbs **ECD only** (no SNAP, no GDR / noise).
Success = `most_likely_bitstring == ground_bitstring`.
Full cell = **20 Hamiltonians × 10 trials = 200**.
A cell that is too slow may first report a **5 H × 4 trial scout** (20 trials); that is labelled.
Protocol: 200 joint SPSA, `a = 0.2 × √(37 / n_params)`, `c=0.15`, vacuum start.
Start L=4; if <90% raise L (soft cap **L=12** — deeper L at this
budget already collapsed on n=8). n=7…11 are **not** re-run (PR #15 / #14).

## Hardware growth

Each transmon = 1 logical bit. Each cavity at `FOCK_CUTOFF=8` = 3 Fock bits (MSB first).
When n exceeds capacity, append one transmon **and** one cavity.
Idle modes (0 assigned bits) stay vacuum and are omitted from the simulated tensor.

| n | hardware plan | simulated | dim | pairs | idle |
|---|---------------|-----------|-----|-------|------|
| 7 | 2T×3C parent (special case) | 1T×2C | 128 | 2 | T1, C2 |
| 8 | 2T×3C | 2T×2C | 256 | 4 | C2 |
| 9–11 | 2T×3C | 2T×3C | 2048 | 6 | C2 bits 1…3 |
| 12 | 3T×4C | 3T×3C | 4096 | 9 | C3 |
| 13–15 | 3T×4C | 3T×4C | 32768 | 12 | — |
| 16 | 4T×5C | 4T×4C | 65536 | 16 | C4 |
| 17–19 | 4T×5C | 4T×5C | 524288 | 20 | — |

`n_params = n_T + 2 n_C + 4 L n_T n_C`.

## Scoreboard (this extension)

| n | L* | k/N | success | n_params | a | dim | pairs | mean p(GS) | wall (s) | status |
|---|----|-----|---------|----------|---|-----|-------|------------|----------|--------|
| 12 | — | 145/200 | 0.725 | 153 | 0.0984 | 4096 | 9 | 0.0116 | 489.0 | capped_L12_below_threshold |
| 13 | — | — | — | 203 | 0.0854 | — | — | — | — | not_started |
| 14 | — | — | — | 203 | 0.0854 | — | — | — | — | not_started |
| 15 | — | — | — | 203 | 0.0854 | — | — | — | — | not_started |

## Depth curves

### n=12

| L | k/N | success | n_params | a | dim | pairs | mean p(GS) | wall (s) | notes |
|---|-----|---------|----------|---|-----|-------|------------|----------|-------|
| 4 | 145/200 | 0.725 | 153 | 0.0984 | 4096 | 9 | 0.0116 | 489.0 |  |
| 5 | 100/200 | 0.500 | 189 | 0.0885 | 4096 | 9 | 0.0044 | 340.8 |  |
| 6 | 6/20 | 0.300 | 225 | 0.0811 | 4096 | 9 | 0.0021 | 41.6 | scout |
| 7 | 4/20 | 0.200 | 261 | 0.0753 | 4096 | 9 | 0.0015 | 47.2 | scout |
| 8 | 6/20 | 0.300 | 297 | 0.0706 | 4096 | 9 | 0.0012 | 52.8 | scout |
| 9 | 1/20 | 0.050 | 333 | 0.0667 | 4096 | 9 | 0.0006 | 59.1 | scout |
| 10 | 1/20 | 0.050 | 369 | 0.0633 | 4096 | 9 | 0.0004 | 64.4 | scout |
| 11 | 0/20 | 0.000 | 405 | 0.0605 | 4096 | 9 | 0.0005 | 69.7 | scout |
| 12 | 1/20 | 0.050 | 441 | 0.0579 | 4096 | 9 | 0.0003 | 75.5 | scout |

### n=13

Not started.

### n=14

Not started.

### n=15

Not started.

## n=7…11 (not re-run; PR #15 / #14)

| n | L* | k/200 | success | n_params | a | dim | pairs | mean p(GS) | wall (s) |
|---|----|-------|---------|----------|---|-----|-------|------------|----------|
| 7 | 4 | 186/200 | 0.930 | 37 | 0.2000 | 128 | 2 | — | — |
| 8 | 4 | 196/200 | 0.980 | 70 | 0.1454 | 256 | 4 | 0.1965 | 85.4 |
| 9 | 4 | 198/200 | 0.990 | 104 | 0.1193 | 2048 | 6 | 0.0380 | 155.1 |
| 10 | 4 | 196/200 | 0.980 | 104 | 0.1193 | 2048 | 6 | 0.0355 | 157.1 |
| 11 | 4 | 188/200 | 0.940 | 104 | 0.1193 | 2048 | 6 | 0.0610 | 158.8 |

## Notes

n=12 finished: best is L=4 at 145/200 (72.5%), never ≥90%. L=5 100/200; L=6–12 are 5H×4 scouts (6,4,6,1,1,0,1 / 20) documenting the same 200-SPSA depth collapse seen on n=8. Soft cap L=12. n=13…15 next.

