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
| 12 | — | — | — | 153 | 0.0984 | — | — | — | — | not_started |
| 13 | — | — | — | 203 | 0.0854 | — | — | — | — | not_started |
| 14 | — | — | — | 203 | 0.0854 | — | — | — | — | not_started |
| 15 | — | — | — | 203 | 0.0854 | — | — | — | — | not_started |

## Depth curves

### n=12

Not started.

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

Higher-n extension scaffolded. Embedding generalized; n=12…15 Hamiltonians and ladder not started yet.

