# ECD system-size scaling — conclusion

Noiseless Gibbs **ECD only** (no SNAP, no GDR / noise).
Success = `most_likely_bitstring == ground_bitstring`.
Each live cell is **20 Hamiltonians × 10 trials = 200**.
Cost = Gibbs `-ln⟨e^{-ηE}⟩` with `sampled_tail` η.
Hardware: n=7…11 stay on the locked 2T×3C map (n=11 exact fill, dim 2048).
For n>11, append one transmon+cavity when the register is full (3T+4C capacity 15, dim 32768 when C3 is live). Idle 0-bit modes are omitted.
PR #15 live ladder **n=8…11** finished at L*=4. This PR continues **n=12…15** at L=4, **200 joint SPSA**, `a = 0.2 × √(37 / n_params)`, soft cap **L=12** (do not blindly go to L=40).
Protocol tag: `200_joint_spsa_noiseless_a_scaled`. Full n=12+ table: `HIGHER_N.md`.

## Summary table (canonical, 200 joint SPSA)

| n | L* | k/N | success | wall (s) | status |
|---|----|-----|---------|----------|--------|
| 7 | 4 | 186/200 | 0.930 | — | prior_data_pr14 |
| 8 | 4 | 196/200 | 0.980 | 85.4 | hit_threshold |
| 9 | 4 | 198/200 | 0.990 | 155.1 | hit_threshold |
| 10 | 4 | 196/200 | 0.980 | 157.1 | hit_threshold |
| 11 | 4 | 188/200 | 0.940 | 158.8 | hit_threshold |
| 12 | — | 145/200 | 0.725 | 829.8 | in_progress |
| 13 | — | — | — | — | not_started |
| 14 | — | — | — | — | not_started |
| 15 | — | — | — | — | not_started |

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

### n=12

| L | k/N | success | wall (s) | SPSA |
|---|-----|---------|----------|------|
| 4 | 145/200 | 0.725 | 489.0 | 200 |
| 5 | 100/200 | 0.500 | 340.8 | 200 |

### n=13

Not started under the 200-SPSA protocol.

### n=14

Not started under the 200-SPSA protocol.

### n=15

Not started under the 200-SPSA protocol.

## Superseded: 70-SPSA L=3…20 (not canonical)

Previous PR #15 cells used **70** joint SPSA and a hard L=20 cap. They never hit 90% for n=8–10; deeper L was systematically worse. Those JSON files are kept under `results_70spsa_superseded/` and **must not** be mixed into the live scoreboard.

| n | best L≥4 (70 SPSA) | k/200 | note |
|---|--------------------|-------|------|
| 8 | 4 | 39/200 | L=4…20 done; never ≥90% |
| 9 | 4 | 51/200 | L=4…20 done; never ≥90% |
| 10 | 4 | 11/200 | L=4…20 done; never ≥90% |
| 11 | 4 | 2/200 | L=4…10 on disk; cancelled |

Why deeper L looked worse: not a unitarity/decoding bug (n=7 ECD matches production QuTiP; gates stay norm-preserving at L=20/40). At fixed 70 SPSA, extra layers add parameters that the budget cannot train — both ⟨H⟩ and p_ground degrade. n=7 L=4 was 168/200 at 70 SPSA vs **186/200 at 200 SPSA** in PR #14. This restart tests whether 200 joint SPSA plus uncapped L recovers ≥90% for n=8…11.

Noisy GDR-in-loop / comprehensive κ_φ τ = 0.5 κτ is **deferred** to the n=7 default-redo agent. This ladder is noiseless mode-finding; κ_φ does not enter the cost.

## Notes

n=12 L=4 145/200 (72.5%); L=5 100/200 (50%) collapsed. Remaining L=6…12 will be 5H×4 scouts, not full 200.

