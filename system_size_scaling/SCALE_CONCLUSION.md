# ECD system-size scaling — conclusion

Noiseless Gibbs **ECD only** (no SNAP, no GDR / noise).
Success = `most_likely_bitstring == ground_bitstring`.
Each live cell is **20 Hamiltonians × 10 trials = 200**.
Cost = Gibbs `-ln⟨e^{-ηE}⟩` with `sampled_tail` η.
Hardware target: **2 transmons × 3 cavities × 8 levels = dim 2048** (n=11 exact fill).
Live ladder is **n=8…11 starting at L=4**, **200 joint SPSA**, increment L until ≥90% or soft cap **L=40** (not a hard stop at 20).
Protocol tag: `200_joint_spsa_noiseless`.

## Summary table (canonical, 200 joint SPSA)

| n | L* | k/200 | success | wall (s) | status |
|---|----|-------|---------|----------|--------|
| 7 | 4 | 186/200 | 0.930 | — | prior_data_pr14 |
| 8 | — | — | — | — | not_started |
| 9 | — | — | — | — | not_started |
| 10 | — | — | — | — | not_started |
| 11 | — | — | — | — | not_started |

## n=7 prior data (not re-run here)

Official L* comes from [PR #14](https://github.com/zach102824/qumode/pull/14) noiseless ECD **L4: 186/200 = 93%** (20 H × 10 trials, **200** joint SPSA, vacuum, `sampled_tail` η, production 1q+2cav / original `four_sat` fleet).

This folder’s n=7 L=3 / 70-SPSA smoke was **158/200 = 79%** and is **not** the scoreboard L*.

## Depth curves (live ladder, 200 joint SPSA, L≥4)

### n=8

Not started under the 200-SPSA protocol.

### n=9

Not started under the 200-SPSA protocol.

### n=10

Not started under the 200-SPSA protocol.

### n=11

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

Diagnosis in progress (see DIAGNOSIS.md). Live 200-SPSA L-sweep is paused until n=8 L=4/8/12 controlled study finishes.

