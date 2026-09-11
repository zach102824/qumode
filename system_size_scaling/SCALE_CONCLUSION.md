# ECD system-size scaling — conclusion

Noiseless Gibbs **ECD only** (no SNAP, no GDR / noise).
Success = `most_likely_bitstring == ground_bitstring`.
Each cell is **20 Hamiltonians × 10 trials = 200**.
Cost = Gibbs `-ln⟨e^{-ηE}⟩` with `sampled_tail` η.
Hardware target: **2 transmons × 3 cavities × 8 levels = dim 2048** (n=11 exact fill).

## Summary table

| n | L* | k/200 | success | wall (s) | status |
|---|----|-------|---------|----------|--------|
| 7 | — | — | — | — | Hamiltonians ready; sweep not started |
| 8 | — | — | — | — | Hamiltonians ready; sweep not started |
| 9 | — | — | — | — | Hamiltonians ready; sweep not started |
| 10 | — | — | — | — | Hamiltonians ready; sweep not started |
| 11 | — | — | — | — | Hamiltonians ready; sweep not started |

## Notes

20 unique-planted 4-SAT instances per n are in `Hamiltonians/n{N}/` at clause
density 18/7 (targets 18, 21, 23, 26, 28). n=7 ECD statevectors match
production `qumode_vqe.circuit.prepare_state` to numerical precision.
The L=3…20 bitstring ladder has not been run yet.
