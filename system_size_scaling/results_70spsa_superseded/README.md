# Superseded 70-SPSA cells (not canonical)

These JSON files are the previous PR #15 noiseless ECD ladder run with
**70 joint SPSA** and a hard cap at L=20. They are kept for audit only.

Do not resume them. Do not copy them into `../results/`. The live protocol
is **200 joint SPSA**, start L=4, increment until ≥90% or soft cap L=40.

| n | best L≥4 | k/200 | status |
|---|----------|-------|--------|
| 8 | 4 | 39/200 | L=4…20 done; never ≥90% |
| 9 | 4 | 51/200 | L=4…20 done; never ≥90% |
| 10 | 4 | 11/200 | L=4…20 done; never ≥90% |
| 11 | 4 | 2/200 | L=4…10 on disk; agent cancelled |

Deeper L was worse because a fixed 70-step SPSA budget cannot train the
extra ECD parameters: both ⟨H⟩ and p_ground fall as L grows. That is
under-optimization, not a broken circuit (n=7 ECD still matches production
QuTiP; truncated displacements stay unitary on the 8-level Fock space).
