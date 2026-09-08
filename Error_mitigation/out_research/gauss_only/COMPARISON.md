# Fully Gaussian twins vs default mix (`gdr_param` only)

H000 / instance 0, shots=8192, n_train=40, seed=`SEED_BASE=2026`, `--twin-design adaptive` (span on random, U(0.5,1) on optimized).

- **Arm A (default mix):** `--n-rank2 10`. Equal to omitted/auto for `n_train=40` (`n_train//4`). Explicit 10 reuses existing `out_research/cache/*_nr10_*` keys (`nrauto` would miss them).
- **Arm B (fully Gaussian):** `--n-rank2 0` → new `*_nr0_*` cache keys, all `t_free=0`.
- Methods: `raw`, `gdr_param`. Readout: `ideal` + `readout_realistic` (no `readout_strong`).

Δ = TVD(`gdr_param` | default mix) − TVD(`gdr_param` | fully Gaussian). Positive ⇒ fully Gaussian better. Tie if |Δ| < 0.002.

No paired cells yet. Smoke cell is next.

## Headline

| metric | value |
|---|---:|
| paired cells | 0 |
| mean Δ | — |
| median Δ | — |
| gauss better / tie / mix better | 0 / 0 / 0 |
