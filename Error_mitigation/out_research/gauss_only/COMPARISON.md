# Fully Gaussian twins vs default mix (`gdr_param` only)

H000 / instance 0, shots=8192, n_train=40, seed=`SEED_BASE=2026`, `--twin-design adaptive` (span on random, U(0.5,1) on optimized).

- **Arm A (default mix):** `--n-rank2 10`. Equal to omitted/auto for `n_train=40` (`n_train//4`). Explicit 10 reuses existing `out_research/cache/*_nr10_*` keys (`nrauto` would miss them).
- **Arm B (fully Gaussian):** `--n-rank2 0` → new `*_nr0_*` cache keys, all `t_free=0`.
- Methods: `raw`, `gdr_param`. Readout: `ideal` + `readout_realistic` (no `readout_strong`).

Δ = TVD(`gdr_param` | default mix) − TVD(`gdr_param` | fully Gaussian). Positive ⇒ fully Gaussian better. Tie if |Δ| < 0.002.

**Recommendation:** pending — 1 paired cell(s); need the ECD priority matrix (≥20 cells) before a keep-vs-switch call.

## Headline

| metric | value |
|---|---:|
| paired cells | 1 |
| mean Δ | 0.0004 |
| median Δ | 0.0004 |
| gauss better / tie / mix better | 0 / 1 / 0 |

## By slice

| slice | n | mean Δ | median Δ | gauss better | tie | mix better |
|---|---:|---:|---:|---:|---:|---:|
| ansatz=ecd | 1 | 0.0004 | 0.0004 | 0 | 1 | 0 |
| params=random | 1 | 0.0004 | 0.0004 | 0 | 1 | 0 |
| kappa_tau=0.003 | 1 | 0.0004 | 0.0004 | 0 | 1 | 0 |
| family=loss | 1 | 0.0004 | 0.0004 | 0 | 1 | 0 |

## Per-cell TVD

| ansatz | params | family | κτ | readout | raw mix | raw gauss | gdr mix | gdr gauss | Δ | verdict |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| ecd | random | loss | 0.003 | ideal | 0.0434 | 0.0434 | 0.0448 | 0.0444 | 0.0004 | tie |

## Notes

- Physical histograms for arm A reuse `Error_mitigation/out_research/cache/` `nr10` adaptive keys (span on random, default on optimized).
- Arm B writes new `nr0` keys into that same shared cache.
- Production adaptive defaults are **not** changed in this PR.
