# GDR improvement lab notebook

Branch: `cursor/gdr-improve-30h-v2-4f00`  
Start: from PR #6 `cursor/error-mitigation-smoke-full-5302` @ `b7b7a8c`  
Hard rules: no `src/` edits; do not overwrite `out/` or `out_smoke/`; one heavy sim at a time.

## Phase 0 (2026-09-04)

Read `Error_mitigation/{mitigation,twins,noise_models,metrics,run_mitigation_experiment}.py` and `out/summary.txt`.

Tests (this VM, after installing qutip/scipy/matplotlib/pytest):

```
python3 -m pytest tests/test_error_mitigation.py -q   # 10 passed, then 16 after research tests
python3 -m pytest -q                                 # 95 passed (slow excluded)
```

Did **not** re-run the full 108-cell baseline.

### Baseline scoreboard (PR #6, shots=8192, n_train=40, 108 cells)

| item | value |
|------|------:|
| product-state TVD max | 1.13e-15 |
| wall | ~64 min (`FULL_START` 09:52 → `FULL_EXIT` 10:56) |
| ECD optimum | E ≈ −6.230 (maxiter=200, 3 restarts) |
| SNAP optimum | E ≈ −5.178 |
| `gdr_param` beats raw | 91 / 108 |
| `gdr_param` worse than raw | 15 / 108 (12 ECD random κτ≥0.03; 3 SNAP random mild ideal) |
| `readout_only` worse than raw | 4 / 108 (ECD comprehensive κτ=0.1, both param sets, realistic+strong) |

Headline cells the plan called out:

| cell | raw | gdr_param | oracle | zne | note |
|------|----:|----------:|-------:|----:|------|
| ECD opt loss κτ=0.003 ideal | 0.0363 | **0.0075** | 0.0069 | 0.0148 | GDR works; oracle *residual* TVD(Mp,q)=**0.070** |
| ECD opt loss κτ=0.003 strong | 0.1534 | **0.0075** | 0.0062 | 0.1516 | idle-ZNE ≈ raw under readout |
| ECD random loss κτ=0.003 ideal | 0.0461 | 0.0459 | 0.0525 | 0.1555 | tiny GDR gain; fit set nth1=0.30 (true 0) |
| ECD random loss κτ=0.1 ideal | 0.2991 | **0.3725** | 0.5377 | 0.2297 | GDR over-corrects; ZNE wins TVD |

Identifiability: on ECD opt loss, η2 stays at the true cumulative η while η1 / nth1 / p_down wander. Mild-random ideal-readout GDR is over-parameterized.

### Method hypotheses (to drop if microbench loses)

| id | change | target failure |
|----|--------|----------------|
| A | ridge + holdout-λ + damp-to-readout (`gdr_ridge` / `gdr_holdout` / `gdr_damped` / `gdr_reg`) | mild random overfit; GDR worse than raw at high κτ |
| B | `readout_then_zne`, `zne_then_readout` | idle-ZNE bias under readout |
| C | log-spaced \|α\| ∈ [0.25, 1.35] (`twin_design=span`) | (η, p_nn) Fisher |
| D | t_free-weighted fit + oracle+residual hops (`gdr_tfree`, `gdr_residual`) | ECD oracle residual ~0.07 (interleaving) |
| E | `gdr_mid` = fit only (η, readout) | structured middle vs `gdr_full` |
| F | energy-weighted ridge (`gdr_energy`) | only if TVD↑ but \|ΔE\| lags |

New runs go under `out_research/<tag>/`. Physical histograms are cached in `out_research/cache/` so fit-only replays do not resimulate.

Optimized params copied from `out/` (not deleted):

- `optimized_params_ecd_h000_nd5.json`
- `optimized_params_snap_h000_nd2.json`

## Phase 1

Added:

- subset flags on `run_mitigation_experiment.py` (`--families`, `--kappa-tau`, `--params`, `--n-rank2`, `--mag-scale-min/max`)
- `run_ablation.py` with cache, method list, and markdown scoreboard vs PR #6
- research fits in `mitigation.py` (no `src/` edits)

## Phase 2 microbenches

See `out_research/<tag>/ablation_summary.md` for numbers. Same-run `gdr_param` is the controlled baseline; `base_*` columns are PR #6 (8192 / 40) and are not shot-matched on cheap loops.
