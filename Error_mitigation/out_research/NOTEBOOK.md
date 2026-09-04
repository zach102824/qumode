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

### `micro_ab` — ECD loss, default twins, shots=4096, n_train=20, κτ∈{0.003,0.1}

12 cells. Wall ~3 min (sim 3–9 s/block; fits dominate).

| verdict | method | evidence |
|---------|--------|----------|
| **keep (A)** | `gdr_damped` | Fixes GDR-worse-than-raw on ECD **random κτ=0.1**: TVD 0.29 vs gdr 0.36–0.41 (and beats raw). α≈0.3–0.4. On optimized mild, α=0 (does not spoil a good unfold). |
| **keep (A/E)** | `gdr_mid` / `gdr_holdout` | Random κτ=0.1 ideal: mid 0.307 vs gdr 0.362 vs raw 0.317. Small help on optimized mild. |
| **drop as default damp** | `gdr_reg` on optimized high-κτ | α=0.20–0.25 *hurts* when unfold is already good (0.30 vs gdr 0.20). |
| **keep (B)** | `readout_then_zne` | Optimized loss κτ=0.003 **strong readout**: ZNE 0.145 → **0.025** (PR #6 ZNE was 0.152 ≈ raw 0.153). Still above gdr 0.009, but the idle-ZNE failure mode is fixed. Random mild ZNE remains bad (~0.22). |
| **keep (D) on optimized** | `gdr_residual` | **Best on all 6 optimized cells.** Mild: 0.0093 / 0.0121 / 0.0076 vs gdr 0.0114 / 0.0179 / 0.0090 (matches oracle). High κτ: ~0.192–0.204 vs gdr 0.197–0.213. Residual hops ≈0 on mild; small leak at κτ=0.1. |
| **drop (D) on random** | `gdr_residual` | Overfits (p_up≈0.15 at κτ=0.1); TVD 0.49 vs raw 0.32. |
| **keep as combo** | `gdr_select` | Holdout picks residual vs damped vs mid vs safe. Added after `micro_ab`. |

Random mild ideal is still tiny (raw 0.0618, gdr 0.0639, damped 0.0622) — next lever is twin span (C), not more regularization.

PR #6 numbers on these cells differ by shot count (8192/40 vs 4096/20) but ranking matches.

`gdr_select` v1 leaked an extra damp step and *hurt* optimized κτ=0.1 (0.30 vs gdr 0.20). Fixed: damp is now an explicit holdout candidate; oracle is preferred when within 5% of the best twin score.

### `micro_span` — same grid, log-spaced |α| ∈ [0.25, 1.35], 5 rank-2 twins

**C is a keep.** Same-run target histograms (same raw TVD); only the training twins changed.

| cell | default `gdr_param` | span `gdr_param` | note |
|------|--------------------:|-----------------:|------|
| ECD random κτ=0.1 ideal | 0.362 (worse than raw 0.317) | **0.237** | flips the overfit failure |
| ECD random κτ=0.1 realistic | 0.390 | **0.233** | + damped 0.220 |
| ECD random κτ=0.1 strong | 0.405 | **0.268** | + damped 0.254 |
| ECD random κτ=0.003 ideal | 0.064 | 0.068 | still tiny / slightly worse; `gdr_select`→safe 0.062 |
| ECD opt κτ=0.003 ideal | 0.0114 | **0.0095** | residual/oracle still 0.0093 |
| ECD opt κτ=0.1 | 0.197–0.213 | 0.207–0.220 | residual still best (~0.192–0.204) |

Working recipe so far: **span twins + `gdr_param`**, plus `gdr_damped` on noisy random, `gdr_residual`/`oracle` on optimized, `readout_then_zne` when reporting ZNE under readout.

### `micro_hard` — ECD thermal + comprehensive, span twins, 4096/20

Span again flips PR #6 GDR-worse-than-raw cells:

| cell | PR #6 gdr | this gdr_param | best new |
|------|----------:|---------------:|----------|
| ECD random thermal κτ=0.1 ideal | 0.407 (> raw 0.332) | **0.274** (< raw 0.331) | damped 0.272 |
| ECD random comprehensive κτ=0.1 ideal | 0.539 (> raw 0.399) | **0.368** (< raw 0.402) | damped **0.328** |
| ECD opt thermal κτ=0.003 | 0.019 | 0.021 | **residual 0.014–0.018** |
| ECD opt thermal κτ=0.1 | 0.270 | 0.317 | **residual 0.249–0.259** |
| ECD opt comprehensive κτ=0.003 | 0.053 | 0.051–0.063 | **gdr_mid 0.043–0.058** |

`gdr_tfree` wins the ECD random comprehensive κτ=0.003 slice (~0.004 better than gdr_param). Residual remains a random-circuit loser.

### `micro_snap` — SNAP loss, span twins, 4096/20

| cell | PR #6 gdr | this gdr_param | best new |
|------|----------:|---------------:|----------|
| SNAP random κτ=0.003 ideal | 0.0223 (> raw 0.0194) | **0.0419** (< raw 0.0473)* | residual 0.039 |
| SNAP random κτ=0.1 | 0.233–0.258 | **0.219–0.227** | select/reg **0.194–0.196** |
| SNAP opt κτ=0.003 | 0.013–0.019 | 0.014–0.020 | residual/tfree 0.013–0.016 |
| SNAP opt κτ=0.1 | 0.367–0.416 | 0.403–0.418 | tfree **0.385–0.394** |

\*shot-mismatched raw; same-run GDR now beats raw on SNAP random mild ideal (the PR #6 failure).

## Phase 3 — shot-matched matrix (8192 / 40 / span, 108 cells)

`out_research/phase3/`. Same shots, twin count, seed, instance as PR #6; only the twin **design** and extra methods changed.

Headline vs PR #6 `gdr_param`:

| success-bar cell | PR #6 raw / gdr | phase3 gdr_param | best new |
|------------------|----------------:|-----------------:|----------|
| ECD random loss κτ=0.1 ideal | 0.299 / **0.373 (worse)** | **0.209** | select 0.201 |
| ECD random comprehensive κτ=0.1 ideal | 0.399 / **0.539 (worse)** | **0.377** | damped **0.342** |
| ECD random thermal κτ=0.1 ideal | 0.332 / **0.407 (worse)** | **0.263** | damped 0.258 |
| SNAP random loss κτ=0.003 ideal | 0.019 / **0.022 (worse)** | **0.022 < raw 0.028** | mid 0.022 |
| ECD opt loss κτ=0.003 strong (ZNE≈raw) | 0.153 / 0.0075 / ZNE 0.152 | gdr 0.017 | **readout_then_zne 0.010** (ZNE now 0.119) |
| ECD opt loss κτ=0.003 ideal | 0.036 / 0.0075 | 0.0096 | **readout_then_zne / ZNE 0.0064** |
| ECD opt thermal κτ=0.1 ideal | 0.767 / 0.270 | 0.318 | **residual 0.258** |
| ECD opt comprehensive κτ=0.003 ideal | 0.182 / 0.053 | 0.054 | **readout_then_zne 0.025** |

Counts:
- `gdr_param` (span) beats PR #6 `gdr_param` on **66 / 108** cells.
- PR #6 GDR-worse-than-raw: **15 → 2**. Remaining two are tiny mild-random cells; `gdr_damped` fixes the ECD one (0.0427 < raw 0.0434). SNAP random comprehensive κτ=0.003 ideal is still +0.003 over raw after damping.
- New methods beat same-run `gdr_param` on: damped 44, mid 43, residual 36, select 33, `readout_then_zne` 11.

Regressions to not hide: span `gdr_param` is a bit worse than PR #6 on ECD/SNAP **optimized comprehensive κτ=0.1** (e.g. 0.416 vs 0.343). Residual/mid recover some but not all. Use PR #6 `gdr_param` numbers as the floor there; the random-circuit failure mode was the one we set out to fix.

## Phase 4

Wired into `run_mitigation_experiment.py` (default `--twin-design span`):
`gdr_damped`, `gdr_mid`, `gdr_residual`, `gdr_select`, `readout_then_zne`.
README methods/caveats updated. `--twin-design default` restores the PR #6 U(0.5,1) mix.
No `src/` edits. Tests: 16 mitigation + full non-slow suite green.
