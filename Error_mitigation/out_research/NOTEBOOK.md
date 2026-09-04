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

Wired into `run_mitigation_experiment.py` (default `--twin-design span` at this point):
`gdr_damped`, `gdr_mid`, `gdr_residual`, `gdr_select`, `readout_then_zne`.
README methods/caveats updated. `--twin-design default` restores the PR #6 U(0.5,1) mix.
No `src/` edits. Tests: 16 mitigation + full non-slow suite green.

## Phase 5 — leftover regressions (cache replay + default twins)

### `phase3_v2` — same 8192/40 span cache, new methods

`gdr_afterburn` ≈ `gdr_residual` on every cell that matters (30 afterburn wins vs gdr are the same residual wins). `gdr_blend` is not a keep.

`gdr_select` with a Gaussian / t_free holdout still **never** picked residual (0/23). Rank-2 twins are too close to end-of-circuit GDR. A hop-cap also blocked the high-κτ optimized cells where residual hops grow with noise (0.11–0.16) even though residual is best there.

Blind `optimized → residual` would catch those 23 span cells, but the next run shows that rule is twin-design specific.

### `opt_default` — optimized only, PR #6 U(0.5,1) twins, 8192/40, 54 cells

This is the span regression. Default `gdr_param` vs span `gdr_param`: **35 better, 19 tie, 0 worse**.

| cell | PR #6 gdr | span gdr | default gdr | default residual |
|------|----------:|---------:|------------:|-----------------:|
| ECD opt comprehensive κτ=0.1 ideal | **0.343** | 0.416 | **0.343** | 0.414 (≈ oracle) |
| ECD opt comprehensive κτ=0.1 strong | 0.351 | 0.403 | **0.338** | 0.406 |
| ECD opt thermal κτ=0.1 ideal | 0.270 | 0.318 | **0.277** | 0.261 |
| ECD opt loss κτ=0.1 ideal | 0.202 | 0.230 | **0.201** | 0.195 |
| SNAP opt comprehensive κτ=0.1 ideal | 0.632 | 0.662 | **0.611** | 0.747 |

Span twins learned a map close to the physical oracle (~0.41), which is the *wrong* end-of-circuit map for interleaved comprehensive noise. The narrower PR #6 mix luckily fits a better effective kernel. Residual-on-optimized is then harmful once default twins are used.

### Shipped recipe (evidence)

1. **`--twin-design adaptive` (new official default):** span on random, U(0.5,1) on optimized.
2. Always report `gdr_param`.
3. `gdr_damped` on random / leftover over-correction.
4. `gdr_residual` as an extra on optimized **loss / thermal** (not the select default).
5. `readout_then_zne` whenever reporting ZNE under readout.
6. `gdr_select`: `gdr_param` on optimized; holdout among `{safe, gdr, mid, damped}` on random.

Post-hoc adaptive hybrid (span+damped on random, default `gdr_param` on optimized): beats PR #6 `gdr_param` on **86/108**, beats raw on **107/108**, worse than raw on **1/108** (SNAP random comprehensive κτ=0.003 ideal, +0.003).

`gdr_afterburn` / `gdr_blend` stay in the ablation driver; afterburn is not a default.

No `src/` edits. `out/` and `out_smoke/` untouched.

## Phase 6 — leftover SNAP cell, interleaving kernel, energy lag

### Energy lag (drop F)

On 36 optimized loss/thermal cells (span and default twins): **no** method improved TVD by >0.002 while making |ΔE| worse by >0.02 vs `gdr_param`. Skip energy-weighted / GS-support fit.

### SNAP random comprehensive κτ=0.003 ideal (keep, gated)

`leftover_floor` / `leftover_floor2` reused the phase3 span cache (54 random cells).

Ungated conservative damp (`slack=0.003`) kills the SNAP leftover (0.0369 < raw 0.0372) but flattens mid-κτ ECD wins (up to +0.02 vs damped) and drops random-vs-PR #6 from 48 → 45.

Gated floor (`slack=0.003` only if safe is within 0.01 of the best twin TVD) still kills SNAP ideal, but applying it to *all* random mild cells turns several good damps back into raw.

**Shipped gate:** use the conservative floor only on **random + comprehensive + κτ≤0.003** (the coherent-error slice where the histogram kernel is wrong at tiny noise). That is 3 cells:

| readout | raw | damped | floor (= new damped) | PR #6 gdr |
|---------|----:|-------:|---------------------:|----------:|
| ideal | 0.0372 | 0.0402 (lose) | **0.0369** | 0.0445 |
| realistic | 0.0535 | **0.0374** | 0.0412 | 0.0324 |
| strong | 0.0892 | **0.0465** | 0.0495 | 0.0410 |

Ideal is the only previous loss-to-raw. Realistic/strong still beat raw. Adaptive vs PR #6 count is unchanged (ideal still a win; the other two already lost to PR #6). Adaptive worse-than-raw: **1 → 0**.

### Interleaving kernel (drop)

`gdr_interleave` = thermal(η_early) → hops/leak → thermal(η_late) → readout. Not equivalent to one binomial.

`leftover_il` (span, ECD opt loss, t_free-only fit) and `leftover_il_all` (all 40 twins):

| κτ | residual | interleave (t_free) | interleave (all) |
|---:|---------:|--------------------:|-----------------:|
| 0.003 ideal | **0.0092** | 0.0161 | 0.0096 (= gdr) |
| 0.1 ideal | **0.1954** | 0.5217 | 0.2541 |

Forward oracle residual TVD(M p, q) is still **0.0705** at κτ=0.003 (the original interleaving caveat). Shot-noise TVD floor at 8192 is ~0.0005, so unfold TVD 0.0092 is real leftover model error — but the two-stage histogram kernel does not beat residual/oracle. Drop; keep `gdr_residual` as the extra on optimized loss/thermal.

README status header updated (smoke + full on this branch are done).

## Phase 7 — leftover loop (structured middle, shots, transfer)

Adaptive twins + gated damped floor stay. No `src/` edits. `out/` / `out_smoke/` untouched.

### Structured middle vs `gdr_full` (drop)

`leftover_mid` reused the `opt_default` cache (ECD optimized comprehensive, default twins, 8192 / 40, `--n-rank2 10`). Two ridged extras on top of `gdr_param`:

- `gdr_split` — per-register extra hops + leak, L2 toward 0 (`lam=0.05`)
- `gdr_band` — signed n±1 band residual, same ridge

Keep bar: beat `gdr_param` on optimized comprehensive **and** do not regress 86/108.

| cell | gdr_param | split | band | mid | residual |
|------|----------:|------:|-----:|----:|---------:|
| κτ=0.003 ideal | 0.0522 | 0.0506 | 0.0510 | **0.0451** | 0.1048 |
| κτ=0.1 ideal | **0.3434** | 0.3434 | 0.3433 | 0.3484 | 0.4138 |
| κτ=0.1 strong | **0.3381** | 0.3379 | 0.3376 | 0.3465 | 0.4057 |

At κτ=0.1 the ridge pins `gdr_split` hops to ~2e-4 (identical to `gdr_param`). `gdr_band` moves TVD by ≤0.0005. `gdr_mid` still wins mild comprehensive and still loses the high-κτ cell — already known, already not the select default. `gdr_full` already loses this cell in PR #6 (0.44). **Drop**; do not change the 86/108 recipe.

### Shot robustness 2048 vs 8192 (keep recipe)

Cache replay only (physical hists unchanged). Hard cells: ECD random κτ=0.1 loss / comprehensive (span) and ECD opt comprehensive κτ=0.1 (default). Official `gdr_select` / `gdr_damped`.

| cell | shots | raw | gdr_param | damped | select |
|------|------:|----:|----------:|-------:|-------:|
| ECD random loss 0.1 ideal | 8192 | 0.298 | 0.209 | 0.203 | **0.201** |
| | 2048 | 0.309 | 0.277 | **0.252** | **0.252** |
| ECD random comprehensive 0.1 ideal | 8192 | 0.403 | 0.377 | **0.342** | **0.342** |
| | 2048 | 0.419 | 0.376 | **0.345** | **0.345** |
| ECD random comprehensive 0.1 realistic | 2048 | 0.415 | 0.431 | **0.374** | **0.374** |
| ECD opt comprehensive 0.1 ideal | 8192 | 0.909 | **0.343** | 0.427 | **0.343** |
| | 2048 | 0.906 | **0.351** | 0.434 | **0.351** |

At 2048 shots, `gdr_param` can lose to raw on one random comprehensive+readout cell (0.431 vs 0.415); **damped / select still win**. Optimized comprehensive: `gdr_param` moves 0.343 → 0.351 (shot noise); damped still hurts, so select correctly keeps `gdr_param`. Recipe is robust; no method change.

### Transfer H001 (keep recipe; do not expand)

`mixed_p_spin` H001 (`mixed_p_spin_p2-4_001.npz`, E0≈−6.032, ground `|1,4,6⟩`). No cached opt. One ECD 200-iter / 3-restart opt into `out_research/optimized_params_ecd_h001_nd5.json` (33 min, best **E=−3.922**; restarts −3.08 / −2.44 / −3.92). Same budget that reached −6.23 on H000. Physics cache tagged `_h001` so H000 histograms are not reused.

Small matrix: optimized only, default twins, 8192 / 40, loss + comprehensive, κτ ∈ {0.003, 0.1}, ideal + strong. `leftover_xfer_h001/`.

| cell | raw | gdr_param / select | damped | residual | vs raw |
|------|----:|-------------------:|-------:|---------:|:------:|
| loss 0.003 ideal | 0.039 | **0.023** | 0.026 | 0.021 | win |
| loss 0.003 strong | 0.247 | **0.038** | 0.045 | 0.031 | win |
| loss 0.1 ideal | 0.587 | **0.495** | 0.519 | 0.958 | win |
| loss 0.1 strong | 0.652 | **0.528** | 0.546 | 0.966 | win |
| comprehensive 0.003 ideal | 0.176 | **0.112** | 0.149 | 0.086 | win |
| comprehensive 0.003 strong | 0.342 | **0.130** | 0.159 | 0.090 | win |
| comprehensive 0.1 ideal | **0.788** | 0.888 | 0.867 | 0.970 | lose |
| comprehensive 0.1 strong | **0.810** | 0.874 | 0.855 | 0.965 | lose |

Mild + high-κτ **loss** transfer. High-κτ **comprehensive** does not: every learned map over-corrects, damped still loses to raw, residual/oracle collapse. Switching optimized select to damped would regress the H000 comprehensive 0.1 cell (0.343 → 0.427). **Do not change the official recipe.** Do not spend the window on 20 Hamiltonians.

H000 vs this weak H001 opt (same cells, `gdr_param`):

| cell | H000 | H001 |
|------|-----:|-----:|
| loss 0.003 ideal | 0.0075 | 0.023 |
| comprehensive 0.003 ideal | 0.052 | 0.112 |
| comprehensive 0.1 ideal | 0.343 | 0.888 |

The H001 circuit is closer to a random / poorly optimized ECD than to the H000 VQE. That is the transfer caveat, not a reason to flip the 86/108 default.

### Phase 7 ship

No official-recipe change. Adaptive twins + gated floor stay. Tests: 25 `test_error_mitigation`, 110 non-slow. Stop new ideas.

## Phase 8 — H001 lose, twin count, tail bins

Adaptive / gated damped / optimized `gdr_param` unchanged. No `src/` edits. `out/` / `out_smoke/` untouched.

### H001 comprehensive κτ=0.1 (keep recipe; stop expanding)

Noiseless energies on H001: **E_random=+0.311**, first-budget **E_opt=−3.922**, E0=−6.032. The weak opt is not a random circuit.

| circuit | twins | raw | gdr_param | damped | select |
|---------|-------|----:|----------:|-------:|-------:|
| H001 random comprehensive 0.1 ideal | span | 0.313 | 0.198 | **0.187** | **0.187** |
| H001 weak opt (−3.92) | default | **0.788** | 0.888 | 0.867 | 0.888 |
| H001 weak opt (−3.92) | span | **0.788** | 0.829 | 0.815 | 0.829 |
| H001 extra opt (−4.82) | default | **0.795** | 0.943 | 0.894 | 0.943 |

Random + official random recipe **wins**. Span twins on the weak opt still lose. One extra opt budget (H000 warm start −3.968; two new random restarts; best **−4.819** from `random_4`, 37 min) still loses, and over-corrects more. H000 warm start does not transfer.

**Verdict:** the lose is this mid-quality VQE under comprehensive κτ=0.1, not a bad label and not “one more restart.” Transfer of the *optimized* recipe needs a near-E0 noiseless opt (H000: −6.23 vs E0 −6.23). This instance did not yield one in 6 starts. **Do not change the official recipe. Do not open 20 Hamiltonians.**

Artifacts: `leftover_xfer_h001_rand/`, `leftover_xfer_h001_spanopt/`, `leftover_xfer_h001_opt2/`.

### Twin-count sweep (keep n_train=40)

Fit-only subsets of the official 40-twin H000 caches (`--fit-n-train`). Four hard cells at 8192, ideal readout:

| cell | n=10 select | n=20 select | n=40 select |
|------|------------:|------------:|------------:|
| ECD random loss 0.1 | 0.230 | 0.230 | **0.201** |
| ECD random comprehensive 0.1 | 0.361 | 0.360 | **0.342** |
| ECD opt loss 0.1 | 0.203 | 0.201 | 0.195 |
| ECD opt comprehensive 0.1 | 0.343 | 0.341 | 0.343 |

Optimized `gdr_param` is already at the n=40 number with **10** twins. Random high-κτ still gains ~0.02–0.03 from 40 (and at n=20, random comprehensive `gdr_param` 0.406 **loses** to raw 0.403; damped/select still win at 0.360). Official `n_train=40` stays — it is the random/span requirement, not the optimized one.

### Tail-bin truncation (drop)

Zero twin bins below 0.5/shots or 2/shots, refit `gdr_param` (25 s). Occupancy at 0.5/shots equals the full hist (multinomial min is 1/8192). TVD change ≤0.001 on the three hard ideal cells. **Drop.**

### Phase 8 ship

No official-recipe change. Tests: 26 `test_error_mitigation`. Stop new ideas.

## Phase 9 — `params=auto`, SNAP, bootstrap

Adaptive / gated damped / optimized `gdr_param` / `n_train=40` unchanged. No `src/` edits.

### `params=auto` (drop as a default)

`classify_opt_quality(E_opt, E0, gap)` treats the **optimized circuit** as random (span twins + holdout/damped select) if `E_opt - E0 > max(abs_tol, rel_gap * gap)` with defaults `abs_tol=0.5`, `rel_gap=0.2`. Ablation flag `--params auto` (not an official default).

H000 ECD: E_opt=−6.230, E0=−7.111, deficit=**0.881**, gap=0.778 → default thresh 0.5 **fires**.  
H001 ECD: deficit=**2.110** → fires.  
SNAP H000: E_opt=−5.178, deficit=**1.933** → fires.

| cell | recipe | raw | gdr_param | select | vs keep-bar |
|------|--------|----:|----------:|-------:|-------------|
| H001 opt comprehensive 0.1 (must flip lose→win) | auto=random | **0.788** | 0.829 | 0.817 (damped) | **still lose** |
| H000 opt comprehensive 0.1 (must keep 0.343) | auto=random | 0.909 | 0.416 | 0.589 (damped) | **regress** |
| H000 same, `abs_tol=1.0` | optimized | 0.909 | **0.343** | **0.343** | no regress, H001 still lose |
| SNAP opt comprehensive 0.1 | auto=random | 0.791 | 0.662 | 0.689 | worse than default ~0.61 |

No threshold both flips H001 and preserves the H000 0.343 cell: H001’s lose is the mid-quality VQE under comprehensive κτ=0.1 (Phase 8), not the recipe label. **Drop as a default.** `--params auto` stays research-only, default unused.

### Bootstrap ±TVD (keep as notebook error bars)

8 independent 8192-shot resamples, official recipe, H000 caches, 145 s.

| cell | raw | gdr_param | select |
|------|----:|----------:|-------:|
| ECD random loss 0.1 | 0.301 ± 0.006 | 0.213 ± 0.012 | **0.208 ± 0.012** |
| ECD random comprehensive 0.1 | 0.407 ± 0.005 | 0.337 ± 0.015 | **0.314 ± 0.013** |
| ECD opt comprehensive 0.1 | 0.908 ± 0.003 | **0.346 ± 0.008** | **0.346 ± 0.008** |
| ECD opt loss 0.1 | 0.697 ± 0.006 | **0.202 ± 0.003** | **0.202 ± 0.003** |
| SNAP random comprehensive 0.003 | 0.039 ± 0.002 | 0.036 ± 0.004 | **0.036 ± 0.004** |

Single-draw headlines sit inside these bands (ECD opt comprehensive 0.343 vs 0.346 ± 0.008). SNAP mild comprehensive select mean 0.036 still beats raw 0.039; the gated 0.0369 draw is typical, not a one-seed fluke.

### Phase 9 ship

Official defaults frozen. Tests: 27 `test_error_mitigation`.

## Phase 10 — defaults frozen (polish only)

No method change. Adaptive / gated damped / optimized `gdr_param` / `n_train=40` stay. No `src/` edits. `out/` / `out_smoke/` untouched. Did **not** reopen `params=auto`, interleave, middle/split/band, energy-weighted fit, or tail bins.

### Research smoke (keep, documented)

`run_mitigation_experiment.py --preset research_smoke` was **not** added: that driver's `DEFAULT_OUTDIR` is `Error_mitigation/out/`. The one-command CI-ish slice lives on the ablation driver (writes only under `out_research/`):

```
python -u Error_mitigation/run_ablation.py --preset research_smoke
```

ECD optimized loss κτ=0.003, `--twin-design adaptive` (→ PR #6 mix), 2048 shots, `n_train=40`, `n_rank2=10`, ideal + realistic, `{raw, gdr_param, gdr_damped, gdr_select}`. Cache key `ecd_optimized_loss_kt0.003_n40_default_nr10_lo0.25_hi1.35_x0`. Cache hit (~24 s). Select keeps `gdr_param` on optimized:

| readout | raw | gdr_param | damped | select |
|---------|----:|----------:|-------:|-------:|
| ideal | 0.0380 | **0.0096** | 0.0238 | **0.0096** |
| realistic | 0.0884 | **0.0192** | 0.0369 | **0.0192** |

### Hard-cell figure (keep)

`Error_mitigation/plot_hard_cells.py` writes `out_research/figures/hard_cells_adaptive.png` from the existing headline numbers (PR #6 `out/` vs adaptive hybrid; select error bars from the Phase 9 8×8192 bootstrap).

### SNAP opt mild hybrid-ZNE (document only; `opt_default` cache)

No new sim. SNAP optimized, default twins, κτ=0.003:

| family | readout | raw | gdr_param | zne_idle | readout_then_zne |
|--------|---------|----:|----------:|---------:|-----------------:|
| loss | ideal | 0.057 | 0.018 | 0.027 | 0.027 |
| loss | realistic | 0.097 | 0.013 | 0.076 | **0.063** |
| loss | strong | 0.189 | 0.014 | 0.162 | **0.053** |
| thermal | ideal | 0.060 | 0.013 | 0.059 | 0.059 |
| thermal | realistic | 0.099 | 0.018 | 0.049 | **0.029** |
| thermal | strong | 0.180 | 0.010 | 0.185 | **0.056** |

Hybrid ZNE still wins under readout. GDR still beats either ZNE on these mild SNAP opt cells. Recipe unchanged.

### Phase 10 ship

README methods table matches the shipped recipe. NOTEBOOK frozen at Phase 10. Tests: 28 `test_error_mitigation`; 113 non-slow. Stop until the next steer.
