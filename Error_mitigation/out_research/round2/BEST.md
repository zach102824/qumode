# Round 2 best

**No beat of adaptive; recipe unchanged.**

Official frozen recipe from PR #8 stays the default. Ban list in `DROPPED.md`.
Official runner was **not** rewired.

Keep bar: beat same-run `gdr_select` by >0.005 TVD on a hard cell, with no
>0.003 regression on protect cells (ECD opt comprehensive 0.1; SNAP gated
floor; multi-H mild H000/H004/H009). ECD random loss/comprehensive 0.1 must
not get worse.

## Pass 1 (fit-only, 8 cached cells)

See `micro_scoreboard.md`. Closest: `gdr_fisher` mean Δ −0.0003. `gdr_then_rtz`
won ECD random loss 0.1 (0.194 vs 0.201) and **regressed 0.343 → 0.408**.

## Pass 2

### 1. Shot-robust ensemble (K=5 twin bootstrap)

Average of 5 `gdr_param` unfolds. `stage_b_scoreboard.md`.

| cell | select | ensemble | member mean±std |
|------|-------:|---------:|----------------:|
| ECD random loss 0.1 | **0.2011** | 0.2188 | 0.229±0.026 |
| ECD random comprehensive 0.1 | **0.3424** | 0.3831 | 0.396±0.035 |
| ECD opt comprehensive 0.1 \* | **0.3419** | 0.3461 | 0.346±0.004 |
| H000 / H004 / H009 mild \* | 0.1012 / 0.0763 / 0.0932 | 0.1012 / 0.0784 / 0.1050 | — |
| SNAP opt comprehensive+rr 0.1 | 0.5903 | **0.5633** | 0.563±0.030 |

**drop.** Beats SNAP opt 0.1 by 0.027 but regresses the 0.343 cell (+0.0042)
and H009 mild (+0.012). Worse on both ECD random high-κτ cells. Variance on
random high-κτ is large (σ≈0.03); averaging does not beat damped/select.

### 2. SNAP Nd=2 optimized transfer (H000)

H004 SNAP is **not** near-E0 (E=−3.93 vs E0=−5.35, deficit 1.41) — skipped.
H000 SNAP deficit 1.93. Default twins, comprehensive + readout_realistic:

| κτ | raw | adaptive select (= `gdr_param`) | ensemble | joint |
|---:|----:|--------------------------------:|---------:|------:|
| 0.003 | 0.1411 | **0.0230** | 0.0230 | 0.0230 |
| 0.03 | 0.5710 | **0.2030** | 0.2239 | 0.2030 |
| 0.1 | 0.7952 | 0.5903 | 0.5633 | 0.5904 |

Adaptive is still best at 0.003 and 0.03. Ensemble’s 0.1 win is the same
method that regresses ECD 0.343 — not a SNAP-only ship.

### 3. Joint histogram + top-8 |E| bins (holdout λ)

**drop.** Holdout almost always picks λ=0 (equals `gdr_param`). Max |Δ| vs
select on protect cells ≈ 0. Distinct from banned energy-weighted *twins*;
still not a keep.

### 4. Active +10 Fisher-greedy Gaussian twins

New sims of 10 extra t_free=0 twins (pool of 80 noiseless, η-Fisher ×
diversity). Refit on 50 twins, same target shots. `active_scoreboard.md`.

| cell | select40 | gdr50 | select50 | Δ select50−select40 |
|------|---------:|------:|---------:|--------------------:|
| ECD random loss 0.1 | **0.2011** | 0.2187 | 0.2094 | **+0.0083 (worse)** |
| ECD random comprehensive 0.1 | 0.3424 | 0.3767 | 0.3366 | −0.0058 |

The comprehensive cell is a 0.0058 select win, but ECD random **loss** 0.1
gets worse — that hard cell must not regress. **drop.**

## Verdict

Honest negative after two passes. Ban list + adaptive remains best.

Reproduce:

```
python -u Error_mitigation/run_round2.py --stage micro
python -u Error_mitigation/run_round2.py --stage b
python -u Error_mitigation/run_round2.py --stage active
```
