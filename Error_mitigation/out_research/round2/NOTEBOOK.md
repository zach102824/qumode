# Round 2 lab notebook

Branch: `cursor/gdr-round2-search-cfbc` (PR #10)  
Start: PR #9 `cursor/gdr-multi-h-validation-c75a`.  
Hard rules: no `src/` edits; do not overwrite `out/` or `out_smoke/`; one heavy sim at a time. Ban list: `DROPPED.md`.

**Answer:** **No beat of adaptive; recipe unchanged.** Official defaults unchanged.

Keep bar: >0.005 TVD vs same-run `gdr_select`, no >0.003 on protect cells; ECD random loss/comprehensive 0.1 must not get worse.

## Pass 1 — fit-only 8 cells (wall 298 s)

`micro_scoreboard.md`. Anneal / eta / fisher / select_kt / mild residual / readout-then-GDR / GDR-then-RTZ / Chebyshev grid (not run). All dropped. Details in the Pass-1 `BEST.md` table.

## Pass 2 — not just the same 8-cell refit

### Ensemble K=5 (`--stage b`, wall 420 s)

11 cells (8 hard + 3 SNAP opt comprehensive+realistic). `stage_b_scoreboard.md`.

- Random high-κτ: ensemble **worse** than select (0.219 vs 0.201 loss; 0.383 vs 0.342 comprehensive). Member σ ≈ 0.03.
- Protect ECD 0.343: 0.346 (**+0.0042**, over the 0.003 bar).
- H009 mild: 0.105 vs 0.093 (**+0.012**).
- SNAP opt 0.1: 0.563 vs 0.590 (would be a beat in isolation).

**drop** — the SNAP 0.1 win cannot be taken without ECD/H009 regressions.

### SNAP Nd=2 H000 opt, comprehensive + readout_realistic

Caches already existed (default twins). H004 SNAP deficit 1.41 — not near-E0, skipped.

| κτ | raw | adaptive | note |
|---:|----:|---------:|------|
| 0.003 | 0.141 | **0.023** | GDR works; adaptive = `gdr_param` |
| 0.03 | 0.571 | **0.203** | adaptive still best |
| 0.1 | 0.795 | 0.590 | ensemble 0.563, not shippable (see above) |

No different *allowed* method wins on SNAP without breaking ECD protect cells.

### Joint GDR (histogram + top-8 |E| bins)

Holdout λ ∈ {0, 1e-4, …, 1e-2}. Almost always λ=0. Not energy-weighted twins (banned). **drop.**

### Active +10 Fisher twins (`--stage active`, wall 30 s)

`x_random` matches cache p_ideal at TVD 6e-16. 10 extra noisy histograms cached under `round2/cache/active10_*.npz`.

select50 beats select40 on comprehensive 0.1 (−0.0058) but **loses** on loss 0.1 (+0.008). **drop.**

## Verdict

Two passes, four new directions, no keep. Stop new ideas; adaptive + ban list remains the recipe.
