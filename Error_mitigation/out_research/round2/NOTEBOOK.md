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

## Pass 3 — shots, family η, cross-H, RL unfold

### Shot sweep (`--stage shots`, wall 49 s)

`shots_scoreboard.md`. GDR vs raw on 4 hard cells at 2048 / 8192 / 32768:

- Random high-κτ: GDR’s TVD win **grows with shots** (loss 0.057 → 0.097 → 0.109).
- ECD opt 0.343: win is huge and **shot-stable** (0.351 / 0.342 / 0.341) — leftover model error.
- SNAP random comprehensive 0.003: select ≈ raw at 2048, **worse** than raw at 8192 (0.0415 vs 0.0372), better at 32768.

Ungated extra α=0.25 at 2048 beat random high-κτ (−0.0098 / −0.0053) and **destroyed 0.343** (0.351→0.489). Gated to **random only**: 2048 beats stand, 0.343 untouched, 8192 identical (floor=0). **Optional, not official.** Official defaults unchanged.

### Family-conditional η (`--stage family` / `c`, wall 140 s)

Opposite of anneal: η ridge 0.3 at κτ=0.003, 10 at 0.1; holdout λ; comprehensive+rr only. 0.343 no-op (ideal readout).

H004 mild −0.0033 (under bar). H009 mild **+0.0031** (protect). Often λ=0. **drop.**

### Cross-H twin bank (`--stage xfer`, wall 12 s)

Same-H diagonal matches. Off-diagonal mean Δ **+0.041** vs same-H `gdr_param`. H009→H004 −0.010 is a one-off; H004→H000 **+0.094**. M does not transfer. **drop.**

### RL soft-clip + early-stop

Holdout picks `n_iter=8` almost always. Soft-clip ≈ `gdr_param`. Early-stop **0.343→0.464**. **drop.**

## Verdict

Three passes. Official 8192 adaptive + ban list remains the recipe. Optional 2048 random extra-damp is a research note only; official runner not wired.
