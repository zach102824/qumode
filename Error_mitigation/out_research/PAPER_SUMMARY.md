# GDR error mitigation — one-page scoreboard

PR #8 on `cursor/gdr-improve-30h-v2-4f00`, from PR #6 (`cursor/error-mitigation-smoke-full-5302` @ `b7b7a8c`). No `src/` edits. Scoreboard is H000 (`mixed_p_spin_p2-4_000.npz`), 108 cells, 8192 shots / 40 twins. Official defaults are **frozen**.

## Shipped recipe

1. `--twin-design adaptive`: log-spaced \(|\alpha|\in[0.25,1.35]\) on **random**; PR #6 \(U(0.5,1)\) on **optimized**.
2. Always report `gdr_param` (official choice on optimized).
3. `gdr_damped` on random; conservative floor **only** on random + comprehensive + \(\kappa\tau\le 0.003\).
4. `gdr_select`: `gdr_param` on optimized; holdout \(\{\mathrm{safe},\,\mathrm{gdr},\,\mathrm{mid},\,\mathrm{damped}\}\) on random.
5. `readout_then_zne` when reporting ZNE under readout.
6. `n_train=40`. `gdr_residual` is an extra on optimized loss/thermal, not the select default.

## Headline vs PR #6

| metric | count |
|--------|------:|
| beats PR #6 `gdr_param` | **86 / 108** |
| beats same-run raw | **108 / 108** |
| worse than raw | **0 / 108** |

## Hard cells (bootstrap ± is 8×8192, official select)

| cell | raw | PR #6 `gdr_param` | adaptive (single) | select bootstrap |
|------|----:|------------------:|------------------:|-----------------:|
| ECD random loss \(\kappa\tau=0.1\) ideal | 0.298 | 0.373 (lose) | **0.203** | **0.208 ± 0.012** |
| ECD random comprehensive \(0.1\) ideal | 0.403 | 0.539 (lose) | **0.342** | **0.314 ± 0.013** |
| ECD opt comprehensive \(0.1\) ideal | 0.909 | **0.343** | **0.343** | **0.346 ± 0.008** |
| SNAP random comprehensive \(0.003\) ideal | 0.037 | 0.045 (lose) | **0.0369** | **0.036 ± 0.004** |

Plot: `out_research/figures/hard_cells_adaptive.png`. Adaptive = span+damped (gated) on random, default-twin `gdr_param` on optimized.

## Transfer caveat (H001)

The optimized recipe needs a **near-\(E_0\) noiseless VQE**. H000 ECD: \(E=-6.230\) vs \(E_0=-7.111\) (deficit 0.881) — comprehensive \(\kappa\tau=0.1\) **wins** (0.343). H001 ECD: first budget \(E=-3.922\), extra budget \(E=-4.819\) vs \(E_0=-6.032\) (deficit 1.21–2.11). Random+span **wins** (0.187); mid-quality opt comprehensive \(0.1\) **loses** to raw (0.888 / 0.943 vs 0.79). No `params=auto` threshold both flips H001 and keeps the H000 0.343 cell. Do **not** open 20 Hamiltonians; do **not** ship `params=auto`.

## Tried and dropped

| idea | why dropped |
|------|-------------|
| Span twins as default on **optimized** | ECD comprehensive \(0.1\): 0.416 vs PR #6 0.343 |
| Blind optimized → `gdr_residual` | Hurts comprehensive / SNAP high-\(\kappa\tau\) on default twins |
| Ungated conservative damp on all random | Flattens mid-\(\kappa\tau\) ECD; 48→45 vs PR #6 |
| `gdr_afterburn`, `gdr_blend`, energy-weighted fit | Lost microbenches; no TVD↑ / \(\lvert\Delta E\rvert\) lag |
| `gdr_interleave` | Worse than residual/oracle; leftover model error stays |
| `gdr_split` / `gdr_band` | \(\le 0.0005\) vs `gdr_param` at \(\kappa\tau=0.1\) |
| Tail-bin truncation | TVD change \(\le 0.001\) |
| `params=auto` as a default | Fires on H000 (deficit 0.881) and **regresses 0.343→0.589**; H001 still loses |

Research-only leftovers (`--params auto`, split/band/interleave) stay off the official path.

## How to reproduce the CI slice

```
python -u Error_mitigation/run_ablation.py --preset research_smoke
```

Writes only under `out_research/`. Do not point `run_mitigation_experiment.py` at `out/`.
