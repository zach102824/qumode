# Adaptive recipe vs PR #6

Hybrid scoreboard (no new target shots beyond `phase3` + `opt_default`):

- **random** cells: `phase3` span twins + `gdr_damped` (conservative floor on comprehensive κτ≤0.003)
- **optimized** cells: `opt_default` U(0.5,1) twins + `gdr_param`

| metric | count |
|--------|------:|
| beats PR #6 `gdr_param` | **86 / 108** |
| beats same-run raw | **108 / 108** |
| worse than raw | **0 / 108** after gated floor on random comprehensive κτ≤0.003 |

Headline cells the original plan called out:

| cell | raw | PR #6 gdr | span gdr | adaptive |
|------|----:|----------:|---------:|---------:|
| ECD random loss κτ=0.1 ideal | 0.298 | 0.373 | 0.209 | **0.203** |
| ECD random comprehensive κτ=0.1 ideal | 0.403 | 0.539 | 0.377 | **0.342** |
| ECD random thermal κτ=0.1 ideal | 0.336 | 0.407 | 0.263 | **0.258** |
| ECD opt loss κτ=0.003 ideal | 0.038 | **0.0075** | 0.0096 | 0.0093 |
| ECD opt loss κτ=0.003 strong | 0.163 | **0.0075** | 0.0165 | 0.0133 |
| ECD opt thermal κτ=0.1 ideal | 0.770 | 0.270 | 0.318 | **0.277** |
| ECD opt comprehensive κτ=0.1 ideal | 0.909 | **0.343** | 0.416 | **0.343** |
| SNAP random comprehensive κτ=0.003 ideal | 0.037 | 0.045 | 0.042 (damped lose) | **0.0369** (gated floor) |
| SNAP random loss κτ=0.003 ideal | 0.028 | 0.022 | 0.022 | **0.022** |
| SNAP opt comprehensive κτ=0.1 ideal | 0.791 | 0.632 | 0.662 | **0.611** |

`opt_default` (54 optimized cells): default `gdr_param` vs span `gdr_param` is **35 better / 19 tie / 0 worse**.

Summary plot of the four headline cells: `out_research/figures/hard_cells_adaptive.png`.
