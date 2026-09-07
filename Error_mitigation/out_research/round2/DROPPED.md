# Round 2 ban list

Do **not** spend runtime promoting these. Code may still exist for replay of
old tags. They are off the official path and off the round-2 search.

Frozen baseline to beat: PR #8 adaptive recipe (`--twin-design adaptive`,
`gdr_param` on optimized, gated `gdr_damped` / `gdr_select` on random,
`n_train=40`, 8192 shots). Scoreboard: 86/108 vs PR #6 `gdr_param`, 108/108 vs
raw on H000.

## Methods (do not revive as contenders)

| item | why it is banned |
|------|------------------|
| `gdr_full` | Unconstrained Kronecker ALS. Over-parametrized; not a contender. |
| `gdr_interleave` | Two-stage loss-hop-loss. Lost to residual/oracle; leftover model error stays. |
| `gdr_split` | Per-register hops on `gdr_param`. ≤0.0005 TVD vs `gdr_param` at κτ=0.1. |
| `gdr_band` | n±1 band residual. Same: ≤0.0005, not a keep. |
| `gdr_afterburn` | ≈ `gdr_residual` on the cells that matter; 0/23 Phase-3 Gaussian holdout picks. |
| `gdr_blend` | Mix GDR/oracle. Lost microbenches. |

## Fit / selector choices (do not revive)

| item | why it is banned |
|------|------------------|
| Energy-weighted fit (`gdr_energy`) | No TVD↑ / \|ΔE\|-down lag. Not a keep. |
| Ungated conservative damp on **all** random | Flattened mid-κτ ECD wins (48→45 vs PR #6). Floor stays **gated** (random + comprehensive + κτ≤0.003). |
| `params=auto` | Fires on H000 (deficit 0.881) and **regresses 0.343→0.589**. H001 still loses. Research flag only. |

## Twin / recipe choices (do not revive)

| item | why it is banned |
|------|------------------|
| Span twins as default on **optimized** circuits | ECD opt comprehensive κτ=0.1: 0.416 vs PR #6 **0.343**. `opt_default`: 35 better / 19 tie / **0 worse** for U(0.5,1) vs span. |
| Blind default-to-`gdr_residual` on optimized comprehensive | Hurts comprehensive and SNAP high-κτ on default twins (ECD comprehensive 0.1: residual ~0.41 vs `gdr_param` 0.34). |

## Pass 1 also dropped (not a keep)

`gdr_anneal`, `gdr_fisher`, `gdr_eta`, `gdr_select_kt`, `gdr_mild_residual`,
`readout_then_gdr`, `gdr_then_rtz`. Closest: `gdr_fisher` mean Δ −0.0003.
`gdr_then_rtz` won ECD random loss 0.1 (0.194 vs 0.201) and **regressed
0.343→0.408**.

## Pass 2 also dropped (not a keep)

| item | why it is dropped |
|------|-------------------|
| `gdr_ensemble` (K=5 twin bootstrap) | Worse on ECD random high-κτ. SNAP opt 0.1 0.563 vs 0.590, but ECD 0.343 **+0.0042** and H009 mild **+0.012**. |
| `gdr_joint` (histogram + top-8 \|E\| bins) | Holdout almost always picks λ=0 (= `gdr_param`). Distinct from banned energy-weighted twins; still not a keep. |
| SNAP Nd=2 opt transfer as a different winner | Adaptive still best at κτ 0.003/0.03 on H000 comprehensive+rr. H004 SNAP not near-E0 (skipped). |
| Active +10 Fisher-greedy Gaussian twins | select50 comprehensive 0.1 −0.0058, but ECD random **loss** 0.1 **+0.008 worse**. |

## Still allowed (round-2 search)

Pass-1/pass-2 ideas above are **not** still allowed as contenders. Frozen
adaptive remains the bar. Do not revive the original ban list. No new search
unless asked.
