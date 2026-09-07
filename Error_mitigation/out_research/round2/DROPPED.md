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

## Still allowed (round-2 search)

- Stronger / annealed regularization on `gdr_param` with holdout (`gdr_anneal`).
- Twin redesign **on random only**: amplitude grid / Fisher weights. Not span-on-optimized.
- Two-stage readout ↔ GDR / `readout_then_zne` mixes under `readout_realistic`.
- κτ-conditional selector trained on holdout twins (`gdr_select_kt`).
- Mild-only residual **gated** so comprehensive high-κτ never selects it (`gdr_mild_residual`).
- η-only light GDR (`gdr_eta`).
