# Round 2 conclusion

**Adaptive is unbeaten. Ban list stays banned. Official runner was not wired.**

Three passes of honest negatives. Official reported method is always `gdr_param` (random and optimized).
Twin design stays `--twin-design adaptive`; gated `gdr_damped` remains an
optional extra. `gdr_select` is ablation-only, **not** the official default.
Protect cells still
hold: ECD opt comprehensive κτ=0.1 **0.3419**, SNAP gated floor, multi-H
mild H000/H004/H009 **0.1012 / 0.0763 / 0.0932**. ECD random
loss/comprehensive 0.1 remains **0.2011 / 0.3424**.

No more kernel search unless asked. Details: `BEST.md`, `NOTEBOOK.md`,
`DROPPED.md`.

## Dropped round-2 ideas

Pre-round-2 ban list (do not revive): `gdr_full`, `gdr_interleave`,
`gdr_split`, `gdr_band`, `gdr_afterburn`, `gdr_blend`, energy-weighted fit,
ungated damp on all random, `params=auto`, span twins as default on
optimized, blind default-to-`gdr_residual` on optimized comprehensive.

| idea | pass | why dropped |
|------|-----:|-------------|
| `gdr_anneal` | 1 | Mild→identity η pull. No keep. |
| `gdr_fisher` | 1 | Closest mean Δ −0.0003. H004 mild −0.0048. |
| `gdr_eta` | 1 | Light η-only map. No 0.005 beat. |
| `gdr_select_kt` | 1 | κτ-conditional selector. No keep. |
| `gdr_mild_residual` | 1 | Gated residual. No keep. |
| `readout_then_gdr` | 1 | No keep. |
| `gdr_then_rtz` | 1 | Won ECD random loss 0.1 (0.194 vs 0.201); **0.343→0.408**. |
| Chebyshev twin grid | 1 | Not run; later twin redesigns dropped. |
| `gdr_ensemble` (K=5) | 2 | Worse on random high-κτ. SNAP opt 0.1 0.563 vs 0.590, but ECD 0.343 **+0.0042** and H009 mild **+0.012**. |
| `gdr_joint` (histogram + top-8 \|E\| bins) | 2 | Holdout almost always λ=0 (= `gdr_param`). Not energy-weighted twins. |
| SNAP Nd=2 as a different winner | 2 | Adaptive still best at κτ 0.003/0.03 on H000 comprehensive+rr. H004 SNAP not near-E0. |
| Active +10 Fisher-greedy twins | 2 | Comprehensive 0.1 −0.0058; ECD random **loss** 0.1 **+0.008 worse**. |
| Ungated shot-damp on optimized | 3 | 2048 extra α=0.25 mixed 0.343 toward raw: **0.351→0.489**. |
| `gdr_family_eta` | 3 | H004 mild −0.0033 (under bar); H009 mild **+0.0031**. Often λ=0. |
| Cross-H twin-bank M | 3 | Mean Δ vs same-H param **+0.041**. H009→H004 −0.010 does not offset H004→H000 +0.094. |
| `gdr_rl` / `gdr_rl_stop` / `gdr_rl_soft` | 3 | Holdout n_iter=8 over-unfolds. **0.343→0.464**. Soft-clip ≈ `gdr_param`. |

## Key physics takeaways

**Shot scaling.** On random high-κτ ECD, GDR’s TVD win vs raw **grows with
shots** (loss 0.1: 0.057 at 2048 → 0.097 at 8192 → 0.109 at 32768). More
shots stabilize the twin MLE; the unfold has more to work with. Adaptive
already uses that at the official 8192-shot scoreboard.

**0.343 is model error, not shot noise.** ECD optimized comprehensive κτ=0.1
is almost shot-independent (select 0.351 / 0.342 / 0.341 at 2048 / 8192 /
32768). Raw stays ~0.91. Extra shots do not close the gap. Mixing the
unfold toward the raw/safe histogram at low shots **hurts** this cell.
Leftover interleaved-vs-end-of-circuit model error, not Poisson variance.

**M is Hamiltonian-specific. Cross-H fails.** Fitting `gdr_param` on H000
twins and unfolding H004/H009 (and the reverse) at the same
comprehensive+readout_realistic κτ=0.003 yields mean Δ **+0.041** vs
same-H fit. Same-H diagonal matches; off-diagonal does not. A noise map
learned on one hybrid instance is not a drop-in map for another, even
under identical circuit noise.

## Optional research note (not a default)

Random-gated extra damp at ≤2048 shots (`gdr_shot_damp`, α=0.25 toward
safe, random circuits only) beat ECD random loss/comprehensive 0.1 by
0.0098 / 0.0053 TVD at 2048 and is identical to adaptive at 8192 (floor
0). It is a research-only schedule for low-shot random cells. It is **not**
the official default, is **not** wired into the official runner, and must
not be applied to optimized circuits (ungated, it moved 0.351→0.489 on
the 0.343 cell).
