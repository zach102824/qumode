# Round 2 lab notebook

Branch: `cursor/gdr-round2-search-cfbc`  
Start: PR #9 `cursor/gdr-multi-h-validation-c75a` (PR #8 adaptive recipe + multi-H caches).  
Hard rules: no `src/` edits; do not overwrite `out/` or `out_smoke/`; one heavy sim at a time. Ban list: `DROPPED.md`.

**Question:** can anything that is *not* on the ban list beat frozen adaptive `gdr_select` on the hard cells, without regressing ECD opt comprehensive κτ=0.1 (0.343) or the multi-H mild H000/H004/H009 wins?

**Answer:** **No.** See `BEST.md`. Official defaults unchanged.

## Setup

- Official recipe left frozen.
- Microbench: 8 cached cells, 8192 shots, `n_train=40`, fit-only (`Error_mitigation/run_round2.py`). Wall **298.5 s**.
- Keep rule: beat same-run `gdr_select` by >0.005 TVD on at least one hard cell, and no protect-cell regression >0.003.

Same-run adaptive `gdr_select` matched the frozen headlines on the cells that matter:

| cell | this select | PR #8 / multi-H headline |
|------|------------:|-------------------------:|
| ECD random loss 0.1 | 0.2011 | 0.203 |
| ECD random comprehensive 0.1 | 0.3424 | 0.342 |
| ECD opt comprehensive 0.1 | 0.3419 | 0.343 |
| SNAP random comprehensive 0.003 (select) | 0.0417 | holdout can pick mid |
| SNAP same (gated `gdr_damped`) | **0.0369** | **0.0369** |
| H000 / H004 / H009 mild realistic | 0.1012 / 0.0763 / 0.0932 | same |

## Ideas tried

| id | method | keep? | numbers |
|----|--------|:-----:|---------|
| 1 | `gdr_anneal` (η→1 at mild κτ, holdout λ) | **drop** | mean Δ +0.0053. H004 −0.0028; 0.343 cell +0.0022. No 0.005 beat. |
| 2 | `gdr_fisher` (n(n−1) twin weights, not energy) | **drop** | mean Δ −0.0003. H004 −0.0048, mild-random comprehensive −0.0036, 0.343 −0.0007. Below keep bar. |
| 2b | Chebyshev \|α\|² grid on random | **not run** | Cheap twin-v2 *is* Fisher reweight of existing span twins. No keep → skip new sims. |
| 3a | `readout_then_gdr` (invert readout, η-only GDR) | **drop** | Collapses to `gdr_eta`. Regresses all protected optimized cells (+0.04 to +0.09). |
| 3b | `gdr_then_rtz` (mix GDR with readout_then_zne / ZNE) | **drop** | Beats ECD random loss 0.1 (0.1944 vs 0.2011) but **0.343 → 0.408**. ECD random comprehensive 0.1 0.3424 → 0.3449. |
| 4 | `gdr_select_kt` | **drop** | Identical to select on 7/8; picks fisher on ECD random comprehensive 0.003 (−0.0036). |
| 5 | `gdr_mild_residual` | **drop** | Gate (optimized loss/thermal κτ≤0.01) never fired on this set. Equals `gdr_param`. Comprehensive high-κτ correctly refused residual. |
| 6 | `gdr_eta` | **drop** | Under-parameterized on optimized comprehensive / readout. Protect regressions. |

Full table: `micro_scoreboard.md`. Raw JSON: `micro_results.json`.

## Notes

- `gdr_select` on SNAP random comprehensive 0.003 still picks `gdr_mid` (0.0417 > raw 0.0372). The shipped adaptive *scoreboard* already uses gated `gdr_damped` (0.0369) for that cell class. Round-2 did not find a holdout rule that recovers the floor without peeking at the target.
- Banned methods were not revived. `CHEAP_METHODS` / `ROUND2_METHODS` omit `gdr_full`, interleave, split, band, afterburn, blend, energy-weighted fit.
- Twin redesign that would put span on **optimized** was not run.

## Verdict

Honest negative. Ban list + adaptive remains best. Do not wire the official runner.
