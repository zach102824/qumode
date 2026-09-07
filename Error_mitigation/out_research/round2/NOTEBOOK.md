# Round 2 lab notebook

Branch: `cursor/gdr-round2-search-cfbc`  
Start: PR #9 `cursor/gdr-multi-h-validation-c75a` (PR #8 adaptive recipe + multi-H caches).  
Hard rules: no `src/` edits; do not overwrite `out/` or `out_smoke/`; one heavy sim at a time. Ban list: `DROPPED.md`.

**Question:** can anything that is *not* on the ban list beat frozen adaptive `gdr_select` on the hard cells, without regressing ECD opt comprehensive κτ=0.1 (0.343) or the multi-H mild H000/H004/H009 wins?

## Setup

- Official recipe left frozen until a method clearly wins.
- Microbench: 8 cached cells, 8192 shots, `n_train=40`, fit-only replay (`Error_mitigation/run_round2.py`).
- Keep rule: beat same-run `gdr_select` by >0.005 TVD on at least one hard cell, and no protect-cell regression >0.003.

Cells:

| id | cell | cache |
|----|------|-------|
| `ecd_rand_loss_0.1` | ECD random loss κτ=0.1 ideal | PR #8 span |
| `ecd_rand_comp_0.1` | ECD random comprehensive κτ=0.1 ideal | PR #8 span |
| `ecd_opt_comp_0.1` | ECD opt comprehensive κτ=0.1 ideal (**protect**, 0.343) | PR #8 default twins |
| `snap_rand_comp_0.003` | SNAP random comprehensive κτ=0.003 ideal (**protect**, gated floor) | PR #8 span |
| `h000_opt_comp_rr_0.003` | H000 opt comprehensive+realistic κτ=0.003 (**protect**) | multi-H |
| `h004_opt_comp_rr_0.003` | H004 same (**protect**) | multi-H |
| `h009_opt_comp_rr_0.003` | H009 same (**protect**) | multi-H |
| `ecd_rand_comp_0.003` | ECD random comprehensive κτ=0.003 ideal | PR #8 span |

## Ideas (not on the ban list)

| id | method | hypothesis |
|----|--------|------------|
| 1 | `gdr_anneal` | Pull η toward 1 at mild κτ; loosen at high κτ; holdout λ. |
| 2 | `gdr_fisher` | Reweight twins by n(n−1) Fisher scale (not energy). |
| 2b | `twin-design grid` | Chebyshev \|α\|² nodes on **random** only. New sims only if cache-replay of (1)/(2) is promising. |
| 3 | `readout_then_gdr` / `gdr_then_rtz` | Invert readout then light η-GDR; or mix GDR with `readout_then_zne`. |
| 4 | `gdr_select_kt` | Holdout selector with κτ/family gate; residual never on comprehensive high-κτ. |
| 5 | `gdr_mild_residual` | Residual only on optimized loss/thermal κτ≤0.01. |
| 6 | `gdr_eta` | Fit only (η1, η2). |

## Results

Pending `python -u Error_mitigation/run_round2.py`. Scoreboard: `micro_scoreboard.md`.

## Verdict (fill after microbench)

See `BEST.md`.
