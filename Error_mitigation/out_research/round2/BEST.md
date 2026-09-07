# Round 2 best

**No beat of adaptive; recipe unchanged.**

Official frozen recipe from PR #8 stays the default:

- `--twin-design adaptive` (span on random, \(U(0.5,1)\) on optimized)
- `gdr_param` on optimized
- gated `gdr_damped` on random comprehensive \(\kappa\tau\le 0.003\)
- `gdr_select` holdout on random
- `n_train=40`, 8192 shots when scoring

Round-2 search (8 cached hard cells, fit-only, 8192 shots, `fit_maxiter=120`,
wall 5.0 min) found **no** method that beat same-run adaptive `gdr_select` by
>0.005 TVD without regressing a protected cell. Ban list in `DROPPED.md`.
Official runner was **not** rewired.

## Same-run adaptive vs new methods (TVD)

Adaptive column is `gdr_select`. Protect cells are marked \*.

| cell | raw | adaptive select | anneal | fisher | eta | select_kt | mild_res | ro→gdr | gdr→rtz |
|------|----:|----------------:|-------:|-------:|----:|----------:|---------:|-------:|--------:|
| ECD random loss 0.1 | 0.2983 | **0.2011** | 0.2083 | 0.2083 | 0.2179 | 0.2011 | 0.2087 | 0.2179 | 0.1944 |
| ECD random comprehensive 0.1 | 0.4031 | **0.3424** | 0.3763 | 0.3436 | 0.4177 | 0.3424 | 0.3765 | 0.4177 | 0.3449 |
| ECD opt comprehensive 0.1 \* | 0.9086 | **0.3419** | 0.3440 | 0.3412 | 0.3864 | 0.3419 | 0.3419 | 0.3864 | 0.4085 |
| SNAP random comprehensive 0.003 \* | 0.0372 | 0.0417† | 0.0415 | 0.0416 | 0.0417 | 0.0417 | 0.0415 | 0.0417 | 0.0407 |
| H000 opt comprehensive+realistic 0.003 \* | 0.3247 | **0.1012** | 0.1012 | 0.1013 | 0.1882 | 0.1012 | 0.1012 | 0.1930 | 0.1012 |
| H004 opt comprehensive+realistic 0.003 \* | 0.2007 | **0.0763** | 0.0735 | 0.0715 | 0.1254 | 0.0763 | 0.0763 | 0.1196 | 0.0726 |
| H009 opt comprehensive+realistic 0.003 \* | 0.2764 | **0.0932** | 0.0956 | 0.0917 | 0.1556 | 0.0932 | 0.0932 | 0.1534 | 0.0953 |
| ECD random comprehensive 0.003 | 0.0816 | **0.0723** | 0.0723 | 0.0686 | 0.0692 | 0.0686 | 0.0723 | 0.0692 | 0.0723 |

† Holdout `gdr_select` picked `gdr_mid` (0.0417, worse than raw). The PR #8
adaptive *scoreboard* number 0.0369 is gated `gdr_damped` on this cell class
(this run: **0.0369**). Round-2 methods did not beat that floor.

## Why nothing shipped

| method | verdict | why |
|--------|---------|-----|
| `gdr_anneal` | drop | Mean ΔTVD +0.0053 vs select. No 0.005 beat. |
| `gdr_fisher` | drop | Closest: mean Δ −0.0003, no protect regression, but no 0.005 beat (best H004 −0.0048). |
| `gdr_eta` | drop | Regresses all four protect-optimized cells. |
| `gdr_select_kt` | drop | Matches select on 7/8; Fisher pick on one mild-random cell (−0.0036). |
| `gdr_mild_residual` | drop | Gate never fired (no optimized loss/thermal cells in the set). Falls back to `gdr_param`. |
| `readout_then_gdr` | drop | Same failure mode as `gdr_eta` under readout. |
| `gdr_then_rtz` | drop | Beats ECD random loss 0.1 (0.1944 vs 0.2011) but **regresses the 0.343 cell to 0.408**. Also slightly worse on ECD random comprehensive 0.1. |
| Chebyshev \|α\|² grid | not run | Fisher reweight of existing span twins is the cheap twin-v2. No keep → no new density-matrix pass. |

Reproduce: `python -u Error_mitigation/run_round2.py`
(uses caches under `out_research/cache/` and `out_research/multi_h/cache/`).
