# Round 2 best

**No beat of adaptive; recipe unchanged.**

Official frozen recipe from PR #8 stays the default. Ban list in `DROPPED.md`.
Official runner was **not** rewired.

Keep bar: beat same-run `gdr_select` by >0.005 TVD on a hard cell, with no
>0.003 regression on protect cells (ECD opt comprehensive 0.1; SNAP gated
floor; multi-H mild H000/H004/H009). ECD random loss/comprehensive 0.1 must
not get worse.

## Pass 1 (fit-only, 8 cached cells)

See `micro_scoreboard.md`. Closest: `gdr_fisher` mean Δ −0.0003. `gdr_then_rtz`
won ECD random loss 0.1 (0.194 vs 0.201) and **regressed 0.343 → 0.408**.

## Pass 2

### 1. Shot-robust ensemble (K=5 twin bootstrap)

Average of 5 `gdr_param` unfolds. `stage_b_scoreboard.md`.

| cell | select | ensemble | member mean±std |
|------|-------:|---------:|----------------:|
| ECD random loss 0.1 | **0.2011** | 0.2188 | 0.229±0.026 |
| ECD random comprehensive 0.1 | **0.3424** | 0.3831 | 0.396±0.035 |
| ECD opt comprehensive 0.1 \* | **0.3419** | 0.3461 | 0.346±0.004 |
| H000 / H004 / H009 mild \* | 0.1012 / 0.0763 / 0.0932 | 0.1012 / 0.0784 / 0.1050 | — |
| SNAP opt comprehensive+rr 0.1 | 0.5903 | **0.5633** | 0.563±0.030 |

**drop.** Beats SNAP opt 0.1 by 0.027 but regresses the 0.343 cell (+0.0042)
and H009 mild (+0.012). Worse on both ECD random high-κτ cells. Variance on
random high-κτ is large (σ≈0.03); averaging does not beat damped/select.

### 2. SNAP Nd=2 optimized transfer (H000)

H004 SNAP is **not** near-E0 (E=−3.93 vs E0=−5.35, deficit 1.41) — skipped.
H000 SNAP deficit 1.93. Default twins, comprehensive + readout_realistic:

| κτ | raw | adaptive select (= `gdr_param`) | ensemble | joint |
|---:|----:|--------------------------------:|---------:|------:|
| 0.003 | 0.1411 | **0.0230** | 0.0230 | 0.0230 |
| 0.03 | 0.5710 | **0.2030** | 0.2239 | 0.2030 |
| 0.1 | 0.7952 | 0.5903 | 0.5633 | 0.5904 |

Adaptive is still best at 0.003 and 0.03. Ensemble’s 0.1 win is the same
method that regresses ECD 0.343 — not a SNAP-only ship.

### 3. Joint histogram + top-8 |E| bins (holdout λ)

**drop.** Holdout almost always picks λ=0 (equals `gdr_param`). Max |Δ| vs
select on protect cells ≈ 0. Distinct from banned energy-weighted *twins*;
still not a keep.

### 4. Active +10 Fisher-greedy Gaussian twins

New sims of 10 extra t_free=0 twins (pool of 80 noiseless, η-Fisher ×
diversity). Refit on 50 twins, same target shots. `active_scoreboard.md`.

| cell | select40 | gdr50 | select50 | Δ select50−select40 |
|------|---------:|------:|---------:|--------------------:|
| ECD random loss 0.1 | **0.2011** | 0.2187 | 0.2094 | **+0.0083 (worse)** |
| ECD random comprehensive 0.1 | 0.3424 | 0.3767 | 0.3366 | −0.0058 |

The comprehensive cell is a 0.0058 select win, but ECD random **loss** 0.1
gets worse — that hard cell must not regress. **drop.**

## Pass 3

### 1. Shot sweep `{2048, 8192, 32768}` (`shots_scoreboard.md`)

Not a new M. Same caches, re-observed. Adaptive vs raw:

| cell | 2048 raw / sel | 8192 raw / sel | 32768 raw / sel |
|------|---------------:|---------------:|----------------:|
| ECD random loss 0.1 | 0.3091 / **0.2521** | 0.2983 / **0.2011** | 0.2950 / **0.1864** |
| ECD random comprehensive 0.1 | 0.4186 / **0.3445** | 0.4031 / **0.3424** | 0.4076 / **0.3039** |
| ECD opt comprehensive 0.1 \* | 0.9060 / **0.3514** | 0.9086 / **0.3419** | 0.9081 / **0.3414** |
| SNAP random comprehensive 0.003 \* | 0.0607 / 0.0594 | **0.0372** / 0.0415 | 0.0334 / **0.0289** |

GDR’s win on random high-κτ **grows with shots** (loss: 0.057 → 0.097 →
0.109 vs raw). The 0.343 cell is almost shot-independent — leftover model
error, not shot noise. SNAP gated floor: select is slightly **worse** than
raw at 8192 (why the floor exists) and recovers at 32768.

Ungated extra damp (α=0.25 toward safe) at 2048 beat both random high-κτ
cells (−0.0098 / −0.0053) but **regressed 0.351 → 0.489** on the optimized
cell (mixing toward raw≈0.91). **Gated to random only:**

| cell | 2048 select | 2048 shot-damp | Δ |
|------|------------:|---------------:|--:|
| ECD random loss 0.1 | 0.2521 | **0.2423** | **−0.0098** |
| ECD random comprehensive 0.1 | 0.3445 | **0.3392** | **−0.0053** |
| ECD opt comprehensive 0.1 \* | **0.3514** | 0.3514 | 0 |
| SNAP random comprehensive 0.003 \* | 0.0594 | 0.0581 | −0.0012 |

8192 / 32768: floor=0, identical to adaptive. **Optional research
schedule** (not official): extra α=0.25 toward safe on **random** circuits
at ≤2048 shots. Official 8192 recipe unchanged. Not wired into the
official runner.

### 2. Family-conditional η prior (`stage_c_scoreboard.md`)

Opposite of dropped anneal: looser η ridge at κτ=0.003 (weight 0.3),
tighter at 0.1 (weight 10), oracle prior, holdout λ. Applied only under
comprehensive+readout_realistic. 0.343 (ideal readout) is a no-op.

| cell | select | family_eta | λ | Δ |
|------|-------:|-----------:|--:|--:|
| H000 mild rr \* | **0.1012** | 0.1012 | 0 | 0 |
| H004 mild rr \* | 0.0763 | 0.0730 | 0.03 | −0.0033 |
| H009 mild rr \* | **0.0932** | 0.0963 | 0.001 | **+0.0031** |
| SNAP opt rr 0.003 | **0.0230** | 0.0252 | 0.1 | +0.0022 |
| H000 / H004 / H009 rr 0.1 | 0.8688 / 0.7612 / 0.9670 | 0.8719 / 0.7612 / 0.9670 | 0.001 / 0 / 0 | +0.003 / 0 / 0 |
| ECD opt 0.1 \* (ideal) | **0.3419** | 0.3419 | — | 0 |

**drop.** Closest: H004 mild −0.0033 (under the 0.005 bar). H009 mild is a
protect regression. Holdout often picks λ=0.

### 3. Cross-H twin bank (`xfer_scoreboard.md`)

Fit `gdr_param` M on one H’s twins, unfold another H’s target.
comprehensive+readout_realistic κτ=0.003.

| source → target | xfer | same-H param | Δ |
|-----------------|-----:|-------------:|--:|
| H000 → H004 | 0.0918 | **0.0763** | +0.016 |
| H000 → H009 | 0.1486 | **0.0932** | +0.055 |
| H004 → H000 | 0.1954 | **0.1012** | +0.094 |
| H004 → H009 | 0.1261 | **0.0932** | +0.033 |
| H009 → H000 | 0.1584 | **0.1012** | +0.057 |
| H009 → H004 | **0.0662** | 0.0763 | −0.010 |

Mean Δ vs same-H param **+0.041**. One pair (H009→H004) beats by 0.010;
the reverse and the others regress. The noise map is **not** portable
across Hamiltonians. **drop.**

### 4. RL soft-clip / holdout early-stop

Pure unfold tweak on the same `gdr_param` M. Holdout almost always picks
`n_iter=8` (vs default 80). Soft-clip alone ≈ `gdr_param`. Early-stop
beats some high-κτ rr cells (H000/H009 rr 0.1) but **regresses 0.343 →
0.464** and H009 mild +0.022. **drop.**

## Verdict

Official 8192 adaptive recipe still unbeaten. Ban list stays banned.
Optional 2048-shot extra damp on **random** circuits is a research note
only — not a default change.

Reproduce:

```
python -u Error_mitigation/run_round2.py --stage micro
python -u Error_mitigation/run_round2.py --stage b
python -u Error_mitigation/run_round2.py --stage active
python -u Error_mitigation/run_round2.py --stage c
```
