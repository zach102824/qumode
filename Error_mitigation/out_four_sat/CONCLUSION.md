# 4-SAT noiseless Gibbs + noisy adaptive GDR

Pipeline on `four_sat_000` … `four_sat_019` (`Hamiltonians/four_sat/`).
Official GDR defaults are unchanged. `params=auto` is not shipped.
`Error_mitigation/out/` was not written. Do not merge this PR.

Headline method is **`gdr_select`** under `--twin-design adaptive`
(span twins on random, U(0.5,1) on optimized; gated damped as in PR #8).
Noise: `comprehensive` + `readout_realistic`, κτ ∈ {0.003, 0.03, 0.1}.
Scoreboard shots: **8192**, `n_train=40`. Smoke used 4000 shots / 12 twins.

## Scout depths

Near-E0 gate: `energy_physical − E0 ≤ 0.5` (4-SAT E0 = 0).
Most-likely ground-state bitstring is **not** enough: ECD often has `success_gs=true` with ⟨H⟩ ~ 1.

### SNAP (H000, 5 trials, seed 3100, 200 joint SPSA)

| depth | best ⟨H⟩ | trials ≤ 0.5 |
|------:|----------:|-------------:|
| L1 | 1.000 | 0/5 |
| L2 | 0.602 | 0/5 |
| **L3** | **0.344** | **3/5** |
| L4 | 0.407 | 3/5 |

Locked **SNAP L3**: smallest depth that reliably produced near-E0 trials.
L2 never cleared 0.5. L4 also had 3/5 passes but was not smaller.
1-trial seed-3000 scout already showed L3 0.354 / L4 0.359 vs L2 0.861.

### ECD (H000, 8 trials, seed 3100, 200 joint SPSA)

| depth | best ⟨H⟩ | trials ≤ 0.5 |
|------:|----------:|-------------:|
| L1 | 0.996 | 0/8 |
| L2 | 0.954 | 0/8 |
| L3 | 0.768 | 0/8 |
| L4 | 0.814 | 0/8 |
| L5 | 0.774 | 0/8 |
| L6 | 0.757 | 0/8 |
| L7 | 0.860 | 0/8 |
| L8 | 0.961 | 0/8 |

H000 ECD never cleared 0.5. Best was **0.757 at L6**.
A 400-step L6 retry (3 trials, seed 3300) was worse (best 0.851).

1-trial alt-H scout (ids 1–19, L4/L5/L6, seed 3200) produced one pass:
**H018 L4 = 0.464**. That locked **ECD L4** as the smallest depth with a pass.

## Near-E0 fleet

8 trials, seed 4000, 200 joint SPSA, workers=4. Filter:
`Error_mitigation/out_four_sat/near_e0_filter.json`.

| ansatz | depth | pass / 20 | passers (⟨H⟩) |
|--------|------:|----------:|---------------|
| SNAP | L3 | **20/20** | all H000–H019; best H010 0.147, H000 0.376 |
| ECD | L4 | **2 + 1 borderline** | H016 0.326, H010 0.483; H018 0.501 kept under ~0.5 |

ECD extra 8 trials (seed 5000) on near-miss ids 2,4,8,10,13,18,19.
That is how H010 (0.483) and H018 (0.501) entered the GDR set.
H000 ECD fleet best remained 0.831 — **not** sent to noisy GDR.

SNAP fleet ⟨H⟩ (best trial):

H000 0.376, H001 0.284, H002 0.439, H003 0.291, H004 0.365, H005 0.363, H006 0.200, H007 0.360, H008 0.262, H009 0.294, H010 0.147, H011 0.393, H012 0.417, H013 0.230, H014 0.444, H015 0.309, H016 0.306, H017 0.235, H018 0.313, H019 0.386.

ECD fleet ⟨H⟩ (best trial, including extra seed-5000 where used):

H000 0.831, H001 0.839, H002 0.534, H003 0.671, H004 0.551, H005 0.848, H006 0.816, H007 0.714, H008 0.563, H009 0.726, H010 0.483, H011 0.708, H012 0.769, H013 0.565, H014 0.774, H015 0.681, H016 0.326, H017 0.579, H018 0.501, H019 0.547.

## Noisy adaptive GDR (8192 shots)

GDR loaded Gibbs **prep + x** from `results/gibbs_four_sat_{snap,ecd}.json`
(no energy L-BFGS-B). Twins skip the vacuum product-state assertion when
prep is not vacuum (`Error_mitigation/twins.py`); vacuum path unchanged.

Coverage: SNAP **20/20** H @8192; ECD passers H016, H010, H018 smoke then 8192;
SNAP H000 smoke then 8192. Remaining SNAP H skipped smoke and went straight to 8192.

### SNAP optimized — raw → `gdr_select`

Wins vs raw: **20/20** at κτ=0.003, **20/20** at 0.03, **17/20** at 0.1.
Mean TVD: 0.192→0.046, 0.548→0.212, 0.772→0.601 (Δ −0.146 / −0.336 / −0.170).

| H | κτ=0.003 | κτ=0.03 | κτ=0.1 |
|--:|----------|---------|--------|
| 000 | 0.165 → 0.029 | 0.374 → 0.057 | 0.657 → 0.207 |
| 001 | 0.208 → 0.051 | 0.636 → 0.322 | 0.847 → 0.800 |
| 002 | 0.236 → 0.048 | 0.645 → 0.220 | 0.846 → 0.803 |
| 003 | 0.202 → 0.036 | 0.606 → 0.068 | 0.769 → 0.458 |
| 004 | 0.235 → 0.040 | 0.632 → 0.251 | 0.828 → 0.794 |
| 005 | 0.193 → 0.056 | 0.527 → 0.251 | 0.737 → 0.644 |
| 006 | 0.199 → 0.044 | 0.542 → 0.144 | 0.781 → 0.587 |
| 007 | 0.195 → 0.056 | 0.536 → 0.358 | 0.733 → 0.687 |
| 008 | 0.133 → 0.039 | 0.576 → 0.201 | 0.848 → 0.389 |
| 009 | 0.191 → 0.032 | 0.465 → 0.251 | 0.713 → 0.719 † |
| 010 | 0.208 → 0.023 | 0.629 → 0.125 | 0.855 → 0.342 |
| 011 | 0.138 → 0.029 | 0.555 → 0.169 | 0.778 → 0.676 |
| 012 | 0.177 → 0.078 | 0.445 → 0.345 | 0.638 → 0.601 |
| 013 | 0.120 → 0.036 | 0.522 → 0.154 | 0.818 → 0.401 |
| 014 | 0.200 → 0.055 | 0.496 → 0.162 | 0.729 → 0.817 † |
| 015 | 0.180 → 0.050 | 0.434 → 0.263 | 0.662 → 0.504 |
| 016 | 0.217 → 0.047 | 0.610 → 0.346 | 0.809 → 0.978 † |
| 017 | 0.184 → 0.024 | 0.545 → 0.072 | 0.824 → 0.179 |
| 018 | 0.220 → 0.078 | 0.588 → 0.274 | 0.771 → 0.659 |
| 019 | 0.238 → 0.073 | 0.601 → 0.213 | 0.786 → 0.778 |

† = select does **not** beat raw. SNAP opt κτ=0.1 losses: H009 0.713→0.719,
H014 0.729→0.817, H016 0.809→0.978 (unfold blow-up).

H000 (the original scout): **0.165 / 0.374 / 0.657 → 0.029 / 0.057 / 0.207**.

### ECD optimized — raw → `gdr_select`

| H | ⟨H⟩ noiseless | κτ=0.003 | κτ=0.03 | κτ=0.1 |
|--:|--------------:|----------|---------|--------|
| 010 | 0.483 | 0.163 → 0.088 | 0.481 → 0.225 | 0.785 → 0.417 |
| 016 | 0.326 | 0.236 → 0.114 | 0.641 → 0.448 | 0.825 → 0.951 † |
| 018 | 0.501 | 0.209 → 0.106 | 0.515 → 0.619 † | 0.645 → 0.910 † |

H010 beats raw at all three κτ. H016 loses only at 0.1 (0.825→0.951).
Borderline H018 loses at 0.03 and 0.1 (0.515→0.619, 0.645→0.910).
Smoke (4000 shots) agreed on those win/loss cells.

### Random circuits (same adaptive recipe)

SNAP random select vs raw wins: **18/20** at 0.003, **16/20** at 0.03, **12/20** at 0.1.
H000 random κτ=0.1 still loses (0.328→0.392), same pattern as mixed p-spin high-κτ random.
ECD random: only H010 κτ=0.003 loses (0.075→0.081); H016 and H018 random beat raw at all three κτ.

## Failures and limits

- **ECD depth:** H000 never near-E0 at L1–L8 (best 0.757). Extra SPSA (400 steps) did not help.
  Optimized GDR is only claimed on the filtered ECD set (H016, H010, H018).
- **Near-E0 bar is load-bearing.** H018 at 0.501 already fails optimized GDR at κτ≥0.03.
  Sending H000 ECD (⟨H⟩=0.831) would be the known high-noise unfold failure, so it was skipped.
- **SNAP κτ=0.1 is not universal.** 3/20 optimized H lose (H009 barely; H014, H016 clearly).
  Mean still improves because the other 17 wins are large (e.g. H000 0.657→0.207, H017 0.824→0.179).
- **Random high-κτ** remains the weaker cell: SNAP random 0.1 only 12/20 wins.
- **Twins:** official vacuum product-state TVD < 1e-6 assert is skipped when Gibbs prep is
  the initial state. GDR still fits on statevector histograms. Vacuum circuits unchanged.
- **Not changed:** official runner defaults, `Error_mitigation/out/`, `params=auto`.
- Plotter warning `No artists with labels found to put in legend` is cosmetic.

## Exact commands

Env for all of the below:

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

### Noiseless scout and fleet

```bash
# SNAP H000 scout (1 trial, then 5 trials)
python3 scripts/Gibbs_and_adaptive_optim_SNAP.py --four-sat --hamiltonian-ids 0 \
  --n-trials 1 --ndepths 1 2 3 4 --seed-base 3000 --workers 4 \
  --output results/gibbs_four_sat_snap_scout.json
python3 scripts/Gibbs_and_adaptive_optim_SNAP.py --four-sat --hamiltonian-ids 0 \
  --n-trials 5 --ndepths 1 2 3 4 --seed-base 3100 --workers 4 \
  --output results/gibbs_four_sat_snap_scout_n5.json

# ECD H000 scout (1 trial, then 8 trials, depths 1–8)
python3 scripts/Gibbs_and_adaptive_optim_ECD.py --four-sat --hamiltonian-ids 0 \
  --n-trials 1 --ndepths 1 2 3 4 5 6 7 8 --seed-base 3000 --workers 4 \
  --output results/gibbs_four_sat_ecd_scout.json
python3 scripts/Gibbs_and_adaptive_optim_ECD.py --four-sat --hamiltonian-ids 0 \
  --n-trials 8 --ndepths 1 2 3 4 5 6 7 8 --seed-base 3100 --workers 4 \
  --output results/gibbs_four_sat_ecd_scout_n8.json

# ECD alt-H scout (lock L4 via H018 0.464)
python3 scripts/Gibbs_and_adaptive_optim_ECD.py --four-sat \
  --hamiltonian-ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --n-trials 1 --ndepths 4 5 6 --seed-base 3200 --workers 4 \
  --output results/gibbs_four_sat_ecd_scout_altH.json

# ECD H000 L6 extra budget (did not help)
python3 scripts/Gibbs_and_adaptive_optim_ECD.py --four-sat --hamiltonian-ids 0 \
  --n-trials 3 --ndepths 6 --outer-iter 400 --seed-base 3300 --workers 4 \
  --output results/gibbs_four_sat_ecd_scout_L6_400.json

# Fleet
python3 scripts/Gibbs_and_adaptive_optim_ECD.py --four-sat --n-trials 8 --ndepths 4 \
  --seed-base 4000 --workers 4 --output results/gibbs_four_sat_ecd.json
python3 scripts/Gibbs_and_adaptive_optim_SNAP.py --four-sat --n-trials 8 --ndepths 3 \
  --seed-base 4000 --workers 4 --output results/gibbs_four_sat_snap.json
python3 scripts/Gibbs_and_adaptive_optim_ECD.py --four-sat \
  --hamiltonian-ids 2 4 8 10 13 18 19 --n-trials 8 --ndepths 4 --seed-base 5000 --workers 4 \
  --output results/gibbs_four_sat_ecd_extra_nearmiss.json
```

### Noisy adaptive GDR

Driver: `Error_mitigation/out_four_sat/run_gdr.sh` (skips dirs that already have `results.json`).
Log: `Error_mitigation/out_four_sat/COMMANDS.log`.

```bash
python3 -u Error_mitigation/run_mitigation_experiment.py \
  --preset full --family four_sat --instance N --ansatz snap --ndepth 3 \
  --families comprehensive --readout readout_realistic \
  --kappa-tau 0.003,0.03,0.1 --gibbs-json results/gibbs_four_sat_snap.json \
  --twin-design adaptive --shots 8192 --n-train 40 --params both \
  --outdir Error_mitigation/out_four_sat/snap_hNNN_s8192

# ECD passers: --ansatz ecd --ndepth 4 --gibbs-json results/gibbs_four_sat_ecd.json
# Smoke: --preset smoke --shots 4000 --n-train 12 --outdir .../${ansatz}_hNNN_smoke
```

Outputs live under `Error_mitigation/out_four_sat/{snap,ecd}_hXXX_{smoke,s8192}/`.

