# Matched 4-SAT: mode-finding + adaptive GDR

Fair SNAP vs ECD comparison on `four_sat_000` … `four_sat_019`.
**Energy / near-E0 bar was not used.** Primary metric is JSON `success`:

`most_likely_bitstring == ground_bitstring`.

Same trial budget, same depths, same seeds:

| | SNAP | ECD |
|---|---|---|
| depth | **L3** | **L4** |
| Hamiltonians | 20 | 20 |
| trials / H | **10** | **10** |
| overall trials | 200 | 200 |
| `seed_base` | 4000 | 4000 |
| SPSA | 200 joint | 200 joint |
| prep start | vacuum | vacuum |
| η | `sampled_tail` | `sampled_tail` |

No ECD extra near-miss / seed-5000 runs. Official GDR defaults are
unchanged. `params=auto` is not shipped. `Error_mitigation/out/` and
`Error_mitigation/out_four_sat/` were not written.

GDR circuit params: `--gibbs-pick success_then_cost` loads, for each H,
the **lowest Gibbs cost among successful trials** (every H had ≥1
success). That is the fleet pick used for reporting mode-finding, not
lowest ⟨H⟩. Pick metadata is in each
`optimized_params_*_hXXX_nd*.json` (`gibbs_trial`, `success`, `cost`).

## 1. Noiseless success %

| ansatz | depth | success | success % | mean ⟨H⟩ (footnote only) |
|--------|------:|--------:|----------:|-------------------------:|
| SNAP | L3 | **193/200** | **96.5%** | 0.574 |
| ECD | L4 | **186/200** | **93.0%** | 0.837 |

On this matched 20×10 budget ECD is within a few points of SNAP on
**mode-finding**. ECD still has higher leftover mass (mean ⟨H⟩ 0.837 vs
0.574); that is a concentration fact, not the headline metric.

Every Hamiltonian has at least one success for both ansatzes
(SNAP min 7/10 on H012; ECD min 7/10 on H005).

### Per-H success (optional)

| H | SNAP | ECD | SNAP pick ⟨H⟩ | ECD pick ⟨H⟩ |
|--:|-----:|----:|--------------:|-------------:|
| 000 | 9/10 | 9/10 | 0.376 | 0.758 |
| 001 | 10/10 | 10/10 | 0.361 | 0.755 |
| 002 | 9/10 | 10/10 | 0.391 | 0.621 |
| 003 | 10/10 | 8/10 | 0.291 | 0.671 |
| 004 | 10/10 | 9/10 | 0.633 | 0.647 |
| 005 | 10/10 | 7/10 | 0.363 | 0.795 |
| 006 | 10/10 | 10/10 | 0.200 | 0.816 |
| 007 | 10/10 | 9/10 | 0.382 | 0.714 |
| 008 | 10/10 | 10/10 | 0.262 | 0.563 |
| 009 | 9/10 | 10/10 | 0.292 | 0.766 |
| 010 | 10/10 | 10/10 | 0.147 | 0.531 |
| 011 | 10/10 | 10/10 | 0.281 | 0.708 |
| 012 | 7/10 | 8/10 | 0.418 | 0.769 |
| 013 | 10/10 | 9/10 | 0.230 | 0.565 |
| 014 | 10/10 | 10/10 | 0.444 | 0.774 |
| 015 | 9/10 | 9/10 | 0.262 | 0.681 |
| 016 | 10/10 | 10/10 | 0.335 | 0.326 |
| 017 | 10/10 | 10/10 | 0.235 | 0.579 |
| 018 | 10/10 | 9/10 | 0.398 | 0.511 |
| 019 | 10/10 | 9/10 | 0.307 | 0.552 |

JSON: `results/gibbs_four_sat_snap_matched_n10.json`,
`results/gibbs_four_sat_ecd_matched_n10.json`
(`success_metric`, `n_success`, `by_hamiltonian`).

## 2. Adaptive GDR (all 20 H, both ansatzes)

Noise: `comprehensive` + `readout_realistic`, κτ ∈ {0.003, 0.03, 0.1}.
Headline method `gdr_select` under `--twin-design adaptive` (PR #8:
span twins on random, U(0.5,1) on optimized). Scoreboard shots **8192**,
`n_train=40`. Smoke first on H000 (4000 shots / 12 twins).

**Ideal optimized circuit has GS mode on 20/20 H for both SNAP and ECD**
(`success_gs_ideal` from the noiseless histogram of the picked
(prep, x)). Mode-finding on the ideal state is not the GDR bottleneck.

### SNAP optimized — raw → `gdr_select` TVD

Wins vs raw: **20/20** at κτ=0.003, **20/20** at 0.03, **13/20** at 0.1.
Mean TVD: 0.186→0.046, 0.539→0.205, 0.766→0.639.

GS mode after mitigation (most-likely bin == ground):
**20/20**, **20/20**, **8/20** select vs raw **20/20**, **18/20**, **4/20**.

| H | κτ=0.003 | κτ=0.03 | κτ=0.1 |
|--:|----------|---------|--------|
| 000 | 0.165 → 0.029 | 0.374 → 0.057 | 0.657 → 0.207 |
| 001 | 0.188 → 0.070 | 0.557 → 0.147 | 0.771 → 0.712 |
| 002 | 0.240 → 0.050 | 0.703 → 0.223 | 0.915 → 0.828 |
| 003 | 0.202 → 0.036 | 0.606 → 0.068 | 0.769 → 0.458 |
| 004 | 0.173 → 0.087 | 0.442 → 0.242 | 0.648 → 0.831 † |
| 005 | 0.193 → 0.055 | 0.527 → 0.251 | 0.737 → 0.644 |
| 006 | 0.199 → 0.044 | 0.542 → 0.144 | 0.781 → 0.587 |
| 007 | 0.186 → 0.052 | 0.524 → 0.355 | 0.737 → 0.699 |
| 008 | 0.133 → 0.039 | 0.576 → 0.200 | 0.848 → 0.389 |
| 009 | 0.190 → 0.029 | 0.465 → 0.251 | 0.713 → 0.717 † |
| 010 | 0.208 → 0.023 | 0.629 → 0.125 | 0.855 → 0.342 |
| 011 | 0.154 → 0.046 | 0.590 → 0.286 | 0.820 → 0.742 |
| 012 | 0.182 → 0.062 | 0.472 → 0.278 | 0.701 → 0.795 † |
| 013 | 0.120 → 0.036 | 0.522 → 0.154 | 0.818 → 0.401 |
| 014 | 0.200 → 0.055 | 0.496 → 0.163 | 0.729 → 0.817 † |
| 015 | 0.157 → 0.040 | 0.388 → 0.254 | 0.620 → 0.844 † |
| 016 | 0.214 → 0.041 | 0.629 → 0.381 | 0.831 → 0.990 † |
| 017 | 0.184 → 0.024 | 0.545 → 0.072 | 0.824 → 0.179 |
| 018 | 0.208 → 0.053 | 0.569 → 0.223 | 0.745 → 0.877 † |
| 019 | 0.224 → 0.057 | 0.630 → 0.234 | 0.806 → 0.719 |

† = select does **not** beat raw. SNAP opt κτ=0.1 losses: H004, H009,
H012, H014, H015, H016, H018.

### ECD optimized — raw → `gdr_select` TVD

Wins vs raw: **20/20** at 0.003, **19/20** at 0.03, **10/20** at 0.1.
Mean TVD: 0.156→0.083, 0.422→0.266, 0.655→0.622.

GS mode after mitigation: **20/20**, **19/20**, **7/20** select vs raw
**20/20**, **19/20**, **1/20**.

Ideal GS mode remains 20/20 even on high-⟨H⟩ ECD picks (e.g. H000
⟨H⟩=0.758, H005 0.795). GDR at κτ=0.1 is mixed once leftover mass is
large; that is expected and is **not** gated by a near-E0 cut in this
run.

| H | pick ⟨H⟩ | κτ=0.003 | κτ=0.03 | κτ=0.1 |
|--:|---------:|----------|---------|--------|
| 000 | 0.758 | 0.161 → 0.083 | 0.388 → 0.382 | 0.567 → 0.961 † |
| 001 | 0.755 | 0.159 → 0.082 | 0.427 → 0.250 | 0.642 → 0.614 |
| 002 | 0.621 | 0.197 → 0.089 | 0.518 → 0.354 | 0.777 → 0.720 |
| 003 | 0.671 | 0.165 → 0.093 | 0.400 → 0.322 | 0.616 → 0.893 † |
| 004 | 0.647 | 0.146 → 0.086 | 0.426 → 0.315 | 0.620 → 0.797 † |
| 005 | 0.795 | 0.154 → 0.108 | 0.329 → 0.348 † | 0.491 → 0.644 † |
| 006 | 0.816 | 0.129 → 0.068 | 0.316 → 0.225 | 0.607 → 0.602 |
| 007 | 0.714 | 0.173 → 0.116 | 0.390 → 0.255 | 0.586 → 0.646 † |
| 008 | 0.563 | 0.136 → 0.100 | 0.421 → 0.198 | 0.727 → 0.402 |
| 009 | 0.766 | 0.118 → 0.064 | 0.325 → 0.221 | 0.528 → 0.575 † |
| 010 | 0.531 | 0.157 → 0.075 | 0.423 → 0.212 | 0.667 → 0.325 |
| 011 | 0.708 | 0.140 → 0.071 | 0.455 → 0.306 | 0.734 → 0.805 † |
| 012 | 0.769 | 0.131 → 0.065 | 0.337 → 0.187 | 0.502 → 0.608 † |
| 013 | 0.565 | 0.121 → 0.053 | 0.459 → 0.102 | 0.794 → 0.198 |
| 014 | 0.774 | 0.145 → 0.082 | 0.385 → 0.258 | 0.608 → 0.411 |
| 015 | 0.681 | 0.126 → 0.102 | 0.259 → 0.232 | 0.444 → 0.381 |
| 016 | 0.326 | 0.236 → 0.114 | 0.641 → 0.448 | 0.825 → 0.951 † |
| 017 | 0.579 | 0.144 → 0.060 | 0.443 → 0.197 | 0.789 → 0.388 |
| 018 | 0.511 | 0.184 → 0.071 | 0.549 → 0.310 | 0.778 → 0.937 † |
| 019 | 0.552 | 0.190 → 0.069 | 0.540 → 0.196 | 0.806 → 0.575 |

κτ=0.03 loss: H005 only. κτ=0.1 losses: H000, H003, H004, H005, H007,
H009, H011, H012, H016, H018.

Machine-readable scoreboard: `Error_mitigation/out_four_sat_matched/scoreboard.json`.

### Random circuits (same adaptive recipe)

SNAP random select vs raw wins: **18/20**, **16/20**, **12/20** at
0.003 / 0.03 / 0.1. ECD random: **18/20**, **20/20**, **18/20**.
Random circuits do not have GS mode (0/20), as expected.

## 3. Energy bar was not used

- Noiseless headline is **193/200 vs 186/200** bitstring successes, not
  `⟨H⟩ − E0 ≤ 0.5`.
- GDR ran on **all 20 H** for both ansatzes, including ECD H000
  (pick ⟨H⟩=0.758) which the previous near-E0 filter skipped.
- ⟨H⟩ is logged only as a footnote / pick diagnostic.

## 4. Exact commands

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

### Noiseless matched fleet

```bash
python3 scripts/Gibbs_and_adaptive_optim_SNAP.py --four-sat \
  --n-trials 10 --ndepths 3 --seed-base 4000 --workers 4 \
  --output results/gibbs_four_sat_snap_matched_n10.json

python3 scripts/Gibbs_and_adaptive_optim_ECD.py --four-sat \
  --n-trials 10 --ndepths 4 --seed-base 4000 --workers 4 \
  --output results/gibbs_four_sat_ecd_matched_n10.json
```

### Noisy adaptive GDR

Smoke (H000), then 8192 on all 20. Driver:
`Error_mitigation/out_four_sat_matched/run_gdr.sh` (sequential) or
`run_gdr_8192.py` (2-wide). Log: `COMMANDS.log`, `gdr_run.log`.

```bash
python3 -u Error_mitigation/run_mitigation_experiment.py \
  --preset full --family four_sat --instance N --ansatz snap --ndepth 3 \
  --families comprehensive --readout readout_realistic \
  --kappa-tau 0.003,0.03,0.1 \
  --gibbs-json results/gibbs_four_sat_snap_matched_n10.json \
  --gibbs-pick success_then_cost --twin-design adaptive \
  --shots 8192 --n-train 40 --params both \
  --outdir Error_mitigation/out_four_sat_matched/snap_hNNN_s8192

# ECD: --ansatz ecd --ndepth 4 \
#   --gibbs-json results/gibbs_four_sat_ecd_matched_n10.json
# Smoke: --preset smoke --shots 4000 --n-train 12 --outdir .../${ansatz}_h000_smoke
```

`--gibbs-pick` default remains `energy` (lowest ⟨H⟩). This run used
`success_then_cost` only. Default GDR twin design is still `adaptive`.

```bash
python3 Error_mitigation/out_four_sat_matched/build_scoreboard.py
```

## Thin hooks

- JSON `success` is explicitly `most_likely_bitstring == ground_bitstring`.
- Gibbs payload `by_hamiltonian` records per-H success and the
  success-then-cost pick.
- `--gibbs-pick {energy,success_then_cost}` on the mitigation runner
  (default `energy`, official path unchanged).
- `compare_histograms` reports `success_gs` / `success_gs_ideal` (mode
  of mitigated / ideal histogram vs ground QNM).
