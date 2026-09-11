# Matched 4-SAT: mode-finding + adaptive GDR

Fair SNAP vs ECD comparison on `four_sat_000` … `four_sat_019`.
**Energy / near-E0 bar was not used.** Primary metric is JSON `success`:

`most_likely_bitstring == ground_bitstring`.

Same trial budget, same seeds, three matched depth pairs:

| | SNAP L3 | ECD L4 | SNAP L2 | ECD L3 | SNAP L1 | ECD L2 |
|---|---|---|---|---|---|---|
| depth | **L3** | **L4** | **L2** | **L3** | **L1** | **L2** |
| Hamiltonians | 20 | 20 | 20 | 20 | 20 | 20 |
| trials / H | **10** | **10** | **10** | **10** | **10** | **10** |
| overall trials | 200 | 200 | 200 | 200 | 200 | 200 |
| `seed_base` | 4000 | 4000 | 4000 | 4000 | 4000 | 4000 |
| SPSA | 200 joint | 200 joint | 200 joint | 200 joint | 200 joint | 200 joint |
| prep start | vacuum | vacuum | vacuum | vacuum | vacuum | vacuum |
| η | `sampled_tail` | `sampled_tail` | `sampled_tail` | `sampled_tail` | `sampled_tail` | `sampled_tail` |

No ECD extra near-miss / seed-5000 runs. Official GDR defaults are
unchanged. `params=auto` is not shipped. `Error_mitigation/out/` and
`Error_mitigation/out_four_sat/` were not written.

GDR circuit params: `--gibbs-pick success_then_cost` loads, for each H,
the **lowest Gibbs cost among successful trials**. If a Hamiltonian has
zero noiseless successes, the pick falls back to lowest Gibbs cost among
all 10 trials (`pick_rule=cost_fallback`, `pick_fallback=true` in
`by_hamiltonian`). Every H had ≥1 success on all three pairs, so the
fallback was not used. That is the fleet pick used for reporting
mode-finding, not lowest ⟨H⟩. Pick metadata is in each
`optimized_params_*_hXXX_nd*.json` (`gibbs_trial`, `success`, `cost`).

Headline noiseless success (bitstring equality, not energy):

| pair | SNAP | ECD |
|---|---|---|
| L3 vs L4 | **193/200 (96.5%)** | **186/200 (93.0%)** |
| L2 vs L3 | **177/200 (88.5%)** | **162/200 (81.0%)** |
| L1 vs L2 | **79/200 (39.5%)** | **123/200 (61.5%)** |

---

## SNAP L3 vs ECD L4

### 1. Noiseless success %

| ansatz | depth | success | success % | mean ⟨H⟩ (footnote only) |
|--------|------:|--------:|----------:|-------------------------:|
| SNAP | L3 | **193/200** | **96.5%** | 0.574 |
| ECD | L4 | **186/200** | **93.0%** | 0.837 |

On this matched 20×10 budget ECD is within a few points of SNAP on
**mode-finding**. ECD still has higher leftover mass (mean ⟨H⟩ 0.837 vs
0.574); that is a concentration fact, not the headline metric.

Every Hamiltonian has at least one success for both ansatzes
(SNAP min 7/10 on H012; ECD min 7/10 on H005).

#### Per-H success (optional)

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

### 2. Adaptive GDR (all 20 H, both ansatzes)

Noise: `comprehensive` + `readout_realistic`, κτ ∈ {0.003, 0.03, 0.1}.
Headline method `gdr_select` under `--twin-design adaptive` (PR #8:
span twins on random, U(0.5,1) on optimized). Scoreboard shots **8192**,
`n_train=40`. Smoke first on H000 (4000 shots / 12 twins).

**Ideal optimized circuit has GS mode on 20/20 H for both SNAP L3 and
ECD L4** (`success_gs_ideal` from the noiseless histogram of the picked
(prep, x)). Mode-finding on the ideal state is not the GDR bottleneck.

#### SNAP L3 optimized — raw → `gdr_select` TVD

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

† = select does **not** beat raw. SNAP L3 opt κτ=0.1 losses: H004, H009,
H012, H014, H015, H016, H018.

#### ECD L4 optimized — raw → `gdr_select` TVD

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

#### Random circuits (same adaptive recipe)

SNAP L3 random select vs raw wins: **18/20**, **16/20**, **12/20** at
0.003 / 0.03 / 0.1. ECD L4 random: **18/20**, **20/20**, **18/20**.
Random circuits do not have GS mode (0/20), as expected.

---

## SNAP L2 vs ECD L3

Same protocol as L3/L4: 20 H × 10 trials, `seed_base=4000`, 200 joint
SPSA, vacuum, `sampled_tail` η, exact matched budget, no ECD extra
near-miss runs. GDR is adaptive + `success_then_cost` on **all 20 H**,
comprehensive + readout_realistic, κτ ∈ {0.003, 0.03, 0.1}, 8192 shots.

JSON: `results/gibbs_four_sat_snap_matched_n10_L2.json`,
`results/gibbs_four_sat_ecd_matched_n10_L3.json`.
GDR dirs: `snap_l2_hXXX_s8192/`, `ecd_l3_hXXX_s8192/` (prefixed so they
do not overwrite the L3/L4 `snap_hXXX_s8192/` / `ecd_hXXX_s8192/`
fleet).

### 1. Noiseless success %

| ansatz | depth | success | success % | mean ⟨H⟩ (footnote only) |
|--------|------:|--------:|----------:|-------------------------:|
| SNAP | L2 | **177/200** | **88.5%** | 0.679 |
| ECD | L3 | **162/200** | **81.0%** | 0.873 |

Shallower circuits lose ~8 points of mode-finding vs the L3/L4 pair
(SNAP 96.5% → 88.5%, ECD 93.0% → 81.0%). The SNAP–ECD gap widens
slightly (7.5 points vs 3.5). Every Hamiltonian still has ≥1 success
(SNAP L2 min 7/10 on H015; ECD L3 min 6/10 on H001 and H004), so the
`success_then_cost` GDR pick is well-defined on all 20.

#### Per-H success (optional)

| H | SNAP L2 | ECD L3 | SNAP pick ⟨H⟩ | ECD pick ⟨H⟩ |
|--:|--------:|-------:|--------------:|-------------:|
| 000 | 8/10 | 9/10 | 0.399 | 0.954 |
| 001 | 10/10 | 6/10 | 0.438 | 0.618 |
| 002 | 9/10 | 7/10 | 0.305 | 0.576 |
| 003 | 9/10 | 8/10 | 0.267 | 0.854 |
| 004 | 9/10 | 6/10 | 0.324 | 0.665 |
| 005 | 10/10 | 9/10 | 0.481 | 0.834 |
| 006 | 10/10 | 8/10 | 0.521 | 0.857 |
| 007 | 10/10 | 9/10 | 0.586 | 0.861 |
| 008 | 9/10 | 8/10 | 0.501 | 0.707 |
| 009 | 9/10 | 9/10 | 0.551 | 0.782 |
| 010 | 9/10 | 7/10 | 0.361 | 0.754 |
| 011 | 9/10 | 10/10 | 0.274 | 0.615 |
| 012 | 9/10 | 10/10 | 0.561 | 0.765 |
| 013 | 8/10 | 8/10 | 0.270 | 0.581 |
| 014 | 9/10 | 8/10 | 0.574 | 0.887 |
| 015 | 7/10 | 8/10 | 0.602 | 0.562 |
| 016 | 9/10 | 8/10 | 0.241 | 0.605 |
| 017 | 8/10 | 7/10 | 0.280 | 0.703 |
| 018 | 8/10 | 10/10 | 0.365 | 0.370 |
| 019 | 8/10 | 7/10 | 0.404 | 0.543 |

### 2. Adaptive GDR (all 20 H, both ansatzes)

Same noise / twin / shot settings as L3/L4. Smoke first on H000
(`snap_l2_h000_smoke`, `ecd_l3_h000_smoke`; 4000 shots / 12 twins).

**Ideal optimized circuit has GS mode on 20/20 H for both SNAP L2 and
ECD L3.** Mode-finding on the ideal picked state is again not the GDR
bottleneck.

#### SNAP L2 optimized — raw → `gdr_select` TVD

Wins vs raw: **20/20** at κτ=0.003, **20/20** at 0.03, **17/20** at 0.1.
Mean TVD: 0.149→0.029, 0.442→0.132, 0.694→0.456.

GS mode after mitigation:
**20/20**, **20/20**, **15/20** select vs raw **20/20**, **20/20**, **3/20**.

| H | pick ⟨H⟩ | κτ=0.003 | κτ=0.03 | κτ=0.1 |
|--:|---------:|----------|---------|--------|
| 000 | 0.399 | 0.146 → 0.036 | 0.360 → 0.120 | 0.623 → 0.419 |
| 001 | 0.438 | 0.172 → 0.039 | 0.526 → 0.198 | 0.758 → 0.534 |
| 002 | 0.305 | 0.202 → 0.025 | 0.568 → 0.177 | 0.775 → 0.938 † |
| 003 | 0.267 | 0.170 → 0.017 | 0.522 → 0.050 | 0.813 → 0.328 |
| 004 | 0.324 | 0.198 → 0.026 | 0.583 → 0.138 | 0.828 → 0.711 |
| 005 | 0.481 | 0.148 → 0.026 | 0.399 → 0.100 | 0.610 → 0.269 |
| 006 | 0.521 | 0.142 → 0.031 | 0.423 → 0.160 | 0.730 → 0.435 |
| 007 | 0.586 | 0.153 → 0.039 | 0.428 → 0.195 | 0.653 → 0.526 |
| 008 | 0.501 | 0.086 → 0.032 | 0.384 → 0.162 | 0.735 → 0.546 |
| 009 | 0.551 | 0.106 → 0.024 | 0.211 → 0.065 | 0.395 → 0.149 |
| 010 | 0.361 | 0.149 → 0.033 | 0.407 → 0.100 | 0.668 → 0.175 |
| 011 | 0.274 | 0.143 → 0.024 | 0.559 → 0.049 | 0.785 → 0.595 |
| 012 | 0.561 | 0.134 → 0.032 | 0.312 → 0.121 | 0.558 → 0.345 |
| 013 | 0.270 | 0.106 → 0.022 | 0.490 → 0.042 | 0.721 → 0.164 |
| 014 | 0.574 | 0.136 → 0.033 | 0.351 → 0.070 | 0.593 → 0.190 |
| 015 | 0.602 | 0.114 → 0.024 | 0.287 → 0.126 | 0.543 → 0.307 |
| 016 | 0.241 | 0.190 → 0.026 | 0.611 → 0.377 | 0.811 → 0.945 † |
| 017 | 0.280 | 0.153 → 0.021 | 0.436 → 0.047 | 0.761 → 0.097 |
| 018 | 0.365 | 0.173 → 0.032 | 0.480 → 0.175 | 0.747 → 0.482 |
| 019 | 0.404 | 0.150 → 0.041 | 0.499 → 0.168 | 0.774 → 0.962 † |

† = select does **not** beat raw. SNAP L2 opt κτ=0.1 losses: H002, H016,
H019.

#### ECD L3 optimized — raw → `gdr_select` TVD

Wins vs raw: **20/20** at 0.003, **19/20** at 0.03, **13/20** at 0.1.
Mean TVD: 0.135→0.063, 0.375→0.182, 0.592→0.531.

GS mode after mitigation: **20/20**, **18/20**, **8/20** select vs raw
**20/20**, **18/20**, **2/20**.

Ideal GS mode remains 20/20 even on high-⟨H⟩ ECD L3 picks (e.g. H000
⟨H⟩=0.954). GDR at κτ=0.1 is mixed; again **not** gated by a near-E0
cut.

| H | pick ⟨H⟩ | κτ=0.003 | κτ=0.03 | κτ=0.1 |
|--:|---------:|----------|---------|--------|
| 000 | 0.954 | 0.123 → 0.057 | 0.283 → 0.172 | 0.438 → 0.415 |
| 001 | 0.618 | 0.166 → 0.065 | 0.496 → 0.141 | 0.680 → 0.754 † |
| 002 | 0.576 | 0.180 → 0.081 | 0.483 → 0.211 | 0.729 → 0.601 |
| 003 | 0.854 | 0.144 → 0.058 | 0.440 → 0.090 | 0.704 → 0.729 † |
| 004 | 0.665 | 0.187 → 0.056 | 0.485 → 0.106 | 0.700 → 0.612 |
| 005 | 0.834 | 0.170 → 0.086 | 0.473 → 0.286 | 0.618 → 0.690 † |
| 006 | 0.857 | 0.139 → 0.069 | 0.299 → 0.228 | 0.547 → 0.360 |
| 007 | 0.861 | 0.137 → 0.079 | 0.308 → 0.210 | 0.442 → 0.422 |
| 008 | 0.707 | 0.123 → 0.062 | 0.401 → 0.227 | 0.656 → 0.585 |
| 009 | 0.782 | 0.091 → 0.069 | 0.248 → 0.191 | 0.480 → 0.410 |
| 010 | 0.754 | 0.109 → 0.046 | 0.274 → 0.079 | 0.496 → 0.335 |
| 011 | 0.615 | 0.090 → 0.043 | 0.390 → 0.116 | 0.642 → 0.801 † |
| 012 | 0.765 | 0.108 → 0.052 | 0.290 → 0.186 | 0.495 → 0.453 |
| 013 | 0.581 | 0.107 → 0.063 | 0.334 → 0.106 | 0.589 → 0.199 |
| 014 | 0.887 | 0.112 → 0.074 | 0.271 → 0.156 | 0.462 → 0.319 |
| 015 | 0.562 | 0.128 → 0.059 | 0.238 → 0.131 | 0.428 → 0.255 |
| 016 | 0.605 | 0.144 → 0.049 | 0.472 → 0.137 | 0.732 → 0.831 † |
| 017 | 0.703 | 0.108 → 0.062 | 0.348 → 0.136 | 0.606 → 0.225 |
| 018 | 0.370 | 0.182 → 0.069 | 0.519 → 0.584 † | 0.691 → 0.925 † |
| 019 | 0.543 | 0.149 → 0.056 | 0.453 → 0.150 | 0.699 → 0.706 † |

κτ=0.03 loss: H018 only. κτ=0.1 losses: H001, H003, H005, H011, H016,
H018, H019.

#### Random circuits (same adaptive recipe)

SNAP L2 random select vs raw wins: **20/20**, **20/20**, **15/20** at
0.003 / 0.03 / 0.1. ECD L3 random: **19/20**, **20/20**, **20/20**.
Random circuits do not have GS mode (0/20), as expected.

---

## SNAP L1 vs ECD L2

Same protocol as the deeper pairs: 20 H × 10 trials, `seed_base=4000`,
200 joint SPSA, vacuum, `sampled_tail` η, exact matched budget, no ECD
extra near-miss runs. GDR is adaptive + `success_then_cost` on **all
20 H**, comprehensive + readout_realistic, κτ ∈ {0.003, 0.03, 0.1},
8192 shots.

JSON: `results/gibbs_four_sat_snap_matched_n10_L1.json`,
`results/gibbs_four_sat_ecd_matched_n10_L2.json`.
GDR dirs: `snap_l1_hXXX_s8192/`, `ecd_l2_hXXX_s8192/` (prefixed so they
do not overwrite SNAP L2 / ECD L3 files).

### 1. Noiseless success %

| ansatz | depth | success | success % | mean ⟨H⟩ (footnote only) |
|--------|------:|--------:|----------:|-------------------------:|
| SNAP | L1 | **79/200** | **39.5%** | 0.951 |
| ECD | L2 | **123/200** | **61.5%** | 0.948 |

This is the first matched pair where **ECD leads mode-finding** (22
points). SNAP L1 is a single D–SNAP–D–SNAP layer (18+5 params); ECD L2
has two R–ECD layers (16+5 params, 8 primitive gates). Mean ⟨H⟩ is
almost the same (~0.95); the gap is which bitstring is most likely, not
energy. Every Hamiltonian still has ≥1 success (SNAP L1 min 2/10 on
H002 / H012 / H013; ECD L2 min 1/10 on H019), so `success_then_cost`
needed no cost fallback.

#### Per-H success (optional)

| H | SNAP L1 | ECD L2 | SNAP pick ⟨H⟩ | ECD pick ⟨H⟩ |
|--:|--------:|-------:|--------------:|-------------:|
| 000 | 4/10 | 10/10 | 1.227 | 0.931 |
| 001 | 8/10 | 5/10 | 0.664 | 0.660 |
| 002 | 2/10 | 8/10 | 0.885 | 0.493 |
| 003 | 3/10 | 7/10 | 0.787 | 0.845 |
| 004 | 3/10 | 5/10 | 0.730 | 0.999 |
| 005 | 5/10 | 6/10 | 0.894 | 0.914 |
| 006 | 3/10 | 5/10 | 1.031 | 1.089 |
| 007 | 4/10 | 6/10 | 0.981 | 0.877 |
| 008 | 4/10 | 5/10 | 0.854 | 0.773 |
| 009 | 3/10 | 7/10 | 0.935 | 0.947 |
| 010 | 3/10 | 6/10 | 0.828 | 0.729 |
| 011 | 8/10 | 4/10 | 0.756 | 0.767 |
| 012 | 2/10 | 7/10 | 0.853 | 0.888 |
| 013 | 2/10 | 5/10 | 0.798 | 0.814 |
| 014 | 3/10 | 6/10 | 1.051 | 1.070 |
| 015 | 4/10 | 9/10 | 0.920 | 0.892 |
| 016 | 3/10 | 8/10 | 0.643 | 0.203 |
| 017 | 4/10 | 6/10 | 0.788 | 0.857 |
| 018 | 7/10 | 7/10 | 0.497 | 0.650 |
| 019 | 4/10 | 1/10 | 0.715 | 0.576 |

### 2. Adaptive GDR (all 20 H, both ansatzes)

Same noise / twin / shot settings as the deeper pairs. Smoke first on
H000 (`snap_l1_h000_smoke`, `ecd_l2_h000_smoke`; 4000 shots / 12 twins),
then 8192 on all 20. Results pending in this revision; filled after the
`--suite l1l2` fleet finishes.

Machine-readable scoreboard (all three pairs):
`Error_mitigation/out_four_sat_matched/scoreboard.json`
(keys `snap_L3`, `ecd_L4`, `snap_L2`, `ecd_L3`, `snap_L1`, `ecd_L2`;
`snap`/`ecd` aliases remain the L3/L4 pair).

---

## Energy bar was not used

- Noiseless headlines are **193/200 vs 186/200** (L3/L4),
  **177/200 vs 162/200** (L2/L3), and **79/200 vs 123/200** (L1/L2)
  bitstring successes, not `⟨H⟩ − E0 ≤ 0.5`.
- GDR ran on **all 20 H** for every pair, including high-⟨H⟩ ECD picks
  (L4 H000 ⟨H⟩=0.758; L3 H000 ⟨H⟩=0.954; L2 H000 ⟨H⟩=0.931) which a
  near-E0 filter would skip.
- ⟨H⟩ is logged only as a footnote / pick diagnostic.

## Exact commands

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

### Noiseless matched fleets

```bash
# SNAP L3 / ECD L4
python3 scripts/Gibbs_and_adaptive_optim_SNAP.py --four-sat \
  --n-trials 10 --ndepths 3 --seed-base 4000 --workers 4 \
  --output results/gibbs_four_sat_snap_matched_n10.json

python3 scripts/Gibbs_and_adaptive_optim_ECD.py --four-sat \
  --n-trials 10 --ndepths 4 --seed-base 4000 --workers 4 \
  --output results/gibbs_four_sat_ecd_matched_n10.json

# SNAP L2 / ECD L3
python3 scripts/Gibbs_and_adaptive_optim_SNAP.py --four-sat \
  --n-trials 10 --ndepths 2 --seed-base 4000 --workers 4 \
  --output results/gibbs_four_sat_snap_matched_n10_L2.json

python3 scripts/Gibbs_and_adaptive_optim_ECD.py --four-sat \
  --n-trials 10 --ndepths 3 --seed-base 4000 --workers 4 \
  --output results/gibbs_four_sat_ecd_matched_n10_L3.json

# SNAP L1 / ECD L2
python3 scripts/Gibbs_and_adaptive_optim_SNAP.py --four-sat \
  --n-trials 10 --ndepths 1 --seed-base 4000 --workers 4 \
  --output results/gibbs_four_sat_snap_matched_n10_L1.json

python3 scripts/Gibbs_and_adaptive_optim_ECD.py --four-sat \
  --n-trials 10 --ndepths 2 --seed-base 4000 --workers 4 \
  --output results/gibbs_four_sat_ecd_matched_n10_L2.json
```

### Noisy adaptive GDR

Smoke (H000), then 8192 on all 20. Drivers:
`Error_mitigation/out_four_sat_matched/run_gdr.sh` (L3/L4 sequential) or
`run_gdr_8192.py` (2-wide; `--suite l3l4` default, `--suite l2l3` for
the shallower pair). Log: `COMMANDS.log`, `gdr_run.log`.

```bash
python3 -u Error_mitigation/run_mitigation_experiment.py \
  --preset full --family four_sat --instance N --ansatz snap --ndepth 3 \
  --families comprehensive --readout readout_realistic \
  --kappa-tau 0.003,0.03,0.1 \
  --gibbs-json results/gibbs_four_sat_snap_matched_n10.json \
  --gibbs-pick success_then_cost --twin-design adaptive \
  --shots 8192 --n-train 40 --params both \
  --outdir Error_mitigation/out_four_sat_matched/snap_hNNN_s8192

# ECD L4: --ansatz ecd --ndepth 4 \
#   --gibbs-json results/gibbs_four_sat_ecd_matched_n10.json
# SNAP L2: --ansatz snap --ndepth 2 \
#   --gibbs-json results/gibbs_four_sat_snap_matched_n10_L2.json \
#   --outdir .../snap_l2_hNNN_s8192
# ECD L3: --ansatz ecd --ndepth 3 \
#   --gibbs-json results/gibbs_four_sat_ecd_matched_n10_L3.json \
#   --outdir .../ecd_l3_hNNN_s8192
# SNAP L1: --ansatz snap --ndepth 1 \
#   --gibbs-json results/gibbs_four_sat_snap_matched_n10_L1.json \
#   --outdir .../snap_l1_hNNN_s8192
# ECD L2: --ansatz ecd --ndepth 2 \
#   --gibbs-json results/gibbs_four_sat_ecd_matched_n10_L2.json \
#   --outdir .../ecd_l2_hNNN_s8192
# Smoke: --preset smoke --shots 4000 --n-train 12 --outdir .../${tag}_h000_smoke

python3 -u Error_mitigation/out_four_sat_matched/run_gdr_8192.py --suite l3l4
python3 -u Error_mitigation/out_four_sat_matched/run_gdr_8192.py --suite l2l3
python3 -u Error_mitigation/out_four_sat_matched/run_gdr_8192.py --suite l1l2
python3 Error_mitigation/out_four_sat_matched/build_scoreboard.py
```

`--gibbs-pick` default remains `energy` (lowest ⟨H⟩). This run used
`success_then_cost` only. Default GDR twin design is still `adaptive`.

## Thin hooks

- JSON `success` is explicitly `most_likely_bitstring == ground_bitstring`.
- Gibbs payload `by_hamiltonian` records per-H success and the
  success-then-cost pick (`pick_rule`, `pick_fallback` when no trial
  found the ground bitstring).
- `--gibbs-pick {energy,success_then_cost}` on the mitigation runner
  (default `energy`, official path unchanged).
- `compare_histograms` reports `success_gs` / `success_gs_ideal` (mode
  of mitigated / ideal histogram vs ground QNM).
