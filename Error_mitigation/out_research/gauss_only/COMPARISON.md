# Fully Gaussian twins vs default mix (`gdr_param` only)

H000 / instance 0, shots=8192, n_train=40, seed=`SEED_BASE=2026`, `--twin-design adaptive` (span on random, U(0.5,1) on optimized).

- **Arm A (default mix):** `--n-rank2 10`. Equal to omitted/auto for `n_train=40` (`n_train//4`). Explicit 10 reuses existing `out_research/cache/*_nr10_*` keys (`nrauto` would miss them).
- **Arm B (fully Gaussian):** `--n-rank2 0` → new `*_nr0_*` cache keys, all `t_free=0`.
- Methods: `raw`, `gdr_param`. Readout: `ideal` + `readout_realistic` (no `readout_strong`).

Δ = TVD(`gdr_param` | default mix) − TVD(`gdr_param` | fully Gaussian). Positive ⇒ fully Gaussian better. Tie if |Δ| < 0.002.

**Recommendation:** keep the ~25% `t_free=2` mix (`n_rank2 = n_train//4`). Fully Gaussian is not strictly better; the mix still helps enough on `gdr_param`.

## Headline

| metric | value |
|---|---:|
| paired cells | 72 |
| mean Δ | -0.0260 |
| median Δ | -0.0079 |
| gauss better / tie / mix better | 11 / 15 / 46 |

## By slice

| slice | n | mean Δ | median Δ | gauss better | tie | mix better |
|---|---:|---:|---:|---:|---:|---:|
| ansatz=ecd | 36 | -0.0085 | -0.0001 | 11 | 9 | 16 |
| ansatz=snap | 36 | -0.0436 | -0.0179 | 0 | 6 | 30 |
| params=optimized | 36 | -0.0354 | -0.0026 | 11 | 7 | 18 |
| params=random | 36 | -0.0167 | -0.0174 | 0 | 8 | 28 |
| kappa_tau=0.003 | 24 | -0.0041 | -0.0015 | 2 | 11 | 11 |
| kappa_tau=0.03 | 24 | -0.0302 | -0.0164 | 5 | 2 | 17 |
| kappa_tau=0.1 | 24 | -0.0438 | -0.0277 | 4 | 2 | 18 |
| family=comprehensive | 24 | -0.0318 | -0.0177 | 6 | 2 | 16 |
| family=loss | 24 | -0.0220 | -0.0076 | 1 | 9 | 14 |
| family=loss_thermal_dephasing | 24 | -0.0243 | -0.0041 | 4 | 4 | 16 |

ECD priority only (loss + comprehensive, no thermal): **24 cells, mean Δ = −0.0097, median Δ = +0.0001, 7 / 7 / 10**. Still keep the mix.

All 11 fully-Gaussian wins are **ECD optimized** (all 6 comprehensive cells, 4 thermal, 1 loss). Random ECD and all SNAP cells are mix-better or ties. SNAP optimized is where the mix helps most (mean Δ ≈ −0.07).

## Per-cell TVD

| ansatz | params | family | κτ | readout | raw mix | raw gauss | gdr mix | gdr gauss | Δ | verdict |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| ecd | optimized | comprehensive | 0.003 | ideal | 0.1842 | 0.1842 | 0.0522 | 0.0489 | 0.0034 | gauss_better |
| ecd | optimized | comprehensive | 0.003 | readout_realistic | 0.2074 | 0.2074 | 0.0539 | 0.0499 | 0.0039 | gauss_better |
| ecd | optimized | comprehensive | 0.03 | ideal | 0.5653 | 0.5653 | 0.1374 | 0.1289 | 0.0086 | gauss_better |
| ecd | optimized | comprehensive | 0.03 | readout_realistic | 0.5879 | 0.5879 | 0.1367 | 0.1289 | 0.0078 | gauss_better |
| ecd | optimized | comprehensive | 0.1 | ideal | 0.9086 | 0.9086 | 0.3434 | 0.3258 | 0.0176 | gauss_better |
| ecd | optimized | comprehensive | 0.1 | readout_realistic | 0.9116 | 0.9116 | 0.3522 | 0.3338 | 0.0184 | gauss_better |
| ecd | optimized | loss | 0.003 | ideal | 0.0383 | 0.0383 | 0.0093 | 0.0093 | -0.0000 | tie |
| ecd | optimized | loss | 0.003 | readout_realistic | 0.0788 | 0.0788 | 0.0107 | 0.0104 | 0.0004 | tie |
| ecd | optimized | loss | 0.03 | ideal | 0.3145 | 0.3145 | 0.0714 | 0.0698 | 0.0017 | tie |
| ecd | optimized | loss | 0.03 | readout_realistic | 0.3286 | 0.3286 | 0.0699 | 0.0676 | 0.0023 | gauss_better |
| ecd | optimized | loss | 0.1 | ideal | 0.6961 | 0.6961 | 0.2011 | 0.1997 | 0.0014 | tie |
| ecd | optimized | loss | 0.1 | readout_realistic | 0.7090 | 0.7090 | 0.2067 | 0.2081 | -0.0014 | tie |
| ecd | optimized | loss_thermal_dephasing | 0.003 | ideal | 0.0489 | 0.0489 | 0.0175 | 0.0176 | -0.0002 | tie |
| ecd | optimized | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0894 | 0.0894 | 0.0164 | 0.0163 | 0.0000 | tie |
| ecd | optimized | loss_thermal_dephasing | 0.03 | ideal | 0.3966 | 0.3966 | 0.1174 | 0.1120 | 0.0054 | gauss_better |
| ecd | optimized | loss_thermal_dephasing | 0.03 | readout_realistic | 0.4164 | 0.4164 | 0.1210 | 0.1174 | 0.0036 | gauss_better |
| ecd | optimized | loss_thermal_dephasing | 0.1 | ideal | 0.7696 | 0.7696 | 0.2769 | 0.2745 | 0.0024 | gauss_better |
| ecd | optimized | loss_thermal_dephasing | 0.1 | readout_realistic | 0.7792 | 0.7792 | 0.2698 | 0.2673 | 0.0024 | gauss_better |
| ecd | random | comprehensive | 0.003 | ideal | 0.0816 | 0.0816 | 0.0723 | 0.0839 | -0.0117 | mix_better |
| ecd | random | comprehensive | 0.003 | readout_realistic | 0.1016 | 0.1016 | 0.0776 | 0.1002 | -0.0226 | mix_better |
| ecd | random | comprehensive | 0.03 | ideal | 0.2417 | 0.2417 | 0.1497 | 0.1912 | -0.0415 | mix_better |
| ecd | random | comprehensive | 0.03 | readout_realistic | 0.2237 | 0.2237 | 0.1731 | 0.2078 | -0.0346 | mix_better |
| ecd | random | comprehensive | 0.1 | ideal | 0.4031 | 0.4031 | 0.3765 | 0.4246 | -0.0481 | mix_better |
| ecd | random | comprehensive | 0.1 | readout_realistic | 0.3980 | 0.3980 | 0.3410 | 0.3788 | -0.0378 | mix_better |
| ecd | random | loss | 0.003 | ideal | 0.0434 | 0.0434 | 0.0448 | 0.0444 | 0.0004 | tie |
| ecd | random | loss | 0.003 | readout_realistic | 0.0590 | 0.0590 | 0.0470 | 0.0468 | 0.0003 | tie |
| ecd | random | loss | 0.03 | ideal | 0.1423 | 0.1423 | 0.0909 | 0.1102 | -0.0193 | mix_better |
| ecd | random | loss | 0.03 | readout_realistic | 0.1539 | 0.1539 | 0.0898 | 0.1133 | -0.0235 | mix_better |
| ecd | random | loss | 0.1 | ideal | 0.2983 | 0.2983 | 0.2087 | 0.2491 | -0.0404 | mix_better |
| ecd | random | loss | 0.1 | readout_realistic | 0.3078 | 0.3078 | 0.2221 | 0.2409 | -0.0188 | mix_better |
| ecd | random | loss_thermal_dephasing | 0.003 | ideal | 0.0535 | 0.0535 | 0.0456 | 0.0477 | -0.0021 | mix_better |
| ecd | random | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0606 | 0.0606 | 0.0476 | 0.0505 | -0.0028 | mix_better |
| ecd | random | loss_thermal_dephasing | 0.03 | ideal | 0.1744 | 0.1744 | 0.1157 | 0.1347 | -0.0190 | mix_better |
| ecd | random | loss_thermal_dephasing | 0.03 | readout_realistic | 0.1915 | 0.1915 | 0.1279 | 0.1468 | -0.0189 | mix_better |
| ecd | random | loss_thermal_dephasing | 0.1 | ideal | 0.3356 | 0.3356 | 0.2633 | 0.2813 | -0.0180 | mix_better |
| ecd | random | loss_thermal_dephasing | 0.1 | readout_realistic | 0.3368 | 0.3368 | 0.2642 | 0.2886 | -0.0244 | mix_better |
| snap | optimized | comprehensive | 0.003 | ideal | 0.0979 | 0.0979 | 0.0181 | 0.0333 | -0.0153 | mix_better |
| snap | optimized | comprehensive | 0.003 | readout_realistic | 0.1411 | 0.1411 | 0.0230 | 0.0431 | -0.0201 | mix_better |
| snap | optimized | comprehensive | 0.03 | ideal | 0.5494 | 0.5494 | 0.1792 | 0.2956 | -0.1165 | mix_better |
| snap | optimized | comprehensive | 0.03 | readout_realistic | 0.5710 | 0.5710 | 0.2030 | 0.3431 | -0.1402 | mix_better |
| snap | optimized | comprehensive | 0.1 | ideal | 0.7914 | 0.7914 | 0.6111 | 0.7210 | -0.1099 | mix_better |
| snap | optimized | comprehensive | 0.1 | readout_realistic | 0.7952 | 0.7952 | 0.5904 | 0.7194 | -0.1290 | mix_better |
| snap | optimized | loss | 0.003 | ideal | 0.0567 | 0.0567 | 0.0177 | 0.0261 | -0.0083 | mix_better |
| snap | optimized | loss | 0.003 | readout_realistic | 0.0971 | 0.0971 | 0.0134 | 0.0203 | -0.0069 | mix_better |
| snap | optimized | loss | 0.03 | ideal | 0.3656 | 0.3656 | 0.0838 | 0.1570 | -0.0732 | mix_better |
| snap | optimized | loss | 0.03 | readout_realistic | 0.3888 | 0.3888 | 0.0751 | 0.1599 | -0.0848 | mix_better |
| snap | optimized | loss | 0.1 | ideal | 0.6603 | 0.6603 | 0.3820 | 0.4823 | -0.1003 | mix_better |
| snap | optimized | loss | 0.1 | readout_realistic | 0.6680 | 0.6680 | 0.3929 | 0.4938 | -0.1009 | mix_better |
| snap | optimized | loss_thermal_dephasing | 0.003 | ideal | 0.0598 | 0.0598 | 0.0131 | 0.0193 | -0.0062 | mix_better |
| snap | optimized | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0988 | 0.0988 | 0.0180 | 0.0218 | -0.0039 | mix_better |
| snap | optimized | loss_thermal_dephasing | 0.03 | ideal | 0.3928 | 0.3928 | 0.0841 | 0.1567 | -0.0726 | mix_better |
| snap | optimized | loss_thermal_dephasing | 0.03 | readout_realistic | 0.4219 | 0.4219 | 0.0868 | 0.1628 | -0.0759 | mix_better |
| snap | optimized | loss_thermal_dephasing | 0.1 | ideal | 0.6967 | 0.6967 | 0.3878 | 0.5391 | -0.1514 | mix_better |
| snap | optimized | loss_thermal_dephasing | 0.1 | readout_realistic | 0.7029 | 0.7029 | 0.4071 | 0.5430 | -0.1359 | mix_better |
| snap | random | comprehensive | 0.003 | ideal | 0.0372 | 0.0372 | 0.0415 | 0.0434 | -0.0019 | tie |
| snap | random | comprehensive | 0.003 | readout_realistic | 0.0535 | 0.0535 | 0.0385 | 0.0409 | -0.0023 | mix_better |
| snap | random | comprehensive | 0.03 | ideal | 0.2450 | 0.2450 | 0.1372 | 0.1447 | -0.0075 | mix_better |
| snap | random | comprehensive | 0.03 | readout_realistic | 0.2406 | 0.2406 | 0.1349 | 0.1330 | 0.0019 | tie |
| snap | random | comprehensive | 0.1 | ideal | 0.4728 | 0.4728 | 0.3747 | 0.4218 | -0.0471 | mix_better |
| snap | random | comprehensive | 0.1 | readout_realistic | 0.4685 | 0.4685 | 0.3334 | 0.3712 | -0.0378 | mix_better |
| snap | random | loss | 0.003 | ideal | 0.0283 | 0.0283 | 0.0217 | 0.0219 | -0.0003 | tie |
| snap | random | loss | 0.003 | readout_realistic | 0.0499 | 0.0499 | 0.0346 | 0.0355 | -0.0009 | tie |
| snap | random | loss | 0.03 | ideal | 0.1304 | 0.1304 | 0.0809 | 0.0875 | -0.0066 | mix_better |
| snap | random | loss | 0.03 | readout_realistic | 0.1520 | 0.1520 | 0.0736 | 0.0875 | -0.0139 | mix_better |
| snap | random | loss | 0.1 | ideal | 0.3351 | 0.3351 | 0.1956 | 0.2123 | -0.0167 | mix_better |
| snap | random | loss | 0.1 | readout_realistic | 0.3457 | 0.3457 | 0.2140 | 0.2331 | -0.0191 | mix_better |
| snap | random | loss_thermal_dephasing | 0.003 | ideal | 0.0291 | 0.0291 | 0.0285 | 0.0296 | -0.0011 | tie |
| snap | random | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0487 | 0.0487 | 0.0287 | 0.0290 | -0.0003 | tie |
| snap | random | loss_thermal_dephasing | 0.03 | ideal | 0.1505 | 0.1505 | 0.0893 | 0.0932 | -0.0039 | mix_better |
| snap | random | loss_thermal_dephasing | 0.03 | readout_realistic | 0.1741 | 0.1741 | 0.0907 | 0.0950 | -0.0043 | mix_better |
| snap | random | loss_thermal_dephasing | 0.1 | ideal | 0.3433 | 0.3433 | 0.1987 | 0.2274 | -0.0287 | mix_better |
| snap | random | loss_thermal_dephasing | 0.1 | readout_realistic | 0.3460 | 0.3460 | 0.1773 | 0.2039 | -0.0266 | mix_better |

## Notes

- Physical histograms for arm A reuse `Error_mitigation/out_research/cache/` `nr10` adaptive keys (span on random, default on optimized).
- Arm B writes new `nr0` keys into that same shared cache.
- Production adaptive defaults are **not** changed in this PR.
