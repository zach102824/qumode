# Ablation summary

tag=`gauss_only/nr0_thermal` shots=8192 n_train=40 twin=adaptive ansatz=both params=both families=loss_thermal_dephasing kappa=0.003,0.03,0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | random | loss_thermal_dephasing | 0.003 | ideal | 0.0535 | 0.0477 | — | — | 0.0488 | 0.0458 | — |
| ecd | random | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0606 | 0.0505 | — | — | 0.0610 | 0.0523 | — |
| ecd | random | loss_thermal_dephasing | 0.03 | ideal | 0.1744 | 0.1347 | — | — | 0.1773 | 0.1447 | — |
| ecd | random | loss_thermal_dephasing | 0.03 | readout_realistic | 0.1915 | 0.1468 | — | — | 0.1824 | 0.1740 | — |
| ecd | random | loss_thermal_dephasing | 0.1 | ideal | 0.3356 | 0.2813 | — | — | 0.3319 | 0.4067 | — |
| ecd | random | loss_thermal_dephasing | 0.1 | readout_realistic | 0.3368 | 0.2886 | — | — | 0.3286 | 0.3960 | — |
| ecd | optimized | loss_thermal_dephasing | 0.003 | ideal | 0.0489 | 0.0176 | — | — | 0.0542 | 0.0193 | — |
| ecd | optimized | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0894 | 0.0163 | — | — | 0.0873 | 0.0181 | — |
| ecd | optimized | loss_thermal_dephasing | 0.03 | ideal | 0.3966 | 0.1120 | — | — | 0.3950 | 0.1317 | — |
| ecd | optimized | loss_thermal_dephasing | 0.03 | readout_realistic | 0.4164 | 0.1174 | — | — | 0.4263 | 0.1252 | — |
| ecd | optimized | loss_thermal_dephasing | 0.1 | ideal | 0.7696 | 0.2745 | — | — | 0.7671 | 0.2700 | — |
| ecd | optimized | loss_thermal_dephasing | 0.1 | readout_realistic | 0.7792 | 0.2673 | — | — | 0.7786 | 0.2825 | — |
| snap | random | loss_thermal_dephasing | 0.003 | ideal | 0.0291 | 0.0296 | — | — | 0.0293 | 0.0304 | — |
| snap | random | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0487 | 0.0290 | — | — | 0.0487 | 0.0299 | — |
| snap | random | loss_thermal_dephasing | 0.03 | ideal | 0.1505 | 0.0932 | — | — | 0.1568 | 0.0972 | — |
| snap | random | loss_thermal_dephasing | 0.03 | readout_realistic | 0.1741 | 0.0950 | — | — | 0.1596 | 0.1065 | — |
| snap | random | loss_thermal_dephasing | 0.1 | ideal | 0.3433 | 0.2274 | — | — | 0.3562 | 0.2419 | — |
| snap | random | loss_thermal_dephasing | 0.1 | readout_realistic | 0.3460 | 0.2039 | — | — | 0.3466 | 0.2560 | — |
| snap | optimized | loss_thermal_dephasing | 0.003 | ideal | 0.0598 | 0.0193 | — | — | 0.0556 | 0.0129 | — |
| snap | optimized | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0988 | 0.0218 | — | — | 0.1027 | 0.0161 | — |
| snap | optimized | loss_thermal_dephasing | 0.03 | ideal | 0.3928 | 0.1567 | — | — | 0.3860 | 0.0923 | — |
| snap | optimized | loss_thermal_dephasing | 0.03 | readout_realistic | 0.4219 | 0.1628 | — | — | 0.4155 | 0.0980 | — |
| snap | optimized | loss_thermal_dephasing | 0.1 | ideal | 0.6967 | 0.5391 | — | — | 0.6952 | 0.4322 | — |
| snap | optimized | loss_thermal_dephasing | 0.1 | readout_realistic | 0.7029 | 0.5430 | — | — | 0.7087 | 0.4417 | — |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   random    loss_thermal_dephasing 0.003 ideal              raw                  0.0535   0.0785
ecd   random    loss_thermal_dephasing 0.003 ideal              gdr_param            0.0477   0.0753
ecd   random    loss_thermal_dephasing 0.003 readout_realistic  raw                  0.0606   0.0642
ecd   random    loss_thermal_dephasing 0.003 readout_realistic  gdr_param            0.0505   0.0568
ecd   random    loss_thermal_dephasing 0.030 ideal              raw                  0.1744   0.1670
ecd   random    loss_thermal_dephasing 0.030 ideal              gdr_param            0.1347   0.0628
ecd   random    loss_thermal_dephasing 0.030 readout_realistic  raw                  0.1915   0.0646
ecd   random    loss_thermal_dephasing 0.030 readout_realistic  gdr_param            0.1468   0.0761
ecd   random    loss_thermal_dephasing 0.100 ideal              raw                  0.3356   0.2086
ecd   random    loss_thermal_dephasing 0.100 ideal              gdr_param            0.2813   0.0194
ecd   random    loss_thermal_dephasing 0.100 readout_realistic  raw                  0.3368   0.1924
ecd   random    loss_thermal_dephasing 0.100 readout_realistic  gdr_param            0.2886   0.0940
ecd   optimized loss_thermal_dephasing 0.003 ideal              raw                  0.0489   0.1663
ecd   optimized loss_thermal_dephasing 0.003 ideal              gdr_param            0.0176   0.0814
ecd   optimized loss_thermal_dephasing 0.003 readout_realistic  raw                  0.0894   0.2934
ecd   optimized loss_thermal_dephasing 0.003 readout_realistic  gdr_param            0.0163   0.0679
ecd   optimized loss_thermal_dephasing 0.030 ideal              raw                  0.3966   1.3804
ecd   optimized loss_thermal_dephasing 0.030 ideal              gdr_param            0.1120   0.6061
ecd   optimized loss_thermal_dephasing 0.030 readout_realistic  raw                  0.4164   1.5117
ecd   optimized loss_thermal_dephasing 0.030 readout_realistic  gdr_param            0.1174   0.6568
ecd   optimized loss_thermal_dephasing 0.100 ideal              raw                  0.7696   3.0330
ecd   optimized loss_thermal_dephasing 0.100 ideal              gdr_param            0.2745   1.5802
ecd   optimized loss_thermal_dephasing 0.100 readout_realistic  raw                  0.7792   3.0364
ecd   optimized loss_thermal_dephasing 0.100 readout_realistic  gdr_param            0.2673   1.4724
snap  random    loss_thermal_dephasing 0.003 ideal              raw                  0.0291   0.0401
snap  random    loss_thermal_dephasing 0.003 ideal              gdr_param            0.0296   0.0462
snap  random    loss_thermal_dephasing 0.003 readout_realistic  raw                  0.0487   0.0561
snap  random    loss_thermal_dephasing 0.003 readout_realistic  gdr_param            0.0290   0.0250
snap  random    loss_thermal_dephasing 0.030 ideal              raw                  0.1505   0.0180
snap  random    loss_thermal_dephasing 0.030 ideal              gdr_param            0.0932   0.1004
snap  random    loss_thermal_dephasing 0.030 readout_realistic  raw                  0.1741   0.0711
snap  random    loss_thermal_dephasing 0.030 readout_realistic  gdr_param            0.0950   0.2545
snap  random    loss_thermal_dephasing 0.100 ideal              raw                  0.3433   0.2473
snap  random    loss_thermal_dephasing 0.100 ideal              gdr_param            0.2274   0.4562
snap  random    loss_thermal_dephasing 0.100 readout_realistic  raw                  0.3460   0.1975
snap  random    loss_thermal_dephasing 0.100 readout_realistic  gdr_param            0.2039   0.3154
snap  optimized loss_thermal_dephasing 0.003 ideal              raw                  0.0598   0.1503
snap  optimized loss_thermal_dephasing 0.003 ideal              gdr_param            0.0193   0.0701
snap  optimized loss_thermal_dephasing 0.003 readout_realistic  raw                  0.0988   0.2458
snap  optimized loss_thermal_dephasing 0.003 readout_realistic  gdr_param            0.0218   0.0564
snap  optimized loss_thermal_dephasing 0.030 ideal              raw                  0.3928   1.4126
snap  optimized loss_thermal_dephasing 0.030 ideal              gdr_param            0.1567   0.7405
snap  optimized loss_thermal_dephasing 0.030 readout_realistic  raw                  0.4219   1.4935
snap  optimized loss_thermal_dephasing 0.030 readout_realistic  gdr_param            0.1628   0.7101
snap  optimized loss_thermal_dephasing 0.100 ideal              raw                  0.6967   3.2082
snap  optimized loss_thermal_dephasing 0.100 ideal              gdr_param            0.5391   2.3515
snap  optimized loss_thermal_dephasing 0.100 readout_realistic  raw                  0.7029   3.3283
snap  optimized loss_thermal_dephasing 0.100 readout_realistic  gdr_param            0.5430   2.3050
```
