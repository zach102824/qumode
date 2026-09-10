# Ablation summary

tag=`gauss_only/default_thermal` shots=8192 n_train=40 twin=adaptive ansatz=both params=both families=loss_thermal_dephasing kappa=0.003,0.03,0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | random | loss_thermal_dephasing | 0.003 | ideal | 0.0535 | 0.0456 | — | — | 0.0488 | 0.0458 | — |
| ecd | random | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0606 | 0.0476 | — | — | 0.0610 | 0.0523 | — |
| ecd | random | loss_thermal_dephasing | 0.03 | ideal | 0.1744 | 0.1157 | — | — | 0.1773 | 0.1447 | — |
| ecd | random | loss_thermal_dephasing | 0.03 | readout_realistic | 0.1915 | 0.1279 | — | — | 0.1824 | 0.1740 | — |
| ecd | random | loss_thermal_dephasing | 0.1 | ideal | 0.3356 | 0.2633 | — | — | 0.3319 | 0.4067 | — |
| ecd | random | loss_thermal_dephasing | 0.1 | readout_realistic | 0.3368 | 0.2642 | — | — | 0.3286 | 0.3960 | — |
| ecd | optimized | loss_thermal_dephasing | 0.003 | ideal | 0.0489 | 0.0175 | — | — | 0.0542 | 0.0193 | — |
| ecd | optimized | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0894 | 0.0164 | — | — | 0.0873 | 0.0181 | — |
| ecd | optimized | loss_thermal_dephasing | 0.03 | ideal | 0.3966 | 0.1174 | — | — | 0.3950 | 0.1317 | — |
| ecd | optimized | loss_thermal_dephasing | 0.03 | readout_realistic | 0.4164 | 0.1210 | — | — | 0.4263 | 0.1252 | — |
| ecd | optimized | loss_thermal_dephasing | 0.1 | ideal | 0.7696 | 0.2769 | — | — | 0.7671 | 0.2700 | — |
| ecd | optimized | loss_thermal_dephasing | 0.1 | readout_realistic | 0.7792 | 0.2698 | — | — | 0.7786 | 0.2825 | — |
| snap | random | loss_thermal_dephasing | 0.003 | ideal | 0.0291 | 0.0285 | — | — | 0.0293 | 0.0304 | — |
| snap | random | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0487 | 0.0287 | — | — | 0.0487 | 0.0299 | — |
| snap | random | loss_thermal_dephasing | 0.03 | ideal | 0.1505 | 0.0893 | — | — | 0.1568 | 0.0972 | — |
| snap | random | loss_thermal_dephasing | 0.03 | readout_realistic | 0.1741 | 0.0907 | — | — | 0.1596 | 0.1065 | — |
| snap | random | loss_thermal_dephasing | 0.1 | ideal | 0.3433 | 0.1987 | — | — | 0.3562 | 0.2419 | — |
| snap | random | loss_thermal_dephasing | 0.1 | readout_realistic | 0.3460 | 0.1773 | — | — | 0.3466 | 0.2560 | — |
| snap | optimized | loss_thermal_dephasing | 0.003 | ideal | 0.0598 | 0.0131 | — | — | 0.0556 | 0.0129 | — |
| snap | optimized | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0988 | 0.0180 | — | — | 0.1027 | 0.0161 | — |
| snap | optimized | loss_thermal_dephasing | 0.03 | ideal | 0.3928 | 0.0841 | — | — | 0.3860 | 0.0923 | — |
| snap | optimized | loss_thermal_dephasing | 0.03 | readout_realistic | 0.4219 | 0.0868 | — | — | 0.4155 | 0.0980 | — |
| snap | optimized | loss_thermal_dephasing | 0.1 | ideal | 0.6967 | 0.3878 | — | — | 0.6952 | 0.4322 | — |
| snap | optimized | loss_thermal_dephasing | 0.1 | readout_realistic | 0.7029 | 0.4071 | — | — | 0.7087 | 0.4417 | — |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   random    loss_thermal_dephasing 0.003 ideal              raw                  0.0535   0.0785
ecd   random    loss_thermal_dephasing 0.003 ideal              gdr_param            0.0456   0.0848
ecd   random    loss_thermal_dephasing 0.003 readout_realistic  raw                  0.0606   0.0642
ecd   random    loss_thermal_dephasing 0.003 readout_realistic  gdr_param            0.0476   0.0718
ecd   random    loss_thermal_dephasing 0.030 ideal              raw                  0.1744   0.1670
ecd   random    loss_thermal_dephasing 0.030 ideal              gdr_param            0.1157   0.1690
ecd   random    loss_thermal_dephasing 0.030 readout_realistic  raw                  0.1915   0.0646
ecd   random    loss_thermal_dephasing 0.030 readout_realistic  gdr_param            0.1279   0.0135
ecd   random    loss_thermal_dephasing 0.100 ideal              raw                  0.3356   0.2086
ecd   random    loss_thermal_dephasing 0.100 ideal              gdr_param            0.2633   0.1769
ecd   random    loss_thermal_dephasing 0.100 readout_realistic  raw                  0.3368   0.1924
ecd   random    loss_thermal_dephasing 0.100 readout_realistic  gdr_param            0.2642   0.0637
ecd   optimized loss_thermal_dephasing 0.003 ideal              raw                  0.0489   0.1663
ecd   optimized loss_thermal_dephasing 0.003 ideal              gdr_param            0.0175   0.0812
ecd   optimized loss_thermal_dephasing 0.003 readout_realistic  raw                  0.0894   0.2934
ecd   optimized loss_thermal_dephasing 0.003 readout_realistic  gdr_param            0.0164   0.0681
ecd   optimized loss_thermal_dephasing 0.030 ideal              raw                  0.3966   1.3804
ecd   optimized loss_thermal_dephasing 0.030 ideal              gdr_param            0.1174   0.6206
ecd   optimized loss_thermal_dephasing 0.030 readout_realistic  raw                  0.4164   1.5117
ecd   optimized loss_thermal_dephasing 0.030 readout_realistic  gdr_param            0.1210   0.6675
ecd   optimized loss_thermal_dephasing 0.100 ideal              raw                  0.7696   3.0330
ecd   optimized loss_thermal_dephasing 0.100 ideal              gdr_param            0.2769   1.6281
ecd   optimized loss_thermal_dephasing 0.100 readout_realistic  raw                  0.7792   3.0364
ecd   optimized loss_thermal_dephasing 0.100 readout_realistic  gdr_param            0.2698   1.4880
snap  random    loss_thermal_dephasing 0.003 ideal              raw                  0.0291   0.0401
snap  random    loss_thermal_dephasing 0.003 ideal              gdr_param            0.0285   0.0510
snap  random    loss_thermal_dephasing 0.003 readout_realistic  raw                  0.0487   0.0561
snap  random    loss_thermal_dephasing 0.003 readout_realistic  gdr_param            0.0287   0.0292
snap  random    loss_thermal_dephasing 0.030 ideal              raw                  0.1505   0.0180
snap  random    loss_thermal_dephasing 0.030 ideal              gdr_param            0.0893   0.0607
snap  random    loss_thermal_dephasing 0.030 readout_realistic  raw                  0.1741   0.0711
snap  random    loss_thermal_dephasing 0.030 readout_realistic  gdr_param            0.0907   0.1992
snap  random    loss_thermal_dephasing 0.100 ideal              raw                  0.3433   0.2473
snap  random    loss_thermal_dephasing 0.100 ideal              gdr_param            0.1987   0.3407
snap  random    loss_thermal_dephasing 0.100 readout_realistic  raw                  0.3460   0.1975
snap  random    loss_thermal_dephasing 0.100 readout_realistic  gdr_param            0.1773   0.2249
snap  optimized loss_thermal_dephasing 0.003 ideal              raw                  0.0598   0.1503
snap  optimized loss_thermal_dephasing 0.003 ideal              gdr_param            0.0131   0.0363
snap  optimized loss_thermal_dephasing 0.003 readout_realistic  raw                  0.0988   0.2458
snap  optimized loss_thermal_dephasing 0.003 readout_realistic  gdr_param            0.0180   0.0196
snap  optimized loss_thermal_dephasing 0.030 ideal              raw                  0.3928   1.4126
snap  optimized loss_thermal_dephasing 0.030 ideal              gdr_param            0.0841   0.4825
snap  optimized loss_thermal_dephasing 0.030 readout_realistic  raw                  0.4219   1.4935
snap  optimized loss_thermal_dephasing 0.030 readout_realistic  gdr_param            0.0868   0.4610
snap  optimized loss_thermal_dephasing 0.100 ideal              raw                  0.6967   3.2082
snap  optimized loss_thermal_dephasing 0.100 ideal              gdr_param            0.3878   1.3606
snap  optimized loss_thermal_dephasing 0.100 readout_realistic  raw                  0.7029   3.3283
snap  optimized loss_thermal_dephasing 0.100 readout_realistic  gdr_param            0.4071   1.3426
```
