# Ablation summary

tag=`gauss_only/default_snap_priority` shots=8192 n_train=40 twin=adaptive ansatz=snap params=both families=loss,comprehensive kappa=0.003,0.03,0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| snap | random | loss | 0.003 | ideal | 0.0283 | 0.0217 | — | — | 0.0194 | 0.0223 | — |
| snap | random | loss | 0.003 | readout_realistic | 0.0499 | 0.0346 | — | — | 0.0424 | 0.0308 | — |
| snap | random | loss | 0.03 | ideal | 0.1304 | 0.0809 | — | — | 0.1427 | 0.0851 | — |
| snap | random | loss | 0.03 | readout_realistic | 0.1520 | 0.0736 | — | — | 0.1551 | 0.0853 | — |
| snap | random | loss | 0.1 | ideal | 0.3351 | 0.1956 | — | — | 0.3330 | 0.2329 | — |
| snap | random | loss | 0.1 | readout_realistic | 0.3457 | 0.2140 | — | — | 0.3444 | 0.2449 | — |
| snap | random | comprehensive | 0.003 | ideal | 0.0372 | 0.0415 | — | — | 0.0413 | 0.0445 | — |
| snap | random | comprehensive | 0.003 | readout_realistic | 0.0535 | 0.0385 | — | — | 0.0523 | 0.0324 | — |
| snap | random | comprehensive | 0.03 | ideal | 0.2450 | 0.1372 | — | — | 0.2380 | 0.1436 | — |
| snap | random | comprehensive | 0.03 | readout_realistic | 0.2406 | 0.1349 | — | — | 0.2528 | 0.1406 | — |
| snap | random | comprehensive | 0.1 | ideal | 0.4728 | 0.3747 | — | — | 0.4663 | 0.4109 | — |
| snap | random | comprehensive | 0.1 | readout_realistic | 0.4685 | 0.3334 | — | — | 0.4725 | 0.4005 | — |
| snap | optimized | loss | 0.003 | ideal | 0.0567 | 0.0177 | — | — | 0.0546 | 0.0146 | — |
| snap | optimized | loss | 0.003 | readout_realistic | 0.0971 | 0.0134 | — | — | 0.0954 | 0.0191 | — |
| snap | optimized | loss | 0.03 | ideal | 0.3656 | 0.0838 | — | — | 0.3584 | 0.0971 | — |
| snap | optimized | loss | 0.03 | readout_realistic | 0.3888 | 0.0751 | — | — | 0.3812 | 0.0908 | — |
| snap | optimized | loss | 0.1 | ideal | 0.6603 | 0.3820 | — | — | 0.6658 | 0.4163 | — |
| snap | optimized | loss | 0.1 | readout_realistic | 0.6680 | 0.3929 | — | — | 0.6552 | 0.3673 | — |
| snap | optimized | comprehensive | 0.003 | ideal | 0.0979 | 0.0181 | — | — | 0.1033 | 0.0270 | — |
| snap | optimized | comprehensive | 0.003 | readout_realistic | 0.1411 | 0.0230 | — | — | 0.1443 | 0.0270 | — |
| snap | optimized | comprehensive | 0.03 | ideal | 0.5494 | 0.1792 | — | — | 0.5508 | 0.2035 | — |
| snap | optimized | comprehensive | 0.03 | readout_realistic | 0.5710 | 0.2030 | — | — | 0.5728 | 0.2285 | — |
| snap | optimized | comprehensive | 0.1 | ideal | 0.7914 | 0.6111 | — | — | 0.7933 | 0.6320 | — |
| snap | optimized | comprehensive | 0.1 | readout_realistic | 0.7952 | 0.5904 | — | — | 0.7958 | 0.6214 | — |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
snap  random    loss                   0.003 ideal              raw                  0.0283   0.0117
snap  random    loss                   0.003 ideal              gdr_param            0.0217   0.0201
snap  random    loss                   0.003 readout_realistic  raw                  0.0499   0.0166
snap  random    loss                   0.003 readout_realistic  gdr_param            0.0346   0.0436
snap  random    loss                   0.030 ideal              raw                  0.1304   0.0735
snap  random    loss                   0.030 ideal              gdr_param            0.0809   0.1679
snap  random    loss                   0.030 readout_realistic  raw                  0.1520   0.0385
snap  random    loss                   0.030 readout_realistic  gdr_param            0.0736   0.0739
snap  random    loss                   0.100 ideal              raw                  0.3351   0.2530
snap  random    loss                   0.100 ideal              gdr_param            0.1956   0.2912
snap  random    loss                   0.100 readout_realistic  raw                  0.3457   0.1593
snap  random    loss                   0.100 readout_realistic  gdr_param            0.2140   0.1790
snap  random    comprehensive          0.003 ideal              raw                  0.0372   0.0154
snap  random    comprehensive          0.003 ideal              gdr_param            0.0415   0.0473
snap  random    comprehensive          0.003 readout_realistic  raw                  0.0535   0.0128
snap  random    comprehensive          0.003 readout_realistic  gdr_param            0.0385   0.0576
snap  random    comprehensive          0.030 ideal              raw                  0.2450   0.0030
snap  random    comprehensive          0.030 ideal              gdr_param            0.1372   0.1881
snap  random    comprehensive          0.030 readout_realistic  raw                  0.2406   0.0486
snap  random    comprehensive          0.030 readout_realistic  gdr_param            0.1349   0.2571
snap  random    comprehensive          0.100 ideal              raw                  0.4728   0.4223
snap  random    comprehensive          0.100 ideal              gdr_param            0.3747   0.4585
snap  random    comprehensive          0.100 readout_realistic  raw                  0.4685   0.3440
snap  random    comprehensive          0.100 readout_realistic  gdr_param            0.3334   0.4294
snap  optimized loss                   0.003 ideal              raw                  0.0567   0.1546
snap  optimized loss                   0.003 ideal              gdr_param            0.0177   0.0515
snap  optimized loss                   0.003 readout_realistic  raw                  0.0971   0.2515
snap  optimized loss                   0.003 readout_realistic  gdr_param            0.0134   0.0367
snap  optimized loss                   0.030 ideal              raw                  0.3656   1.2167
snap  optimized loss                   0.030 ideal              gdr_param            0.0838   0.3812
snap  optimized loss                   0.030 readout_realistic  raw                  0.3888   1.2538
snap  optimized loss                   0.030 readout_realistic  gdr_param            0.0751   0.3149
snap  optimized loss                   0.100 ideal              raw                  0.6603   2.8109
snap  optimized loss                   0.100 ideal              gdr_param            0.3820   1.0610
snap  optimized loss                   0.100 readout_realistic  raw                  0.6680   2.8581
snap  optimized loss                   0.100 readout_realistic  gdr_param            0.3929   1.1559
snap  optimized comprehensive          0.003 ideal              raw                  0.0979   0.2830
snap  optimized comprehensive          0.003 ideal              gdr_param            0.0181   0.0903
snap  optimized comprehensive          0.003 readout_realistic  raw                  0.1411   0.4172
snap  optimized comprehensive          0.003 readout_realistic  gdr_param            0.0230   0.1156
snap  optimized comprehensive          0.030 ideal              raw                  0.5494   2.1276
snap  optimized comprehensive          0.030 ideal              gdr_param            0.1792   0.6928
snap  optimized comprehensive          0.030 readout_realistic  raw                  0.5710   2.2040
snap  optimized comprehensive          0.030 readout_realistic  gdr_param            0.2030   0.7287
snap  optimized comprehensive          0.100 ideal              raw                  0.7914   3.4906
snap  optimized comprehensive          0.100 ideal              gdr_param            0.6111   1.8540
snap  optimized comprehensive          0.100 readout_realistic  raw                  0.7952   3.5215
snap  optimized comprehensive          0.100 readout_realistic  gdr_param            0.5904   1.9559
```
