# Ablation summary

tag=`gauss_only/nr0_ecd_priority` shots=8192 n_train=40 twin=adaptive ansatz=ecd params=both families=loss,comprehensive kappa=0.003,0.03,0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | random | loss | 0.003 | ideal | 0.0434 | 0.0444 | — | — | 0.0461 | 0.0459 | — |
| ecd | random | loss | 0.003 | readout_realistic | 0.0590 | 0.0468 | — | — | 0.0613 | 0.0499 | — |
| ecd | random | loss | 0.03 | ideal | 0.1423 | 0.1102 | — | — | 0.1423 | 0.1283 | — |
| ecd | random | loss | 0.03 | readout_realistic | 0.1539 | 0.1133 | — | — | 0.1540 | 0.1304 | — |
| ecd | random | loss | 0.1 | ideal | 0.2983 | 0.2491 | — | — | 0.2991 | 0.3725 | — |
| ecd | random | loss | 0.1 | readout_realistic | 0.3078 | 0.2409 | — | — | 0.2978 | 0.3940 | — |
| ecd | random | comprehensive | 0.003 | ideal | 0.0816 | 0.0839 | — | — | 0.0801 | 0.0713 | — |
| ecd | random | comprehensive | 0.003 | readout_realistic | 0.1016 | 0.1002 | — | — | 0.1007 | 0.0847 | — |
| ecd | random | comprehensive | 0.03 | ideal | 0.2417 | 0.1912 | — | — | 0.2380 | 0.2841 | — |
| ecd | random | comprehensive | 0.03 | readout_realistic | 0.2237 | 0.2078 | — | — | 0.2357 | 0.2897 | — |
| ecd | random | comprehensive | 0.1 | ideal | 0.4031 | 0.4246 | — | — | 0.3990 | 0.5392 | — |
| ecd | random | comprehensive | 0.1 | readout_realistic | 0.3980 | 0.3788 | — | — | 0.4052 | 0.5224 | — |
| ecd | optimized | loss | 0.003 | ideal | 0.0383 | 0.0093 | — | — | 0.0363 | 0.0075 | — |
| ecd | optimized | loss | 0.003 | readout_realistic | 0.0788 | 0.0104 | — | — | 0.0758 | 0.0107 | — |
| ecd | optimized | loss | 0.03 | ideal | 0.3145 | 0.0698 | — | — | 0.3143 | 0.0724 | — |
| ecd | optimized | loss | 0.03 | readout_realistic | 0.3286 | 0.0676 | — | — | 0.3385 | 0.0731 | — |
| ecd | optimized | loss | 0.1 | ideal | 0.6961 | 0.1997 | — | — | 0.7064 | 0.2015 | — |
| ecd | optimized | loss | 0.1 | readout_realistic | 0.7090 | 0.2081 | — | — | 0.7082 | 0.2064 | — |
| ecd | optimized | comprehensive | 0.003 | ideal | 0.1842 | 0.0489 | — | — | 0.1823 | 0.0534 | — |
| ecd | optimized | comprehensive | 0.003 | readout_realistic | 0.2074 | 0.0499 | — | — | 0.2158 | 0.0523 | — |
| ecd | optimized | comprehensive | 0.03 | ideal | 0.5653 | 0.1289 | — | — | 0.5762 | 0.1434 | — |
| ecd | optimized | comprehensive | 0.03 | readout_realistic | 0.5879 | 0.1289 | — | — | 0.5832 | 0.1484 | — |
| ecd | optimized | comprehensive | 0.1 | ideal | 0.9086 | 0.3258 | — | — | 0.9082 | 0.3429 | — |
| ecd | optimized | comprehensive | 0.1 | readout_realistic | 0.9116 | 0.3338 | — | — | 0.9094 | 0.3519 | — |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   random    loss                   0.003 ideal              raw                  0.0434   0.0785
ecd   random    loss                   0.003 ideal              gdr_param            0.0444   0.0892
ecd   random    loss                   0.003 readout_realistic  raw                  0.0590   0.0360
ecd   random    loss                   0.003 readout_realistic  gdr_param            0.0468   0.0397
ecd   random    loss                   0.030 ideal              raw                  0.1423   0.0060
ecd   random    loss                   0.030 ideal              gdr_param            0.1102   0.0571
ecd   random    loss                   0.030 readout_realistic  raw                  0.1539   0.0233
ecd   random    loss                   0.030 readout_realistic  gdr_param            0.1133   0.1447
ecd   random    loss                   0.100 ideal              raw                  0.2983   0.0834
ecd   random    loss                   0.100 ideal              gdr_param            0.2491   0.0334
ecd   random    loss                   0.100 readout_realistic  raw                  0.3078   0.0560
ecd   random    loss                   0.100 readout_realistic  gdr_param            0.2409   0.0760
ecd   random    comprehensive          0.003 ideal              raw                  0.0816   0.0698
ecd   random    comprehensive          0.003 ideal              gdr_param            0.0839   0.0716
ecd   random    comprehensive          0.003 readout_realistic  raw                  0.1016   0.0501
ecd   random    comprehensive          0.003 readout_realistic  gdr_param            0.1002   0.0554
ecd   random    comprehensive          0.030 ideal              raw                  0.2417   0.1276
ecd   random    comprehensive          0.030 ideal              gdr_param            0.1912   0.0408
ecd   random    comprehensive          0.030 readout_realistic  raw                  0.2237   0.0031
ecd   random    comprehensive          0.030 readout_realistic  gdr_param            0.2078   0.2085
ecd   random    comprehensive          0.100 ideal              raw                  0.4031   0.1404
ecd   random    comprehensive          0.100 ideal              gdr_param            0.4246   0.1003
ecd   random    comprehensive          0.100 readout_realistic  raw                  0.3980   0.1416
ecd   random    comprehensive          0.100 readout_realistic  gdr_param            0.3788   0.1709
ecd   optimized loss                   0.003 ideal              raw                  0.0383   0.1227
ecd   optimized loss                   0.003 ideal              gdr_param            0.0093   0.0441
ecd   optimized loss                   0.003 readout_realistic  raw                  0.0788   0.2615
ecd   optimized loss                   0.003 readout_realistic  gdr_param            0.0104   0.0462
ecd   optimized loss                   0.030 ideal              raw                  0.3145   1.0306
ecd   optimized loss                   0.030 ideal              gdr_param            0.0698   0.3593
ecd   optimized loss                   0.030 readout_realistic  raw                  0.3286   1.1126
ecd   optimized loss                   0.030 readout_realistic  gdr_param            0.0676   0.3550
ecd   optimized loss                   0.100 ideal              raw                  0.6961   2.4681
ecd   optimized loss                   0.100 ideal              gdr_param            0.1997   1.1033
ecd   optimized loss                   0.100 readout_realistic  raw                  0.7090   2.5730
ecd   optimized loss                   0.100 readout_realistic  gdr_param            0.2081   1.1397
ecd   optimized comprehensive          0.003 ideal              raw                  0.1842   0.7821
ecd   optimized comprehensive          0.003 ideal              gdr_param            0.0489   0.2496
ecd   optimized comprehensive          0.003 readout_realistic  raw                  0.2074   0.8398
ecd   optimized comprehensive          0.003 readout_realistic  gdr_param            0.0499   0.2365
ecd   optimized comprehensive          0.030 ideal              raw                  0.5653   2.1211
ecd   optimized comprehensive          0.030 ideal              gdr_param            0.1289   0.7376
ecd   optimized comprehensive          0.030 readout_realistic  raw                  0.5879   2.2317
ecd   optimized comprehensive          0.030 readout_realistic  gdr_param            0.1289   0.7709
ecd   optimized comprehensive          0.100 ideal              raw                  0.9086   3.7039
ecd   optimized comprehensive          0.100 ideal              gdr_param            0.3258   1.7684
ecd   optimized comprehensive          0.100 readout_realistic  raw                  0.9116   3.7765
ecd   optimized comprehensive          0.100 readout_realistic  gdr_param            0.3338   1.7863
```
