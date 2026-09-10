# Ablation summary

tag=`gauss_only/default_ecd_priority` shots=8192 n_train=40 twin=adaptive ansatz=ecd params=both families=loss,comprehensive kappa=0.003,0.03,0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | random | loss | 0.003 | ideal | 0.0434 | 0.0448 | — | — | 0.0461 | 0.0459 | — |
| ecd | random | loss | 0.003 | readout_realistic | 0.0590 | 0.0470 | — | — | 0.0613 | 0.0499 | — |
| ecd | random | loss | 0.03 | ideal | 0.1423 | 0.0909 | — | — | 0.1423 | 0.1283 | — |
| ecd | random | loss | 0.03 | readout_realistic | 0.1539 | 0.0898 | — | — | 0.1540 | 0.1304 | — |
| ecd | random | loss | 0.1 | ideal | 0.2983 | 0.2087 | — | — | 0.2991 | 0.3725 | — |
| ecd | random | loss | 0.1 | readout_realistic | 0.3078 | 0.2221 | — | — | 0.2978 | 0.3940 | — |
| ecd | random | comprehensive | 0.003 | ideal | 0.0816 | 0.0723 | — | — | 0.0801 | 0.0713 | — |
| ecd | random | comprehensive | 0.003 | readout_realistic | 0.1016 | 0.0776 | — | — | 0.1007 | 0.0847 | — |
| ecd | random | comprehensive | 0.03 | ideal | 0.2417 | 0.1497 | — | — | 0.2380 | 0.2841 | — |
| ecd | random | comprehensive | 0.03 | readout_realistic | 0.2237 | 0.1731 | — | — | 0.2357 | 0.2897 | — |
| ecd | random | comprehensive | 0.1 | ideal | 0.4031 | 0.3765 | — | — | 0.3990 | 0.5392 | — |
| ecd | random | comprehensive | 0.1 | readout_realistic | 0.3980 | 0.3410 | — | — | 0.4052 | 0.5224 | — |
| ecd | optimized | loss | 0.003 | ideal | 0.0383 | 0.0093 | — | — | 0.0363 | 0.0075 | — |
| ecd | optimized | loss | 0.003 | readout_realistic | 0.0788 | 0.0107 | — | — | 0.0758 | 0.0107 | — |
| ecd | optimized | loss | 0.03 | ideal | 0.3145 | 0.0714 | — | — | 0.3143 | 0.0724 | — |
| ecd | optimized | loss | 0.03 | readout_realistic | 0.3286 | 0.0699 | — | — | 0.3385 | 0.0731 | — |
| ecd | optimized | loss | 0.1 | ideal | 0.6961 | 0.2011 | — | — | 0.7064 | 0.2015 | — |
| ecd | optimized | loss | 0.1 | readout_realistic | 0.7090 | 0.2067 | — | — | 0.7082 | 0.2064 | — |
| ecd | optimized | comprehensive | 0.003 | ideal | 0.1842 | 0.0522 | — | — | 0.1823 | 0.0534 | — |
| ecd | optimized | comprehensive | 0.003 | readout_realistic | 0.2074 | 0.0539 | — | — | 0.2158 | 0.0523 | — |
| ecd | optimized | comprehensive | 0.03 | ideal | 0.5653 | 0.1374 | — | — | 0.5762 | 0.1434 | — |
| ecd | optimized | comprehensive | 0.03 | readout_realistic | 0.5879 | 0.1367 | — | — | 0.5832 | 0.1484 | — |
| ecd | optimized | comprehensive | 0.1 | ideal | 0.9086 | 0.3434 | — | — | 0.9082 | 0.3429 | — |
| ecd | optimized | comprehensive | 0.1 | readout_realistic | 0.9116 | 0.3522 | — | — | 0.9094 | 0.3519 | — |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   random    loss                   0.003 ideal              raw                  0.0434   0.0785
ecd   random    loss                   0.003 ideal              gdr_param            0.0448   0.0824
ecd   random    loss                   0.003 readout_realistic  raw                  0.0590   0.0360
ecd   random    loss                   0.003 readout_realistic  gdr_param            0.0470   0.0335
ecd   random    loss                   0.030 ideal              raw                  0.1423   0.0060
ecd   random    loss                   0.030 ideal              gdr_param            0.0909   0.0248
ecd   random    loss                   0.030 readout_realistic  raw                  0.1539   0.0233
ecd   random    loss                   0.030 readout_realistic  gdr_param            0.0898   0.0466
ecd   random    loss                   0.100 ideal              raw                  0.2983   0.0834
ecd   random    loss                   0.100 ideal              gdr_param            0.2087   0.1075
ecd   random    loss                   0.100 readout_realistic  raw                  0.3078   0.0560
ecd   random    loss                   0.100 readout_realistic  gdr_param            0.2221   0.0070
ecd   random    comprehensive          0.003 ideal              raw                  0.0816   0.0698
ecd   random    comprehensive          0.003 ideal              gdr_param            0.0723   0.0514
ecd   random    comprehensive          0.003 readout_realistic  raw                  0.1016   0.0501
ecd   random    comprehensive          0.003 readout_realistic  gdr_param            0.0776   0.0336
ecd   random    comprehensive          0.030 ideal              raw                  0.2417   0.1276
ecd   random    comprehensive          0.030 ideal              gdr_param            0.1497   0.1222
ecd   random    comprehensive          0.030 readout_realistic  raw                  0.2237   0.0031
ecd   random    comprehensive          0.030 readout_realistic  gdr_param            0.1731   0.1659
ecd   random    comprehensive          0.100 ideal              raw                  0.4031   0.1404
ecd   random    comprehensive          0.100 ideal              gdr_param            0.3765   0.3634
ecd   random    comprehensive          0.100 readout_realistic  raw                  0.3980   0.1416
ecd   random    comprehensive          0.100 readout_realistic  gdr_param            0.3410   0.0932
ecd   optimized loss                   0.003 ideal              raw                  0.0383   0.1227
ecd   optimized loss                   0.003 ideal              gdr_param            0.0093   0.0442
ecd   optimized loss                   0.003 readout_realistic  raw                  0.0788   0.2615
ecd   optimized loss                   0.003 readout_realistic  gdr_param            0.0107   0.0490
ecd   optimized loss                   0.030 ideal              raw                  0.3145   1.0306
ecd   optimized loss                   0.030 ideal              gdr_param            0.0714   0.3652
ecd   optimized loss                   0.030 readout_realistic  raw                  0.3286   1.1126
ecd   optimized loss                   0.030 readout_realistic  gdr_param            0.0699   0.3609
ecd   optimized loss                   0.100 ideal              raw                  0.6961   2.4681
ecd   optimized loss                   0.100 ideal              gdr_param            0.2011   1.1393
ecd   optimized loss                   0.100 readout_realistic  raw                  0.7090   2.5730
ecd   optimized loss                   0.100 readout_realistic  gdr_param            0.2067   1.1731
ecd   optimized comprehensive          0.003 ideal              raw                  0.1842   0.7821
ecd   optimized comprehensive          0.003 ideal              gdr_param            0.0522   0.2727
ecd   optimized comprehensive          0.003 readout_realistic  raw                  0.2074   0.8398
ecd   optimized comprehensive          0.003 readout_realistic  gdr_param            0.0539   0.2552
ecd   optimized comprehensive          0.030 ideal              raw                  0.5653   2.1211
ecd   optimized comprehensive          0.030 ideal              gdr_param            0.1374   0.7872
ecd   optimized comprehensive          0.030 readout_realistic  raw                  0.5879   2.2317
ecd   optimized comprehensive          0.030 readout_realistic  gdr_param            0.1367   0.7999
ecd   optimized comprehensive          0.100 ideal              raw                  0.9086   3.7039
ecd   optimized comprehensive          0.100 ideal              gdr_param            0.3434   1.8609
ecd   optimized comprehensive          0.100 readout_realistic  raw                  0.9116   3.7765
ecd   optimized comprehensive          0.100 readout_realistic  gdr_param            0.3522   1.8790
```
