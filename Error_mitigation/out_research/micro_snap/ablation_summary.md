# Ablation summary

tag=`micro_snap` shots=4096 n_train=20 twin=span ansatz=snap params=both families=loss kappa=0.003,0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| snap | random | loss | 0.003 | ideal | 0.0473 | 0.0419 | 0.0391 | gdr_residual | 0.0194 | 0.0223 | -0.0028 |
| snap | random | loss | 0.003 | readout_realistic | 0.0453 | 0.0451 | 0.0419 | gdr_reg | 0.0424 | 0.0308 | -0.0031 |
| snap | random | loss | 0.003 | readout_strong | 0.0935 | 0.0436 | 0.0432 | gdr_mid | 0.0974 | 0.0356 | -0.0004 |
| snap | random | loss | 0.1 | ideal | 0.3418 | 0.2268 | 0.1964 | gdr_select | 0.3330 | 0.2329 | -0.0304 |
| snap | random | loss | 0.1 | readout_realistic | 0.3518 | 0.2189 | 0.1946 | gdr_reg | 0.3444 | 0.2449 | -0.0243 |
| snap | random | loss | 0.1 | readout_strong | 0.3547 | 0.2257 | 0.1941 | gdr_reg | 0.3545 | 0.2580 | -0.0316 |
| snap | optimized | loss | 0.003 | ideal | 0.0446 | 0.0139 | 0.0134 | gdr_tfree | 0.0546 | 0.0146 | -0.0005 |
| snap | optimized | loss | 0.003 | readout_realistic | 0.0966 | 0.0173 | 0.0128 | gdr_residual | 0.0954 | 0.0191 | -0.0045 |
| snap | optimized | loss | 0.003 | readout_strong | 0.1818 | 0.0202 | 0.0163 | gdr_residual | 0.1830 | 0.0127 | -0.0039 |
| snap | optimized | loss | 0.1 | ideal | 0.6576 | 0.4034 | 0.3852 | gdr_tfree | 0.6658 | 0.4163 | -0.0182 |
| snap | optimized | loss | 0.1 | readout_realistic | 0.6713 | 0.4177 | 0.3943 | gdr_tfree | 0.6552 | 0.3673 | -0.0233 |
| snap | optimized | loss | 0.1 | readout_strong | 0.6781 | 0.4117 | 0.3849 | gdr_tfree | 0.6794 | 0.4051 | -0.0269 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
snap  random    loss                   0.003 ideal              raw                  0.0473   0.0095
snap  random    loss                   0.003 ideal              oracle_binomial      0.0394   0.0206
snap  random    loss                   0.003 ideal              gdr_param            0.0419   0.0146
snap  random    loss                   0.003 ideal              gdr_damped           0.0431   0.0130
snap  random    loss                   0.003 ideal              gdr_ridge            0.0419   0.0146
snap  random    loss                   0.003 ideal              gdr_holdout          0.0419   0.0137
snap  random    loss                   0.003 ideal              gdr_reg              0.0430   0.0127
snap  random    loss                   0.003 ideal              gdr_mid              0.0422   0.0102
snap  random    loss                   0.003 ideal              gdr_tfree            0.0421   0.0169
snap  random    loss                   0.003 ideal              gdr_residual         0.0391   0.0323
snap  random    loss                   0.003 ideal              zne_idle             0.1640   0.1057
snap  random    loss                   0.003 ideal              readout_then_zne     0.1640   0.1057
snap  random    loss                   0.003 ideal              zne_then_readout     0.1640   0.1057
snap  random    loss                   0.003 ideal              gdr_select           0.0394   0.0206
snap  random    loss                   0.003 readout_realistic  raw                  0.0453   0.0014
snap  random    loss                   0.003 readout_realistic  readout_only         0.0384   0.0163
snap  random    loss                   0.003 readout_realistic  oracle_binomial      0.0453   0.0302
snap  random    loss                   0.003 readout_realistic  gdr_param            0.0451   0.0184
snap  random    loss                   0.003 readout_realistic  gdr_damped           0.0433   0.0180
snap  random    loss                   0.003 readout_realistic  gdr_ridge            0.0443   0.0163
snap  random    loss                   0.003 readout_realistic  gdr_holdout          0.0443   0.0164
snap  random    loss                   0.003 readout_realistic  gdr_reg              0.0419   0.0164
snap  random    loss                   0.003 readout_realistic  gdr_mid              0.0441   0.0212
snap  random    loss                   0.003 readout_realistic  gdr_tfree            0.0445   0.0176
snap  random    loss                   0.003 readout_realistic  gdr_residual         0.0471   0.0457
snap  random    loss                   0.003 readout_realistic  zne_idle             0.1639   0.1179
snap  random    loss                   0.003 readout_realistic  readout_then_zne     0.1658   0.1433
snap  random    loss                   0.003 readout_realistic  zne_then_readout     0.1690   0.1394
snap  random    loss                   0.003 readout_realistic  gdr_select           0.0453   0.0302
snap  random    loss                   0.003 readout_strong     raw                  0.0935   0.0638
snap  random    loss                   0.003 readout_strong     readout_only         0.0471   0.0096
snap  random    loss                   0.003 readout_strong     oracle_binomial      0.0440   0.0250
snap  random    loss                   0.003 readout_strong     gdr_param            0.0436   0.0284
snap  random    loss                   0.003 readout_strong     gdr_damped           0.0434   0.0265
snap  random    loss                   0.003 readout_strong     gdr_ridge            0.0434   0.0285
snap  random    loss                   0.003 readout_strong     gdr_holdout          0.0436   0.0284
snap  random    loss                   0.003 readout_strong     gdr_reg              0.0434   0.0265
snap  random    loss                   0.003 readout_strong     gdr_mid              0.0432   0.0365
snap  random    loss                   0.003 readout_strong     gdr_tfree            0.0436   0.0325
snap  random    loss                   0.003 readout_strong     gdr_residual         0.0436   0.0309
snap  random    loss                   0.003 readout_strong     zne_idle             0.1799   0.1799
snap  random    loss                   0.003 readout_strong     readout_then_zne     0.1951   0.1570
snap  random    loss                   0.003 readout_strong     zne_then_readout     0.1959   0.1451
snap  random    loss                   0.003 readout_strong     gdr_select           0.0440   0.0250
snap  random    loss                   0.100 ideal              raw                  0.3418   0.2823
snap  random    loss                   0.100 ideal              oracle_binomial      0.2559   0.4913
snap  random    loss                   0.100 ideal              gdr_param            0.2268   0.4309
snap  random    loss                   0.100 ideal              gdr_damped           0.1991   0.4012
snap  random    loss                   0.100 ideal              gdr_ridge            0.2259   0.4345
snap  random    loss                   0.100 ideal              gdr_holdout          0.2149   0.4535
snap  random    loss                   0.100 ideal              gdr_reg              0.2015   0.4279
snap  random    loss                   0.100 ideal              gdr_mid              0.2159   0.4426
snap  random    loss                   0.100 ideal              gdr_tfree            0.2220   0.3982
snap  random    loss                   0.100 ideal              gdr_residual         0.2692   1.0347
snap  random    loss                   0.100 ideal              zne_idle             0.2262   0.0329
snap  random    loss                   0.100 ideal              readout_then_zne     0.2262   0.0329
snap  random    loss                   0.100 ideal              zne_then_readout     0.2262   0.0329
snap  random    loss                   0.100 ideal              gdr_select           0.1964   0.3938
snap  random    loss                   0.100 readout_realistic  raw                  0.3518   0.1875
snap  random    loss                   0.100 readout_realistic  readout_only         0.3492   0.2130
snap  random    loss                   0.100 readout_realistic  oracle_binomial      0.2516   0.4069
snap  random    loss                   0.100 readout_realistic  gdr_param            0.2189   0.3073
snap  random    loss                   0.100 readout_realistic  gdr_damped           0.1977   0.2837
snap  random    loss                   0.100 readout_realistic  gdr_ridge            0.2188   0.3143
snap  random    loss                   0.100 readout_realistic  gdr_holdout          0.2135   0.3135
snap  random    loss                   0.100 readout_realistic  gdr_reg              0.1946   0.2934
snap  random    loss                   0.100 readout_realistic  gdr_mid              0.2129   0.3334
snap  random    loss                   0.100 readout_realistic  gdr_tfree            0.2124   0.2547
snap  random    loss                   0.100 readout_realistic  gdr_residual         0.2575   0.9496
snap  random    loss                   0.100 readout_realistic  zne_idle             0.2130   0.1101
snap  random    loss                   0.100 readout_realistic  readout_then_zne     0.2039   0.1371
snap  random    loss                   0.100 readout_realistic  zne_then_readout     0.2054   0.1327
snap  random    loss                   0.100 readout_realistic  gdr_select           0.1987   0.2790
snap  random    loss                   0.100 readout_strong     raw                  0.3547   0.0679
snap  random    loss                   0.100 readout_strong     readout_only         0.3419   0.1264
snap  random    loss                   0.100 readout_strong     oracle_binomial      0.2745   0.3008
snap  random    loss                   0.100 readout_strong     gdr_param            0.2257   0.1456
snap  random    loss                   0.100 readout_strong     gdr_damped           0.2004   0.1408
snap  random    loss                   0.100 readout_strong     gdr_ridge            0.2220   0.1581
snap  random    loss                   0.100 readout_strong     gdr_holdout          0.2177   0.1611
snap  random    loss                   0.100 readout_strong     gdr_reg              0.1941   0.1524
snap  random    loss                   0.100 readout_strong     gdr_mid              0.2113   0.1864
snap  random    loss                   0.100 readout_strong     gdr_tfree            0.2062   0.0598
snap  random    loss                   0.100 readout_strong     gdr_residual         0.2348   0.7577
snap  random    loss                   0.100 readout_strong     zne_idle             0.2988   0.5813
snap  random    loss                   0.100 readout_strong     readout_then_zne     0.2906   0.5076
snap  random    loss                   0.100 readout_strong     zne_then_readout     0.2826   0.5440
snap  random    loss                   0.100 readout_strong     gdr_select           0.2033   0.1417
snap  optimized loss                   0.003 ideal              raw                  0.0446   0.1424
snap  optimized loss                   0.003 ideal              oracle_binomial      0.0206   0.0949
snap  optimized loss                   0.003 ideal              gdr_param            0.0139   0.0623
snap  optimized loss                   0.003 ideal              gdr_damped           0.0156   0.0743
snap  optimized loss                   0.003 ideal              gdr_ridge            0.0139   0.0623
snap  optimized loss                   0.003 ideal              gdr_holdout          0.0138   0.0621
snap  optimized loss                   0.003 ideal              gdr_reg              0.0156   0.0742
snap  optimized loss                   0.003 ideal              gdr_mid              0.0138   0.0624
snap  optimized loss                   0.003 ideal              gdr_tfree            0.0134   0.0587
snap  optimized loss                   0.003 ideal              gdr_residual         0.0143   0.0596
snap  optimized loss                   0.003 ideal              zne_idle             0.0347   0.1588
snap  optimized loss                   0.003 ideal              readout_then_zne     0.0347   0.1588
snap  optimized loss                   0.003 ideal              zne_then_readout     0.0347   0.1588
snap  optimized loss                   0.003 ideal              gdr_select           0.0139   0.0623
snap  optimized loss                   0.003 readout_realistic  raw                  0.0966   0.2440
snap  optimized loss                   0.003 readout_realistic  readout_only         0.0565   0.1386
snap  optimized loss                   0.003 readout_realistic  oracle_binomial      0.0244   0.0767
snap  optimized loss                   0.003 readout_realistic  gdr_param            0.0173   0.0453
snap  optimized loss                   0.003 readout_realistic  gdr_damped           0.0173   0.0453
snap  optimized loss                   0.003 readout_realistic  gdr_ridge            0.0173   0.0451
snap  optimized loss                   0.003 readout_realistic  gdr_holdout          0.0173   0.0453
snap  optimized loss                   0.003 readout_realistic  gdr_reg              0.0173   0.0453
snap  optimized loss                   0.003 readout_realistic  gdr_mid              0.0182   0.0500
snap  optimized loss                   0.003 readout_realistic  gdr_tfree            0.0164   0.0396
snap  optimized loss                   0.003 readout_realistic  gdr_residual         0.0128   0.0372
snap  optimized loss                   0.003 readout_realistic  zne_idle             0.0498   0.1061
snap  optimized loss                   0.003 readout_realistic  readout_then_zne     0.0314   0.0481
snap  optimized loss                   0.003 readout_realistic  zne_then_readout     0.0342   0.0471
snap  optimized loss                   0.003 readout_realistic  gdr_select           0.0173   0.0451
snap  optimized loss                   0.003 readout_strong     raw                  0.1818   0.4767
snap  optimized loss                   0.003 readout_strong     readout_only         0.0501   0.1084
snap  optimized loss                   0.003 readout_strong     oracle_binomial      0.0240   0.0461
snap  optimized loss                   0.003 readout_strong     gdr_param            0.0202   0.0123
snap  optimized loss                   0.003 readout_strong     gdr_damped           0.0203   0.0219
snap  optimized loss                   0.003 readout_strong     gdr_ridge            0.0203   0.0127
snap  optimized loss                   0.003 readout_strong     gdr_holdout          0.0203   0.0127
snap  optimized loss                   0.003 readout_strong     gdr_reg              0.0204   0.0222
snap  optimized loss                   0.003 readout_strong     gdr_mid              0.0201   0.0210
snap  optimized loss                   0.003 readout_strong     gdr_tfree            0.0221   0.0038
snap  optimized loss                   0.003 readout_strong     gdr_residual         0.0163   0.0109
snap  optimized loss                   0.003 readout_strong     zne_idle             0.1559   0.5615
snap  optimized loss                   0.003 readout_strong     readout_then_zne     0.0665   0.2640
snap  optimized loss                   0.003 readout_strong     zne_then_readout     0.0725   0.2617
snap  optimized loss                   0.003 readout_strong     gdr_select           0.0240   0.0461
snap  optimized loss                   0.100 ideal              raw                  0.6576   2.7297
snap  optimized loss                   0.100 ideal              oracle_binomial      0.5159   1.8846
snap  optimized loss                   0.100 ideal              gdr_param            0.4034   1.0257
snap  optimized loss                   0.100 ideal              gdr_damped           0.4663   1.4517
snap  optimized loss                   0.100 ideal              gdr_ridge            0.4038   1.0280
snap  optimized loss                   0.100 ideal              gdr_holdout          0.4146   1.0912
snap  optimized loss                   0.100 ideal              gdr_reg              0.4746   1.5008
snap  optimized loss                   0.100 ideal              gdr_mid              0.4085   1.2125
snap  optimized loss                   0.100 ideal              gdr_tfree            0.3852   0.9108
snap  optimized loss                   0.100 ideal              gdr_residual         0.4549   1.1711
snap  optimized loss                   0.100 ideal              zne_idle             0.5560   1.8743
snap  optimized loss                   0.100 ideal              readout_then_zne     0.5560   1.8743
snap  optimized loss                   0.100 ideal              zne_then_readout     0.5560   1.8743
snap  optimized loss                   0.100 ideal              gdr_select           0.4663   1.4517
snap  optimized loss                   0.100 readout_realistic  raw                  0.6713   2.8819
snap  optimized loss                   0.100 readout_realistic  readout_only         0.6664   2.8320
snap  optimized loss                   0.100 readout_realistic  oracle_binomial      0.5229   1.9616
snap  optimized loss                   0.100 readout_realistic  gdr_param            0.4177   1.2393
snap  optimized loss                   0.100 readout_realistic  gdr_damped           0.4791   1.6375
snap  optimized loss                   0.100 readout_realistic  gdr_ridge            0.4178   1.2407
snap  optimized loss                   0.100 readout_realistic  gdr_holdout          0.4199   1.2552
snap  optimized loss                   0.100 readout_realistic  gdr_reg              0.4808   1.6494
snap  optimized loss                   0.100 readout_realistic  gdr_mid              0.4238   1.4061
snap  optimized loss                   0.100 readout_realistic  gdr_tfree            0.3943   1.1099
snap  optimized loss                   0.100 readout_realistic  gdr_residual         0.4372   1.3994
snap  optimized loss                   0.100 readout_realistic  zne_idle             0.5885   2.2606
snap  optimized loss                   0.100 readout_realistic  readout_then_zne     0.5836   2.2220
snap  optimized loss                   0.100 readout_realistic  zne_then_readout     0.5816   2.1998
snap  optimized loss                   0.100 readout_realistic  gdr_select           0.4668   1.5579
snap  optimized loss                   0.100 readout_strong     raw                  0.6781   2.9431
snap  optimized loss                   0.100 readout_strong     readout_only         0.6598   2.7795
snap  optimized loss                   0.100 readout_strong     oracle_binomial      0.5229   2.0439
snap  optimized loss                   0.100 readout_strong     gdr_param            0.4117   1.3104
snap  optimized loss                   0.100 readout_strong     gdr_damped           0.4729   1.6777
snap  optimized loss                   0.100 readout_strong     gdr_ridge            0.4112   1.3096
snap  optimized loss                   0.100 readout_strong     gdr_holdout          0.4102   1.3131
snap  optimized loss                   0.100 readout_strong     gdr_reg              0.4595   1.6064
snap  optimized loss                   0.100 readout_strong     gdr_mid              0.4299   1.5069
snap  optimized loss                   0.100 readout_strong     gdr_tfree            0.3849   1.1510
snap  optimized loss                   0.100 readout_strong     gdr_residual         0.4397   1.3579
snap  optimized loss                   0.100 readout_strong     zne_idle             0.6147   2.3207
snap  optimized loss                   0.100 readout_strong     readout_then_zne     0.6005   2.1368
snap  optimized loss                   0.100 readout_strong     zne_then_readout     0.5921   2.0999
snap  optimized loss                   0.100 readout_strong     gdr_select           0.4102   1.3131
```
