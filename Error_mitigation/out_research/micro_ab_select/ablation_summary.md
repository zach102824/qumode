# Ablation summary

tag=`micro_ab_select` shots=4096 n_train=20 twin=default ansatz=ecd params=both families=loss kappa=0.003,0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | random | loss | 0.003 | ideal | 0.0618 | 0.0639 | 0.0622 | gdr_damped | 0.0461 | 0.0459 | -0.0016 |
| ecd | random | loss | 0.003 | readout_realistic | 0.0760 | 0.0657 | 0.0652 | gdr_reg | 0.0613 | 0.0499 | -0.0006 |
| ecd | random | loss | 0.003 | readout_strong | 0.1156 | 0.0885 | 0.0875 | gdr_reg | 0.0996 | 0.0606 | -0.0010 |
| ecd | random | loss | 0.1 | ideal | 0.3169 | 0.3624 | 0.2887 | readout_then_zne | 0.2991 | 0.3725 | -0.0737 |
| ecd | random | loss | 0.1 | readout_realistic | 0.3024 | 0.3900 | 0.2854 | gdr_select | 0.2978 | 0.3940 | -0.1046 |
| ecd | random | loss | 0.1 | readout_strong | 0.3190 | 0.4052 | 0.2858 | readout_then_zne | 0.3142 | 0.3763 | -0.1194 |
| ecd | optimized | loss | 0.003 | ideal | 0.0373 | 0.0114 | 0.0093 | gdr_residual | 0.0363 | 0.0075 | -0.0021 |
| ecd | optimized | loss | 0.003 | readout_realistic | 0.0835 | 0.0179 | 0.0121 | gdr_residual | 0.0758 | 0.0107 | -0.0058 |
| ecd | optimized | loss | 0.003 | readout_strong | 0.1624 | 0.0090 | 0.0076 | gdr_residual | 0.1534 | 0.0075 | -0.0013 |
| ecd | optimized | loss | 0.1 | ideal | 0.7139 | 0.1972 | 0.1921 | gdr_residual | 0.7064 | 0.2015 | -0.0051 |
| ecd | optimized | loss | 0.1 | readout_realistic | 0.7000 | 0.2002 | 0.1914 | gdr_residual | 0.7082 | 0.2064 | -0.0088 |
| ecd | optimized | loss | 0.1 | readout_strong | 0.7144 | 0.2131 | 0.2040 | gdr_residual | 0.7233 | 0.2087 | -0.0091 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   random    loss                   0.003 ideal              raw                  0.0618   0.0058
ecd   random    loss                   0.003 ideal              oracle_binomial      0.0725   0.0127
ecd   random    loss                   0.003 ideal              gdr_param            0.0639   0.0045
ecd   random    loss                   0.003 ideal              gdr_damped           0.0622   0.0051
ecd   random    loss                   0.003 ideal              gdr_ridge            0.0635   0.0083
ecd   random    loss                   0.003 ideal              gdr_holdout          0.0639   0.0045
ecd   random    loss                   0.003 ideal              gdr_reg              0.0622   0.0051
ecd   random    loss                   0.003 ideal              gdr_mid              0.0637   0.0096
ecd   random    loss                   0.003 ideal              gdr_tfree            0.0632   0.0013
ecd   random    loss                   0.003 ideal              gdr_residual         0.0725   0.0127
ecd   random    loss                   0.003 ideal              zne_idle             0.2226   0.0835
ecd   random    loss                   0.003 ideal              readout_then_zne     0.2226   0.0835
ecd   random    loss                   0.003 ideal              zne_then_readout     0.2226   0.0835
ecd   random    loss                   0.003 ideal              gdr_select           0.0635   0.0083
ecd   random    loss                   0.003 readout_realistic  raw                  0.0760   0.0088
ecd   random    loss                   0.003 readout_realistic  readout_only         0.0665   0.0019
ecd   random    loss                   0.003 readout_realistic  oracle_binomial      0.0676   0.0060
ecd   random    loss                   0.003 readout_realistic  gdr_param            0.0657   0.0165
ecd   random    loss                   0.003 readout_realistic  gdr_damped           0.0653   0.0121
ecd   random    loss                   0.003 readout_realistic  gdr_ridge            0.0656   0.0167
ecd   random    loss                   0.003 readout_realistic  gdr_holdout          0.0655   0.0156
ecd   random    loss                   0.003 readout_realistic  gdr_reg              0.0652   0.0115
ecd   random    loss                   0.003 readout_realistic  gdr_mid              0.0656   0.0168
ecd   random    loss                   0.003 readout_realistic  gdr_tfree            0.0655   0.0158
ecd   random    loss                   0.003 readout_realistic  gdr_residual         0.0676   0.0060
ecd   random    loss                   0.003 readout_realistic  zne_idle             0.2355   0.3394
ecd   random    loss                   0.003 readout_realistic  readout_then_zne     0.2438   0.3446
ecd   random    loss                   0.003 readout_realistic  zne_then_readout     0.2434   0.3555
ecd   random    loss                   0.003 readout_realistic  gdr_select           0.0665   0.0019
ecd   random    loss                   0.003 readout_strong     raw                  0.1156   0.0463
ecd   random    loss                   0.003 readout_strong     readout_only         0.0870   0.0448
ecd   random    loss                   0.003 readout_strong     oracle_binomial      0.0926   0.0335
ecd   random    loss                   0.003 readout_strong     gdr_param            0.0885   0.0523
ecd   random    loss                   0.003 readout_strong     gdr_damped           0.0876   0.0493
ecd   random    loss                   0.003 readout_strong     gdr_ridge            0.0885   0.0523
ecd   random    loss                   0.003 readout_strong     gdr_holdout          0.0888   0.0513
ecd   random    loss                   0.003 readout_strong     gdr_reg              0.0875   0.0481
ecd   random    loss                   0.003 readout_strong     gdr_mid              0.0885   0.0513
ecd   random    loss                   0.003 readout_strong     gdr_tfree            0.0880   0.0524
ecd   random    loss                   0.003 readout_strong     gdr_residual         0.0925   0.0311
ecd   random    loss                   0.003 readout_strong     zne_idle             0.2582   0.0046
ecd   random    loss                   0.003 readout_strong     readout_then_zne     0.2965   0.0097
ecd   random    loss                   0.003 readout_strong     zne_then_readout     0.2944   0.0027
ecd   random    loss                   0.003 readout_strong     gdr_select           0.0888   0.0513
ecd   random    loss                   0.100 ideal              raw                  0.3169   0.1161
ecd   random    loss                   0.100 ideal              oracle_binomial      0.5157   0.0772
ecd   random    loss                   0.100 ideal              gdr_param            0.3624   0.1494
ecd   random    loss                   0.100 ideal              gdr_damped           0.2929   0.0697
ecd   random    loss                   0.100 ideal              gdr_ridge            0.3609   0.1468
ecd   random    loss                   0.100 ideal              gdr_holdout          0.3527   0.1124
ecd   random    loss                   0.100 ideal              gdr_reg              0.3064   0.0667
ecd   random    loss                   0.100 ideal              gdr_mid              0.3073   0.0019
ecd   random    loss                   0.100 ideal              gdr_tfree            0.3279   0.0184
ecd   random    loss                   0.100 ideal              gdr_residual         0.4892   0.1567
ecd   random    loss                   0.100 ideal              zne_idle             0.2887   0.2032
ecd   random    loss                   0.100 ideal              readout_then_zne     0.2887   0.2032
ecd   random    loss                   0.100 ideal              zne_then_readout     0.2887   0.2032
ecd   random    loss                   0.100 ideal              gdr_select           0.3073   0.0019
ecd   random    loss                   0.100 readout_realistic  raw                  0.3024   0.1030
ecd   random    loss                   0.100 readout_realistic  readout_only         0.3000   0.1115
ecd   random    loss                   0.100 readout_realistic  oracle_binomial      0.5423   0.1297
ecd   random    loss                   0.100 readout_realistic  gdr_param            0.3900   0.3294
ecd   random    loss                   0.100 readout_realistic  gdr_damped           0.2887   0.1751
ecd   random    loss                   0.100 readout_realistic  gdr_ridge            0.3880   0.3301
ecd   random    loss                   0.100 readout_realistic  gdr_holdout          0.3829   0.3192
ecd   random    loss                   0.100 readout_realistic  gdr_reg              0.3108   0.2115
ecd   random    loss                   0.100 readout_realistic  gdr_mid              0.3302   0.1924
ecd   random    loss                   0.100 readout_realistic  gdr_tfree            0.3589   0.2508
ecd   random    loss                   0.100 readout_realistic  gdr_residual         0.4987   0.2770
ecd   random    loss                   0.100 readout_realistic  zne_idle             0.3099   0.1511
ecd   random    loss                   0.100 readout_realistic  readout_then_zne     0.3167   0.1685
ecd   random    loss                   0.100 readout_realistic  zne_then_readout     0.3168   0.1530
ecd   random    loss                   0.100 readout_realistic  gdr_select           0.2854   0.1316
ecd   random    loss                   0.100 readout_strong     raw                  0.3190   0.0728
ecd   random    loss                   0.100 readout_strong     readout_only         0.3122   0.0896
ecd   random    loss                   0.100 readout_strong     oracle_binomial      0.5382   0.4608
ecd   random    loss                   0.100 readout_strong     gdr_param            0.4052   0.5968
ecd   random    loss                   0.100 readout_strong     gdr_damped           0.3078   0.3939
ecd   random    loss                   0.100 readout_strong     gdr_ridge            0.4043   0.5928
ecd   random    loss                   0.100 readout_strong     gdr_holdout          0.4026   0.5814
ecd   random    loss                   0.100 readout_strong     gdr_reg              0.3161   0.4093
ecd   random    loss                   0.100 readout_strong     gdr_mid              0.3611   0.4399
ecd   random    loss                   0.100 readout_strong     gdr_tfree            0.3820   0.4983
ecd   random    loss                   0.100 readout_strong     gdr_residual         0.4975   0.6695
ecd   random    loss                   0.100 readout_strong     zne_idle             0.2745   0.3403
ecd   random    loss                   0.100 readout_strong     readout_then_zne     0.2858   0.2789
ecd   random    loss                   0.100 readout_strong     zne_then_readout     0.2859   0.3148
ecd   random    loss                   0.100 readout_strong     gdr_select           0.3390   0.4049
ecd   optimized loss                   0.003 ideal              raw                  0.0373   0.1230
ecd   optimized loss                   0.003 ideal              oracle_binomial      0.0093   0.0471
ecd   optimized loss                   0.003 ideal              gdr_param            0.0114   0.0530
ecd   optimized loss                   0.003 ideal              gdr_damped           0.0114   0.0530
ecd   optimized loss                   0.003 ideal              gdr_ridge            0.0113   0.0528
ecd   optimized loss                   0.003 ideal              gdr_holdout          0.0105   0.0506
ecd   optimized loss                   0.003 ideal              gdr_reg              0.0118   0.0542
ecd   optimized loss                   0.003 ideal              gdr_mid              0.0104   0.0504
ecd   optimized loss                   0.003 ideal              gdr_tfree            0.0095   0.0479
ecd   optimized loss                   0.003 ideal              gdr_residual         0.0093   0.0471
ecd   optimized loss                   0.003 ideal              zne_idle             0.0142   0.0431
ecd   optimized loss                   0.003 ideal              readout_then_zne     0.0142   0.0431
ecd   optimized loss                   0.003 ideal              zne_then_readout     0.0142   0.0431
ecd   optimized loss                   0.003 ideal              gdr_select           0.0105   0.0506
ecd   optimized loss                   0.003 readout_realistic  raw                  0.0835   0.2783
ecd   optimized loss                   0.003 readout_realistic  readout_only         0.0465   0.1493
ecd   optimized loss                   0.003 readout_realistic  oracle_binomial      0.0122   0.0554
ecd   optimized loss                   0.003 readout_realistic  gdr_param            0.0179   0.0720
ecd   optimized loss                   0.003 readout_realistic  gdr_damped           0.0179   0.0720
ecd   optimized loss                   0.003 readout_realistic  gdr_ridge            0.0179   0.0720
ecd   optimized loss                   0.003 readout_realistic  gdr_holdout          0.0171   0.0696
ecd   optimized loss                   0.003 readout_realistic  gdr_reg              0.0171   0.0696
ecd   optimized loss                   0.003 readout_realistic  gdr_mid              0.0179   0.0719
ecd   optimized loss                   0.003 readout_realistic  gdr_tfree            0.0259   0.0932
ecd   optimized loss                   0.003 readout_realistic  gdr_residual         0.0121   0.0552
ecd   optimized loss                   0.003 readout_realistic  zne_idle             0.0778   0.2456
ecd   optimized loss                   0.003 readout_realistic  readout_then_zne     0.0361   0.1201
ecd   optimized loss                   0.003 readout_realistic  zne_then_readout     0.0461   0.1456
ecd   optimized loss                   0.003 readout_realistic  gdr_select           0.0179   0.0720
ecd   optimized loss                   0.003 readout_strong     raw                  0.1624   0.5627
ecd   optimized loss                   0.003 readout_strong     readout_only         0.0446   0.1428
ecd   optimized loss                   0.003 readout_strong     oracle_binomial      0.0076   0.0329
ecd   optimized loss                   0.003 readout_strong     gdr_param            0.0090   0.0440
ecd   optimized loss                   0.003 readout_strong     gdr_damped           0.0090   0.0440
ecd   optimized loss                   0.003 readout_strong     gdr_ridge            0.0090   0.0440
ecd   optimized loss                   0.003 readout_strong     gdr_holdout          0.0090   0.0440
ecd   optimized loss                   0.003 readout_strong     gdr_reg              0.0090   0.0440
ecd   optimized loss                   0.003 readout_strong     gdr_mid              0.0090   0.0440
ecd   optimized loss                   0.003 readout_strong     gdr_tfree            0.0092   0.0450
ecd   optimized loss                   0.003 readout_strong     gdr_residual         0.0076   0.0329
ecd   optimized loss                   0.003 readout_strong     zne_idle             0.1451   0.5134
ecd   optimized loss                   0.003 readout_strong     readout_then_zne     0.0251   0.1222
ecd   optimized loss                   0.003 readout_strong     zne_then_readout     0.0314   0.1384
ecd   optimized loss                   0.003 readout_strong     gdr_select           0.0090   0.0440
ecd   optimized loss                   0.100 ideal              raw                  0.7139   2.5331
ecd   optimized loss                   0.100 ideal              oracle_binomial      0.1953   1.0431
ecd   optimized loss                   0.100 ideal              gdr_param            0.1972   1.1224
ecd   optimized loss                   0.100 ideal              gdr_damped           0.3003   1.4046
ecd   optimized loss                   0.100 ideal              gdr_ridge            0.1978   1.1309
ecd   optimized loss                   0.100 ideal              gdr_holdout          0.1972   1.1224
ecd   optimized loss                   0.100 ideal              gdr_reg              0.3003   1.4046
ecd   optimized loss                   0.100 ideal              gdr_mid              0.1998   1.1426
ecd   optimized loss                   0.100 ideal              gdr_tfree            0.2000   1.1391
ecd   optimized loss                   0.100 ideal              gdr_residual         0.1921   1.0243
ecd   optimized loss                   0.100 ideal              zne_idle             0.4417   1.3990
ecd   optimized loss                   0.100 ideal              readout_then_zne     0.4417   1.3990
ecd   optimized loss                   0.100 ideal              zne_then_readout     0.4417   1.3990
ecd   optimized loss                   0.100 ideal              gdr_select           0.3003   1.4046
ecd   optimized loss                   0.100 readout_realistic  raw                  0.7000   2.5165
ecd   optimized loss                   0.100 readout_realistic  readout_only         0.6941   2.4465
ecd   optimized loss                   0.100 readout_realistic  oracle_binomial      0.1950   1.0363
ecd   optimized loss                   0.100 readout_realistic  gdr_param            0.2002   1.1230
ecd   optimized loss                   0.100 readout_realistic  gdr_damped           0.3234   1.4539
ecd   optimized loss                   0.100 readout_realistic  gdr_ridge            0.2008   1.1376
ecd   optimized loss                   0.100 readout_realistic  gdr_holdout          0.2004   1.1277
ecd   optimized loss                   0.100 readout_realistic  gdr_reg              0.3236   1.4574
ecd   optimized loss                   0.100 readout_realistic  gdr_mid              0.2033   1.1621
ecd   optimized loss                   0.100 readout_realistic  gdr_tfree            0.2042   1.1623
ecd   optimized loss                   0.100 readout_realistic  gdr_residual         0.1914   1.0176
ecd   optimized loss                   0.100 readout_realistic  zne_idle             0.4433   1.4825
ecd   optimized loss                   0.100 readout_realistic  readout_then_zne     0.4299   1.3745
ecd   optimized loss                   0.100 readout_realistic  zne_then_readout     0.4275   1.3826
ecd   optimized loss                   0.100 readout_realistic  gdr_select           0.3234   1.4539
ecd   optimized loss                   0.100 readout_strong     raw                  0.7144   2.7226
ecd   optimized loss                   0.100 readout_strong     readout_only         0.6936   2.4951
ecd   optimized loss                   0.100 readout_strong     oracle_binomial      0.2088   1.1130
ecd   optimized loss                   0.100 readout_strong     gdr_param            0.2131   1.1624
ecd   optimized loss                   0.100 readout_strong     gdr_damped           0.3089   1.4289
ecd   optimized loss                   0.100 readout_strong     gdr_ridge            0.2134   1.1788
ecd   optimized loss                   0.100 readout_strong     gdr_holdout          0.2131   1.1624
ecd   optimized loss                   0.100 readout_strong     gdr_reg              0.3089   1.4289
ecd   optimized loss                   0.100 readout_strong     gdr_mid              0.2148   1.2126
ecd   optimized loss                   0.100 readout_strong     gdr_tfree            0.2146   1.2100
ecd   optimized loss                   0.100 readout_strong     gdr_residual         0.2040   1.0933
ecd   optimized loss                   0.100 readout_strong     zne_idle             0.4684   1.5863
ecd   optimized loss                   0.100 readout_strong     readout_then_zne     0.4278   1.3955
ecd   optimized loss                   0.100 readout_strong     zne_then_readout     0.4143   1.3835
ecd   optimized loss                   0.100 readout_strong     gdr_select           0.3089   1.4289
```
