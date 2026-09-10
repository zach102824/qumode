# Ablation summary

tag=`micro_span` shots=4096 n_train=20 twin=span ansatz=ecd params=both families=loss kappa=0.003,0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | random | loss | 0.003 | ideal | 0.0618 | 0.0678 | 0.0618 | gdr_select | 0.0461 | 0.0459 | -0.0061 |
| ecd | random | loss | 0.003 | readout_realistic | 0.0760 | 0.0640 | 0.0638 | gdr_holdout | 0.0613 | 0.0499 | -0.0003 |
| ecd | random | loss | 0.003 | readout_strong | 0.1156 | 0.0905 | 0.0870 | gdr_damped | 0.0996 | 0.0606 | -0.0035 |
| ecd | random | loss | 0.1 | ideal | 0.3169 | 0.2367 | 0.2314 | gdr_tfree | 0.2991 | 0.3725 | -0.0054 |
| ecd | random | loss | 0.1 | readout_realistic | 0.3024 | 0.2331 | 0.2196 | gdr_damped | 0.2978 | 0.3940 | -0.0135 |
| ecd | random | loss | 0.1 | readout_strong | 0.3190 | 0.2680 | 0.2543 | gdr_damped | 0.3142 | 0.3763 | -0.0137 |
| ecd | optimized | loss | 0.003 | ideal | 0.0373 | 0.0095 | 0.0093 | gdr_residual | 0.0363 | 0.0075 | -0.0002 |
| ecd | optimized | loss | 0.003 | readout_realistic | 0.0835 | 0.0131 | 0.0122 | gdr_residual | 0.0758 | 0.0107 | -0.0009 |
| ecd | optimized | loss | 0.003 | readout_strong | 0.1624 | 0.0133 | 0.0076 | gdr_residual | 0.1534 | 0.0075 | -0.0057 |
| ecd | optimized | loss | 0.1 | ideal | 0.7139 | 0.2130 | 0.1921 | gdr_residual | 0.7064 | 0.2015 | -0.0209 |
| ecd | optimized | loss | 0.1 | readout_realistic | 0.7000 | 0.2073 | 0.1914 | gdr_residual | 0.7082 | 0.2064 | -0.0159 |
| ecd | optimized | loss | 0.1 | readout_strong | 0.7144 | 0.2195 | 0.2040 | gdr_residual | 0.7233 | 0.2087 | -0.0155 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   random    loss                   0.003 ideal              raw                  0.0618   0.0058
ecd   random    loss                   0.003 ideal              oracle_binomial      0.0725   0.0127
ecd   random    loss                   0.003 ideal              gdr_param            0.0678   0.0110
ecd   random    loss                   0.003 ideal              gdr_damped           0.0622   0.0071
ecd   random    loss                   0.003 ideal              gdr_ridge            0.0675   0.0106
ecd   random    loss                   0.003 ideal              gdr_holdout          0.0675   0.0106
ecd   random    loss                   0.003 ideal              gdr_reg              0.0621   0.0070
ecd   random    loss                   0.003 ideal              gdr_mid              0.0672   0.0111
ecd   random    loss                   0.003 ideal              gdr_tfree            0.0672   0.0095
ecd   random    loss                   0.003 ideal              gdr_residual         0.0738   0.0180
ecd   random    loss                   0.003 ideal              zne_idle             0.2226   0.0835
ecd   random    loss                   0.003 ideal              readout_then_zne     0.2226   0.0835
ecd   random    loss                   0.003 ideal              zne_then_readout     0.2226   0.0835
ecd   random    loss                   0.003 ideal              gdr_select           0.0618   0.0058
ecd   random    loss                   0.003 readout_realistic  raw                  0.0760   0.0088
ecd   random    loss                   0.003 readout_realistic  readout_only         0.0665   0.0019
ecd   random    loss                   0.003 readout_realistic  oracle_binomial      0.0676   0.0060
ecd   random    loss                   0.003 readout_realistic  gdr_param            0.0640   0.0115
ecd   random    loss                   0.003 readout_realistic  gdr_damped           0.0653   0.0035
ecd   random    loss                   0.003 readout_realistic  gdr_ridge            0.0643   0.0120
ecd   random    loss                   0.003 readout_realistic  gdr_holdout          0.0638   0.0120
ecd   random    loss                   0.003 readout_realistic  gdr_reg              0.0664   0.0012
ecd   random    loss                   0.003 readout_realistic  gdr_mid              0.0638   0.0124
ecd   random    loss                   0.003 readout_realistic  gdr_tfree            0.0648   0.0124
ecd   random    loss                   0.003 readout_realistic  gdr_residual         0.0689   0.0031
ecd   random    loss                   0.003 readout_realistic  zne_idle             0.2355   0.3394
ecd   random    loss                   0.003 readout_realistic  readout_then_zne     0.2438   0.3446
ecd   random    loss                   0.003 readout_realistic  zne_then_readout     0.2434   0.3555
ecd   random    loss                   0.003 readout_realistic  gdr_select           0.0665   0.0019
ecd   random    loss                   0.003 readout_strong     raw                  0.1156   0.0463
ecd   random    loss                   0.003 readout_strong     readout_only         0.0870   0.0448
ecd   random    loss                   0.003 readout_strong     oracle_binomial      0.0926   0.0335
ecd   random    loss                   0.003 readout_strong     gdr_param            0.0905   0.0620
ecd   random    loss                   0.003 readout_strong     gdr_damped           0.0870   0.0448
ecd   random    loss                   0.003 readout_strong     gdr_ridge            0.0907   0.0595
ecd   random    loss                   0.003 readout_strong     gdr_holdout          0.0907   0.0595
ecd   random    loss                   0.003 readout_strong     gdr_reg              0.0870   0.0448
ecd   random    loss                   0.003 readout_strong     gdr_mid              0.0906   0.0569
ecd   random    loss                   0.003 readout_strong     gdr_tfree            0.0907   0.0605
ecd   random    loss                   0.003 readout_strong     gdr_residual         0.0923   0.0272
ecd   random    loss                   0.003 readout_strong     zne_idle             0.2582   0.0046
ecd   random    loss                   0.003 readout_strong     readout_then_zne     0.2965   0.0097
ecd   random    loss                   0.003 readout_strong     zne_then_readout     0.2944   0.0027
ecd   random    loss                   0.003 readout_strong     gdr_select           0.0898   0.0565
ecd   random    loss                   0.100 ideal              raw                  0.3169   0.1161
ecd   random    loss                   0.100 ideal              oracle_binomial      0.5157   0.0772
ecd   random    loss                   0.100 ideal              gdr_param            0.2367   0.0564
ecd   random    loss                   0.100 ideal              gdr_damped           0.2331   0.0654
ecd   random    loss                   0.100 ideal              gdr_ridge            0.2374   0.0492
ecd   random    loss                   0.100 ideal              gdr_holdout          0.2385   0.0392
ecd   random    loss                   0.100 ideal              gdr_reg              0.2361   0.0469
ecd   random    loss                   0.100 ideal              gdr_mid              0.2631   0.0663
ecd   random    loss                   0.100 ideal              gdr_tfree            0.2314   0.1439
ecd   random    loss                   0.100 ideal              gdr_residual         0.4637   0.4018
ecd   random    loss                   0.100 ideal              zne_idle             0.2887   0.2032
ecd   random    loss                   0.100 ideal              readout_then_zne     0.2887   0.2032
ecd   random    loss                   0.100 ideal              zne_then_readout     0.2887   0.2032
ecd   random    loss                   0.100 ideal              gdr_select           0.2385   0.0392
ecd   random    loss                   0.100 readout_realistic  raw                  0.3024   0.1030
ecd   random    loss                   0.100 readout_realistic  readout_only         0.3000   0.1115
ecd   random    loss                   0.100 readout_realistic  oracle_binomial      0.5423   0.1297
ecd   random    loss                   0.100 readout_realistic  gdr_param            0.2331   0.1138
ecd   random    loss                   0.100 readout_realistic  gdr_damped           0.2196   0.1135
ecd   random    loss                   0.100 readout_realistic  gdr_ridge            0.2315   0.1062
ecd   random    loss                   0.100 readout_realistic  gdr_holdout          0.2307   0.0982
ecd   random    loss                   0.100 readout_realistic  gdr_reg              0.2206   0.0995
ecd   random    loss                   0.100 readout_realistic  gdr_mid              0.2558   0.0097
ecd   random    loss                   0.100 readout_realistic  gdr_tfree            0.2335   0.1874
ecd   random    loss                   0.100 readout_realistic  gdr_residual         0.4922   0.4114
ecd   random    loss                   0.100 readout_realistic  zne_idle             0.3099   0.1511
ecd   random    loss                   0.100 readout_realistic  readout_then_zne     0.3167   0.1685
ecd   random    loss                   0.100 readout_realistic  zne_then_readout     0.3168   0.1530
ecd   random    loss                   0.100 readout_realistic  gdr_select           0.2558   0.0097
ecd   random    loss                   0.100 readout_strong     raw                  0.3190   0.0728
ecd   random    loss                   0.100 readout_strong     readout_only         0.3122   0.0896
ecd   random    loss                   0.100 readout_strong     oracle_binomial      0.5382   0.4608
ecd   random    loss                   0.100 readout_strong     gdr_param            0.2680   0.2998
ecd   random    loss                   0.100 readout_strong     gdr_damped           0.2543   0.2578
ecd   random    loss                   0.100 readout_strong     gdr_ridge            0.2672   0.2949
ecd   random    loss                   0.100 readout_strong     gdr_holdout          0.2672   0.2949
ecd   random    loss                   0.100 readout_strong     gdr_reg              0.2572   0.2641
ecd   random    loss                   0.100 readout_strong     gdr_mid              0.2896   0.3268
ecd   random    loss                   0.100 readout_strong     gdr_tfree            0.2638   0.2760
ecd   random    loss                   0.100 readout_strong     gdr_residual         0.4863   0.7223
ecd   random    loss                   0.100 readout_strong     zne_idle             0.2745   0.3403
ecd   random    loss                   0.100 readout_strong     readout_then_zne     0.2858   0.2789
ecd   random    loss                   0.100 readout_strong     zne_then_readout     0.2859   0.3148
ecd   random    loss                   0.100 readout_strong     gdr_select           0.2896   0.3268
ecd   optimized loss                   0.003 ideal              raw                  0.0373   0.1230
ecd   optimized loss                   0.003 ideal              oracle_binomial      0.0093   0.0471
ecd   optimized loss                   0.003 ideal              gdr_param            0.0095   0.0479
ecd   optimized loss                   0.003 ideal              gdr_damped           0.0095   0.0479
ecd   optimized loss                   0.003 ideal              gdr_ridge            0.0095   0.0480
ecd   optimized loss                   0.003 ideal              gdr_holdout          0.0095   0.0479
ecd   optimized loss                   0.003 ideal              gdr_reg              0.0095   0.0479
ecd   optimized loss                   0.003 ideal              gdr_mid              0.0095   0.0480
ecd   optimized loss                   0.003 ideal              gdr_tfree            0.0095   0.0479
ecd   optimized loss                   0.003 ideal              gdr_residual         0.0093   0.0471
ecd   optimized loss                   0.003 ideal              zne_idle             0.0142   0.0431
ecd   optimized loss                   0.003 ideal              readout_then_zne     0.0142   0.0431
ecd   optimized loss                   0.003 ideal              zne_then_readout     0.0142   0.0431
ecd   optimized loss                   0.003 ideal              gdr_select           0.0373   0.1230
ecd   optimized loss                   0.003 readout_realistic  raw                  0.0835   0.2783
ecd   optimized loss                   0.003 readout_realistic  readout_only         0.0465   0.1493
ecd   optimized loss                   0.003 readout_realistic  oracle_binomial      0.0122   0.0554
ecd   optimized loss                   0.003 readout_realistic  gdr_param            0.0131   0.0593
ecd   optimized loss                   0.003 readout_realistic  gdr_damped           0.0247   0.0908
ecd   optimized loss                   0.003 readout_realistic  gdr_ridge            0.0136   0.0613
ecd   optimized loss                   0.003 readout_realistic  gdr_holdout          0.0134   0.0606
ecd   optimized loss                   0.003 readout_realistic  gdr_reg              0.0266   0.0961
ecd   optimized loss                   0.003 readout_realistic  gdr_mid              0.0137   0.0613
ecd   optimized loss                   0.003 readout_realistic  gdr_tfree            0.0151   0.0647
ecd   optimized loss                   0.003 readout_realistic  gdr_residual         0.0122   0.0554
ecd   optimized loss                   0.003 readout_realistic  zne_idle             0.0778   0.2456
ecd   optimized loss                   0.003 readout_realistic  readout_then_zne     0.0361   0.1201
ecd   optimized loss                   0.003 readout_realistic  zne_then_readout     0.0461   0.1456
ecd   optimized loss                   0.003 readout_realistic  gdr_select           0.0465   0.1493
ecd   optimized loss                   0.003 readout_strong     raw                  0.1624   0.5627
ecd   optimized loss                   0.003 readout_strong     readout_only         0.0446   0.1428
ecd   optimized loss                   0.003 readout_strong     oracle_binomial      0.0076   0.0329
ecd   optimized loss                   0.003 readout_strong     gdr_param            0.0133   0.0567
ecd   optimized loss                   0.003 readout_strong     gdr_damped           0.0133   0.0567
ecd   optimized loss                   0.003 readout_strong     gdr_ridge            0.0131   0.0564
ecd   optimized loss                   0.003 readout_strong     gdr_holdout          0.0123   0.0540
ecd   optimized loss                   0.003 readout_strong     gdr_reg              0.0139   0.0584
ecd   optimized loss                   0.003 readout_strong     gdr_mid              0.0105   0.0487
ecd   optimized loss                   0.003 readout_strong     gdr_tfree            0.0136   0.0575
ecd   optimized loss                   0.003 readout_strong     gdr_residual         0.0076   0.0329
ecd   optimized loss                   0.003 readout_strong     zne_idle             0.1451   0.5134
ecd   optimized loss                   0.003 readout_strong     readout_then_zne     0.0251   0.1222
ecd   optimized loss                   0.003 readout_strong     zne_then_readout     0.0314   0.1384
ecd   optimized loss                   0.003 readout_strong     gdr_select           0.0446   0.1428
ecd   optimized loss                   0.100 ideal              raw                  0.7139   2.5331
ecd   optimized loss                   0.100 ideal              oracle_binomial      0.1953   1.0431
ecd   optimized loss                   0.100 ideal              gdr_param            0.2130   1.1687
ecd   optimized loss                   0.100 ideal              gdr_damped           0.2879   1.3734
ecd   optimized loss                   0.100 ideal              gdr_ridge            0.2139   1.1716
ecd   optimized loss                   0.100 ideal              gdr_holdout          0.2133   1.1697
ecd   optimized loss                   0.100 ideal              gdr_reg              0.2631   1.3060
ecd   optimized loss                   0.100 ideal              gdr_mid              0.2147   1.1797
ecd   optimized loss                   0.100 ideal              gdr_tfree            0.2893   1.3837
ecd   optimized loss                   0.100 ideal              gdr_residual         0.1921   1.0184
ecd   optimized loss                   0.100 ideal              zne_idle             0.4417   1.3990
ecd   optimized loss                   0.100 ideal              readout_then_zne     0.4417   1.3990
ecd   optimized loss                   0.100 ideal              zne_then_readout     0.4417   1.3990
ecd   optimized loss                   0.100 ideal              gdr_select           0.3638   1.5800
ecd   optimized loss                   0.100 readout_realistic  raw                  0.7000   2.5165
ecd   optimized loss                   0.100 readout_realistic  readout_only         0.6941   2.4465
ecd   optimized loss                   0.100 readout_realistic  oracle_binomial      0.1950   1.0363
ecd   optimized loss                   0.100 readout_realistic  gdr_param            0.2073   1.1480
ecd   optimized loss                   0.100 readout_realistic  gdr_damped           0.2801   1.3428
ecd   optimized loss                   0.100 readout_realistic  gdr_ridge            0.2083   1.1482
ecd   optimized loss                   0.100 readout_realistic  gdr_holdout          0.2073   1.1480
ecd   optimized loss                   0.100 readout_realistic  gdr_reg              0.2801   1.3428
ecd   optimized loss                   0.100 readout_realistic  gdr_mid              0.2074   1.1571
ecd   optimized loss                   0.100 readout_realistic  gdr_tfree            0.2585   1.2699
ecd   optimized loss                   0.100 readout_realistic  gdr_residual         0.1914   1.0093
ecd   optimized loss                   0.100 readout_realistic  zne_idle             0.4433   1.4825
ecd   optimized loss                   0.100 readout_realistic  readout_then_zne     0.4299   1.3745
ecd   optimized loss                   0.100 readout_realistic  zne_then_readout     0.4275   1.3826
ecd   optimized loss                   0.100 readout_realistic  gdr_select           0.3537   1.5377
ecd   optimized loss                   0.100 readout_strong     raw                  0.7144   2.7226
ecd   optimized loss                   0.100 readout_strong     readout_only         0.6936   2.4951
ecd   optimized loss                   0.100 readout_strong     oracle_binomial      0.2088   1.1130
ecd   optimized loss                   0.100 readout_strong     gdr_param            0.2195   1.2045
ecd   optimized loss                   0.100 readout_strong     gdr_damped           0.2667   1.3336
ecd   optimized loss                   0.100 readout_strong     gdr_ridge            0.2213   1.2059
ecd   optimized loss                   0.100 readout_strong     gdr_holdout          0.2195   1.2045
ecd   optimized loss                   0.100 readout_strong     gdr_reg              0.2667   1.3336
ecd   optimized loss                   0.100 readout_strong     gdr_mid              0.2209   1.2171
ecd   optimized loss                   0.100 readout_strong     gdr_tfree            0.3069   1.4310
ecd   optimized loss                   0.100 readout_strong     gdr_residual         0.2040   1.0779
ecd   optimized loss                   0.100 readout_strong     zne_idle             0.4684   1.5863
ecd   optimized loss                   0.100 readout_strong     readout_then_zne     0.4278   1.3955
ecd   optimized loss                   0.100 readout_strong     zne_then_readout     0.4143   1.3835
ecd   optimized loss                   0.100 readout_strong     gdr_select           0.3864   1.6571
```
