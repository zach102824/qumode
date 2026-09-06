# Ablation summary

tag=`leftover_floor` shots=8192 n_train=40 twin=span ansatz=both params=random families=loss,loss_thermal_dephasing,comprehensive kappa=0.003,0.03,0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | random | loss | 0.003 | ideal | 0.0434 | 0.0448 | 0.0427 | gdr_damped | 0.0461 | 0.0459 | -0.0021 |
| ecd | random | loss | 0.003 | readout_realistic | 0.0590 | 0.0470 | 0.0471 | gdr_damped | 0.0613 | 0.0499 | 0.0001 |
| ecd | random | loss | 0.003 | readout_strong | 0.1071 | 0.0589 | 0.0585 | gdr_damped | 0.0996 | 0.0606 | -0.0004 |
| ecd | random | loss | 0.03 | ideal | 0.1423 | 0.0909 | 0.0847 | gdr_damped | 0.1423 | 0.1283 | -0.0062 |
| ecd | random | loss | 0.03 | readout_realistic | 0.1539 | 0.0898 | 0.0839 | gdr_damped | 0.1540 | 0.1304 | -0.0059 |
| ecd | random | loss | 0.03 | readout_strong | 0.1856 | 0.1130 | 0.1104 | gdr_damped | 0.1711 | 0.1263 | -0.0026 |
| ecd | random | loss | 0.1 | ideal | 0.2983 | 0.2087 | 0.2013 | gdr_floor | 0.2991 | 0.3725 | -0.0074 |
| ecd | random | loss | 0.1 | readout_realistic | 0.3078 | 0.2221 | 0.2085 | gdr_floor | 0.2978 | 0.3940 | -0.0136 |
| ecd | random | loss | 0.1 | readout_strong | 0.3153 | 0.2322 | 0.2268 | gdr_damped | 0.3142 | 0.3763 | -0.0054 |
| ecd | random | loss_thermal_dephasing | 0.003 | ideal | 0.0535 | 0.0456 | 0.0482 | gdr_damped | 0.0488 | 0.0458 | 0.0026 |
| ecd | random | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0606 | 0.0476 | 0.0481 | gdr_damped | 0.0610 | 0.0523 | 0.0004 |
| ecd | random | loss_thermal_dephasing | 0.003 | readout_strong | 0.1041 | 0.0577 | 0.0588 | gdr_damped | 0.1005 | 0.0655 | 0.0010 |
| ecd | random | loss_thermal_dephasing | 0.03 | ideal | 0.1744 | 0.1157 | 0.1126 | gdr_damped | 0.1773 | 0.1447 | -0.0031 |
| ecd | random | loss_thermal_dephasing | 0.03 | readout_realistic | 0.1915 | 0.1279 | 0.1283 | gdr_damped | 0.1824 | 0.1740 | 0.0004 |
| ecd | random | loss_thermal_dephasing | 0.03 | readout_strong | 0.2126 | 0.1442 | 0.1423 | gdr_damped | 0.2056 | 0.1588 | -0.0018 |
| ecd | random | loss_thermal_dephasing | 0.1 | ideal | 0.3356 | 0.2633 | 0.2580 | gdr_damped | 0.3319 | 0.4067 | -0.0053 |
| ecd | random | loss_thermal_dephasing | 0.1 | readout_realistic | 0.3368 | 0.2642 | 0.2634 | gdr_damped | 0.3286 | 0.3960 | -0.0007 |
| ecd | random | loss_thermal_dephasing | 0.1 | readout_strong | 0.3443 | 0.2793 | 0.2721 | gdr_damped | 0.3361 | 0.4211 | -0.0073 |
| ecd | random | comprehensive | 0.003 | ideal | 0.0816 | 0.0723 | 0.0656 | gdr_floor | 0.0801 | 0.0713 | -0.0066 |
| ecd | random | comprehensive | 0.003 | readout_realistic | 0.1016 | 0.0776 | 0.0696 | gdr_floor | 0.1007 | 0.0847 | -0.0080 |
| ecd | random | comprehensive | 0.003 | readout_strong | 0.1346 | 0.0917 | 0.0787 | gdr_floor | 0.1339 | 0.0911 | -0.0130 |
| ecd | random | comprehensive | 0.03 | ideal | 0.2417 | 0.1497 | 0.1437 | gdr_damped | 0.2380 | 0.2841 | -0.0060 |
| ecd | random | comprehensive | 0.03 | readout_realistic | 0.2237 | 0.1731 | 0.1522 | gdr_floor | 0.2357 | 0.2897 | -0.0210 |
| ecd | random | comprehensive | 0.03 | readout_strong | 0.2608 | 0.2063 | 0.1845 | gdr_floor | 0.2593 | 0.3045 | -0.0218 |
| ecd | random | comprehensive | 0.1 | ideal | 0.4031 | 0.3765 | 0.3353 | gdr_floor | 0.3990 | 0.5392 | -0.0412 |
| ecd | random | comprehensive | 0.1 | readout_realistic | 0.3980 | 0.3410 | 0.2954 | gdr_floor | 0.4052 | 0.5224 | -0.0456 |
| ecd | random | comprehensive | 0.1 | readout_strong | 0.3995 | 0.3397 | 0.3125 | gdr_damped | 0.4035 | 0.5287 | -0.0273 |
| snap | random | loss | 0.003 | ideal | 0.0283 | 0.0217 | 0.0218 | gdr_damped | 0.0194 | 0.0223 | 0.0001 |
| snap | random | loss | 0.003 | readout_realistic | 0.0499 | 0.0346 | 0.0349 | gdr_damped | 0.0424 | 0.0308 | 0.0002 |
| snap | random | loss | 0.003 | readout_strong | 0.0876 | 0.0386 | 0.0379 | gdr_damped | 0.0974 | 0.0356 | -0.0007 |
| snap | random | loss | 0.03 | ideal | 0.1304 | 0.0809 | 0.0708 | gdr_floor | 0.1427 | 0.0851 | -0.0101 |
| snap | random | loss | 0.03 | readout_realistic | 0.1520 | 0.0736 | 0.0671 | gdr_floor | 0.1551 | 0.0853 | -0.0065 |
| snap | random | loss | 0.03 | readout_strong | 0.1833 | 0.0849 | 0.0792 | gdr_floor | 0.1849 | 0.1043 | -0.0058 |
| snap | random | loss | 0.1 | ideal | 0.3351 | 0.1956 | 0.1866 | gdr_floor | 0.3330 | 0.2329 | -0.0090 |
| snap | random | loss | 0.1 | readout_realistic | 0.3457 | 0.2140 | 0.1959 | gdr_floor | 0.3444 | 0.2449 | -0.0181 |
| snap | random | loss | 0.1 | readout_strong | 0.3461 | 0.1953 | 0.1805 | gdr_floor | 0.3545 | 0.2580 | -0.0148 |
| snap | random | loss_thermal_dephasing | 0.003 | ideal | 0.0291 | 0.0285 | 0.0278 | gdr_damped | 0.0293 | 0.0304 | -0.0007 |
| snap | random | loss_thermal_dephasing | 0.003 | readout_realistic | 0.0487 | 0.0287 | 0.0285 | gdr_damped | 0.0487 | 0.0299 | -0.0002 |
| snap | random | loss_thermal_dephasing | 0.003 | readout_strong | 0.0911 | 0.0336 | 0.0290 | gdr_damped | 0.0995 | 0.0380 | -0.0046 |
| snap | random | loss_thermal_dephasing | 0.03 | ideal | 0.1505 | 0.0893 | 0.0755 | gdr_floor | 0.1568 | 0.0972 | -0.0138 |
| snap | random | loss_thermal_dephasing | 0.03 | readout_realistic | 0.1741 | 0.0907 | 0.0889 | gdr_floor | 0.1596 | 0.1065 | -0.0019 |
| snap | random | loss_thermal_dephasing | 0.03 | readout_strong | 0.1978 | 0.0833 | 0.0759 | gdr_floor | 0.1991 | 0.1074 | -0.0074 |
| snap | random | loss_thermal_dephasing | 0.1 | ideal | 0.3433 | 0.1987 | 0.1893 | gdr_floor | 0.3562 | 0.2419 | -0.0095 |
| snap | random | loss_thermal_dephasing | 0.1 | readout_realistic | 0.3460 | 0.1773 | 0.1690 | gdr_floor | 0.3466 | 0.2560 | -0.0083 |
| snap | random | loss_thermal_dephasing | 0.1 | readout_strong | 0.3525 | 0.2055 | 0.1888 | gdr_floor | 0.3667 | 0.2835 | -0.0168 |
| snap | random | comprehensive | 0.003 | ideal | 0.0372 | 0.0415 | 0.0369 | gdr_floor | 0.0413 | 0.0445 | -0.0046 |
| snap | random | comprehensive | 0.003 | readout_realistic | 0.0535 | 0.0385 | 0.0374 | gdr_damped | 0.0523 | 0.0324 | -0.0011 |
| snap | random | comprehensive | 0.003 | readout_strong | 0.0892 | 0.0473 | 0.0465 | gdr_damped | 0.0950 | 0.0410 | -0.0008 |
| snap | random | comprehensive | 0.03 | ideal | 0.2450 | 0.1372 | 0.1257 | gdr_floor | 0.2380 | 0.1436 | -0.0115 |
| snap | random | comprehensive | 0.03 | readout_realistic | 0.2406 | 0.1349 | 0.1302 | gdr_floor | 0.2528 | 0.1406 | -0.0047 |
| snap | random | comprehensive | 0.03 | readout_strong | 0.2619 | 0.1336 | 0.1227 | gdr_floor | 0.2711 | 0.1500 | -0.0108 |
| snap | random | comprehensive | 0.1 | ideal | 0.4728 | 0.3747 | 0.3380 | gdr_floor | 0.4663 | 0.4109 | -0.0366 |
| snap | random | comprehensive | 0.1 | readout_realistic | 0.4685 | 0.3334 | 0.3127 | gdr_floor | 0.4725 | 0.4005 | -0.0207 |
| snap | random | comprehensive | 0.1 | readout_strong | 0.4817 | 0.3432 | 0.3139 | gdr_floor | 0.4744 | 0.4373 | -0.0293 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   random    loss                   0.003 ideal              raw                  0.0434   0.0785
ecd   random    loss                   0.003 ideal              oracle_binomial      0.0545   0.0968
ecd   random    loss                   0.003 ideal              gdr_param            0.0448   0.0824
ecd   random    loss                   0.003 ideal              gdr_damped           0.0427   0.0810
ecd   random    loss                   0.003 ideal              gdr_floor            0.0434   0.0785
ecd   random    loss                   0.003 readout_realistic  raw                  0.0590   0.0360
ecd   random    loss                   0.003 readout_realistic  oracle_binomial      0.0533   0.0335
ecd   random    loss                   0.003 readout_realistic  gdr_param            0.0470   0.0335
ecd   random    loss                   0.003 readout_realistic  gdr_damped           0.0471   0.0338
ecd   random    loss                   0.003 readout_realistic  gdr_floor            0.0486   0.0342
ecd   random    loss                   0.003 readout_strong     raw                  0.1071   0.0127
ecd   random    loss                   0.003 readout_strong     oracle_binomial      0.0633   0.0149
ecd   random    loss                   0.003 readout_strong     gdr_param            0.0589   0.0277
ecd   random    loss                   0.003 readout_strong     gdr_damped           0.0585   0.0251
ecd   random    loss                   0.003 readout_strong     gdr_floor            0.0606   0.0225
ecd   random    loss                   0.030 ideal              raw                  0.1423   0.0060
ecd   random    loss                   0.030 ideal              oracle_binomial      0.2030   0.1209
ecd   random    loss                   0.030 ideal              gdr_param            0.0909   0.0248
ecd   random    loss                   0.030 ideal              gdr_damped           0.0847   0.0220
ecd   random    loss                   0.030 ideal              gdr_floor            0.0978   0.0154
ecd   random    loss                   0.030 readout_realistic  raw                  0.1539   0.0233
ecd   random    loss                   0.030 readout_realistic  oracle_binomial      0.2239   0.2073
ecd   random    loss                   0.030 readout_realistic  gdr_param            0.0898   0.0466
ecd   random    loss                   0.030 readout_realistic  gdr_damped           0.0839   0.0430
ecd   random    loss                   0.030 readout_realistic  gdr_floor            0.0927   0.0357
ecd   random    loss                   0.030 readout_strong     raw                  0.1856   0.0053
ecd   random    loss                   0.030 readout_strong     oracle_binomial      0.2284   0.1728
ecd   random    loss                   0.030 readout_strong     gdr_param            0.1130   0.0037
ecd   random    loss                   0.030 readout_strong     gdr_damped           0.1104   0.0023
ecd   random    loss                   0.030 readout_strong     gdr_floor            0.1191   0.0006
ecd   random    loss                   0.100 ideal              raw                  0.2983   0.0834
ecd   random    loss                   0.100 ideal              oracle_binomial      0.5370   0.0092
ecd   random    loss                   0.100 ideal              gdr_param            0.2087   0.1075
ecd   random    loss                   0.100 ideal              gdr_damped           0.2031   0.1051
ecd   random    loss                   0.100 ideal              gdr_floor            0.2013   0.1015
ecd   random    loss                   0.100 readout_realistic  raw                  0.3078   0.0560
ecd   random    loss                   0.100 readout_realistic  oracle_binomial      0.5470   0.1003
ecd   random    loss                   0.100 readout_realistic  gdr_param            0.2221   0.0070
ecd   random    loss                   0.100 readout_realistic  gdr_damped           0.2110   0.0152
ecd   random    loss                   0.100 readout_realistic  gdr_floor            0.2085   0.0233
ecd   random    loss                   0.100 readout_strong     raw                  0.3153   0.0285
ecd   random    loss                   0.100 readout_strong     oracle_binomial      0.5211   0.2375
ecd   random    loss                   0.100 readout_strong     gdr_param            0.2322   0.0873
ecd   random    loss                   0.100 readout_strong     gdr_damped           0.2268   0.0784
ecd   random    loss                   0.100 readout_strong     gdr_floor            0.2301   0.0717
ecd   random    loss_thermal_dephasing 0.003 ideal              raw                  0.0535   0.0785
ecd   random    loss_thermal_dephasing 0.003 ideal              oracle_binomial      0.0495   0.0843
ecd   random    loss_thermal_dephasing 0.003 ideal              gdr_param            0.0456   0.0848
ecd   random    loss_thermal_dephasing 0.003 ideal              gdr_damped           0.0482   0.0816
ecd   random    loss_thermal_dephasing 0.003 ideal              gdr_floor            0.0535   0.0785
ecd   random    loss_thermal_dephasing 0.003 readout_realistic  raw                  0.0606   0.0642
ecd   random    loss_thermal_dephasing 0.003 readout_realistic  oracle_binomial      0.0545   0.0576
ecd   random    loss_thermal_dephasing 0.003 readout_realistic  gdr_param            0.0476   0.0718
ecd   random    loss_thermal_dephasing 0.003 readout_realistic  gdr_damped           0.0481   0.0723
ecd   random    loss_thermal_dephasing 0.003 readout_realistic  gdr_floor            0.0503   0.0726
ecd   random    loss_thermal_dephasing 0.003 readout_strong     raw                  0.1041   0.0193
ecd   random    loss_thermal_dephasing 0.003 readout_strong     oracle_binomial      0.0644   0.0307
ecd   random    loss_thermal_dephasing 0.003 readout_strong     gdr_param            0.0577   0.0197
ecd   random    loss_thermal_dephasing 0.003 readout_strong     gdr_damped           0.0588   0.0209
ecd   random    loss_thermal_dephasing 0.003 readout_strong     gdr_floor            0.0596   0.0212
ecd   random    loss_thermal_dephasing 0.030 ideal              raw                  0.1744   0.1670
ecd   random    loss_thermal_dephasing 0.030 ideal              oracle_binomial      0.2356   0.0703
ecd   random    loss_thermal_dephasing 0.030 ideal              gdr_param            0.1157   0.1690
ecd   random    loss_thermal_dephasing 0.030 ideal              gdr_damped           0.1126   0.1685
ecd   random    loss_thermal_dephasing 0.030 ideal              gdr_floor            0.1326   0.1678
ecd   random    loss_thermal_dephasing 0.030 readout_realistic  raw                  0.1915   0.0646
ecd   random    loss_thermal_dephasing 0.030 readout_realistic  oracle_binomial      0.2248   0.1571
ecd   random    loss_thermal_dephasing 0.030 readout_realistic  gdr_param            0.1279   0.0135
ecd   random    loss_thermal_dephasing 0.030 readout_realistic  gdr_damped           0.1283   0.0269
ecd   random    loss_thermal_dephasing 0.030 readout_realistic  gdr_floor            0.1417   0.0431
ecd   random    loss_thermal_dephasing 0.030 readout_strong     raw                  0.2126   0.0808
ecd   random    loss_thermal_dephasing 0.030 readout_strong     oracle_binomial      0.2405   0.0511
ecd   random    loss_thermal_dephasing 0.030 readout_strong     gdr_param            0.1442   0.0537
ecd   random    loss_thermal_dephasing 0.030 readout_strong     gdr_damped           0.1423   0.0634
ecd   random    loss_thermal_dephasing 0.030 readout_strong     gdr_floor            0.1518   0.0731
ecd   random    loss_thermal_dephasing 0.100 ideal              raw                  0.3356   0.2086
ecd   random    loss_thermal_dephasing 0.100 ideal              oracle_binomial      0.5238   0.0168
ecd   random    loss_thermal_dephasing 0.100 ideal              gdr_param            0.2633   0.1769
ecd   random    loss_thermal_dephasing 0.100 ideal              gdr_damped           0.2580   0.1848
ecd   random    loss_thermal_dephasing 0.100 ideal              gdr_floor            0.2629   0.1896
ecd   random    loss_thermal_dephasing 0.100 readout_realistic  raw                  0.3368   0.1924
ecd   random    loss_thermal_dephasing 0.100 readout_realistic  oracle_binomial      0.5288   0.1586
ecd   random    loss_thermal_dephasing 0.100 readout_realistic  gdr_param            0.2642   0.0637
ecd   random    loss_thermal_dephasing 0.100 readout_realistic  gdr_damped           0.2634   0.0982
ecd   random    loss_thermal_dephasing 0.100 readout_realistic  gdr_floor            0.2728   0.1259
ecd   random    loss_thermal_dephasing 0.100 readout_strong     raw                  0.3443   0.2201
ecd   random    loss_thermal_dephasing 0.100 readout_strong     oracle_binomial      0.5308   0.2361
ecd   random    loss_thermal_dephasing 0.100 readout_strong     gdr_param            0.2793   0.1272
ecd   random    loss_thermal_dephasing 0.100 readout_strong     gdr_damped           0.2721   0.1630
ecd   random    loss_thermal_dephasing 0.100 readout_strong     gdr_floor            0.2805   0.1809
ecd   random    comprehensive          0.003 ideal              raw                  0.0816   0.0698
ecd   random    comprehensive          0.003 ideal              oracle_binomial      0.0967   0.1660
ecd   random    comprehensive          0.003 ideal              gdr_param            0.0723   0.0514
ecd   random    comprehensive          0.003 ideal              gdr_damped           0.0723   0.0514
ecd   random    comprehensive          0.003 ideal              gdr_floor            0.0656   0.0560
ecd   random    comprehensive          0.003 readout_realistic  raw                  0.1016   0.0501
ecd   random    comprehensive          0.003 readout_realistic  oracle_binomial      0.1038   0.1667
ecd   random    comprehensive          0.003 readout_realistic  gdr_param            0.0776   0.0336
ecd   random    comprehensive          0.003 readout_realistic  gdr_damped           0.0776   0.0336
ecd   random    comprehensive          0.003 readout_realistic  gdr_floor            0.0696   0.0389
ecd   random    comprehensive          0.003 readout_strong     raw                  0.1346   0.0550
ecd   random    comprehensive          0.003 readout_strong     oracle_binomial      0.1076   0.1812
ecd   random    comprehensive          0.003 readout_strong     gdr_param            0.0917   0.0385
ecd   random    comprehensive          0.003 readout_strong     gdr_damped           0.0893   0.0396
ecd   random    comprehensive          0.003 readout_strong     gdr_floor            0.0787   0.0454
ecd   random    comprehensive          0.030 ideal              raw                  0.2417   0.1276
ecd   random    comprehensive          0.030 ideal              oracle_binomial      0.4120   0.1548
ecd   random    comprehensive          0.030 ideal              gdr_param            0.1497   0.1222
ecd   random    comprehensive          0.030 ideal              gdr_damped           0.1437   0.1238
ecd   random    comprehensive          0.030 ideal              gdr_floor            0.1585   0.1249
ecd   random    comprehensive          0.030 readout_realistic  raw                  0.2237   0.0031
ecd   random    comprehensive          0.030 readout_realistic  oracle_binomial      0.4326   0.1091
ecd   random    comprehensive          0.030 readout_realistic  gdr_param            0.1731   0.1659
ecd   random    comprehensive          0.030 readout_realistic  gdr_damped           0.1530   0.1154
ecd   random    comprehensive          0.030 readout_realistic  gdr_floor            0.1522   0.0902
ecd   random    comprehensive          0.030 readout_strong     raw                  0.2608   0.0749
ecd   random    comprehensive          0.030 readout_strong     oracle_binomial      0.4469   0.0274
ecd   random    comprehensive          0.030 readout_strong     gdr_param            0.2063   0.0282
ecd   random    comprehensive          0.030 readout_strong     gdr_damped           0.1892   0.0454
ecd   random    comprehensive          0.030 readout_strong     gdr_floor            0.1845   0.0540
ecd   random    comprehensive          0.100 ideal              raw                  0.4031   0.1404
ecd   random    comprehensive          0.100 ideal              oracle_binomial      0.7381   0.4237
ecd   random    comprehensive          0.100 ideal              gdr_param            0.3765   0.3634
ecd   random    comprehensive          0.100 ideal              gdr_damped           0.3424   0.3077
ecd   random    comprehensive          0.100 ideal              gdr_floor            0.3353   0.2854
ecd   random    comprehensive          0.100 readout_realistic  raw                  0.3980   0.1416
ecd   random    comprehensive          0.100 readout_realistic  oracle_binomial      0.7418   0.4903
ecd   random    comprehensive          0.100 readout_realistic  gdr_param            0.3410   0.0932
ecd   random    comprehensive          0.100 readout_realistic  gdr_damped           0.3027   0.1116
ecd   random    comprehensive          0.100 readout_realistic  gdr_floor            0.2954   0.1177
ecd   random    comprehensive          0.100 readout_strong     raw                  0.3995   0.1153
ecd   random    comprehensive          0.100 readout_strong     oracle_binomial      0.7353   0.5508
ecd   random    comprehensive          0.100 readout_strong     gdr_param            0.3397   0.1603
ecd   random    comprehensive          0.100 readout_strong     gdr_damped           0.3125   0.1527
ecd   random    comprehensive          0.100 readout_strong     gdr_floor            0.3147   0.1506
snap  random    loss                   0.003 ideal              raw                  0.0283   0.0117
snap  random    loss                   0.003 ideal              oracle_binomial      0.0222   0.0249
snap  random    loss                   0.003 ideal              gdr_param            0.0217   0.0201
snap  random    loss                   0.003 ideal              gdr_damped           0.0218   0.0189
snap  random    loss                   0.003 ideal              gdr_floor            0.0283   0.0117
snap  random    loss                   0.003 readout_realistic  raw                  0.0499   0.0166
snap  random    loss                   0.003 readout_realistic  oracle_binomial      0.0361   0.0502
snap  random    loss                   0.003 readout_realistic  gdr_param            0.0346   0.0436
snap  random    loss                   0.003 readout_realistic  gdr_damped           0.0349   0.0418
snap  random    loss                   0.003 readout_realistic  gdr_floor            0.0373   0.0365
snap  random    loss                   0.003 readout_strong     raw                  0.0876   0.0708
snap  random    loss                   0.003 readout_strong     oracle_binomial      0.0380   0.0106
snap  random    loss                   0.003 readout_strong     gdr_param            0.0386   0.0042
snap  random    loss                   0.003 readout_strong     gdr_damped           0.0379   0.0027
snap  random    loss                   0.003 readout_strong     gdr_floor            0.0411   0.0009
snap  random    loss                   0.030 ideal              raw                  0.1304   0.0735
snap  random    loss                   0.030 ideal              oracle_binomial      0.1109   0.2228
snap  random    loss                   0.030 ideal              gdr_param            0.0809   0.1679
snap  random    loss                   0.030 ideal              gdr_damped           0.0779   0.1632
snap  random    loss                   0.030 ideal              gdr_floor            0.0708   0.1396
snap  random    loss                   0.030 readout_realistic  raw                  0.1520   0.0385
snap  random    loss                   0.030 readout_realistic  oracle_binomial      0.1063   0.1383
snap  random    loss                   0.030 readout_realistic  gdr_param            0.0736   0.0739
snap  random    loss                   0.030 readout_realistic  gdr_damped           0.0708   0.0692
snap  random    loss                   0.030 readout_realistic  gdr_floor            0.0671   0.0453
snap  random    loss                   0.030 readout_strong     raw                  0.1833   0.0102
snap  random    loss                   0.030 readout_strong     oracle_binomial      0.1087   0.2016
snap  random    loss                   0.030 readout_strong     gdr_param            0.0849   0.1197
snap  random    loss                   0.030 readout_strong     gdr_damped           0.0804   0.1115
snap  random    loss                   0.030 readout_strong     gdr_floor            0.0792   0.0953
snap  random    loss                   0.100 ideal              raw                  0.3351   0.2530
snap  random    loss                   0.100 ideal              oracle_binomial      0.2445   0.3627
snap  random    loss                   0.100 ideal              gdr_param            0.1956   0.2912
snap  random    loss                   0.100 ideal              gdr_damped           0.1901   0.2874
snap  random    loss                   0.100 ideal              gdr_floor            0.1866   0.2836
snap  random    loss                   0.100 readout_realistic  raw                  0.3457   0.1593
snap  random    loss                   0.100 readout_realistic  oracle_binomial      0.2619   0.2797
snap  random    loss                   0.100 readout_realistic  gdr_param            0.2140   0.1790
snap  random    loss                   0.100 readout_realistic  gdr_damped           0.2024   0.1787
snap  random    loss                   0.100 readout_realistic  gdr_floor            0.1959   0.1782
snap  random    loss                   0.100 readout_strong     raw                  0.3461   0.0873
snap  random    loss                   0.100 readout_strong     oracle_binomial      0.2422   0.2398
snap  random    loss                   0.100 readout_strong     gdr_param            0.1953   0.1587
snap  random    loss                   0.100 readout_strong     gdr_damped           0.1834   0.1591
snap  random    loss                   0.100 readout_strong     gdr_floor            0.1805   0.1594
snap  random    loss_thermal_dephasing 0.003 ideal              raw                  0.0291   0.0401
snap  random    loss_thermal_dephasing 0.003 ideal              oracle_binomial      0.0284   0.0542
snap  random    loss_thermal_dephasing 0.003 ideal              gdr_param            0.0285   0.0510
snap  random    loss_thermal_dephasing 0.003 ideal              gdr_damped           0.0278   0.0493
snap  random    loss_thermal_dephasing 0.003 ideal              gdr_floor            0.0291   0.0401
snap  random    loss_thermal_dephasing 0.003 readout_realistic  raw                  0.0487   0.0561
snap  random    loss_thermal_dephasing 0.003 readout_realistic  oracle_binomial      0.0289   0.0234
snap  random    loss_thermal_dephasing 0.003 readout_realistic  gdr_param            0.0287   0.0292
snap  random    loss_thermal_dephasing 0.003 readout_realistic  gdr_damped           0.0285   0.0309
snap  random    loss_thermal_dephasing 0.003 readout_realistic  gdr_floor            0.0337   0.0376
snap  random    loss_thermal_dephasing 0.003 readout_strong     raw                  0.0911   0.0475
snap  random    loss_thermal_dephasing 0.003 readout_strong     oracle_binomial      0.0357   0.0220
snap  random    loss_thermal_dephasing 0.003 readout_strong     gdr_param            0.0336   0.0215
snap  random    loss_thermal_dephasing 0.003 readout_strong     gdr_damped           0.0290   0.0168
snap  random    loss_thermal_dephasing 0.003 readout_strong     gdr_floor            0.0319   0.0080
snap  random    loss_thermal_dephasing 0.030 ideal              raw                  0.1505   0.0180
snap  random    loss_thermal_dephasing 0.030 ideal              oracle_binomial      0.1088   0.1149
snap  random    loss_thermal_dephasing 0.030 ideal              gdr_param            0.0893   0.0607
snap  random    loss_thermal_dephasing 0.030 ideal              gdr_damped           0.0854   0.0568
snap  random    loss_thermal_dephasing 0.030 ideal              gdr_floor            0.0755   0.0371
snap  random    loss_thermal_dephasing 0.030 readout_realistic  raw                  0.1741   0.0711
snap  random    loss_thermal_dephasing 0.030 readout_realistic  oracle_binomial      0.1043   0.2611
snap  random    loss_thermal_dephasing 0.030 readout_realistic  gdr_param            0.0907   0.1992
snap  random    loss_thermal_dephasing 0.030 readout_realistic  gdr_damped           0.0898   0.1938
snap  random    loss_thermal_dephasing 0.030 readout_realistic  gdr_floor            0.0889   0.1663
snap  random    loss_thermal_dephasing 0.030 readout_strong     raw                  0.1978   0.0373
snap  random    loss_thermal_dephasing 0.030 readout_strong     oracle_binomial      0.1098   0.1853
snap  random    loss_thermal_dephasing 0.030 readout_strong     gdr_param            0.0833   0.1131
snap  random    loss_thermal_dephasing 0.030 readout_strong     gdr_damped           0.0777   0.1038
snap  random    loss_thermal_dephasing 0.030 readout_strong     gdr_floor            0.0759   0.0807
snap  random    loss_thermal_dephasing 0.100 ideal              raw                  0.3433   0.2473
snap  random    loss_thermal_dephasing 0.100 ideal              oracle_binomial      0.2749   0.5027
snap  random    loss_thermal_dephasing 0.100 ideal              gdr_param            0.1987   0.3407
snap  random    loss_thermal_dephasing 0.100 ideal              gdr_damped           0.1905   0.3267
snap  random    loss_thermal_dephasing 0.100 ideal              gdr_floor            0.1893   0.3126
snap  random    loss_thermal_dephasing 0.100 readout_realistic  raw                  0.3460   0.1975
snap  random    loss_thermal_dephasing 0.100 readout_realistic  oracle_binomial      0.2530   0.3609
snap  random    loss_thermal_dephasing 0.100 readout_realistic  gdr_param            0.1773   0.2249
snap  random    loss_thermal_dephasing 0.100 readout_realistic  gdr_damped           0.1707   0.2246
snap  random    loss_thermal_dephasing 0.100 readout_realistic  gdr_floor            0.1690   0.2244
snap  random    loss_thermal_dephasing 0.100 readout_strong     raw                  0.3525   0.1419
snap  random    loss_thermal_dephasing 0.100 readout_strong     oracle_binomial      0.2725   0.3374
snap  random    loss_thermal_dephasing 0.100 readout_strong     gdr_param            0.2055   0.1451
snap  random    loss_thermal_dephasing 0.100 readout_strong     gdr_damped           0.1924   0.1565
snap  random    loss_thermal_dephasing 0.100 readout_strong     gdr_floor            0.1888   0.1623
snap  random    comprehensive          0.003 ideal              raw                  0.0372   0.0154
snap  random    comprehensive          0.003 ideal              oracle_binomial      0.0452   0.0449
snap  random    comprehensive          0.003 ideal              gdr_param            0.0415   0.0473
snap  random    comprehensive          0.003 ideal              gdr_damped           0.0402   0.0441
snap  random    comprehensive          0.003 ideal              gdr_floor            0.0369   0.0218
snap  random    comprehensive          0.003 readout_realistic  raw                  0.0535   0.0128
snap  random    comprehensive          0.003 readout_realistic  oracle_binomial      0.0420   0.0595
snap  random    comprehensive          0.003 readout_realistic  gdr_param            0.0385   0.0576
snap  random    comprehensive          0.003 readout_realistic  gdr_damped           0.0374   0.0524
snap  random    comprehensive          0.003 readout_realistic  gdr_floor            0.0412   0.0320
snap  random    comprehensive          0.003 readout_strong     raw                  0.0892   0.0038
snap  random    comprehensive          0.003 readout_strong     oracle_binomial      0.0529   0.0944
snap  random    comprehensive          0.003 readout_strong     gdr_param            0.0473   0.0918
snap  random    comprehensive          0.003 readout_strong     gdr_damped           0.0465   0.0891
snap  random    comprehensive          0.003 readout_strong     gdr_floor            0.0495   0.0655
snap  random    comprehensive          0.030 ideal              raw                  0.2450   0.0030
snap  random    comprehensive          0.030 ideal              oracle_binomial      0.1800   0.1548
snap  random    comprehensive          0.030 ideal              gdr_param            0.1372   0.1881
snap  random    comprehensive          0.030 ideal              gdr_damped           0.1372   0.1881
snap  random    comprehensive          0.030 ideal              gdr_floor            0.1257   0.1499
snap  random    comprehensive          0.030 readout_realistic  raw                  0.2406   0.0486
snap  random    comprehensive          0.030 readout_realistic  oracle_binomial      0.1574   0.2273
snap  random    comprehensive          0.030 readout_realistic  gdr_param            0.1349   0.2571
snap  random    comprehensive          0.030 readout_realistic  gdr_damped           0.1333   0.2478
snap  random    comprehensive          0.030 readout_realistic  gdr_floor            0.1302   0.2108
snap  random    comprehensive          0.030 readout_strong     raw                  0.2619   0.0265
snap  random    comprehensive          0.030 readout_strong     oracle_binomial      0.1628   0.2075
snap  random    comprehensive          0.030 readout_strong     gdr_param            0.1336   0.2390
snap  random    comprehensive          0.030 readout_strong     gdr_damped           0.1296   0.2287
snap  random    comprehensive          0.030 readout_strong     gdr_floor            0.1227   0.1877
snap  random    comprehensive          0.100 ideal              raw                  0.4728   0.4223
snap  random    comprehensive          0.100 ideal              oracle_binomial      0.5143   0.3349
snap  random    comprehensive          0.100 ideal              gdr_param            0.3747   0.4585
snap  random    comprehensive          0.100 ideal              gdr_damped           0.3504   0.4531
snap  random    comprehensive          0.100 ideal              gdr_floor            0.3380   0.4494
snap  random    comprehensive          0.100 readout_realistic  raw                  0.4685   0.3440
snap  random    comprehensive          0.100 readout_realistic  oracle_binomial      0.4787   0.3774
snap  random    comprehensive          0.100 readout_realistic  gdr_param            0.3334   0.4294
snap  random    comprehensive          0.100 readout_realistic  gdr_damped           0.3177   0.4213
snap  random    comprehensive          0.100 readout_realistic  gdr_floor            0.3127   0.4186
snap  random    comprehensive          0.100 readout_strong     raw                  0.4817   0.2640
snap  random    comprehensive          0.100 readout_strong     oracle_binomial      0.4701   0.2791
snap  random    comprehensive          0.100 readout_strong     gdr_param            0.3432   0.2046
snap  random    comprehensive          0.100 readout_strong     gdr_damped           0.3254   0.2282
snap  random    comprehensive          0.100 readout_strong     gdr_floor            0.3139   0.2439
```
