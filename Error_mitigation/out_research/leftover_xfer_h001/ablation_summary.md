# Ablation summary

tag=`leftover_xfer_h001` shots=8192 n_train=40 twin=default ansatz=ecd params=optimized families=loss,comprehensive kappa=0.003,0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | optimized | loss | 0.003 | ideal | 0.0394 | 0.0225 | 0.0209 | gdr_residual | 0.0363 | 0.0075 | -0.0016 |
| ecd | optimized | loss | 0.003 | readout_strong | 0.2467 | 0.0377 | 0.0308 | gdr_residual | 0.1534 | 0.0075 | -0.0069 |
| ecd | optimized | loss | 0.1 | ideal | 0.5867 | 0.4947 | 0.4947 | gdr_select | 0.7064 | 0.2015 | 0.0000 |
| ecd | optimized | loss | 0.1 | readout_strong | 0.6517 | 0.5277 | 0.5277 | gdr_select | 0.7233 | 0.2087 | 0.0000 |
| ecd | optimized | comprehensive | 0.003 | ideal | 0.1756 | 0.1122 | 0.0862 | gdr_residual | 0.1823 | 0.0534 | -0.0259 |
| ecd | optimized | comprehensive | 0.003 | readout_strong | 0.3423 | 0.1303 | 0.0898 | gdr_residual | 0.2722 | 0.0491 | -0.0405 |
| ecd | optimized | comprehensive | 0.1 | ideal | 0.7875 | 0.8880 | 0.8668 | gdr_damped | 0.9082 | 0.3429 | -0.0213 |
| ecd | optimized | comprehensive | 0.1 | readout_strong | 0.8095 | 0.8742 | 0.8554 | gdr_damped | 0.9062 | 0.3508 | -0.0188 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   optimized loss                   0.003 ideal              raw                  0.0394   0.0698
ecd   optimized loss                   0.003 ideal              oracle_binomial      0.0265   0.0511
ecd   optimized loss                   0.003 ideal              gdr_param            0.0225   0.0393
ecd   optimized loss                   0.003 ideal              gdr_damped           0.0259   0.0454
ecd   optimized loss                   0.003 ideal              gdr_residual         0.0209   0.0347
ecd   optimized loss                   0.003 ideal              gdr_select           0.0225   0.0393
ecd   optimized loss                   0.003 readout_strong     raw                  0.2467   0.5835
ecd   optimized loss                   0.003 readout_strong     oracle_binomial      0.0397   0.0953
ecd   optimized loss                   0.003 readout_strong     gdr_param            0.0377   0.0840
ecd   optimized loss                   0.003 readout_strong     gdr_damped           0.0446   0.0971
ecd   optimized loss                   0.003 readout_strong     gdr_residual         0.0308   0.0694
ecd   optimized loss                   0.003 readout_strong     gdr_select           0.0377   0.0840
ecd   optimized loss                   0.100 ideal              raw                  0.5867   1.4704
ecd   optimized loss                   0.100 ideal              oracle_binomial      0.9539   2.9182
ecd   optimized loss                   0.100 ideal              gdr_param            0.4947   1.2065
ecd   optimized loss                   0.100 ideal              gdr_damped           0.5187   1.2989
ecd   optimized loss                   0.100 ideal              gdr_residual         0.9583   2.8130
ecd   optimized loss                   0.100 ideal              gdr_select           0.4947   1.2065
ecd   optimized loss                   0.100 readout_strong     raw                  0.6517   1.8513
ecd   optimized loss                   0.100 readout_strong     oracle_binomial      0.9542   2.9573
ecd   optimized loss                   0.100 readout_strong     gdr_param            0.5277   1.3474
ecd   optimized loss                   0.100 readout_strong     gdr_damped           0.5464   1.4033
ecd   optimized loss                   0.100 readout_strong     gdr_residual         0.9661   2.8935
ecd   optimized loss                   0.100 readout_strong     gdr_select           0.5277   1.3474
ecd   optimized comprehensive          0.003 ideal              raw                  0.1756   0.4813
ecd   optimized comprehensive          0.003 ideal              oracle_binomial      0.1331   0.3807
ecd   optimized comprehensive          0.003 ideal              gdr_param            0.1122   0.3161
ecd   optimized comprehensive          0.003 ideal              gdr_damped           0.1487   0.4153
ecd   optimized comprehensive          0.003 ideal              gdr_residual         0.0862   0.2456
ecd   optimized comprehensive          0.003 ideal              gdr_select           0.1122   0.3161
ecd   optimized comprehensive          0.003 readout_strong     raw                  0.3423   0.9589
ecd   optimized comprehensive          0.003 readout_strong     oracle_binomial      0.1379   0.4034
ecd   optimized comprehensive          0.003 readout_strong     gdr_param            0.1303   0.3944
ecd   optimized comprehensive          0.003 readout_strong     gdr_damped           0.1589   0.4739
ecd   optimized comprehensive          0.003 readout_strong     gdr_residual         0.0898   0.2631
ecd   optimized comprehensive          0.003 readout_strong     gdr_select           0.1303   0.3944
ecd   optimized comprehensive          0.100 ideal              raw                  0.7875   2.4187
ecd   optimized comprehensive          0.100 ideal              oracle_binomial      0.9913   1.5611
ecd   optimized comprehensive          0.100 ideal              gdr_param            0.8880   2.7462
ecd   optimized comprehensive          0.100 ideal              gdr_damped           0.8668   2.6971
ecd   optimized comprehensive          0.100 ideal              gdr_residual         0.9699   2.1311
ecd   optimized comprehensive          0.100 ideal              gdr_select           0.8880   2.7462
ecd   optimized comprehensive          0.100 readout_strong     raw                  0.8095   2.6158
ecd   optimized comprehensive          0.100 readout_strong     oracle_binomial      0.9888   1.4701
ecd   optimized comprehensive          0.100 readout_strong     gdr_param            0.8742   2.5891
ecd   optimized comprehensive          0.100 readout_strong     gdr_damped           0.8554   2.5577
ecd   optimized comprehensive          0.100 readout_strong     gdr_residual         0.9649   2.1503
ecd   optimized comprehensive          0.100 readout_strong     gdr_select           0.8742   2.5891
```
