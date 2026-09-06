# Ablation summary

tag=`leftover_twins_n10_opt` shots=8192 n_train=40 twin=default ansatz=ecd params=optimized families=loss,comprehensive kappa=0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | optimized | loss | 0.1 | ideal | 0.6961 | 0.2030 | 0.2030 | gdr_select | 0.7064 | 0.2015 | 0.0000 |
| ecd | optimized | loss | 0.1 | readout_realistic | 0.7090 | 0.2062 | 0.2062 | gdr_select | 0.7082 | 0.2064 | 0.0000 |
| ecd | optimized | loss | 0.1 | readout_strong | 0.7207 | 0.2096 | 0.2096 | gdr_select | 0.7233 | 0.2087 | 0.0000 |
| ecd | optimized | comprehensive | 0.1 | ideal | 0.9086 | 0.3429 | 0.3429 | gdr_select | 0.9082 | 0.3429 | 0.0000 |
| ecd | optimized | comprehensive | 0.1 | readout_realistic | 0.9116 | 0.3491 | 0.3491 | gdr_select | 0.9094 | 0.3519 | 0.0000 |
| ecd | optimized | comprehensive | 0.1 | readout_strong | 0.9065 | 0.3395 | 0.3395 | gdr_select | 0.9062 | 0.3508 | 0.0000 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   optimized loss                   0.100 ideal              raw                  0.6961   2.4681
ecd   optimized loss                   0.100 ideal              gdr_param            0.2030   1.1491
ecd   optimized loss                   0.100 ideal              gdr_damped           0.3261   1.4789
ecd   optimized loss                   0.100 ideal              gdr_select           0.2030   1.1491
ecd   optimized loss                   0.100 readout_realistic  raw                  0.7090   2.5730
ecd   optimized loss                   0.100 readout_realistic  gdr_param            0.2062   1.1795
ecd   optimized loss                   0.100 readout_realistic  gdr_damped           0.3304   1.5123
ecd   optimized loss                   0.100 readout_realistic  gdr_select           0.2062   1.1795
ecd   optimized loss                   0.100 readout_strong     raw                  0.7207   2.7631
ecd   optimized loss                   0.100 readout_strong     gdr_param            0.2096   1.1926
ecd   optimized loss                   0.100 readout_strong     gdr_damped           0.3077   1.4617
ecd   optimized loss                   0.100 readout_strong     gdr_select           0.2096   1.1926
ecd   optimized comprehensive          0.100 ideal              raw                  0.9086   3.7039
ecd   optimized comprehensive          0.100 ideal              gdr_param            0.3429   1.8528
ecd   optimized comprehensive          0.100 ideal              gdr_damped           0.4268   2.1305
ecd   optimized comprehensive          0.100 ideal              gdr_select           0.3429   1.8528
ecd   optimized comprehensive          0.100 readout_realistic  raw                  0.9116   3.7765
ecd   optimized comprehensive          0.100 readout_realistic  gdr_param            0.3491   1.8665
ecd   optimized comprehensive          0.100 readout_realistic  gdr_damped           0.4326   2.1483
ecd   optimized comprehensive          0.100 readout_realistic  gdr_select           0.3491   1.8665
ecd   optimized comprehensive          0.100 readout_strong     raw                  0.9065   3.8261
ecd   optimized comprehensive          0.100 readout_strong     gdr_param            0.3395   1.8232
ecd   optimized comprehensive          0.100 readout_strong     gdr_damped           0.4521   2.1982
ecd   optimized comprehensive          0.100 readout_strong     gdr_select           0.3395   1.8232
```
