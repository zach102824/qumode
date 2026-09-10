# Ablation summary

tag=`leftover_twins_n20_opt` shots=8192 n_train=40 twin=default ansatz=ecd params=optimized families=loss,comprehensive kappa=0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | optimized | loss | 0.1 | ideal | 0.6961 | 0.2007 | 0.2007 | gdr_select | 0.7064 | 0.2015 | 0.0000 |
| ecd | optimized | loss | 0.1 | readout_realistic | 0.7090 | 0.2051 | 0.2051 | gdr_select | 0.7082 | 0.2064 | 0.0000 |
| ecd | optimized | loss | 0.1 | readout_strong | 0.7207 | 0.2096 | 0.2096 | gdr_select | 0.7233 | 0.2087 | 0.0000 |
| ecd | optimized | comprehensive | 0.1 | ideal | 0.9086 | 0.3406 | 0.3406 | gdr_select | 0.9082 | 0.3429 | 0.0000 |
| ecd | optimized | comprehensive | 0.1 | readout_realistic | 0.9116 | 0.3457 | 0.3457 | gdr_select | 0.9094 | 0.3519 | 0.0000 |
| ecd | optimized | comprehensive | 0.1 | readout_strong | 0.9065 | 0.3385 | 0.3385 | gdr_select | 0.9062 | 0.3508 | 0.0000 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   optimized loss                   0.100 ideal              raw                  0.6961   2.4681
ecd   optimized loss                   0.100 ideal              gdr_param            0.2007   1.1342
ecd   optimized loss                   0.100 ideal              gdr_damped           0.2748   1.3343
ecd   optimized loss                   0.100 ideal              gdr_select           0.2007   1.1342
ecd   optimized loss                   0.100 readout_realistic  raw                  0.7090   2.5730
ecd   optimized loss                   0.100 readout_realistic  gdr_param            0.2051   1.1598
ecd   optimized loss                   0.100 readout_realistic  gdr_damped           0.3045   1.4300
ecd   optimized loss                   0.100 readout_realistic  gdr_select           0.2051   1.1598
ecd   optimized loss                   0.100 readout_strong     raw                  0.7207   2.7631
ecd   optimized loss                   0.100 readout_strong     gdr_param            0.2096   1.1701
ecd   optimized loss                   0.100 readout_strong     gdr_damped           0.2585   1.3069
ecd   optimized loss                   0.100 readout_strong     gdr_select           0.2096   1.1701
ecd   optimized comprehensive          0.100 ideal              raw                  0.9086   3.7039
ecd   optimized comprehensive          0.100 ideal              gdr_param            0.3406   1.8449
ecd   optimized comprehensive          0.100 ideal              gdr_damped           0.4249   2.1238
ecd   optimized comprehensive          0.100 ideal              gdr_select           0.3406   1.8449
ecd   optimized comprehensive          0.100 readout_realistic  raw                  0.9116   3.7765
ecd   optimized comprehensive          0.100 readout_realistic  gdr_param            0.3457   1.8512
ecd   optimized comprehensive          0.100 readout_realistic  gdr_damped           0.4297   2.1353
ecd   optimized comprehensive          0.100 readout_realistic  gdr_select           0.3457   1.8512
ecd   optimized comprehensive          0.100 readout_strong     raw                  0.9065   3.8261
ecd   optimized comprehensive          0.100 readout_strong     gdr_param            0.3385   1.8037
ecd   optimized comprehensive          0.100 readout_strong     gdr_damped           0.4228   2.0878
ecd   optimized comprehensive          0.100 readout_strong     gdr_select           0.3385   1.8037
```
