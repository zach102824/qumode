# Ablation summary

tag=`leftover_shots2048_opt` shots=2048 n_train=40 twin=default ansatz=ecd params=optimized families=comprehensive kappa=0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | optimized | comprehensive | 0.1 | ideal | 0.9060 | 0.3514 | 0.3514 | gdr_select | 0.9082 | 0.3429 | 0.0000 |
| ecd | optimized | comprehensive | 0.1 | readout_realistic | 0.9173 | 0.3604 | 0.3604 | gdr_select | 0.9094 | 0.3519 | 0.0000 |
| ecd | optimized | comprehensive | 0.1 | readout_strong | 0.9041 | 0.3515 | 0.3515 | gdr_select | 0.9062 | 0.3508 | 0.0000 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   optimized comprehensive          0.100 ideal              raw                  0.9060   3.7350
ecd   optimized comprehensive          0.100 ideal              gdr_param            0.3514   1.9013
ecd   optimized comprehensive          0.100 ideal              gdr_damped           0.4336   2.1764
ecd   optimized comprehensive          0.100 ideal              gdr_select           0.3514   1.9013
ecd   optimized comprehensive          0.100 readout_realistic  raw                  0.9173   3.8847
ecd   optimized comprehensive          0.100 readout_realistic  gdr_param            0.3604   1.9398
ecd   optimized comprehensive          0.100 readout_realistic  gdr_damped           0.4431   2.2282
ecd   optimized comprehensive          0.100 readout_realistic  gdr_select           0.3604   1.9398
ecd   optimized comprehensive          0.100 readout_strong     raw                  0.9041   3.8106
ecd   optimized comprehensive          0.100 readout_strong     gdr_param            0.3515   1.8963
ecd   optimized comprehensive          0.100 readout_strong     gdr_damped           0.4336   2.1636
ecd   optimized comprehensive          0.100 readout_strong     gdr_select           0.3515   1.8963
```
