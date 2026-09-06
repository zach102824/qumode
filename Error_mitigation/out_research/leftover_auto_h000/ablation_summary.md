# Ablation summary

tag=`leftover_auto_h000` shots=8192 n_train=40 twin=default ansatz=ecd params=auto families=comprehensive kappa=0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | optimized | comprehensive | 0.1 | ideal | 0.9086 | 0.4163 | 0.4104 | gdr_mid | 0.9082 | 0.3429 | -0.0058 |
| ecd | optimized | comprehensive | 0.1 | readout_strong | 0.9065 | 0.4034 | 0.4012 | gdr_mid | 0.9062 | 0.3508 | -0.0022 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   optimized comprehensive          0.100 ideal              raw                  0.9086   3.7039
ecd   optimized comprehensive          0.100 ideal              gdr_param            0.4163   2.3902
ecd   optimized comprehensive          0.100 ideal              gdr_damped           0.5639   2.7843
ecd   optimized comprehensive          0.100 ideal              gdr_mid              0.4104   2.3064
ecd   optimized comprehensive          0.100 ideal              gdr_select           0.5885   2.8500
ecd   optimized comprehensive          0.100 readout_strong     raw                  0.9065   3.8261
ecd   optimized comprehensive          0.100 readout_strong     gdr_param            0.4034   2.2756
ecd   optimized comprehensive          0.100 readout_strong     gdr_damped           0.5545   2.7023
ecd   optimized comprehensive          0.100 readout_strong     gdr_mid              0.4012   2.2196
ecd   optimized comprehensive          0.100 readout_strong     gdr_select           0.5797   2.7734
```
