# Ablation summary

tag=`leftover_auto_snap` shots=8192 n_train=40 twin=default ansatz=snap params=auto families=comprehensive kappa=0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| snap | optimized | comprehensive | 0.1 | ideal | 0.7914 | 0.6619 | 0.6833 | gdr_damped | 0.7933 | 0.6320 | 0.0214 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
snap  optimized comprehensive          0.100 ideal              raw                  0.7914   3.4906
snap  optimized comprehensive          0.100 ideal              gdr_param            0.6619   2.2542
snap  optimized comprehensive          0.100 ideal              gdr_damped           0.6833   2.5015
snap  optimized comprehensive          0.100 ideal              gdr_select           0.6886   2.5633
```
