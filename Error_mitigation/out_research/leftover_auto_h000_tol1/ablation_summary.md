# Ablation summary

tag=`leftover_auto_h000_tol1` shots=8192 n_train=40 twin=default ansatz=ecd params=auto families=comprehensive kappa=0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | optimized | comprehensive | 0.1 | ideal | 0.9086 | 0.3434 | 0.3434 | gdr_select | 0.9082 | 0.3429 | 0.0000 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   optimized comprehensive          0.100 ideal              raw                  0.9086   3.7039
ecd   optimized comprehensive          0.100 ideal              gdr_param            0.3434   1.8609
ecd   optimized comprehensive          0.100 ideal              gdr_damped           0.4272   2.1373
ecd   optimized comprehensive          0.100 ideal              gdr_select           0.3434   1.8609
```
