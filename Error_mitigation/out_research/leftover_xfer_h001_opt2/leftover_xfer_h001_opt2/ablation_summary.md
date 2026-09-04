# Ablation summary

tag=`leftover_xfer_h001_opt2` shots=8192 n_train=40 twin=default ansatz=ecd params=optimized families=comprehensive kappa=0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | optimized | comprehensive | 0.1 | ideal | 0.7947 | 0.9427 | 0.8937 | gdr_damped | 0.9082 | 0.3429 | -0.0491 |
| ecd | optimized | comprehensive | 0.1 | readout_strong | 0.7960 | 0.9466 | 0.8942 | gdr_damped | 0.9062 | 0.3508 | -0.0524 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   optimized comprehensive          0.100 ideal              raw                  0.7947   4.8681
ecd   optimized comprehensive          0.100 ideal              oracle_binomial      0.9982   4.5093
ecd   optimized comprehensive          0.100 ideal              gdr_param            0.9427   4.1079
ecd   optimized comprehensive          0.100 ideal              gdr_damped           0.8937   4.2219
ecd   optimized comprehensive          0.100 ideal              gdr_select           0.9427   4.1079
ecd   optimized comprehensive          0.100 readout_strong     raw                  0.7960   4.8828
ecd   optimized comprehensive          0.100 readout_strong     oracle_binomial      0.9982   4.5509
ecd   optimized comprehensive          0.100 readout_strong     gdr_param            0.9466   4.0219
ecd   optimized comprehensive          0.100 readout_strong     gdr_damped           0.8942   4.1536
ecd   optimized comprehensive          0.100 readout_strong     gdr_select           0.9466   4.0219
```
