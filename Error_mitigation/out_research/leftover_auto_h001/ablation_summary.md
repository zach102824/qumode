# Ablation summary

tag=`leftover_auto_h001` shots=8192 n_train=40 twin=default ansatz=ecd params=auto families=comprehensive kappa=0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | optimized | comprehensive | 0.1 | ideal | 0.7875 | 0.8290 | 0.8152 | gdr_damped | 0.9082 | 0.3429 | -0.0138 |
| ecd | optimized | comprehensive | 0.1 | readout_strong | 0.8095 | 0.8336 | 0.8209 | gdr_damped | 0.9062 | 0.3508 | -0.0128 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   optimized comprehensive          0.100 ideal              raw                  0.7875   2.4187
ecd   optimized comprehensive          0.100 ideal              gdr_param            0.8290   2.5621
ecd   optimized comprehensive          0.100 ideal              gdr_damped           0.8152   2.5263
ecd   optimized comprehensive          0.100 ideal              gdr_mid              0.8278   2.3718
ecd   optimized comprehensive          0.100 ideal              gdr_select           0.8174   2.5334
ecd   optimized comprehensive          0.100 readout_strong     raw                  0.8095   2.6158
ecd   optimized comprehensive          0.100 readout_strong     gdr_param            0.8336   2.6501
ecd   optimized comprehensive          0.100 readout_strong     gdr_damped           0.8209   2.5961
ecd   optimized comprehensive          0.100 readout_strong     gdr_mid              0.8275   2.4438
ecd   optimized comprehensive          0.100 readout_strong     gdr_select           0.8241   2.6096
```
