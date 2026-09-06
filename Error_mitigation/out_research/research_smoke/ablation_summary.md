# Ablation summary

tag=`research_smoke` shots=2048 n_train=40 twin=adaptive ansatz=ecd params=optimized families=loss kappa=0.003

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | optimized | loss | 0.003 | ideal | 0.0380 | 0.0096 | 0.0096 | gdr_select | 0.0363 | 0.0075 | 0.0000 |
| ecd | optimized | loss | 0.003 | readout_realistic | 0.0884 | 0.0192 | 0.0192 | gdr_select | 0.0758 | 0.0107 | 0.0000 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   optimized loss                   0.003 ideal              raw                  0.0380   0.1149
ecd   optimized loss                   0.003 ideal              gdr_param            0.0096   0.0382
ecd   optimized loss                   0.003 ideal              gdr_damped           0.0238   0.0766
ecd   optimized loss                   0.003 ideal              gdr_select           0.0096   0.0382
ecd   optimized loss                   0.003 readout_realistic  raw                  0.0884   0.2910
ecd   optimized loss                   0.003 readout_realistic  gdr_param            0.0192   0.0777
ecd   optimized loss                   0.003 readout_realistic  gdr_damped           0.0369   0.1250
ecd   optimized loss                   0.003 readout_realistic  gdr_select           0.0192   0.0777
```
