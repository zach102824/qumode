# Ablation summary

tag=`gauss_only/default_smoke` shots=8192 n_train=40 twin=adaptive ansatz=ecd params=random families=loss kappa=0.003

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | random | loss | 0.003 | ideal | 0.0434 | 0.0448 | — | — | 0.0461 | 0.0459 | — |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   random    loss                   0.003 ideal              raw                  0.0434   0.0785
ecd   random    loss                   0.003 ideal              gdr_param            0.0448   0.0824
```
