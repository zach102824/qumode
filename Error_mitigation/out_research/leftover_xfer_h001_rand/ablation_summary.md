# Ablation summary

tag=`leftover_xfer_h001_rand` shots=8192 n_train=40 twin=span ansatz=ecd params=random families=loss,comprehensive kappa=0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | random | loss | 0.1 | ideal | 0.2231 | 0.1417 | 0.1364 | gdr_damped | 0.2991 | 0.3725 | -0.0053 |
| ecd | random | loss | 0.1 | readout_strong | 0.2349 | 0.1547 | 0.1529 | gdr_select | 0.3142 | 0.3763 | -0.0019 |
| ecd | random | comprehensive | 0.1 | ideal | 0.3125 | 0.1982 | 0.1868 | gdr_damped | 0.3990 | 0.5392 | -0.0114 |
| ecd | random | comprehensive | 0.1 | readout_strong | 0.3225 | 0.2240 | 0.2177 | gdr_damped | 0.4035 | 0.5287 | -0.0063 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   random    loss                   0.100 ideal              raw                  0.2231   0.0907
ecd   random    loss                   0.100 ideal              oracle_binomial      0.3583   0.1636
ecd   random    loss                   0.100 ideal              gdr_param            0.1417   0.0252
ecd   random    loss                   0.100 ideal              gdr_damped           0.1364   0.0318
ecd   random    loss                   0.100 ideal              gdr_select           0.1364   0.0318
ecd   random    loss                   0.100 readout_strong     raw                  0.2349   0.1453
ecd   random    loss                   0.100 readout_strong     oracle_binomial      0.3510   0.1436
ecd   random    loss                   0.100 readout_strong     gdr_param            0.1547   0.0869
ecd   random    loss                   0.100 readout_strong     gdr_damped           0.1531   0.0940
ecd   random    loss                   0.100 readout_strong     gdr_select           0.1529   0.0975
ecd   random    comprehensive          0.100 ideal              raw                  0.3125   0.1742
ecd   random    comprehensive          0.100 ideal              oracle_binomial      0.6425   0.0425
ecd   random    comprehensive          0.100 ideal              gdr_param            0.1982   0.0189
ecd   random    comprehensive          0.100 ideal              gdr_damped           0.1868   0.0004
ecd   random    comprehensive          0.100 ideal              gdr_select           0.1868   0.0004
ecd   random    comprehensive          0.100 readout_strong     raw                  0.3225   0.2090
ecd   random    comprehensive          0.100 readout_strong     oracle_binomial      0.6106   0.0294
ecd   random    comprehensive          0.100 readout_strong     gdr_param            0.2240   0.1381
ecd   random    comprehensive          0.100 readout_strong     gdr_damped           0.2177   0.1456
ecd   random    comprehensive          0.100 readout_strong     gdr_select           0.2194   0.1418
```
