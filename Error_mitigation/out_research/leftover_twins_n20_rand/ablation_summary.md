# Ablation summary

tag=`leftover_twins_n20_rand` shots=8192 n_train=40 twin=span ansatz=ecd params=random families=loss,comprehensive kappa=0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | random | loss | 0.1 | ideal | 0.2983 | 0.2302 | 0.2151 | gdr_damped | 0.2991 | 0.3725 | -0.0152 |
| ecd | random | loss | 0.1 | readout_realistic | 0.3078 | 0.2380 | 0.2210 | gdr_damped | 0.2978 | 0.3940 | -0.0170 |
| ecd | random | loss | 0.1 | readout_strong | 0.3153 | 0.2496 | 0.2396 | gdr_damped | 0.3142 | 0.3763 | -0.0101 |
| ecd | random | comprehensive | 0.1 | ideal | 0.4031 | 0.4062 | 0.3596 | gdr_damped | 0.3990 | 0.5392 | -0.0466 |
| ecd | random | comprehensive | 0.1 | readout_realistic | 0.3980 | 0.3741 | 0.3139 | gdr_damped | 0.4052 | 0.5224 | -0.0602 |
| ecd | random | comprehensive | 0.1 | readout_strong | 0.3995 | 0.3593 | 0.3208 | gdr_damped | 0.4035 | 0.5287 | -0.0385 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   random    loss                   0.100 ideal              raw                  0.2983   0.0834
ecd   random    loss                   0.100 ideal              gdr_param            0.2302   0.0412
ecd   random    loss                   0.100 ideal              gdr_damped           0.2151   0.0475
ecd   random    loss                   0.100 ideal              gdr_select           0.2302   0.0412
ecd   random    loss                   0.100 readout_realistic  raw                  0.3078   0.0560
ecd   random    loss                   0.100 readout_realistic  gdr_param            0.2380   0.0321
ecd   random    loss                   0.100 readout_realistic  gdr_damped           0.2210   0.0181
ecd   random    loss                   0.100 readout_realistic  gdr_select           0.2380   0.0321
ecd   random    loss                   0.100 readout_strong     raw                  0.3153   0.0285
ecd   random    loss                   0.100 readout_strong     gdr_param            0.2496   0.0129
ecd   random    loss                   0.100 readout_strong     gdr_damped           0.2396   0.0174
ecd   random    loss                   0.100 readout_strong     gdr_select           0.2496   0.0129
ecd   random    comprehensive          0.100 ideal              raw                  0.4031   0.1404
ecd   random    comprehensive          0.100 ideal              gdr_param            0.4062   0.1449
ecd   random    comprehensive          0.100 ideal              gdr_damped           0.3596   0.1438
ecd   random    comprehensive          0.100 ideal              gdr_select           0.3596   0.1438
ecd   random    comprehensive          0.100 readout_realistic  raw                  0.3980   0.1416
ecd   random    comprehensive          0.100 readout_realistic  gdr_param            0.3741   0.0232
ecd   random    comprehensive          0.100 readout_realistic  gdr_damped           0.3139   0.0389
ecd   random    comprehensive          0.100 readout_realistic  gdr_select           0.3203   0.0300
ecd   random    comprehensive          0.100 readout_strong     raw                  0.3995   0.1153
ecd   random    comprehensive          0.100 readout_strong     gdr_param            0.3593   0.0388
ecd   random    comprehensive          0.100 readout_strong     gdr_damped           0.3208   0.0738
ecd   random    comprehensive          0.100 readout_strong     gdr_select           0.3221   0.0688
```
