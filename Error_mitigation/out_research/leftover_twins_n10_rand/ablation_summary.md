# Ablation summary

tag=`leftover_twins_n10_rand` shots=8192 n_train=40 twin=span ansatz=ecd params=random families=loss,comprehensive kappa=0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | random | loss | 0.1 | ideal | 0.2983 | 0.2299 | 0.2189 | gdr_damped | 0.2991 | 0.3725 | -0.0110 |
| ecd | random | loss | 0.1 | readout_realistic | 0.3078 | 0.2330 | 0.2186 | gdr_damped | 0.2978 | 0.3940 | -0.0144 |
| ecd | random | loss | 0.1 | readout_strong | 0.3153 | 0.2461 | 0.2363 | gdr_damped | 0.3142 | 0.3763 | -0.0098 |
| ecd | random | comprehensive | 0.1 | ideal | 0.4031 | 0.3841 | 0.3501 | gdr_damped | 0.3990 | 0.5392 | -0.0340 |
| ecd | random | comprehensive | 0.1 | readout_realistic | 0.3980 | 0.3530 | 0.3108 | gdr_damped | 0.4052 | 0.5224 | -0.0422 |
| ecd | random | comprehensive | 0.1 | readout_strong | 0.3995 | 0.3522 | 0.3212 | gdr_damped | 0.4035 | 0.5287 | -0.0310 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   random    loss                   0.100 ideal              raw                  0.2983   0.0834
ecd   random    loss                   0.100 ideal              gdr_param            0.2299   0.0365
ecd   random    loss                   0.100 ideal              gdr_damped           0.2189   0.0412
ecd   random    loss                   0.100 ideal              gdr_select           0.2299   0.0365
ecd   random    loss                   0.100 readout_realistic  raw                  0.3078   0.0560
ecd   random    loss                   0.100 readout_realistic  gdr_param            0.2330   0.0467
ecd   random    loss                   0.100 readout_realistic  gdr_damped           0.2186   0.0305
ecd   random    loss                   0.100 readout_realistic  gdr_select           0.2330   0.0467
ecd   random    loss                   0.100 readout_strong     raw                  0.3153   0.0285
ecd   random    loss                   0.100 readout_strong     gdr_param            0.2461   0.0191
ecd   random    loss                   0.100 readout_strong     gdr_damped           0.2363   0.0239
ecd   random    loss                   0.100 readout_strong     gdr_select           0.2461   0.0191
ecd   random    comprehensive          0.100 ideal              raw                  0.4031   0.1404
ecd   random    comprehensive          0.100 ideal              gdr_param            0.3841   0.1492
ecd   random    comprehensive          0.100 ideal              gdr_damped           0.3501   0.1470
ecd   random    comprehensive          0.100 ideal              gdr_select           0.3612   0.1479
ecd   random    comprehensive          0.100 readout_realistic  raw                  0.3980   0.1416
ecd   random    comprehensive          0.100 readout_realistic  gdr_param            0.3530   0.0773
ecd   random    comprehensive          0.100 readout_realistic  gdr_damped           0.3108   0.0078
ecd   random    comprehensive          0.100 readout_realistic  gdr_select           0.3165   0.0194
ecd   random    comprehensive          0.100 readout_strong     raw                  0.3995   0.1153
ecd   random    comprehensive          0.100 readout_strong     gdr_param            0.3522   0.0005
ecd   random    comprehensive          0.100 readout_strong     gdr_damped           0.3212   0.0558
ecd   random    comprehensive          0.100 readout_strong     gdr_select           0.3224   0.0420
```
