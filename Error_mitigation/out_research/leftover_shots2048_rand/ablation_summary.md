# Ablation summary

tag=`leftover_shots2048_rand` shots=2048 n_train=40 twin=span ansatz=ecd params=random families=loss,comprehensive kappa=0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | random | loss | 0.1 | ideal | 0.3091 | 0.2765 | 0.2521 | gdr_damped | 0.2991 | 0.3725 | -0.0244 |
| ecd | random | loss | 0.1 | readout_realistic | 0.3242 | 0.2944 | 0.2790 | gdr_damped | 0.2978 | 0.3940 | -0.0155 |
| ecd | random | loss | 0.1 | readout_strong | 0.3180 | 0.2994 | 0.2752 | gdr_damped | 0.3142 | 0.3763 | -0.0241 |
| ecd | random | comprehensive | 0.1 | ideal | 0.4186 | 0.3756 | 0.3445 | gdr_damped | 0.3990 | 0.5392 | -0.0311 |
| ecd | random | comprehensive | 0.1 | readout_realistic | 0.4154 | 0.4310 | 0.3739 | gdr_damped | 0.4052 | 0.5224 | -0.0571 |
| ecd | random | comprehensive | 0.1 | readout_strong | 0.4047 | 0.3931 | 0.3447 | gdr_damped | 0.4035 | 0.5287 | -0.0484 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   random    loss                   0.100 ideal              raw                  0.3091   0.2476
ecd   random    loss                   0.100 ideal              gdr_param            0.2765   0.4499
ecd   random    loss                   0.100 ideal              gdr_damped           0.2521   0.4094
ecd   random    loss                   0.100 ideal              gdr_select           0.2521   0.4094
ecd   random    loss                   0.100 readout_realistic  raw                  0.3242   0.1220
ecd   random    loss                   0.100 readout_realistic  gdr_param            0.2944   0.3148
ecd   random    loss                   0.100 readout_realistic  gdr_damped           0.2790   0.2775
ecd   random    loss                   0.100 readout_realistic  gdr_select           0.2816   0.2869
ecd   random    loss                   0.100 readout_strong     raw                  0.3180   0.0559
ecd   random    loss                   0.100 readout_strong     gdr_param            0.2994   0.0959
ecd   random    loss                   0.100 readout_strong     gdr_damped           0.2752   0.0853
ecd   random    loss                   0.100 readout_strong     gdr_select           0.2810   0.0888
ecd   random    comprehensive          0.100 ideal              raw                  0.4186   0.0494
ecd   random    comprehensive          0.100 ideal              gdr_param            0.3756   0.1785
ecd   random    comprehensive          0.100 ideal              gdr_damped           0.3445   0.1398
ecd   random    comprehensive          0.100 ideal              gdr_select           0.3445   0.1398
ecd   random    comprehensive          0.100 readout_realistic  raw                  0.4154   0.1963
ecd   random    comprehensive          0.100 readout_realistic  gdr_param            0.4310   0.0958
ecd   random    comprehensive          0.100 readout_realistic  gdr_damped           0.3739   0.1305
ecd   random    comprehensive          0.100 readout_realistic  gdr_select           0.3739   0.1305
ecd   random    comprehensive          0.100 readout_strong     raw                  0.4047   0.0808
ecd   random    comprehensive          0.100 readout_strong     gdr_param            0.3931   0.0515
ecd   random    comprehensive          0.100 readout_strong     gdr_damped           0.3447   0.0598
ecd   random    comprehensive          0.100 readout_strong     gdr_select           0.3447   0.0598
```
