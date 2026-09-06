# Ablation summary

tag=`leftover_mid` shots=8192 n_train=40 twin=default ansatz=ecd params=optimized families=comprehensive kappa=0.003,0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | optimized | comprehensive | 0.003 | ideal | 0.1842 | 0.0522 | 0.0451 | gdr_mid | 0.1823 | 0.0534 | -0.0072 |
| ecd | optimized | comprehensive | 0.003 | readout_realistic | 0.2074 | 0.0539 | 0.0472 | gdr_mid | 0.2158 | 0.0523 | -0.0067 |
| ecd | optimized | comprehensive | 0.003 | readout_strong | 0.2799 | 0.0543 | 0.0473 | gdr_mid | 0.2722 | 0.0491 | -0.0069 |
| ecd | optimized | comprehensive | 0.1 | ideal | 0.9086 | 0.3434 | 0.3433 | gdr_band | 0.9082 | 0.3429 | -0.0001 |
| ecd | optimized | comprehensive | 0.1 | readout_realistic | 0.9116 | 0.3522 | 0.3520 | gdr_band | 0.9094 | 0.3519 | -0.0001 |
| ecd | optimized | comprehensive | 0.1 | readout_strong | 0.9065 | 0.3381 | 0.3376 | gdr_band | 0.9062 | 0.3508 | -0.0005 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   optimized comprehensive          0.003 ideal              raw                  0.1842   0.7821
ecd   optimized comprehensive          0.003 ideal              gdr_param            0.0522   0.2727
ecd   optimized comprehensive          0.003 ideal              gdr_mid              0.0451   0.2540
ecd   optimized comprehensive          0.003 ideal              gdr_residual         0.1048   0.5892
ecd   optimized comprehensive          0.003 ideal              gdr_split            0.0506   0.2658
ecd   optimized comprehensive          0.003 ideal              gdr_band             0.0510   0.2675
ecd   optimized comprehensive          0.003 ideal              gdr_select           0.0522   0.2727
ecd   optimized comprehensive          0.003 readout_realistic  raw                  0.2074   0.8398
ecd   optimized comprehensive          0.003 readout_realistic  gdr_param            0.0539   0.2552
ecd   optimized comprehensive          0.003 readout_realistic  gdr_mid              0.0472   0.2404
ecd   optimized comprehensive          0.003 readout_realistic  gdr_residual         0.1010   0.5457
ecd   optimized comprehensive          0.003 readout_realistic  gdr_split            0.0523   0.2501
ecd   optimized comprehensive          0.003 readout_realistic  gdr_band             0.0527   0.2511
ecd   optimized comprehensive          0.003 readout_realistic  gdr_select           0.0539   0.2552
ecd   optimized comprehensive          0.003 readout_strong     raw                  0.2799   1.1144
ecd   optimized comprehensive          0.003 readout_strong     gdr_param            0.0543   0.2482
ecd   optimized comprehensive          0.003 readout_strong     gdr_mid              0.0473   0.2310
ecd   optimized comprehensive          0.003 readout_strong     gdr_residual         0.1054   0.5610
ecd   optimized comprehensive          0.003 readout_strong     gdr_split            0.0520   0.2404
ecd   optimized comprehensive          0.003 readout_strong     gdr_band             0.0523   0.2413
ecd   optimized comprehensive          0.003 readout_strong     gdr_select           0.0543   0.2482
ecd   optimized comprehensive          0.100 ideal              raw                  0.9086   3.7039
ecd   optimized comprehensive          0.100 ideal              gdr_param            0.3434   1.8609
ecd   optimized comprehensive          0.100 ideal              gdr_mid              0.3484   1.8943
ecd   optimized comprehensive          0.100 ideal              gdr_residual         0.4138   2.1735
ecd   optimized comprehensive          0.100 ideal              gdr_split            0.3434   1.8609
ecd   optimized comprehensive          0.100 ideal              gdr_band             0.3433   1.8613
ecd   optimized comprehensive          0.100 ideal              gdr_select           0.3434   1.8609
ecd   optimized comprehensive          0.100 readout_realistic  raw                  0.9116   3.7765
ecd   optimized comprehensive          0.100 readout_realistic  gdr_param            0.3522   1.8790
ecd   optimized comprehensive          0.100 readout_realistic  gdr_mid              0.3577   1.9127
ecd   optimized comprehensive          0.100 readout_realistic  gdr_residual         0.4168   2.1877
ecd   optimized comprehensive          0.100 readout_realistic  gdr_split            0.3522   1.8789
ecd   optimized comprehensive          0.100 readout_realistic  gdr_band             0.3520   1.8793
ecd   optimized comprehensive          0.100 readout_realistic  gdr_select           0.3522   1.8790
ecd   optimized comprehensive          0.100 readout_strong     raw                  0.9065   3.8261
ecd   optimized comprehensive          0.100 readout_strong     gdr_param            0.3381   1.8243
ecd   optimized comprehensive          0.100 readout_strong     gdr_mid              0.3465   1.8533
ecd   optimized comprehensive          0.100 readout_strong     gdr_residual         0.4057   2.1295
ecd   optimized comprehensive          0.100 readout_strong     gdr_split            0.3379   1.8239
ecd   optimized comprehensive          0.100 readout_strong     gdr_band             0.3376   1.8286
ecd   optimized comprehensive          0.100 readout_strong     gdr_select           0.3381   1.8243
```
