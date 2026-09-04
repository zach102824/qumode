# Ablation summary

tag=`leftover_il_all` shots=8192 n_train=40 twin=span ansatz=ecd params=optimized families=loss kappa=0.003,0.1

TVD unless noted. `base_*` is PR #6 (`Error_mitigation/out/`, shots=8192, n_train=40).
Same-run `gdr_param` is the controlled baseline for method changes.

| ansatz | params | family | κτ | readout | raw | gdr_param | best new | best name | base_raw | base_gdr | Δ vs same-run gdr |
|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| ecd | optimized | loss | 0.003 | ideal | 0.0383 | 0.0096 | 0.0092 | gdr_residual | 0.0363 | 0.0075 | -0.0004 |
| ecd | optimized | loss | 0.003 | readout_realistic | 0.0788 | 0.0103 | 0.0097 | gdr_residual | 0.0758 | 0.0107 | -0.0005 |
| ecd | optimized | loss | 0.003 | readout_strong | 0.1629 | 0.0165 | 0.0119 | gdr_residual | 0.1534 | 0.0075 | -0.0046 |
| ecd | optimized | loss | 0.1 | ideal | 0.6961 | 0.2298 | 0.1954 | gdr_residual | 0.7064 | 0.2015 | -0.0345 |
| ecd | optimized | loss | 0.1 | readout_realistic | 0.7090 | 0.2340 | 0.2040 | gdr_residual | 0.7082 | 0.2064 | -0.0299 |
| ecd | optimized | loss | 0.1 | readout_strong | 0.7207 | 0.2420 | 0.2072 | gdr_residual | 0.7233 | 0.2087 | -0.0349 |

## Per-method TVD

```
ansatz params    family                    kt readout            method                  TVD       dE
-----------------------------------------------------------------------------------------------------
ecd   optimized loss                   0.003 ideal              raw                  0.0383   0.1227
ecd   optimized loss                   0.003 ideal              oracle_binomial      0.0092   0.0436
ecd   optimized loss                   0.003 ideal              gdr_param            0.0096   0.0449
ecd   optimized loss                   0.003 ideal              gdr_residual         0.0092   0.0436
ecd   optimized loss                   0.003 ideal              gdr_interleave       0.0096   0.0449
ecd   optimized loss                   0.003 readout_realistic  raw                  0.0788   0.2615
ecd   optimized loss                   0.003 readout_realistic  oracle_binomial      0.0097   0.0421
ecd   optimized loss                   0.003 readout_realistic  gdr_param            0.0103   0.0462
ecd   optimized loss                   0.003 readout_realistic  gdr_residual         0.0097   0.0421
ecd   optimized loss                   0.003 readout_realistic  gdr_interleave       0.0103   0.0461
ecd   optimized loss                   0.003 readout_strong     raw                  0.1629   0.5745
ecd   optimized loss                   0.003 readout_strong     oracle_binomial      0.0119   0.0600
ecd   optimized loss                   0.003 readout_strong     gdr_param            0.0165   0.0786
ecd   optimized loss                   0.003 readout_strong     gdr_residual         0.0119   0.0600
ecd   optimized loss                   0.003 readout_strong     gdr_interleave       0.0165   0.0785
ecd   optimized loss                   0.100 ideal              raw                  0.6961   2.4681
ecd   optimized loss                   0.100 ideal              oracle_binomial      0.1975   1.0507
ecd   optimized loss                   0.100 ideal              gdr_param            0.2298   1.2486
ecd   optimized loss                   0.100 ideal              gdr_residual         0.1954   1.0292
ecd   optimized loss                   0.100 ideal              gdr_interleave       0.2541   1.3134
ecd   optimized loss                   0.100 readout_realistic  raw                  0.7090   2.5730
ecd   optimized loss                   0.100 readout_realistic  oracle_binomial      0.2065   1.0996
ecd   optimized loss                   0.100 readout_realistic  gdr_param            0.2340   1.2783
ecd   optimized loss                   0.100 readout_realistic  gdr_residual         0.2040   1.0719
ecd   optimized loss                   0.100 readout_realistic  gdr_interleave       0.2460   1.2958
ecd   optimized loss                   0.100 readout_strong     raw                  0.7207   2.7631
ecd   optimized loss                   0.100 readout_strong     oracle_binomial      0.2099   1.1159
ecd   optimized loss                   0.100 readout_strong     gdr_param            0.2420   1.3022
ecd   optimized loss                   0.100 readout_strong     gdr_residual         0.2072   1.0903
ecd   optimized loss                   0.100 readout_strong     gdr_interleave       0.2717   1.3699
```
