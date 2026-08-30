# Fixed-L=32 state prep (5-qubit register)

Hilbert-space dimension is fixed: L=32, n_qubits=5 for every α.
HEA cannot gain qubits as |α| grows. This is the cleaner expressivity test.

ECD N_d=2 with terminal rotation = **10 real parameters**.
HEA matched = smallest n(L+1) ≥ that count with n_layers≥1 (here 10 params, one CZ layer).
ECD starts: 8 random + 1 constructive seed. Cost `1-F`, L-BFGS-B.
Numbers: `results/stateprep_fixedL32.json`.

## Even cat (best over starts)

| α | L | F_trunc | n | constructive F | ECD N_d=2 | HEA matched | HEA extra | ECD trap rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 32 | 1.000000e+00 | 1 | 1.000000 | 0.996717 | 1.000000 | 1.000000 | 0.00 |
| 1.0 | 32 | 1.000000e+00 | 1 | 1.000000 | 0.948471 | 0.999996 | 0.999996 | 0.00 |
| 1.5 | 32 | 1.000000e+00 | 2 | 1.000000 | 0.944564 | 0.998678 | 0.998678 | 0.44 |
| 2.0 | 32 | 1.000000e+00 | 4 | 1.000000 | 0.964555 | 0.984862 | 0.984863 | 0.56 |
| 2.5 | 32 | 1.000000e+00 | 6 | 1.000000 | 0.976576 | 0.970150 | 0.970482 | 0.89 |
| 3.0 | 32 | 1.000000e+00 | 9 | 1.000000 | 0.983474 | 0.965311 | 0.970893 | 0.33 |

## Fock |n⟩ negative control

| α | n | ECD N_d=2 | HEA matched |
|---:|---:|---:|---:|
| 0.5 | 1 | 0.735759 | 1.000000 |
| 1.0 | 1 | 0.735759 | 1.000000 |
| 1.5 | 2 | 0.541341 | 1.000000 |
| 2.0 | 4 | 0.390734 | 1.000000 |
| 2.5 | 6 | 0.321246 | 1.000000 |
| 3.0 | 9 | 0.263511 | 1.000000 |

## Verdict

Constructive min F=1.000000. **HEA still wins** fair-match cats at fixed L=32 (4 vs 2). Register size is constant (5 qubits), so HEA does not pick up bits as |α| grows. Fock |n⟩ remains the negative control (number-sparse / coherent-dense). Not retuned.
