# Fair-match growing-L state prep (protocol fix)

This run fixes two confounds in the historical `results/stateprep_scaling.json`:
1. HEA match is a **floor**: `n_params >= ECD_params` and `n_layers >= 1` (at least one CZ).
   For n=6 and ECD N_d=2+terminal (10 params) that is 12 HEA params, not a product of 6 Ry.
2. ECD uses **8 random starts plus 1 constructive seed** (Ry(π/2)+ECD(2α), second block small).
   Trap rate = fraction of those starts with F in [0.48, 0.52] (the N_d=1 even/odd trap).

Cost is `1-F`. L-BFGS-B, not Gibbs. Numbers are from `results/stateprep_scaling_fair.json`.
HEA extra (n_layers=5) is a caveat, not the comparison of record.

## Param counts

| ECD | N_d=2, terminal_rotation=True, 10 real params |
| HEA matched | smallest n(L+1) ≥ ECD params with L≥1 |
| Starts | 8 random + 1 constructive ECD seed, maxiter=200 |

| α | L | n_qubits | ECD params | HEA floor params | HEA nearest-to-8 (historical confound) |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 25 | 5 | 10 | 10 | 10 |
| 1.5 | 31 | 5 | 10 | 10 | 10 |
| 2.0 | 36 | 6 | 10 | 12 | 6 |
| 2.5 | 43 | 6 | 10 | 12 | 6 |
| 3.0 | 49 | 6 | 10 | 12 | 6 |
| 3.5 | 57 | 6 | 10 | 12 | 6 |
| 4.0 | 64 | 6 | 10 | 12 | 6 |

## Even cat (best over starts)

| α | L | constructive F | ECD N_d=2 | HEA floor-matched | HEA extra | ECD trap rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 25 | 1.000000 | 0.948471 | 0.999996 | 0.999996 | 0.00 |
| 1.5 | 31 | 1.000000 | 0.944564 | 0.998678 | 0.998678 | 0.44 |
| 2.0 | 36 | 1.000000 | 0.964555 | 0.984862 | 0.984863 | 0.56 |
| 2.5 | 43 | 1.000000 | 0.976576 | 0.970150 | 0.970482 | 0.89 |
| 3.0 | 49 | 1.000000 | 0.983474 | 0.969426 | 0.970893 | 0.33 |
| 3.5 | 57 | 1.000000 | 0.987742 | 0.967137 | 0.970906 | 0.44 |
| 4.0 | 64 | 1.000000 | 0.990556 | 0.966596 | 0.966868 | 0.44 |

## Fock |n⟩ negative control (best over starts)

| α | n | L | ECD N_d=2 | HEA floor-matched | ECD trap rate |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 1 | 25 | 0.735759 | 1.000000 | 0.00 |
| 1.5 | 2 | 31 | 0.541341 | 1.000000 | 0.00 |
| 2.0 | 4 | 36 | 0.390734 | 1.000000 | 0.00 |
| 2.5 | 6 | 43 | 0.321246 | 1.000000 | 0.00 |
| 3.0 | 9 | 49 | 0.263511 | 1.000000 | 0.00 |
| 3.5 | 12 | 57 | 0.228736 | 1.000000 | 0.00 |
| 4.0 | 16 | 64 | 0.198435 | 1.000000 | 0.00 |

## Verdict

Constructive post-select min F=1.000000. Fair-match even-cat: ECD N_d=2 wins 4 α-points, HEA 3. Best-start ECD F: α=1.0:0.9485, α=1.5:0.9446, α=2.0:0.9646, α=2.5:0.9766, α=3.0:0.9835, α=3.5:0.9877, α=4.0:0.9906. HEA floor-matched: α=1.0:1.0000, α=1.5:0.9987, α=2.0:0.9849, α=2.5:0.9701, α=3.0:0.9694, α=3.5:0.9671, α=4.0:0.9666. ECD trap rates (do not hide): α=1.0:0.00, α=1.5:0.44, α=2.0:0.56, α=2.5:0.89, α=3.0:0.33, α=3.5:0.44, α=4.0:0.44. Fock |n⟩: HEA wins 7/7. Variational matched ECD vs HEA is optimizer-sensitive; the thesis claim of record is the constructive O(1) circuit for K=2 coherent-sparse cats. Parameters were not retuned to force an ECD win.
