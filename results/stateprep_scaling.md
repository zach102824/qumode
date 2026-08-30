# ECD vs HEA state preparation

The thesis is about **constructive O(1) ECD** for K=2 coherent-sparse targets
(even two-legged cats) versus a Fock-register qubit HEA.
**Variational** matched ECD vs HEA is a separate, optimizer-sensitive question.
On the cleaner fixed-L=32 fair-match test, **HEA still wins cats** (4 α-points vs 2).

Cost is `1-F` (not Gibbs). This is not a claim that ECD is a new gate, that cats
are new, or that Gibbs / `sampled_tail` helps state prep.

## Files

| File | What |
|---|---|
| `results/stateprep_scaling.json` | **Historical** growing-L run (nearest HEA match, 4 random starts). Do not overwrite. Confounded: at n=6 the nearest match to 8 params is a product of 6 Ry; several ECD starts sat in the F=1/2 trap. |
| `results/stateprep_scaling_nearest_4start.md` | Tables from that historical JSON. |
| `results/stateprep_scaling_fair.json` / `.md` | **A. Growing-L, fair match.** Floor HEA (`n_params ≥ ECD`, ≥1 CZ layer). ECD: 8 random + 1 constructive seed. |
| `results/stateprep_fixedL32.json` / `.md` | **B. Fixed L=32**, n_qubits=5 for every α. Cleaner: HEA cannot gain qubits as |α| grows. |

## A. Growing-L fair match (comparison of record for variational)

Constructive post-select min F=1.000000. Fair-match even-cat: ECD N_d=2 wins 4 α-points, HEA 3. Best-start ECD F: α=1.0:0.9485, α=1.5:0.9446, α=2.0:0.9646, α=2.5:0.9766, α=3.0:0.9835, α=3.5:0.9877, α=4.0:0.9906. HEA floor-matched: α=1.0:1.0000, α=1.5:0.9987, α=2.0:0.9849, α=2.5:0.9701, α=3.0:0.9694, α=3.5:0.9671, α=4.0:0.9666. ECD trap rates (do not hide): α=1.0:0.00, α=1.5:0.44, α=2.0:0.56, α=2.5:0.89, α=3.0:0.33, α=3.5:0.44, α=4.0:0.44. Fock |n⟩: HEA wins 7/7. Variational matched ECD vs HEA is optimizer-sensitive; the thesis claim of record is the constructive O(1) circuit for K=2 coherent-sparse cats. Parameters were not retuned to force an ECD win.

Full table: `results/stateprep_scaling_fair.md`.

## B. Fixed L=32 (5-qubit register)

Constructive min F=1.000000. **HEA still wins** fair-match cats at fixed L=32 (4 vs 2). Register size is constant (5 qubits), so HEA does not pick up bits as |α| grows. Fock |n⟩ remains the negative control (number-sparse / coherent-dense). Not retuned.

Full table: `results/stateprep_fixedL32.md`.

## Citations (not claimed as this work)

- Eickbusch et al., arXiv:2111.06414 — ECD and Fock compilation (≲10 ECD for |7⟩).
- Singh, Royer, Girvin, arXiv:2504.19992 — analytic ECD circuits.
- Zhou and Lucas, arXiv:2608.07696 — N-fold cat lower bound.
- Lu et al., arXiv:2603.09233 — we do not beat position encoding.
- Krastanov et al., Phys. Rev. A 92, 040303 (2015) — SNAP / oscillator control.
- Dutta et al., arXiv:2501.11735 — **negative control / contrast only**: binary knapsack VQE, wrong problem class.
