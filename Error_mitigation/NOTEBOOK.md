# GDR improvement lab notebook

Living notes for the 30-hour GDR pass on `cursor/gdr-improve-30h-b22c`.
Baseline smoke/full is owned by another agent (`Error_mitigation/out_smoke`, `Error_mitigation/out`); this work writes only under `Error_mitigation/out_research/` and `out_ablation/`.

## Phase 0 — failure modes (2026-09-04)

Unit tests: `tests/test_error_mitigation.py` 10/10 green before any edits.

Timed on this VM: ECD \(N_d=5\) noiseless histogram ~0.016 s; **noisy** `density_matrix` ~0.49 s. Physical TVD at κτ=0.003, random ECD, **no shots**: **0.0175**. The README smoke row (raw TVD 0.0616 / gdr 0.0646 at 4000 shots) is therefore **shot-noise dominated**. Unconstrained 11-parameter MLE + 80-step Richardson–Lucy on a near-identity kernel overfits the multinomial noise. That is **under-fit / over-capacity**, not a structural failure of Gaussian twins.

| case | structural? | why |
|------|:-----------:|-----|
| Mild loss, GDR ≥ raw | no | 11 params, 12 twins, η≈0.985, RL unfolds shot noise. Prior + freeze readout + identity-aware unfold. |
| Oracle residual ≫ shot noise on target | **yes** | Loss *between* non-Gaussian gates does not commute to a single Fock kernel. Exact for \(t_{\mathrm{free}}=0\) twins. |
| Joint loss + readout | mixed | \(n\to n-1\) vs \(n\to n\pm1\) degenerate on one circuit; |α|² diversity + freeze calibrated readout. |
| `zne_idle` under readout | **yes** | Idle stretch does not scale the detector. Invert readout per scale, then extrapolate. |
| Heating / η^k moments | **yes** for moments | Thermal-loss kernel with \(n_{\mathrm{th}}\) is still the right parametric family. |
| `comprehensive` | **yes** | Ancilla / coherent ECD errors break phase covariance. Histogram GDR can only absorb them as leak. |

## Methods added (kept old ones)

- `gdr_param_reg` — L2 on (η→1, extras→0, readout→spec), holdout CV over λ_η, shrink-unfold.
- `gdr_eta` / `gdr_eta_nth` — freeze calibrated readout and extra hops; fit only circuit η (and \(n_{\mathrm{th}}\)).
- `gdr_two_stage` — Gaussian twins identify η; \(t_{\mathrm{free}}>0\) twins identify leak/hops.
- `gdr_indep` — per-mode η, \(n_{\mathrm{th}}\) (structured middle ground).
- `gdr_energy` — extra energy-matching term on twins.
- `readout_then_zne` / `zne_then_readout` — hybrid ZNE + calibrated detector.
- Twin `alpha_policy={uniform,wide,stratified}` and `--n-rank2` / `--t-free-rank`.

## Phase 1+ log

(filled as jobs finish)
