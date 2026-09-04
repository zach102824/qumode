# Gaussian Data Regression on a hybrid qumode

## Status for cloud / grok bot (2026-09-04, research v2)

PR #6 baseline (`out/`, 108 cells @ 8192 / 40 twins) is **done** — do not overwrite `out/` or `out_smoke/`. Research runs live in `out_research/`. Do not edit `src/`.

| item | status |
|------|--------|
| Modules under `Error_mitigation/` | done (`noise_models.py`, `twins.py`, `mitigation.py`, `metrics.py`, `run_mitigation_experiment.py`) |
| This README (noises / methods / caveats) | done |
| `src/` | **do not edit** |
| `tests/test_error_mitigation.py` | **passed** (10 tests): kernels column-stochastic, ECD Poisson TVD `<1e-6`, SNAP product-state TVD `<1e-6`, `readout_only` skipped on ideal and beats raw on strong readout, RL inverts binomial blur |
| Smoke `--preset smoke` | **incomplete**. Stopped after ECD random / `loss` κτ=0.003. Partial cache: `Error_mitigation/out_smoke/optimized_params_ecd_h000_nd5.json` (smoke-quality: `maxiter=5`, `success=false`, E≈−3.47 vs E0≈−7.11). **Do not reuse this cache for `full`.** |
| Full `--preset full` | not started |
| Histogram / TVD figures | not written (driver never finished) |
| Existing `pytest` suite | unit file green; full repo pytest not re-run after the interrupt |

**What the bot should do**

1. From repo root, run smoke to completion (one process only), then full:
   ```bash
   python -u Error_mitigation/run_mitigation_experiment.py --preset smoke --ansatz both --outdir Error_mitigation/out_smoke
   python -u Error_mitigation/run_mitigation_experiment.py --preset full --ansatz both --outdir Error_mitigation/out
   ```
2. If `out_smoke/optimized_params_ecd_h000_nd5.json` is still the 5-iter cache, delete it before `full` (or write `full` to `Error_mitigation/out`, which uses a different path and will re-optimize).
3. Confirm in the smoke log: every `t_free=0` twin prints product-state TVD mean/max `≪ 1e-6` at the end (assertion already raises if it fails).
4. Sanity checks on `out_smoke/summary.txt` / `results.json`:
   - `ideal` readout: no `readout_only` method.
   - `readout_realistic`: `readout_only` TVD `<` `raw` TVD.
   - `loss` family: `oracle_binomial` residual near shot noise; if not, interleaving (not readout) is the issue.
5. Run `python -m pytest tests/test_error_mitigation.py -q` and `python -m pytest -q` (existing suite must stay green).
6. Do not edit the plan file. Do not modify `src/`.

**Runtime note.** Each ECD noisy block (target + 12 twins + ZNE scales 2,3) took ~3–4 min locally at `ndepth=5`. Smoke is ~2 ansatz × 2 param sets × 3 families ≈ 12 such blocks plus SNAP (faster). Full is 40 twins × 3 κτ × 3 readout levels and will take hours. Fitting (`gdr_param` MLE) is cheap next to `HybridSimulator.density_matrix`.

**Partial smoke numbers (ECD, random params, `loss`, κτ=0.003, 4000 shots)** — GDR did *not* beat raw; mild loss + shots, possible overfit/over-correct. Revisit after a finished run.

| readout | raw TVD | gdr_param | oracle |
|---------|--------:|----------:|-------:|
| ideal | 0.0616 | 0.0646 | 0.0700 |
| readout_realistic | 0.0683 | 0.0713 | 0.0725 |

---

Error mitigation for the hybrid qubit–qumode VQE, tested on **one mixed p-spin instance** (`mixed_p_spin_p2-4_000.npz`). The device output is a `|q, n, m>` histogram. Circuit noise is simulated with the existing `NoiseConfig` path; readout is a column-stochastic confusion matrix on that histogram, then finite shots.

The idea is Clifford Data Regression with **Gaussian circuits** in place of Cliffords: twins whose ideal photon histogram is a product of truncated Poissons, used to fit a few-parameter transfer map, which is inverted on the target histogram.

No `src/` code is modified. Optimized mixed-p-spin ECD/SNAP parameters are not checked in, so the driver noiselessly optimizes them and caches `out/optimized_params_*.json`.

## How to run

From the repo root:

```bash
python Error_mitigation/run_mitigation_experiment.py --preset smoke
python Error_mitigation/run_mitigation_experiment.py --preset full --ansatz both
```

Useful flags: `--ansatz ecd|snap|both`, `--instance 0`, `--outdir Error_mitigation/out`, `--shots`, `--n-train`, `--seed`, `--readout ideal|readout_realistic|readout_strong|all`, `--families`, `--kappa-tau`, `--params`, `--twin-design span|default|adaptive`.

Cheap research loops (writes only under `out_research/`)::

```bash
python -u Error_mitigation/run_ablation.py --tag micro --ansatz ecd --families loss --kappa-tau 0.003,0.1 --shots 4096 --n-train 20
```

| preset | shots | twins | κτ | readout | optimizer |
|--------|------:|------:|----|---------|-----------|
| `smoke` | 4000 | 12 | 0.003 | ideal + realistic | 1× L-BFGS-B, 5 iter |
| `full` | 20000 | 40 | 0.003, 0.03, 0.1 | all three | 3× L-BFGS-B, 200 iter |

Each ansatz is run at **random** parameters and at the **noiselessly optimized** parameters. ECD uses \(N_d=5\) (40 params); SNAP uses \(N_d=2\) (36 params). Vacuum start.

## Circuit noise

Applied between layers by `HybridSimulator.density_matrix` (`src/qumode_vqe/noise.py`). Sweep: `kappa_tau ∈ {0.003, 0.03, 0.1}` per application.

| family | what it is | why it is here |
|--------|------------|----------------|
| `loss` | Paper amplitude-damping Kraus, one application per UER/SNAP layer. Pure photon loss, \(n_{\mathrm{th}}=0\). | Phase-covariant. Binomial unfolding should be exact **up to interleaving**. |
| `loss_thermal_dephasing` | Lindblad cavity loss with \(n_{\mathrm{th}}=0.05\) and number dephasing \(\kappa_\phi \tau = 0.5\,\kappa\tau\). | Still phase-covariant. Factorial moments are **not** a pure \(\eta^k\) rescaling once there is heating. |
| `comprehensive` | Lindblad loss (\(n_{\mathrm{th}}=0.01\)), transmon T1/T2, cavity self-Kerr, 1% ECD-amplitude and rotation errors. Per ECD/SNAP pair. | Ancilla errors break phase covariance. Tests whether a histogram-level map still transfers from Gaussian twins. |

Idle-time ZNE (`scale_noise`) multiplies \(\kappa\tau\), \(\kappa_\phi\), and \(1/T_1^{\mathrm{q}}\), \(1/T_2^{\mathrm{q}}\). It does **not** scale readout: stretching idles does not change the detector. That is the failure mode `zne_idle` is expected to show under `readout_*`.

## Readout noise

Applied **only** to the final histogram (`apply_confusion` in `src/qumode_vqe/measurement.py`). Never on the density matrix. Static, uncorrelated, one confusion matrix per register.

| level | qubit \(p_{01}, p_{10}\) | Fock \(p_{n\to n\pm 1}\) |
|-------|--------------------------|--------------------------|
| `ideal` | 0, 0 | 0 |
| `readout_realistic` | 0.01, 0.03 (asymmetric: \(\lvert 1\rangle\to\lvert 0\rangle\) relaxation dominates) | 0.03 |
| `readout_strong` | 0.03, 0.08 | 0.10 (brackets the 13.5% raw Fock TVD of Curtis et al., PRA 103, 023705) |

\(p_{nn}=0.03\) is chosen so readout-only raw TVD is the same order as loss-only raw TVD at \(\kappa\tau=0.003\). All levels use finite shots.

## Gaussian twins

Same gate count and depth as the target.

- **ECD:** snap each qubit angle \(\theta\) onto \(\{0,\pi\}\). Then every ECD is an unconditional displacement plus a bit flip, and the ideal state is \(\lvert q\rangle\lvert\alpha_1\rangle\lvert\alpha_2\rangle\). The analytic histogram is the truncated 1-mode product evolution (ECD reduced to unconditional \(D(\pm\beta/2)\); SNAP is already local). Asserted against `statevector` at TVD \(<10^{-6}\) for every \(t_{\mathrm{free}}=0\) twin. A Poisson formula in the tracked \(|\alpha|^2\) is stored as a diagnostic: truncated \(D(\alpha)\) is not an ideal coherent state at large \(|\alpha|\), so Poisson TVD is allowed to be larger for SNAP.
- **SNAP:** replace each SNAP phase list \(\theta_n\) by its least-squares affine fit \(b n\) (\(\theta_0=0\)). Affine SNAP is a rotation.
- Default (`--twin-design span`): \(\lvert\beta\rvert\) / \(\lvert\alpha\rvert\) log-spaced in \([0.25, 1.35]\) (capped so \(\lvert\alpha\rvert\) stays inside the Fock grid). Wider \(\lvert\alpha\rvert^2\) span improves Fisher information for \((\eta, p_{nn})\). PR #6 used `U(0.5, 1)` (`--twin-design default`). `--twin-design adaptive` uses span on random circuits and the PR #6 mix on optimized ones (span regresses on some optimized comprehensive / high-κτ cells).
- Default mix: 75% fully Gaussian (\(t_{\mathrm{free}}=0\)), 25% with \(t_{\mathrm{free}}=2\) non-Gaussian gates left in (Gaussian rank \(2^t\)).

Twins are measured with the **same** readout level and shot count as the target. The Poisson check uses physical probabilities, never the readout-corrupted histogram.

## Methods

| name | histogram? | what it does |
|------|:----------:|--------------|
| `raw` | yes | Shot histogram. Baseline. |
| `readout_only` | yes | Invert only the calibrated readout confusion (Maciejewski et al., Quantum 4, 257). Skipped when readout is ideal. |
| `oracle_binomial` | yes | Known-model end-of-circuit thermal-loss kernel with the true cumulative \(\eta=e^{-\sum\kappa\tau}\), composed with the **true** readout confusion. No learning. Residual TVD\((M p_{\mathrm{ideal}}, q_{\mathrm{noisy}})\) is the interleaved-vs-end-of-circuit error plus shot noise. |
| `gdr_param` | yes | Fit \(M(\eta_1,\eta_2,n_{\mathrm{th}},p_\downarrow,p_\uparrow,\varepsilon,p_{01},p_{10},p_{nn})\) by multinomial MLE on the twins, then Richardson–Lucy unfold. Loss and readout are fitted **jointly**. |
| `gdr_damped` | yes | Same fit as `gdr_param`, then mix the unfold with the readout-inverted (or raw) histogram. Mix weight α is chosen on twins. Stops over-correction on mild/random high-κτ cells. |
| `gdr_mid` | yes | Fit only \((\eta_1,\eta_2,p_{01},p_{10},p_{nn})\); freeze heating/hops/leak. Structured middle ground vs `gdr_full`. |
| `gdr_residual` | yes | Oracle end-of-circuit kernel composed with a small extra hop/leak fitted on twins (especially \(t_{\mathrm{free}}>0\)). Aimed at ECD interleaving residual. |
| `gdr_afterburn` | yes | Richer residual: oracle plus extra thermal-loss / hops / leak, fitted on \(t_{\mathrm{free}}>0\) twins. |
| `gdr_blend` | yes | Convex mix of the `gdr_param` and `oracle_binomial` unfolds; mix weight chosen on twins. |
| `gdr_select` | yes | Circuit-class recipe plus holdout. On **optimized** parameters pick `gdr_residual` (or `gdr_afterburn` if it wins the rank-2 twins). On **random** circuits, Gaussian holdout among `{safe, gdr_param, gdr_mid, gdr_damped}` — residual overfits random twins. |
| `gdr_full` | yes | Unstructured column-stochastic \(C_q\otimes C_1\otimes C_2\), alternating NNLS, initialized from `gdr_param`. Shows the cost of over-parametrization. |
| `scalar_cdr` | no | Classic CDR on the energy only: \(E_{\mathrm{ideal}}\approx a_1 E_{\mathrm{noisy}}+a_0\). |
| `zne_idle` | yes | Target at noise scales \((1,2,3)\); Richardson extrapolate each bin; clip and renormalize. Readout not scaled. |
| `readout_then_zne` | yes | Invert calibrated readout on each idle-stretched histogram, then ZNE. Fixes the idle-ZNE bias under `readout_*`. |

Unfolding is Richardson–Lucy on the simplex; NNLS is stored as a `gdr_param` cross-check.

## Outputs (`Error_mitigation/out/`)

- `results.json` — metrics, fitted parameters vs true \((\eta, p_{nn}, p_{01}, p_{10})\), factorial-moment diagnostic, histograms.
- `summary.txt` — full table, plus a headline at \(\kappa\tau=0.003\): `readout_only` vs `gdr_param` vs `oracle_binomial` for each readout level (optimized parameters).
- `hist_<ansatz>_<params>_<family>_<readout>.png` — grouped bars on the top-12 ideal bins and per-mode photon-number marginals.
- `summary_tvd_<ansatz>.png` — TVD and \(\lvert\Delta E\rvert\) vs \(\kappa\tau\); one color per method, line style per readout level, panels per noise family and parameter set.
- `optimized_params_*.json` — cached noiseless optima.

## What to look for

- **Ideal readout:** `readout_only` is skipped. Other methods should match a no-readout run (same seed).
- **`readout_*`:** `readout_only` must beat `raw` in TVD. It is the floor that `gdr_param` has to beat if circuit noise is also present.
- **`loss` family, `oracle_binomial` residual:** should sit near the shot-noise floor. If it does not, interleaved loss (not the readout model) is the culprit.
- **Heating:** \(g_k^{\mathrm{noisy}}/g_k^{\mathrm{ideal}}\) vs \(\eta^k\) should fail for `loss_thermal_dephasing`.
- **Identifiability:** loss \(n\to n-1\) and readout \(n\to n\pm 1\) look the same on a single circuit. Twins that span a range of \(\lvert\alpha\rvert^2\) break the degeneracy because loss scales with \(n\) and \(p_{nn}\) does not. Check fitted \(\eta\) vs true cumulative \(\eta\), and fitted \(p_{nn}\) vs the `MeasurementConfig`.
- **`zne_idle` under readout:** circuit noise is scaled, the detector is not, so extrapolation is biased. Use `readout_then_zne`.
- **Research scoreboard:** `Error_mitigation/out_research/NOTEBOOK.md` and `out_research/phase3/`. Do not overwrite `out/`.

## Caveats (known, not bugs)

1. A single histogram-level matrix \(q = M p\) is exact when a phase-covariant channel acts **after** the ideal circuit. Loss **between** non-Gaussian gates depends on intermediate coherences; commuting it to the end is an approximation (exact for Gaussian twins).
2. Thermal photons are not an independent classical addition on the probability generating function. Dividing factorial moments by \(\eta^k\) is exact only for pure loss.
3. Readout is a static uncorrelated per-register confusion. Real bitwise photon-number readout has correlated multi-bit errors and loss *during* the readout sequence (the HMM treatment of Curtis et al. is out of scope).
4. `comprehensive` coherent ECD errors change the circuit, not a stochastic histogram kernel. GDR can only absorb them as an effective leak.
5. Sampling overhead of binomial unfolding grows as \((2/\eta-1)^{n_{\max}}\). Keep mean photon number inside the Fock grid; the twins are amplitude-capped for that reason.
