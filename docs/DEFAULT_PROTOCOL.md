# Default noiseless + noisy protocol (n=7)

This is the **canonical** 7-qubit 4-SAT recipe. It replaces the old
“noiseless optimize → one noisy eval + GDR” pipeline.

Locked depths: **ECD L4** (32 ansatz + 5 prep parameters) and **SNAP L3**
(54 ansatz + 5 prep). Instances: `four_sat_000` … `four_sat_019`
(`Hamiltonians/four_sat/`). Ten trials each, `seed_base=4000`, vacuum
start, `sampled_tail` Gibbs η, **200 joint SPSA** steps (prep never frozen).

Success is **mode finding**: `most_likely_bitstring == ground_bitstring`.
Report **k/200**.

Official mitigation is **`gdr_param` only** (PR #16). `gdr_select` is not
part of this protocol.

## Noise model

Circuit family: **`comprehensive`** (`Error_mitigation.noise_models.circuit_noise`).

After each ECD/SNAP pair:

* Lindblad cavity loss with \(n_{\mathrm{th}}=0.01\)
* cavity number-dephasing \(\kappa_\phi\,\tau_{\mathrm{app}} = 0.5\,\kappa\tau\)
  (same factor as `loss_thermal_dephasing`; previously `comprehensive` often
  had \(\kappa_\phi=0\))
* transmon T1/T2, cavity self-Kerr, 1% ECD-amplitude and rotation errors

Noisy runs also turn **`readout_realistic`** on (qubit \(p_{01}=0.01\),
\(p_{10}=0.03\); Fock \(p_{n\to n\pm1}=0.03\)).

κτ sweep: \(\{0.003,\,0.03,\,0.1\}\).

`Error_mitigation/out/` is the frozen PR #6 GDR *histogram* baseline. Do
not overwrite it. The comprehensive \(\kappa_\phi\) default applies to new
simulations only.

## Noiseless

Every SPSA iterate \(i=0,\ldots,200\) logs Gibbs cost, physical energy
\(\langle H\rangle\), Born \(p(\mathrm{GS})\), the most-likely bitstring, and η.

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u scripts/run_default_protocol.py noiseless --ansatz ecd --workers 4
python -u scripts/run_default_protocol.py noiseless --ansatz snap --workers 4
```

Equivalent Gibbs CLI (same seeds / depths / 4-SAT family):

```bash
python -u scripts/Gibbs_and_adaptive_optim_ECD.py --four-sat --n-trials 10 \
  --ndepths 4 --seed-base 4000 --workers 4 \
  --output results/protocol/gibbs_four_sat_ecd_l4_noiseless.json
python -u scripts/Gibbs_and_adaptive_optim_SNAP.py --four-sat --n-trials 10 \
  --ndepths 3 --seed-base 4000 --workers 4 \
  --output results/protocol/gibbs_four_sat_snap_l3_noiseless.json
```

## Noisy simulation — GDR inside the SPSA loop

**Not** a final-only unfold. Each SPSA probe:

1. noisy forward (`comprehensive` + κτ + \(\kappa_\phi\) + readout confusion)
2. `gdr_param` Richardson–Lucy unfold with a **fixed** M
3. Gibbs cost from the mitigated histogram
4. SPSA update

**M policy B:** fit M once per `(Hamiltonian, κτ, ansatz)` from Gaussian
twins of a random same-depth vacuum circuit *before* the loop. Reuse that
M for all 200 steps. Do not refit every iteration.

In-loop histograms apply readout confusion **without extra multinomial
shot noise** so the SPSA gradient is not drowned. Twin fits for M still
use 8192 shots (`n_train=40`, span design). After 200 steps, report
mitigated vs raw bitstring success and \(p(\mathrm{GS})\).

```bash
python -u scripts/run_default_protocol.py noisy --ansatz ecd --kappa-tau 0.003 0.03 0.1 --workers 4
python -u scripts/run_default_protocol.py noisy --ansatz snap --kappa-tau 0.003 0.03 0.1 --workers 4
python -u scripts/run_default_protocol.py plots
python -u scripts/run_default_protocol.py conclude
```

Noisy-in-loop is expensive. Triage order: (1) code + \(\kappa_\phi\) + logging,
(2) wipe unused research dumps, (3) full noiseless ECD+SNAP, (4) smoke + at
least one κτ full noisy per ansatz, then remaining κτ. `CONCLUSION.md`
states honestly what completed.

## Outputs

Under `results/protocol/`:

* `gibbs_four_sat_{ecd,snap}_l*_noiseless.json` — trials with `step_trace`
* `gibbs_four_sat_{ecd,snap}_l*_noisy_kt*.json` — in-loop GDR trials
* `gdr_M/` — cached policy-B θ (11 `gdr_param` numbers) per (H, κτ, ansatz)
* `figures/` — \(\langle H\rangle\) and \(p(\mathrm{GS})\) vs step
* `CONCLUSION.md` — k/200 tables

## What this is not

* Not `gdr_select`, `gdr_damped`, or ZNE.
* Not the mixed p-spin GDR 108-cell matrix (`Error_mitigation/out/`).
* Not the superseded “matched” 4-SAT noiseless-then-final-GDR dumps
  (`out_four_sat/`, `out_four_sat_matched/`). Those are not in this tree.
