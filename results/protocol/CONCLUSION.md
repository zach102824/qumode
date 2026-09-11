# Default protocol conclusion (n=7 4-SAT)

Canonical noiseless + noisy-in-loop recipe for **ECD L4** and **SNAP L3**.
Success = `most_likely_bitstring == ground_bitstring`. Pooled over 20 Hamiltonians × 10 trials = 200.

Noise: `comprehensive` with cavity number-dephasing `κ_φ τ_app = 0.5 κτ`, plus `readout_realistic`.
Mitigation: **gdr_param only**, M policy B (fit once per (H, κτ, ansatz), reuse all 200 SPSA steps).

## Noiseless

| ansatz | depth | k/200 | rate | mean ⟨H⟩ | mean p(GS) | wall (s) | file |
|--------|------:|------:|-----:|---------:|-----------:|---------:|------|
| ecd | L4 | 186/200 | 0.930 | 0.837 | 0.247 | 400.0 | `gibbs_four_sat_ecd_l4_noiseless.json` |
| snap | L3 | 193/200 | 0.965 | 0.574 | 0.508 | 331.8 | `gibbs_four_sat_snap_l3_noiseless.json` |

## Noisy-in-loop (gdr_param, policy B)

_No noisy-in-loop JSONs written yet._

## Commands

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u scripts/run_default_protocol.py noiseless --ansatz ecd --workers 4
python -u scripts/run_default_protocol.py noiseless --ansatz snap --workers 4
python -u scripts/run_default_protocol.py noisy --ansatz ecd --kappa-tau 0.003 --workers 4
python -u scripts/run_default_protocol.py noisy --ansatz snap --kappa-tau 0.003 --workers 4
python -u scripts/run_default_protocol.py noisy --ansatz ecd --kappa-tau 0.03 --workers 4
python -u scripts/run_default_protocol.py noisy --ansatz snap --kappa-tau 0.03 --workers 4
python -u scripts/run_default_protocol.py noisy --ansatz ecd --kappa-tau 0.1 --workers 4
python -u scripts/run_default_protocol.py noisy --ansatz snap --kappa-tau 0.1 --workers 4
python -u scripts/run_default_protocol.py plots
python -u scripts/run_default_protocol.py conclude
```

In-loop cost uses the unfolded histogram after readout confusion (no extra shot noise).
Twin fits for M use 8192 shots. See `docs/DEFAULT_PROTOCOL.md`.

