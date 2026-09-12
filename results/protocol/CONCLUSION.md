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

| ansatz | depth | κτ | mit k/N | raw k/N | mean ⟨H⟩ | wall (s) | file |
|--------|------:|---:|--------:|--------:|---------:|---------:|------|
| ecd | L4 | 0.003 | 186/200 | 186/200 | 0.880 | 2163.6 | `gibbs_four_sat_ecd_l4_noisy_kt0.003.json` |
| ecd | L4 | 0.03 | 183/200 | 131/200 | 1.024 | 3481.0 | `gibbs_four_sat_ecd_l4_noisy_kt0.03.json` |
| ecd | L4 | 0.1 | 110/200 | 5/200 | 1.093 | 4211.3 | `gibbs_four_sat_ecd_l4_noisy_kt0.1.json` |
| snap | L3 | 0.003 | 188/200 | 188/200 | 0.681 | 1808.1 | `gibbs_four_sat_snap_l3_noisy_kt0.003.json` |
| snap | L3 | 0.03 | 184/200 | 183/200 | 0.977 | 2643.4 | `gibbs_four_sat_snap_l3_noisy_kt0.03.json` |
| snap | L3 | 0.1 | 157/200 | 21/200 | 1.117 | 3078.3 | `gibbs_four_sat_snap_l3_noisy_kt0.1.json` |

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
Twin fits for M use 8192 shots. Every cell above is 20 H × 10 trials × 201-step traces.

## Coverage

All eight canonical cells are complete (2 noiseless + 6 noisy κτ). No rush-cut.
In-loop GDR is a no-op at κτ=0.003 (mit=raw), lifts ECD at 0.03 (131→183),
and is essential at 0.1 (ECD 5→110, SNAP 21→157).
See `docs/DEFAULT_PROTOCOL.md`.

