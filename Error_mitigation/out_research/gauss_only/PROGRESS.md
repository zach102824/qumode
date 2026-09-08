# gauss_only progress

Adaptive twin design (`--twin-design adaptive`: span on random, U(0.5,1) on optimized).
Arm A = `n_rank2=10` (`n_train//4` auto for `n_train=40`; reuses `nr10` caches).
Arm B = `n_rank2=0` (new `nr0` cache keys).

Driver: `Error_mitigation/run_gauss_only_abate.py`.

Stages (priority order):

1. `smoke` — ECD random loss κτ=0.003 ideal, both arms
2. `ecd_priority` — ECD random+optimized, loss+comprehensive, 3 κτ, ideal+realistic
3. `snap_priority` — SNAP same
4. `thermal` — both ansatz, `loss_thermal_dephasing` last

Not started yet.
2026-09-08T04:18:53Z  driver stages=['smoke'] arms=['default', 'nr0']
2026-09-08T04:18:53Z  START arm=default stage=smoke tag=gauss_only/default_smoke n_rank2=10
2026-09-08T04:18:54Z  DONE  arm=default stage=smoke wall=1.0s -> /workspace/Error_mitigation/out_research/gauss_only/default_smoke/results.json
2026-09-08T04:18:54Z  PAIRED n=0 meanΔ=None W/T/L=0/0/0
2026-09-08T04:18:54Z  START arm=nr0 stage=smoke tag=gauss_only/nr0_smoke n_rank2=0
2026-09-08T04:19:03Z  DONE  arm=nr0 stage=smoke wall=9.6s -> /workspace/Error_mitigation/out_research/gauss_only/nr0_smoke/results.json
2026-09-08T04:19:03Z  PAIRED n=1 meanΔ=0.0003660540718782093 W/T/L=0/1/0
