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
