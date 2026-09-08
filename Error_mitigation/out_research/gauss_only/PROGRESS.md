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

**Finished.** 72 paired cells. Keep the ~25% `t_free=2` mix.
2026-09-08T04:18:53Z  driver stages=['smoke'] arms=['default', 'nr0']
2026-09-08T04:18:53Z  START arm=default stage=smoke tag=gauss_only/default_smoke n_rank2=10
2026-09-08T04:18:54Z  DONE  arm=default stage=smoke wall=1.0s -> /workspace/Error_mitigation/out_research/gauss_only/default_smoke/results.json
2026-09-08T04:18:54Z  PAIRED n=0 meanΔ=None W/T/L=0/0/0
2026-09-08T04:18:54Z  START arm=nr0 stage=smoke tag=gauss_only/nr0_smoke n_rank2=0
2026-09-08T04:19:03Z  DONE  arm=nr0 stage=smoke wall=9.6s -> /workspace/Error_mitigation/out_research/gauss_only/nr0_smoke/results.json
2026-09-08T04:19:03Z  PAIRED n=1 meanΔ=0.0003660540718782093 W/T/L=0/1/0
2026-09-08T04:19:26Z  driver stages=['ecd_priority', 'snap_priority', 'thermal'] arms=['default', 'nr0']
2026-09-08T04:19:26Z  START arm=default stage=ecd_priority tag=gauss_only/default_ecd_priority n_rank2=10
2026-09-08T04:20:32Z  DONE  arm=default stage=ecd_priority wall=65.6s -> /workspace/Error_mitigation/out_research/gauss_only/default_ecd_priority/results.json
2026-09-08T04:20:32Z  PAIRED n=1 meanΔ=0.0003660540718782093 W/T/L=0/1/0
2026-09-08T04:20:32Z  START arm=nr0 stage=ecd_priority tag=gauss_only/nr0_ecd_priority n_rank2=0
2026-09-08T04:24:37Z  DONE  arm=nr0 stage=ecd_priority wall=245.4s -> /workspace/Error_mitigation/out_research/gauss_only/nr0_ecd_priority/results.json
2026-09-08T04:24:37Z  PAIRED n=24 meanΔ=-0.00972576129803823 W/T/L=7/7/10
2026-09-08T04:24:37Z  START arm=default stage=snap_priority tag=gauss_only/default_snap_priority n_rank2=10
2026-09-08T04:25:24Z  DONE  arm=default stage=snap_priority wall=46.8s -> /workspace/Error_mitigation/out_research/gauss_only/default_snap_priority/results.json
2026-09-08T04:25:24Z  PAIRED n=24 meanΔ=-0.00972576129803823 W/T/L=7/7/10
2026-09-08T04:25:24Z  START arm=nr0 stage=snap_priority tag=gauss_only/nr0_snap_priority n_rank2=0
2026-09-08T04:27:33Z  DONE  arm=nr0 stage=snap_priority wall=128.8s -> /workspace/Error_mitigation/out_research/gauss_only/nr0_snap_priority/results.json
2026-09-08T04:27:33Z  PAIRED n=48 meanΔ=-0.0268940196142554 W/T/L=7/11/30
2026-09-08T04:27:33Z  START arm=default stage=thermal tag=gauss_only/default_thermal n_rank2=10
2026-09-08T04:28:34Z  DONE  arm=default stage=thermal wall=60.9s -> /workspace/Error_mitigation/out_research/gauss_only/default_thermal/results.json
2026-09-08T04:28:34Z  PAIRED n=48 meanΔ=-0.0268940196142554 W/T/L=7/11/30
2026-09-08T04:28:34Z  START arm=nr0 stage=thermal tag=gauss_only/nr0_thermal n_rank2=0
2026-09-08T04:32:07Z  DONE  arm=nr0 stage=thermal wall=212.9s -> /workspace/Error_mitigation/out_research/gauss_only/nr0_thermal/results.json
2026-09-08T04:32:07Z  PAIRED n=72 meanΔ=-0.026018566907184067 W/T/L=11/15/46
