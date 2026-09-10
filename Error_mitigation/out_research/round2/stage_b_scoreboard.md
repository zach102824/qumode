# Round-2 microbench

stage=round2_stage_b shots=8192 seed=2026 fit_maxiter=120 wall=420.4s  any_keep=False

| id | raw | gdr_param | gdr_select | gdr_ensemble | gdr_joint |
|---|---|---|---|---|---|
| ecd_rand_loss_0.1 | 0.2983 | 0.2087 | 0.2011 | 0.2188 | 0.2087 |
| ecd_rand_comp_0.1 | 0.4031 | 0.3765 | 0.3424 | 0.3831 | 0.3765 |
| ecd_opt_comp_0.1 | 0.9086 | 0.3419 | 0.3419 | 0.3461 | 0.3419 |
| snap_rand_comp_0.003 | 0.0372 | 0.0415 | 0.0415 | 0.0425 | 0.0415 |
| h000_opt_comp_rr_0.003 | 0.3247 | 0.1012 | 0.1012 | 0.1012 | 0.1012 |
| h004_opt_comp_rr_0.003 | 0.2007 | 0.0763 | 0.0763 | 0.0784 | 0.0763 |
| h009_opt_comp_rr_0.003 | 0.2764 | 0.0932 | 0.0932 | 0.1050 | 0.0931 |
| ecd_rand_comp_0.003 | 0.0816 | 0.0723 | 0.0723 | 0.0743 | 0.0722 |
| snap_opt_comp_rr_0.003 | 0.1411 | 0.0230 | 0.0230 | 0.0230 | 0.0230 |
| snap_opt_comp_rr_0.03 | 0.5710 | 0.2030 | 0.2030 | 0.2239 | 0.2030 |
| snap_opt_comp_rr_0.1 | 0.7952 | 0.5903 | 0.5903 | 0.5633 | 0.5904 |

## Keep / drop vs adaptive `gdr_select`

| method | keep? | beats | regressions | mean ΔTVD |
|---|---|---|---|---:|
| `gdr_ensemble` | drop | snap_opt_comp_rr_0.1 | ecd_opt_comp_0.1, h009_opt_comp_rr_0.003 | +0.0067 |
| `gdr_joint` | drop | — | — | +0.0038 |
