# Round-2 microbench

stage=round2_stage_c shots=8192 seed=2026 fit_maxiter=120 wall=139.5s  any_keep=False

| id | raw | gdr_param | gdr_select | gdr_family_eta | gdr_shot_damp | gdr_rl | gdr_rl_soft | gdr_rl_stop |
|---|---|---|---|---|---|---|---|---|
| ecd_rand_loss_0.1 | 0.2983 | 0.2087 | 0.2011 | 0.2087 | 0.2011 | 0.2269 | 0.2087 | 0.2269 |
| ecd_rand_comp_0.1 | 0.4031 | 0.3765 | 0.3424 | 0.3765 | 0.3424 | 0.3376 | 0.3764 | 0.3376 |
| ecd_opt_comp_0.1 | 0.9086 | 0.3419 | 0.3419 | 0.3419 | 0.3419 | 0.4639 | 0.3422 | 0.4636 |
| snap_rand_comp_0.003 | 0.0372 | 0.0415 | 0.0415 | 0.0415 | 0.0415 | 0.0416 | 0.0416 | 0.0415 |
| h000_opt_comp_rr_0.003 | 0.3247 | 0.1012 | 0.1012 | 0.1012 | 0.1012 | 0.1013 | 0.1013 | 0.1047 |
| h004_opt_comp_rr_0.003 | 0.2007 | 0.0763 | 0.0763 | 0.0730 | 0.0763 | 0.0804 | 0.0765 | 0.0802 |
| h009_opt_comp_rr_0.003 | 0.2764 | 0.0932 | 0.0932 | 0.0963 | 0.0932 | 0.1148 | 0.0931 | 0.1148 |
| ecd_rand_comp_0.003 | 0.0816 | 0.0723 | 0.0723 | 0.0723 | 0.0723 | 0.0681 | 0.0722 | 0.0681 |
| h000_opt_comp_rr_0.1 | 0.9527 | 0.8688 | 0.8688 | 0.8719 | 0.8688 | 0.8122 | 0.8687 | 0.8122 |
| h004_opt_comp_rr_0.1 | 0.8527 | 0.7612 | 0.7612 | 0.7612 | 0.7612 | 0.8392 | 0.7612 | 0.8392 |
| h009_opt_comp_rr_0.1 | 0.9132 | 0.9670 | 0.9670 | 0.9670 | 0.9670 | 0.8455 | 0.9661 | 0.8455 |
| snap_opt_comp_rr_0.003 | 0.1411 | 0.0230 | 0.0230 | 0.0252 | 0.0230 | 0.0361 | 0.0233 | 0.0356 |
| snap_opt_comp_rr_0.1 | 0.7952 | 0.5903 | 0.5903 | 0.5903 | 0.5903 | 0.6053 | 0.5904 | 0.6052 |

## Keep / drop vs adaptive `gdr_select`

| method | keep? | beats | regressions | mean ΔTVD |
|---|---|---|---|---:|
| `gdr_family_eta` | drop | — | h009_opt_comp_rr_0.003 | +0.0036 |
| `gdr_shot_damp` | drop | — | — | +0.0000 |
| `gdr_rl` | drop | h000_opt_comp_rr_0.1, h009_opt_comp_rr_0.1 | ecd_opt_comp_0.1, h004_opt_comp_rr_0.003, h009_opt_comp_rr_0.003 | +0.0071 |
| `gdr_rl_soft` | drop | — | — | +0.0032 |
| `gdr_rl_stop` | drop | h000_opt_comp_rr_0.1, h009_opt_comp_rr_0.1 | ecd_opt_comp_0.1, h000_opt_comp_rr_0.003, h004_opt_comp_rr_0.003, h009_opt_comp_rr_0.003 | +0.0073 |
