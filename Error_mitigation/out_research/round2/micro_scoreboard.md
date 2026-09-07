# Round-2 microbench

shots=8192 seed=2026 fit_maxiter=120 wall=298.5s  any_keep=False

| id | raw | gdr_param | gdr_select | gdr_anneal | gdr_eta | gdr_fisher | gdr_select_kt | gdr_mild_residual | readout_then_gdr | gdr_then_rtz |
|---|---|---|---|---|---|---|---|---|---|---|
| ecd_rand_loss_0.1 | 0.2983 | 0.2087 | 0.2011 | 0.2083 | 0.2179 | 0.2083 | 0.2011 | 0.2087 | 0.2179 | 0.1944 |
| ecd_rand_comp_0.1 | 0.4031 | 0.3765 | 0.3424 | 0.3763 | 0.4177 | 0.3436 | 0.3424 | 0.3765 | 0.4177 | 0.3449 |
| ecd_opt_comp_0.1 | 0.9086 | 0.3419 | 0.3419 | 0.3440 | 0.3864 | 0.3412 | 0.3419 | 0.3419 | 0.3864 | 0.4085 |
| snap_rand_comp_0.003 | 0.0372 | 0.0415 | 0.0417 | 0.0415 | 0.0417 | 0.0416 | 0.0417 | 0.0415 | 0.0417 | 0.0407 |
| h000_opt_comp_rr_0.003 | 0.3247 | 0.1012 | 0.1012 | 0.1012 | 0.1882 | 0.1013 | 0.1012 | 0.1012 | 0.1930 | 0.1012 |
| h004_opt_comp_rr_0.003 | 0.2007 | 0.0763 | 0.0763 | 0.0735 | 0.1254 | 0.0715 | 0.0763 | 0.0763 | 0.1196 | 0.0726 |
| h009_opt_comp_rr_0.003 | 0.2764 | 0.0932 | 0.0932 | 0.0956 | 0.1556 | 0.0917 | 0.0932 | 0.0932 | 0.1534 | 0.0953 |
| ecd_rand_comp_0.003 | 0.0816 | 0.0723 | 0.0723 | 0.0723 | 0.0692 | 0.0686 | 0.0686 | 0.0723 | 0.0692 | 0.0723 |

## Keep / drop vs adaptive `gdr_select`

| method | keep? | beats | regressions | mean ΔTVD |
|---|---|---|---|---:|
| `gdr_anneal` | drop | — | — | +0.0053 |
| `gdr_eta` | drop | — | ecd_opt_comp_0.1, h000_opt_comp_rr_0.003, h004_opt_comp_rr_0.003, h009_opt_comp_rr_0.003 | +0.0415 |
| `gdr_fisher` | drop | — | — | -0.0003 |
| `gdr_select_kt` | drop | — | — | -0.0005 |
| `gdr_mild_residual` | drop | — | — | +0.0052 |
| `readout_then_gdr` | drop | — | ecd_opt_comp_0.1, h000_opt_comp_rr_0.003, h004_opt_comp_rr_0.003, h009_opt_comp_rr_0.003 | +0.0411 |
| `gdr_then_rtz` | drop | ecd_rand_loss_0.1 | ecd_opt_comp_0.1 | +0.0075 |
