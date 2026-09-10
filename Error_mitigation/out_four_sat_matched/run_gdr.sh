#!/usr/bin/env bash
# Matched 4-SAT noisy + adaptive GDR on SNAP L3 and ECD L4 (all 20 H).
# Does not write Error_mitigation/out/ or change official GDR defaults.
# Does not overwrite Error_mitigation/out_four_sat/.
set -euo pipefail
cd /workspace
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl
mkdir -p "$MPLCONFIGDIR" Error_mitigation/out_four_sat_matched

LOG=Error_mitigation/out_four_sat_matched/COMMANDS.log
GDRLOG=Error_mitigation/out_four_sat_matched/gdr_run.log

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG" | tee -a "$GDRLOG"; }

gdr() {
  local ansatz="$1" hid="$2" ndepth="$3" tag="$4" shots="$5" ntrain="$6" preset="$7"
  local outdir="Error_mitigation/out_four_sat_matched/${ansatz}_h$(printf '%03d' "$hid")_${tag}"
  local gibbs="results/gibbs_four_sat_${ansatz}_matched_n10.json"
  if [[ -d "$outdir" && -f "$outdir/results.json" ]]; then
    log "SKIP existing $outdir"
    return 0
  fi
  log "START $ansatz H${hid} nd=${ndepth} $tag shots=${shots} n_train=${ntrain}"
  python3 -u Error_mitigation/run_mitigation_experiment.py \
    --preset "$preset" \
    --family four_sat \
    --instance "$hid" \
    --ansatz "$ansatz" \
    --ndepth "$ndepth" \
    --families comprehensive \
    --readout readout_realistic \
    --kappa-tau 0.003,0.03,0.1 \
    --gibbs-json "$gibbs" \
    --gibbs-pick success_then_cost \
    --twin-design adaptive \
    --shots "$shots" \
    --n-train "$ntrain" \
    --params both \
    --outdir "$outdir"
  log "DONE $ansatz H${hid} $tag"
}

# Smoke first on H000, then 8192 on all 20 for both ansatzes.
gdr snap 0 3 smoke 4000 12 smoke
gdr ecd 0 4 smoke 4000 12 smoke

for hid in $(seq 0 19); do
  gdr snap "$hid" 3 s8192 8192 40 full
done
for hid in $(seq 0 19); do
  gdr ecd "$hid" 4 s8192 8192 40 full
done

log "GDR loop finished"
exit 0
