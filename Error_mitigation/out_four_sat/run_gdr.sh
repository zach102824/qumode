#!/usr/bin/env bash
# 4-SAT noisy + adaptive GDR on near-E0 Gibbs circuits.
# Does not write Error_mitigation/out/ or change official GDR defaults.
set -euo pipefail
cd /workspace
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl
mkdir -p "$MPLCONFIGDIR" Error_mitigation/out_four_sat

LOG=Error_mitigation/out_four_sat/COMMANDS.log
GDRLOG=Error_mitigation/out_four_sat/gdr_run.log
DEADLINE_UNIX=${DEADLINE_UNIX:-$(($(date +%s) + 10*3600))}

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG" | tee -a "$GDRLOG"; }
remaining() { echo $((DEADLINE_UNIX - $(date +%s))); }
timed_out() { [[ $(remaining) -lt 120 ]]; }

gdr() {
  local ansatz="$1" hid="$2" ndepth="$3" tag="$4" shots="$5" ntrain="$6" preset="$7"
  local outdir="Error_mitigation/out_four_sat/${ansatz}_h$(printf '%03d' "$hid")_${tag}"
  local gibbs="results/gibbs_four_sat_${ansatz}.json"
  if [[ -d "$outdir" && -f "$outdir/results.json" ]]; then
    log "SKIP existing $outdir"
    return 0
  fi
  if timed_out; then
    log "STOP deadline before $ansatz H${hid} $tag"
    return 2
  fi
  log "START $ansatz H${hid} nd=${ndepth} $tag shots=${shots} n_train=${ntrain} remain=$(remaining)s"
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
    --twin-design adaptive \
    --shots "$shots" \
    --n-train "$ntrain" \
    --params both \
    --outdir "$outdir"
  log "DONE $ansatz H${hid} $tag remain=$(remaining)s"
}

# 1) SNAP scout H000 smoke then 8192
gdr snap 0 3 smoke 4000 12 smoke
gdr snap 0 3 s8192 8192 40 full

# 2) ECD passers starting from best (H016), then H010, H018
gdr ecd 16 4 smoke 4000 12 smoke
gdr ecd 16 4 s8192 8192 40 full
gdr ecd 10 4 smoke 4000 12 smoke
gdr ecd 10 4 s8192 8192 40 full
gdr ecd 18 4 smoke 4000 12 smoke
gdr ecd 18 4 s8192 8192 40 full

# 3) SNAP 8192 on remaining filtered H (all 20 pass; skip 0 already done)
for hid in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
  gdr snap "$hid" 3 s8192 8192 40 full || break
done

log "GDR loop finished remain=$(remaining)s"
exit 0
