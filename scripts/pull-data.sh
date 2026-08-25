#!/bin/bash
# Pull SentinelAI datasets from GCS onto the current VM.
# Datasets live in a bucket now (not on the boot disk), so a fresh small-disk VM
# pulls only what an experiment needs.
#
# Usage: bash scripts/pull-data.sh [xd|ucf|kinetics|all]   (default: xd)
#   xd  = XD-Violence (I3D visual + AST audio + text features) ~14G — fusion exps
#   ucf = UCF-Crime I3D ~62G  |  kinetics = Kinetics-400 ~2G  |  all = everything
set -e
BUCKET=gs://sentinelai-data-just-aloe-499321
DEST="$HOME/documents/SentinelAI/data"
what="${1:-xd}"
mkdir -p "$DEST"

case "$what" in
  xd)       gcloud storage rsync --recursive "$BUCKET/data/xd-violence" "$DEST/xd-violence" ;;
  ucf)      gcloud storage rsync --recursive "$BUCKET/data/ucf-crime"   "$DEST/ucf-crime" ;;
  kinetics) gcloud storage rsync --recursive "$BUCKET/data/kinetics400" "$DEST/kinetics400" ;;
  all)      gcloud storage rsync --recursive "$BUCKET/data" "$DEST" ;;
  *) echo "unknown: $what (use xd|ucf|kinetics|all)"; exit 1 ;;
esac
echo "pulled '$what' from $BUCKET -> $DEST"
