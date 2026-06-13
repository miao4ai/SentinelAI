#!/bin/bash
# Create (or restart) the SentinelAI CPU dev VM.
# - e2-standard-4: 4 vCPU / 16GB, no GPU (no GPU quota needed)
# - Builds billing history to unlock GPU quota later
# - Anti-runaway-cost: 8h hard auto-stop + idle auto-shutdown (30min)
set -e
PROJECT=just-aloe-499321-q2
ZONE=us-central1-a
NAME=sentinel-dev
DIR="$(cd "$(dirname "$0")" && pwd)"

# If the VM already exists, just start it instead of recreating.
if gcloud compute instances describe "$NAME" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
  gcloud compute instances start "$NAME" --project="$PROJECT" --zone="$ZONE"
  exit 0
fi

gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --max-run-duration=28800s \
  --instance-termination-action=STOP \
  --metadata-from-file=startup-script="$DIR/idle-shutdown.sh" \
  --labels=purpose=sentinelai-dev
