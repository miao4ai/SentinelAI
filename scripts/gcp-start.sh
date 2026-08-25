#!/bin/bash
# Create (or restart) the SentinelAI GPU machine.
# - g2-standard-8: 8 vCPU / 32GB, 1x NVIDIA L4 (24GB)
# - PyTorch + CUDA image (drivers preinstalled)
# - Small 60GB boot disk: datasets live in GCS (gs://sentinelai-data-...),
#   pull what you need with scripts/pull-data.sh (see that script).
# - cloud-platform scope so the VM can read/write the GCS bucket.
# - Anti-runaway-cost: 4h hard auto-stop + idle auto-shutdown (120min)
#
# After first create: run scripts/setup on the VM to install deps + pull data.
# If L4 is stocked out, swap --machine-type to e2-standard-8 (CPU) for non-GPU work.
set -e
PROJECT=just-aloe-499321-q2
ZONE=us-central1-b
NAME=sentinel-gpu
DIR="$(cd "$(dirname "$0")" && pwd)"

# If the VM already exists, just start it instead of recreating.
if gcloud compute instances describe "$NAME" --project="$PROJECT" --zone="$ZONE" >/dev/null 2>&1; then
  gcloud compute instances start "$NAME" --project="$PROJECT" --zone="$ZONE"
  exit 0
fi

gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=60GB \
  --boot-disk-type=pd-balanced \
  --scopes=cloud-platform \
  --max-run-duration=14400s \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --metadata-from-file=startup-script="$DIR/idle-shutdown.sh" \
  --labels=purpose=sentinelai-gpu
