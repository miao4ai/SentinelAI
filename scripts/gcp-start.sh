#!/bin/bash
# Create (or restart) the SentinelAI L4 GPU machine.
# - g2-standard-8: 8 vCPU / 32GB, 1x NVIDIA L4 (24GB)
# - PyTorch + CUDA image (drivers preinstalled)
# - 250GB SSD for datasets / model weights / checkpoints
# - Anti-runaway-cost: 4h hard auto-stop + idle auto-shutdown (120min)
set -e
PROJECT=just-aloe-499321-q2
ZONE=us-central1-a
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
  --boot-disk-size=250GB \
  --boot-disk-type=pd-balanced \
  --max-run-duration=14400s \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --metadata-from-file=startup-script="$DIR/idle-shutdown.sh" \
  --labels=purpose=sentinelai-gpu
