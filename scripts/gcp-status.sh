#!/bin/bash
# Show whether the dev VM is running (i.e. whether it is costing money).
set -e
PROJECT=just-aloe-499321-q2
ZONE=us-central1-a
NAME=sentinel-dev
gcloud compute instances describe "$NAME" --project="$PROJECT" --zone="$ZONE" \
  --format="table(name, status, machineType.basename(), scheduling.maxRunDuration.seconds)" 2>/dev/null \
  || echo "$NAME does not exist (already deleted)."
