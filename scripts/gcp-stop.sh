#!/bin/bash
# Stop the dev VM (keeps the disk; no CPU/GPU billing while stopped).
set -e
PROJECT=just-aloe-499321-q2
ZONE=us-central1-a
NAME=sentinel-gpu
gcloud compute instances stop "$NAME" --project="$PROJECT" --zone="$ZONE"
echo "Stopped $NAME. Disk is kept; start again with gcp-start.sh."
