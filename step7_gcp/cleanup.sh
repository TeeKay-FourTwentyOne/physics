#!/bin/bash
# Cleanup script - deletes GCP resources for Hénon-Heiles training

set -e

#=============================================================================
# CONFIGURATION
#=============================================================================
PROJECT_ID="${GCP_PROJECT_ID:-physics-486301}"
ZONE="us-central1-a"
VM_NAME="henon-heiles-trainer"
BUCKET_NAME="${PROJECT_ID}-henon-heiles"

#=============================================================================
# VALIDATION
#=============================================================================
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "ERROR: Please set PROJECT_ID or export GCP_PROJECT_ID"
    exit 1
fi

echo "=== GCP Cleanup for Hénon-Heiles Training ==="
echo ""
echo "This will DELETE the following resources:"
echo "  VM:     $VM_NAME"
echo "  Bucket: gs://$BUCKET_NAME/ (and all contents)"
echo ""

#=============================================================================
# CONFIRMATION
#=============================================================================
read -p "Download results before deleting? [y/N] " download_first
if [[ "$download_first" =~ ^[Yy] ]]; then
    echo ""
    echo ">>> Downloading results..."
    mkdir -p ./results
    gsutil -m rsync -r "gs://$BUCKET_NAME/results/" ./results/ 2>/dev/null || true
    gsutil -m rsync -r "gs://$BUCKET_NAME/checkpoints/" ./results/checkpoints/ 2>/dev/null || true
    echo "    Downloaded to ./results/"
    echo ""
fi

read -p "Are you sure you want to delete these resources? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy] ]]; then
    echo "Aborted."
    exit 0
fi

echo ""

#=============================================================================
# DELETE VM
#=============================================================================
echo ">>> Deleting VM..."
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
    gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
    echo "    Deleted $VM_NAME"
else
    echo "    VM not found (already deleted?)"
fi

#=============================================================================
# DELETE BUCKET
#=============================================================================
echo ">>> Deleting GCS bucket..."
if gsutil ls -b "gs://$BUCKET_NAME" &>/dev/null; then
    gsutil -m rm -r "gs://$BUCKET_NAME"
    echo "    Deleted gs://$BUCKET_NAME/"
else
    echo "    Bucket not found (already deleted?)"
fi

echo ""
echo "=== Cleanup Complete ==="
