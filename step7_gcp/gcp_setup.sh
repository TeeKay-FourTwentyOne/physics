#!/bin/bash
# GCP Setup Script for 6D Hénon-Heiles Training
# Creates GCS bucket and Spot VM with L4 GPU

set -e

#=============================================================================
# CONFIGURATION - EDIT THESE
#=============================================================================
PROJECT_ID="${GCP_PROJECT_ID:-physics-486301}"
REGION="us-central1"
ZONE="us-central1-a"
VM_NAME="henon-heiles-trainer"
BUCKET_NAME="${PROJECT_ID}-henon-heiles"

# VM Configuration
MACHINE_TYPE="g2-standard-8"  # Includes L4 GPU
BOOT_DISK_SIZE="100GB"
# Alternative: n1-standard-8 with --accelerator=type=nvidia-tesla-t4,count=1

#=============================================================================
# VALIDATION
#=============================================================================
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "ERROR: Please set PROJECT_ID in this script or export GCP_PROJECT_ID"
    echo ""
    echo "  Option 1: Edit this script and set PROJECT_ID=\"your-actual-project\""
    echo "  Option 2: export GCP_PROJECT_ID=\"your-actual-project\" && ./gcp_setup.sh"
    exit 1
fi

echo "=== GCP Setup for Hénon-Heiles Training ==="
echo "Project:  $PROJECT_ID"
echo "Region:   $REGION"
echo "Zone:     $ZONE"
echo "VM:       $VM_NAME"
echo "Bucket:   gs://$BUCKET_NAME/"
echo ""

# Confirm project
gcloud config set project "$PROJECT_ID"

#=============================================================================
# CREATE GCS BUCKET
#=============================================================================
echo ">>> Creating GCS bucket..."
if gsutil ls -b "gs://$BUCKET_NAME" &>/dev/null; then
    echo "    Bucket already exists"
else
    gsutil mb -l "$REGION" "gs://$BUCKET_NAME"
    echo "    Created gs://$BUCKET_NAME/"
fi

#=============================================================================
# UPLOAD TRAINING CODE
#=============================================================================
echo ">>> Uploading training code..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gsutil cp "$SCRIPT_DIR/train_henon_heiles.py" "gs://$BUCKET_NAME/code/"
gsutil cp "$SCRIPT_DIR/startup_script.sh" "gs://$BUCKET_NAME/code/"
echo "    Uploaded to gs://$BUCKET_NAME/code/"

#=============================================================================
# CREATE SPOT VM
#=============================================================================
echo ">>> Creating Spot VM with L4 GPU..."

# Check if VM already exists
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
    echo "    VM already exists. Delete it first with ./cleanup.sh or:"
    echo "    gcloud compute instances delete $VM_NAME --zone=$ZONE"
    exit 1
fi

# Create the VM
gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type=pd-ssd \
    --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --scopes=storage-full \
    --metadata=startup-script-url="gs://$BUCKET_NAME/code/startup_script.sh",bucket-name="$BUCKET_NAME"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "VM is starting. Training will begin automatically in ~5 minutes."
echo ""
echo "Monitor with:"
echo "  ./monitor.sh status      # Check VM status"
echo "  ./monitor.sh log         # View training logs"
echo "  ./monitor.sh checkpoints # List saved checkpoints"
echo "  ./monitor.sh ssh         # SSH into VM"
echo ""
echo "When done:"
echo "  ./cleanup.sh             # Delete VM and bucket"
