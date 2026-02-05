#!/bin/bash
# VM Startup Script for Hénon-Heiles Training
# Downloads code from GCS, resumes from checkpoint if available, starts training

set -e

exec > >(tee -a /var/log/henon-training.log) 2>&1
echo "=== Startup Script Started at $(date) ==="

#=============================================================================
# CONFIGURATION
#=============================================================================
BUCKET_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket-name" -H "Metadata-Flavor: Google")
WORK_DIR="/home/trainer"
CODE_DIR="$WORK_DIR/code"
CHECKPOINT_DIR="$WORK_DIR/checkpoints"
RESULTS_DIR="$WORK_DIR/results"

echo "Bucket: gs://$BUCKET_NAME/"
echo "Work directory: $WORK_DIR"

#=============================================================================
# SETUP DIRECTORIES
#=============================================================================
mkdir -p "$CODE_DIR" "$CHECKPOINT_DIR" "$RESULTS_DIR"

#=============================================================================
# DOWNLOAD CODE
#=============================================================================
echo ">>> Downloading training code..."
gsutil cp "gs://$BUCKET_NAME/code/train_henon_heiles.py" "$CODE_DIR/"
echo "    Downloaded train_henon_heiles.py"

#=============================================================================
# CHECK FOR EXISTING CHECKPOINTS
#=============================================================================
echo ">>> Checking for existing checkpoints..."
LATEST_CHECKPOINT=""

if gsutil ls "gs://$BUCKET_NAME/checkpoints/" &>/dev/null; then
    # Find the latest checkpoint
    LATEST_CHECKPOINT=$(gsutil ls "gs://$BUCKET_NAME/checkpoints/*.pt" 2>/dev/null | sort -V | tail -1 || true)

    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "    Found checkpoint: $LATEST_CHECKPOINT"
        gsutil cp "$LATEST_CHECKPOINT" "$CHECKPOINT_DIR/"
        CHECKPOINT_FILE="$CHECKPOINT_DIR/$(basename "$LATEST_CHECKPOINT")"
        echo "    Downloaded to $CHECKPOINT_FILE"
    fi
fi

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "    No existing checkpoints found. Starting fresh."
fi

#=============================================================================
# SETUP CRON FOR GCS SYNC
#=============================================================================
echo ">>> Setting up checkpoint sync cron job..."
CRON_CMD="*/30 * * * * gsutil -m rsync -r $CHECKPOINT_DIR gs://$BUCKET_NAME/checkpoints/ >> /var/log/gcs-sync.log 2>&1"
(crontab -l 2>/dev/null | grep -v "gsutil.*checkpoints"; echo "$CRON_CMD") | crontab -
echo "    Checkpoints will sync to GCS every 30 minutes"

# Also sync results
RESULTS_CRON="*/30 * * * * gsutil -m rsync -r $RESULTS_DIR gs://$BUCKET_NAME/results/ >> /var/log/gcs-sync.log 2>&1"
(crontab -l 2>/dev/null; echo "$RESULTS_CRON") | crontab -

#=============================================================================
# WAIT FOR GPU
#=============================================================================
echo ">>> Waiting for GPU to be ready..."
for i in {1..30}; do
    if nvidia-smi &>/dev/null; then
        echo "    GPU detected!"
        nvidia-smi
        break
    fi
    echo "    Waiting for GPU... ($i/30)"
    sleep 10
done

#=============================================================================
# VERIFY PYTORCH CUDA
#=============================================================================
echo ">>> Verifying PyTorch CUDA..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

#=============================================================================
# START TRAINING
#=============================================================================
echo ">>> Starting training..."
cd "$CODE_DIR"

# Build command with optional checkpoint resume
TRAIN_CMD="python3 train_henon_heiles.py --checkpoint-dir $CHECKPOINT_DIR --results-dir $RESULTS_DIR"

if [ -n "$CHECKPOINT_FILE" ] && [ -f "$CHECKPOINT_FILE" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $CHECKPOINT_FILE"
fi

echo "Command: $TRAIN_CMD"
echo ""
echo "=== Training Output ==="

# Run training (nohup so it survives SSH disconnection)
nohup $TRAIN_CMD > "$WORK_DIR/training.log" 2>&1 &
TRAIN_PID=$!
echo "Training started with PID $TRAIN_PID"

# Save PID for monitoring
echo $TRAIN_PID > "$WORK_DIR/training.pid"

# Tail the log
tail -f "$WORK_DIR/training.log" &

echo "=== Startup Script Complete at $(date) ==="
