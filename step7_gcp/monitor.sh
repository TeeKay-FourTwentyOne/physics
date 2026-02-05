#!/bin/bash
# Monitor script for Hénon-Heiles training on GCP

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

#=============================================================================
# COMMANDS
#=============================================================================
case "${1:-help}" in
    status)
        echo "=== VM Status ==="
        gcloud compute instances describe "$VM_NAME" --zone="$ZONE" \
            --format="table(name,status,scheduling.provisioningModel,networkInterfaces[0].accessConfigs[0].natIP)"
        ;;

    log)
        echo "=== Training Log (Ctrl+C to exit) ==="
        gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
            --command="tail -f /home/trainer/training.log"
        ;;

    log-startup)
        echo "=== Startup Script Log ==="
        gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
            --command="cat /var/log/henon-training.log"
        ;;

    checkpoints)
        echo "=== Checkpoints in GCS ==="
        gsutil ls -l "gs://$BUCKET_NAME/checkpoints/" 2>/dev/null || echo "No checkpoints yet"
        ;;

    results)
        echo "=== Latest Results ==="
        # Try to get results file
        RESULTS=$(gsutil cat "gs://$BUCKET_NAME/results/final_results.json" 2>/dev/null || true)
        if [ -n "$RESULTS" ]; then
            echo "$RESULTS" | python3 -m json.tool
        else
            # Try training log for latest epoch
            echo "No final results yet. Latest from training log:"
            gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
                --command="grep -E 'Epoch|Energy|Loss' /home/trainer/training.log | tail -20" 2>/dev/null || echo "Training not started yet"
        fi
        ;;

    ssh)
        echo "=== SSH to VM ==="
        gcloud compute ssh "$VM_NAME" --zone="$ZONE"
        ;;

    gpu)
        echo "=== GPU Status ==="
        gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
            --command="nvidia-smi"
        ;;

    sync)
        echo "=== Force sync checkpoints to GCS ==="
        gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
            --command="gsutil -m rsync -r /home/trainer/checkpoints gs://$BUCKET_NAME/checkpoints/"
        echo "Sync complete"
        ;;

    download)
        echo "=== Download results locally ==="
        mkdir -p ./results
        gsutil -m rsync -r "gs://$BUCKET_NAME/results/" ./results/
        gsutil -m rsync -r "gs://$BUCKET_NAME/checkpoints/" ./results/checkpoints/
        echo "Downloaded to ./results/"
        ;;

    start)
        echo "=== Starting VM ==="
        gcloud compute instances start "$VM_NAME" --zone="$ZONE"
        echo "VM starting. Training will resume from latest checkpoint."
        ;;

    stop)
        echo "=== Stopping VM ==="
        # Sync first
        echo "Syncing checkpoints before stop..."
        gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
            --command="gsutil -m rsync -r /home/trainer/checkpoints gs://$BUCKET_NAME/checkpoints/ && gsutil -m rsync -r /home/trainer/results gs://$BUCKET_NAME/results/" 2>/dev/null || true
        gcloud compute instances stop "$VM_NAME" --zone="$ZONE"
        echo "VM stopped. Restart with: ./monitor.sh start"
        ;;

    help|*)
        echo "Usage: ./monitor.sh <command>"
        echo ""
        echo "Commands:"
        echo "  status      - Show VM status"
        echo "  log         - Tail training log (live)"
        echo "  log-startup - View startup script log"
        echo "  checkpoints - List checkpoints in GCS"
        echo "  results     - Show latest results"
        echo "  ssh         - SSH into VM"
        echo "  gpu         - Show GPU status (nvidia-smi)"
        echo "  sync        - Force sync checkpoints to GCS"
        echo "  download    - Download results to local ./results/"
        echo "  start       - Start stopped VM"
        echo "  stop        - Stop VM (syncs checkpoints first)"
        echo ""
        echo "Environment:"
        echo "  GCP_PROJECT_ID=$PROJECT_ID"
        echo "  Bucket: gs://$BUCKET_NAME/"
        ;;
esac
