# Step 7: GCP Deployment for 6D Hénon-Heiles Benchmark

Deploy the LP+PINN solver to Google Cloud Platform for the 6D Hénon-Heiles benchmark.

## Quick Start: Google Colab (Recommended)

The easiest way to run this benchmark is via Google Colab:

1. Open `henon_heiles_colab.ipynb` in Google Colab
2. Go to Runtime → Change runtime type → Select **T4 GPU**
3. Run all cells

Checkpoints are saved to Google Drive for persistence across sessions.

**Colab notebook URL** (after uploading to Drive):
```
gs://physics-486301-henon-heiles/notebooks/henon_heiles_colab.ipynb
```

---

## Alternative: GCP Compute Engine

## Prerequisites

1. **Google Cloud SDK** installed and authenticated:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **APIs enabled**:
   ```bash
   gcloud services enable compute.googleapis.com storage.googleapis.com
   ```

3. **GPU quota**: Ensure you have L4 or T4 GPU quota in `us-central1-a`

## Quick Start

### 1. Configure Project ID

Edit `gcp_setup.sh` and set your project ID:
```bash
PROJECT_ID="your-project-id"
```

### 2. Deploy

```bash
chmod +x *.sh
./gcp_setup.sh
```

This creates:
- GCS bucket: `gs://${PROJECT_ID}-henon-heiles/`
- Spot VM: `henon-heiles-trainer` with L4 GPU

### 3. Monitor Training

```bash
# Check VM status
./monitor.sh status

# View training logs (live tail)
./monitor.sh log

# List checkpoints in GCS
./monitor.sh checkpoints

# View latest results
./monitor.sh results

# SSH into VM
./monitor.sh ssh
```

### 4. Cleanup

```bash
./cleanup.sh
```

## Architecture

### Network
- 8 hidden layers × 512 neurons
- Tanh activation
- Input: 6 (position coordinates x₁...x₆)
- Output: 12 (real + imaginary momentum components)

### Hénon-Heiles Potential (6D)
```
V(x) = ½Σᵢxᵢ² + λΣᵢ(xᵢ²xᵢ₊₁ - xᵢ₊₁³/3)
```
With periodic boundary (x₇ = x₁) and λ = 1/√80 ≈ 0.111803.

### Training
- Collocation points: 100,000 per epoch
- Learning rate: 1e-4 (Adam)
- Max epochs: 50,000
- Checkpoint interval: 1000 epochs
- GCS sync: Every 30 minutes via cron

### Loss Functions
1. **Physics loss**: QHJE residual (p² + iℏ∇·p = 2m(E-V))
2. **Curl loss**: Irrotationality (∂pᵢ/∂xⱼ = ∂pⱼ/∂xᵢ)

## Expected Results

| Metric | Target |
|--------|--------|
| Final energy | 2.95-2.99 a.u. |
| Below harmonic ZPE (3.0) | Yes |
| Error vs ~2.97 | < 2% |

## Cost Estimate

| Configuration | Hourly (Spot) | 24hr Total |
|---------------|---------------|------------|
| g2-standard-8 (L4) | $0.25-0.40 | $6-10 |

Expected training time: ~20 hours → **$5-8 total**

## Preemption Handling

Spot VMs may be preempted. The system handles this via:
1. Checkpoints saved to GCS every 30 minutes
2. Startup script resumes from latest checkpoint
3. VM set to STOP (not DELETE) on termination

If preempted, simply restart:
```bash
gcloud compute instances start henon-heiles-trainer --zone=us-central1-a
```

## Troubleshooting

### VM won't start (quota exceeded)
```bash
# Try T4 instead of L4
gcloud compute instances create henon-heiles-trainer \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  ...
```

### Training not starting
```bash
./monitor.sh ssh
tail -f /var/log/startup-script.log
```

### Check GPU is detected
```bash
./monitor.sh ssh
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```
