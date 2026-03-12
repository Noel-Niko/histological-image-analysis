# Model Download Guide

Guide for downloading trained DINOv2-UperNet models from Databricks to your local machine for inference.

---

## Overview

Trained models are stored in two locations:
1. **MLflow Experiments** - Full experiment tracking with metrics, parameters, and model artifacts
2. **DBFS FileStore** - Persistent model files at `/dbfs/FileStore/allen_brain_data/models/`

**Model Size:** ~1.2-1.5 GB per model (excluded from git)

### Compute Requirements for Local Inference

✅ **Yes, the model can run on a laptop!**

| Hardware | Requirement |
|----------|------------|
| **RAM** | Minimum 8 GB, 16 GB recommended |
| **Disk** | 2 GB free space (model + results) |
| **CPU** | Any modern processor (2+ cores) |
| **GPU** | Optional - speeds up inference significantly |

**Performance:**
- **CPU only:** ~10-30 seconds per image (Intel i5/i7 or AMD Ryzen 5/7)
- **Laptop GPU:** ~2-5 seconds per image (NVIDIA MX/GTX series)
- **Desktop GPU:** ~1-2 seconds per image (NVIDIA RTX 3060+)

The model (342M parameters, 1.2 GB) fits comfortably in laptop RAM. GPU is optional but recommended for batch processing (10+ images).

---

## Prerequisites

### 1. Databricks CLI

```bash
# Install Databricks CLI
pip install databricks-cli

# Verify installation
databricks --version
```

### 2. Configure Authentication

```bash
# List configured profiles
databricks auth profiles

# Should show 'dev' profile targeting:
# grainger-gtg-mlops-dev-use1-dbx-shared-ws.cloud.databricks.com
```

If not configured, set up authentication:

```bash
# Configure with personal access token
databricks configure --token

# Enter:
# - Host: https://grainger-gtg-mlops-dev-use1-dbx-shared-ws.cloud.databricks.com
# - Token: <your-databricks-token>
```

---

## Available Models

| Model | Classes | Backbone | Path | MLflow Run ID |
|-------|---------|----------|------|---------------|
| Coarse | 6 | Frozen | `dbfs:/FileStore/allen_brain_data/models/coarse_6class` | TBD |
| Depth-2 | 19 | Frozen | `dbfs:/FileStore/allen_brain_data/models/depth2` | TBD |
| Full | 1,328 | Frozen | `dbfs:/FileStore/allen_brain_data/models/full` | TBD |
| **Full (unfrozen)** | 1,328 | Last 4 blocks unfrozen | `dbfs:/FileStore/allen_brain_data/models/unfrozen` | `6cc49e1ccb0d4b30b371e9a071dcbe6f` |

---

## Method 1: Download from DBFS (Recommended)

### Step 1: Create Local Directory

```bash
cd /Users/xnxn040/PycharmProjects/Personal-Projects/histological-image-analysis
mkdir -p models
```

### Step 2: Download Model

```bash
# Download unfrozen model (~1.2 GB)
databricks fs cp -r \
  dbfs:/FileStore/allen_brain_data/models/unfrozen \
  ./models/dinov2-upernet-unfrozen

# Download takes 2-5 minutes depending on connection speed
```

### Step 3: Verify Download

```bash
# List downloaded files
ls -lh ./models/dinov2-upernet-unfrozen/

# Expected output:
# config.json                 (~1 KB)    - Model architecture config
# preprocessor_config.json    (~1 KB)    - Image processor config
# model.safetensors          (~1.2 GB)   - Model weights
# training_args.bin          (optional)  - Training configuration
# optimizer.pt               (optional)  - Can be deleted
# scheduler.pt               (optional)  - Can be deleted

# Check total size
du -sh ./models/dinov2-upernet-unfrozen/
# Expected: ~1.2G
```

### Step 4: Clean Up Optional Files (Optional)

```bash
# Remove optimizer and scheduler (not needed for inference)
cd ./models/dinov2-upernet-unfrozen/
rm -f optimizer.pt scheduler.pt rng_state*.pth

# This reduces disk usage by ~2-3 GB if present
```

---

## Method 2: Download from MLflow

### Step 1: Set Environment Variables

```bash
export DATABRICKS_HOST="https://grainger-gtg-mlops-dev-use1-dbx-shared-ws.cloud.databricks.com"
export DATABRICKS_TOKEN="<your-token>"
```

### Step 2: Download via MLflow CLI

```bash
cd /Users/xnxn040/PycharmProjects/Personal-Projects/histological-image-analysis
mkdir -p models

# Download model artifacts from MLflow run
mlflow artifacts download \
  --run-id 6cc49e1ccb0d4b30b371e9a071dcbe6f \
  --artifact-path model \
  --dst-path ./models/dinov2-upernet-unfrozen
```

### Step 3: Verify Download

Same as Method 1, Step 3 above.

---

## Method 3: Load Directly from MLflow (No Download)

For quick testing without downloading, load the model directly from MLflow:

```python
import mlflow.transformers

# Set up MLflow tracking URI
mlflow.set_tracking_uri("databricks")

# Load model directly from MLflow
model = mlflow.transformers.load_model(
    "runs:/6cc49e1ccb0d4b30b371e9a071dcbe6f/model"
)

# This downloads the model to a temporary cache directory
# Subsequent loads will use the cached version
```

**Note:** This requires `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables.

---

## Verify Model Installation

Create a test script to verify the model loads correctly:

```python
#!/usr/bin/env python3
"""Verify downloaded model loads correctly."""

from transformers import UperNetForSemanticSegmentation, AutoImageProcessor
import os

MODEL_PATH = "./models/dinov2-upernet-unfrozen"

print("=" * 60)
print("MODEL VERIFICATION")
print("=" * 60)

# Check required files exist
print("\n1. Checking files...")
required_files = ["config.json", "preprocessor_config.json"]
for filename in required_files:
    path = os.path.join(MODEL_PATH, filename)
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"   {status} {filename}")
    if not exists:
        print(f"      ERROR: Missing required file!")
        exit(1)

# Check model weights
print("\n2. Checking model weights...")
safetensors_path = os.path.join(MODEL_PATH, "model.safetensors")
pytorch_path = os.path.join(MODEL_PATH, "pytorch_model.bin")

if os.path.exists(safetensors_path):
    size_gb = os.path.getsize(safetensors_path) / 1e9
    print(f"   ✅ model.safetensors ({size_gb:.2f} GB)")
elif os.path.exists(pytorch_path):
    size_gb = os.path.getsize(pytorch_path) / 1e9
    print(f"   ✅ pytorch_model.bin ({size_gb:.2f} GB)")
else:
    print(f"   ❌ No model weights found!")
    exit(1)

# Load image processor
print("\n3. Loading image processor...")
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    print(f"   ✅ Processor loaded")
    print(f"      Size: {processor.size}")
    print(f"      Do rescale: {processor.do_rescale}")
    print(f"      Do normalize: {processor.do_normalize}")
except Exception as e:
    print(f"   ❌ Failed to load processor: {e}")
    exit(1)

# Load model
print("\n4. Loading model...")
try:
    model = UperNetForSemanticSegmentation.from_pretrained(MODEL_PATH)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model loaded")
    print(f"      Total parameters: {num_params:,}")
    print(f"      Number of labels: {model.config.num_labels}")
    print(f"      Hidden size: {model.config.hidden_size}")
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ SUCCESS! Model is ready for inference.")
print("=" * 60)
```

Save as `verify_model.py` and run:

```bash
python verify_model.py
```

---

## Quick Start: Using the Inference Script

**Easiest way for PhD researchers to test the model:**

```bash
# 1. Download the model (see sections above)
databricks fs cp -r dbfs:/FileStore/allen_brain_data/models/unfrozen ./models/dinov2-upernet-unfrozen

# 2. Run inference on a single image
python scripts/run_inference.py --image path/to/brain_slice.jpg --output results/

# 3. Or batch process a directory of images
python scripts/run_inference.py --image-dir images/ --output results/

# 4. Force CPU if no GPU available
python scripts/run_inference.py --image image.png --output results/ --cpu
```

**What you get:**
- `<name>_mask.png` - Raw segmentation mask (518×518, uint16)
- `<name>_mask_resized.png` - Segmentation at original image size (uint16)
- `<name>_visualization.png` - Side-by-side comparison with color-coded regions

**Example output:**
```
Processing 1 image(s)...
============================================================

brain_slice_001.jpg
  ✓ Saved: results/brain_slice_001_mask.png
  ✓ Saved: results/brain_slice_001_mask_resized.png
  ✓ Saved: results/brain_slice_001_visualization.png
  → Detected 87 / 1328 possible classes

============================================================
✓ Processing complete!
  Results saved to: results/
```

**Script features:**
- ✅ Runs on laptop (CPU or GPU)
- ✅ Batch processing support
- ✅ Progress bar for multiple images
- ✅ Automatic GPU detection
- ✅ Saves masks + visualizations
- ✅ Reports detected brain regions

---

## Inference Example (Python Code)

If you prefer to write your own inference code, here's a basic example:

```python
from transformers import UperNetForSemanticSegmentation, AutoImageProcessor
from PIL import Image
import torch
import numpy as np

# Load model and processor
MODEL_PATH = "./models/dinov2-upernet-unfrozen"
model = UperNetForSemanticSegmentation.from_pretrained(MODEL_PATH)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load and preprocess image
image = Image.open("path/to/brain_slice.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # Shape: [1, 1328, H, W]

# Get predicted class for each pixel
prediction = logits.argmax(dim=1)[0]  # Shape: [H, W]
prediction_np = prediction.cpu().numpy()

print(f"Input image size: {image.size}")
print(f"Prediction shape: {prediction_np.shape}")
print(f"Unique predicted classes: {len(np.unique(prediction_np))}")

# Visualize (optional)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title("Input Image")
axes[0].axis("off")

axes[1].imshow(prediction_np, cmap="nipy_spectral")
axes[1].set_title("Predicted Segmentation")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("segmentation_result.png", dpi=150, bbox_inches="tight")
print("Saved visualization to segmentation_result.png")
```

---

## Troubleshooting

### Error: `databricks: command not found`

**Solution:** Install Databricks CLI:
```bash
pip install databricks-cli
```

### Error: `Authentication is not configured`

**Solution:** Configure Databricks authentication:
```bash
databricks configure --token
# Enter host and token when prompted
```

### Error: `FileNotFoundError: Model not found`

**Cause:** Model hasn't been saved to DBFS yet, or path is incorrect.

**Solution:**
1. Check the training notebook completed successfully (Cell 7)
2. Verify path on DBFS:
   ```bash
   databricks fs ls dbfs:/FileStore/allen_brain_data/models/
   ```
3. If model is missing, re-run the training notebook Cell 7, or run the `mlflow_log_unfrozen_model.ipynb` notebook

### Error: `OSError: preprocessor_config.json not found`

**Cause:** Processor config wasn't saved with the model.

**Solution:** Run `notebooks/mlflow_log_unfrozen_model.ipynb` which will download and save the processor config.

### Download is very slow

**Cause:** Large file transfer over network.

**Solutions:**
- Use a wired connection instead of WiFi
- Download during off-peak hours
- Consider using Method 3 (load directly from MLflow) for testing

### Error: `RuntimeError: CUDA out of memory`

**Cause:** Model is too large for GPU.

**Solutions:**
```python
# Use CPU for inference
device = torch.device("cpu")
model = model.to(device)

# Or use smaller batch sizes / tile-based inference
# Process image in 512x512 tiles with overlap
```

---

## Model File Structure

```
models/dinov2-upernet-unfrozen/
├── config.json                 # Model architecture configuration
│   ├── num_labels: 1328
│   ├── hidden_size: 1024
│   └── backbone: "facebook/dinov2-large"
├── preprocessor_config.json    # Image preprocessing configuration
│   ├── size: {"height": 518, "width": 518}
│   ├── do_rescale: true
│   └── do_normalize: true
├── model.safetensors          # Model weights (safetensors format)
│   └── 342M parameters (~1.2 GB)
├── training_args.bin          # Training hyperparameters (optional)
├── optimizer.pt               # Optimizer state (can delete)
└── scheduler.pt               # LR scheduler state (can delete)
```

---

## Model Metadata

**Unfrozen Model (`6cc49e1ccb0d4b30b371e9a071dcbe6f`):**

| Attribute | Value |
|-----------|-------|
| Classes | 1,328 |
| Architecture | DINOv2-Large (304M) + UperNet (38M) |
| Total Parameters | 342M |
| Trainable Parameters | 88M (25.8%) |
| Backbone | Last 4 blocks unfrozen (blocks 20-23) |
| Training Data | Allen CCFv3 10µm Nissl mouse brain (1,016 slices) |
| Input Size | 518×518 (DINOv2 native resolution) |
| Validation mIoU | 60.3% → **TBD** (unfrozen) |
| Training Time | ~11 hours (100 epochs, 1x L40S 48GB) |
| Differential LR | Backbone 1e-5, Head 1e-4 |

---

## Next Steps

1. **Download the model** using Method 1 (DBFS) or Method 2 (MLflow)
2. **Verify installation** with the verification script
3. **Run inference** on your histological images
4. **See also:**
   - [docs/finetuning_recommendations.md](finetuning_recommendations.md) - Further training improvements
   - [docs/dinov2_model_research.md](dinov2_model_research.md) - Model architecture analysis
   - [notebooks/finetune_unfrozen.ipynb](../notebooks/finetune_unfrozen.ipynb) - Training notebook

---

## Model Not in Git

Models are excluded from git via `.gitignore` because:
- Model files are ~1.2-1.5 GB (exceeds GitHub's 100 MB limit)
- Binary files don't benefit from git version control
- Models are already versioned in MLflow

To share models with collaborators:
1. Share the MLflow run ID
2. Share the DBFS path
3. Or upload to HuggingFace Hub (optional)

---

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Verify Databricks CLI authentication: `databricks auth profiles`
3. Check MLflow experiment UI for model artifacts
4. Consult the training notebooks for model saving code
