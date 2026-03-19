#!/usr/bin/env python3
"""Verify downloaded model loads correctly."""

from transformers import UperNetForSemanticSegmentation, AutoImageProcessor
import os
import sys

MODEL_PATH = "./models/dinov2-upernet-final"

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
        print(f"\n   Model path checked: {os.path.abspath(MODEL_PATH)}")
        print(f"   Have you downloaded the model yet?")
        print(f"   Run: databricks fs cp -r dbfs:/FileStore/allen_brain_data/models/final-200ep {MODEL_PATH}")
        sys.exit(1)

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
    print(f"      Looked for: model.safetensors or pytorch_model.bin")
    sys.exit(1)

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
    sys.exit(1)

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
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ SUCCESS! Model is ready for inference.")
print("=" * 60)
print("\nYou can now use the model for inference.")
print("See docs/model_download_guide.md for inference examples.")
