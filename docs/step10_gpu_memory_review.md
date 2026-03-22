# Step 10 GPU Memory Review: Full Mapping Training Runs

## Summary

Three attempts to train the DINOv2-Large + UperNet model with full 1,328-class mapping
produced three different outcomes. This document captures root causes and lessons learned.

## Run Results

| Notebook | Cluster | Config | Result |
|---|---|---|---|
| `step10_full.ipynb` | Single GPU, single node (L40S 48 GB) | batch=4, no grad accum | **Succeeded** (50 epochs, ~5.5 hrs) |
| `step10_finetune_full.ipynb` | Multi-GPU, multi-node (8x L40S) | batch=4, no grad accum | **OOM** during backward pass |
| `step10_full_single_gpu.ipynb` | Single GPU, single node (L40S 48 GB) | batch=2, grad accum=2, grad ckpt=True | **ValueError** (UperNet doesn't support gradient checkpointing) |

All three notebooks used identical model code: DINOv2-Large frozen backbone + UperNet decoder,
1,328 classes, 518x518 crop, fp16 training.

## Root Cause Analysis

### Why `step10_full.ipynb` succeeded (single GPU)

On the single-GPU single-node cluster, the training process is the **sole occupant** of the GPU.
The full 44.53 GiB is available. Memory budget:

| Component | Estimate |
|---|---|
| DINOv2-Large backbone (frozen, fp16 weights) | ~600 MB |
| UperNet decoder (trainable, fp16 weights + fp32 optimizer states) | ~200 MB + ~400 MB |
| Backbone output features (4 stages, stored for decoder backward) | ~2-4 GB |
| Decoder intermediate activations | ~4-6 GB |
| Logits (1328 x 518 x 518 x batch=4, fp16 forward) | ~2.85 GB |
| Logits gradients (fp32 for loss backward) | ~5.7 GB |
| PyTorch CUDA allocator overhead | ~2-4 GB |
| **Total** | **~18-24 GB** |

This fits within 44.53 GiB with ~20 GB headroom.

### Why `step10_finetune_full.ipynb` OOM'd (multi-GPU)

The error message revealed **two processes** on GPU 0:

```
Process 16555 has 5.61 GiB memory in use.
Process 22872 has 36.71 GiB memory in use.
```

On a multi-GPU Databricks cluster, HF Trainer auto-detects multiple GPUs and launches
Distributed Data Parallel (DDP). The second process (5.61 GiB) is either:

- A DDP replica assigned to the same physical GPU
- The Databricks driver/spark process with CUDA context
- Another notebook sharing the node

This 5.61 GiB reduces available memory from 44.53 GiB to ~38.9 GiB. Combined with
5.16 GiB of CUDA memory fragmentation (reserved but unallocated), effective free memory
during the backward pass was only ~2.19 GiB — not enough for the ~5.31 GiB gradient
allocation.

**Key takeaway:** Multi-GPU != more memory per GPU. DDP replicates the entire model
on each GPU, and shared GPU nodes may have other processes consuming VRAM.

### Why `step10_full_single_gpu.ipynb` failed (ValueError)

`UperNetForSemanticSegmentation` does not implement `supports_gradient_checkpointing`
in HuggingFace Transformers. When `gradient_checkpointing=True` is passed to
`TrainingArguments`, the Trainer calls `model.gradient_checkpointing_enable()` at the
start of training, which raises:

```
ValueError: UperNetForSemanticSegmentation does not support gradient checkpointing.
```

Additionally, gradient checkpointing would provide **no benefit** in this setup:
- The backbone is frozen (`requires_grad=False` on all backbone params)
- PyTorch already skips storing intermediate backbone activations when no gradients
  flow through them
- The memory bottleneck is the logits tensor (1328 classes x 518x518 x batch),
  not backbone activations

## Lessons Learned

### 1. Single GPU is sufficient for this workload

`step10_full.ipynb` proves that batch=4 with 1,328 classes fits on a single L40S 48 GB
with ~20 GB headroom. Multi-GPU provides speed (data parallelism) but requires careful
memory accounting due to per-GPU overhead from DDP and Databricks processes.

### 2. Multi-GPU requires explicit memory planning

Before running on multi-GPU clusters:
- Account for DDP process memory overhead per GPU (~2-6 GB for CUDA context, NCCL buffers)
- Account for Databricks driver processes that may hold GPU memory
- Consider reducing per-device batch size and using gradient accumulation
- Use `nvidia-smi` or `torch.cuda.memory_summary()` to audit GPU occupancy before training

### 3. Not all HF models support gradient checkpointing

Always check `model.supports_gradient_checkpointing` before enabling it in TrainingArguments.
Composite models like UperNet (which wraps a backbone + decoder) may not implement the
required `_set_gradient_checkpointing` method.

For UperNet specifically, manual gradient checkpointing on the backbone submodule is
possible (`model.backbone.gradient_checkpointing_enable()`) but unnecessary when the
backbone is frozen.

### 4. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a good default

The OOM error reported 5.16 GiB "reserved by PyTorch but unallocated" — fragmentation
waste. Setting `expandable_segments:True` allows the CUDA allocator to grow segments
instead of allocating fixed-size blocks, reducing fragmentation. This is low-risk and
should be set in all training notebooks.

### 5. Always clear GPU cache between notebook cells

After a sanity-check forward pass, call:
```python
del dummy, out
torch.cuda.empty_cache()
```
to release cached CUDA allocations before the Trainer takes over. Without this,
PyTorch's caching allocator holds onto memory from the sanity check, reducing
available memory for training.

### 6. Verify training args before sending to a cluster

The `gradient_checkpointing` failure could have been caught locally by instantiating
`TrainingArguments` and calling `model.gradient_checkpointing_enable()` in a test,
rather than discovering it after cluster startup and data loading.

## Corrected Notebook: `step10_full_single_gpu.ipynb`

The fixed version removes `gradient_checkpointing` entirely and retains:
- **Batch size 2 + gradient accumulation 2** (same effective batch=4, halved peak VRAM)
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (reduces fragmentation)
- **`torch.cuda.empty_cache()`** after sanity check (clean GPU state before training)

These are conservative optimizations that provide extra headroom without changing training
dynamics. On the single-GPU cluster, `step10_full.ipynb` already works at batch=4 without
any of these — the optimizations are insurance for tighter memory situations.

## Recommendation for Multi-GPU Runs

If you need to run on the multi-GPU cluster for speed:
1. Use batch=2 + gradient_accumulation_steps=2 (preserves effective batch=4)
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
3. Verify GPU occupancy on the cluster before training with:
   ```python
   import subprocess
   print(subprocess.check_output(["nvidia-smi"]).decode())
   ```
4. If other processes occupy GPU memory, either kill them or further reduce batch size
