# DINOv2 Model Variant Research: Backbone Selection for Brain Segmentation

## Context

Current pipeline uses **DINOv2-Large** (304M params) + UperNet for semantic segmentation.
Results so far with frozen backbone, head-only training:

| Run | Granularity | mIoU | Overall Acc | Runtime | Cluster |
|-----|-------------|------|-------------|---------|---------|
| 2 | Coarse 6-class | **88.0%** | 95.8% | 23 min | 1x L40S |
| 3 | Depth-2 19-class | **69.6%** | 95.5% | 25 min | 1x L40S |
| 4 | Full 1,328-class | **60.3%** | 88.9% | 5.5 hrs | 1x L40S |

Question: would a larger backbone (DINOv2-Giant) improve these results?

---

## Available DINOv2 Variants

| Variant | HF Model ID | Params | Layers | Hidden Size | Patch | Out Features (UperNet) |
|---------|-------------|--------|--------|-------------|-------|------------------------|
| Small | `facebook/dinov2-small` | 21M | 12 | 384 | 14 | stage3, stage6, stage9, stage12 |
| Base | `facebook/dinov2-base` | 86M | 12 | 768 | 14 | stage3, stage6, stage9, stage12 |
| **Large** | `facebook/dinov2-large` | 304M | 24 | 1024 | 14 | stage6, stage12, stage18, stage24 |
| **Giant** | `facebook/dinov2-giant` | 1.1B | 40 | 1536 | 14 | stage10, stage20, stage30, stage40 |

All variants use the same `Dinov2Config` class in HuggingFace Transformers. All use 14x14 patches
and produce 37x37 feature maps at 518x518 input resolution.

**"With registers" variants** also exist (e.g., `facebook/dinov2-large-with-registers`). Register
tokens are extra learnable tokens appended to the patch sequence that absorb global information,
reducing artifacts in attention maps. These are drop-in replacements with the same config structure
and may produce slightly cleaner feature maps for dense prediction tasks like segmentation.

---

## Memory Analysis: Giant vs Large on L40S 48 GB

### Frozen backbone, 1,328 classes, batch=4, fp16

| Component | Large (current) | Giant |
|-----------|----------------|-------|
| Backbone weights (frozen, fp16) | ~0.6 GB | ~2.2 GB |
| UperNet head (trainable, fp16 + fp32 optimizer) | ~0.6 GB | ~0.8 GB |
| Backbone output features (4 stages, for decoder) | ~2-4 GB | ~3-5 GB |
| Decoder activations | ~4-6 GB | ~5-7 GB |
| Logits (1328 x 518 x 518 x 4, fp16) | ~2.85 GB | ~2.85 GB |
| Logits gradients (fp32) | ~5.7 GB | ~5.7 GB |
| CUDA allocator overhead | ~2-4 GB | ~2-4 GB |
| **Total estimate** | **~18-24 GB** | **~22-28 GB** |
| **Headroom on L40S 48 GB** | **~20-26 GB** | **~16-22 GB** |

**Conclusion: Giant fits on a single L40S 48 GB** with ~16-22 GB headroom. The logits tensor
(determined by num_labels, not backbone size) remains the dominant memory consumer. Giant adds
~4 GB over Large, primarily from larger backbone weights and wider feature channels.

For multi-GPU DDP: requires ~22-28 GB per GPU. Fits if no other process occupies GPU memory,
but leaves less margin than Large. Recommend batch=2 + grad_accum=2 on multi-GPU to be safe.

---

## Code Impact Assessment

### Current code in `training.py`

```python
DINOV2_LARGE_OUT_FEATURES = ["stage6", "stage12", "stage18", "stage24"]  # line 28

# Used in create_model() at two points:
# 1. Loading backbone config (line 180)
bc = Dinov2Config.from_pretrained(
    pretrained_backbone_path,
    out_features=DINOV2_LARGE_OUT_FEATURES,  # HARDCODED
    ...
)
# 2. Loading pretrained weights (line 210)
pretrained_backbone = AutoBackbone.from_pretrained(
    pretrained_backbone_path,
    out_features=DINOV2_LARGE_OUT_FEATURES,  # HARDCODED
)
```

**Problem:** `DINOV2_LARGE_OUT_FEATURES` is hardcoded for 24-layer models. Giant (40 layers)
requires `["stage10", "stage20", "stage30", "stage40"]`. The existing `backbone_config` parameter
in `create_model()` can bypass the first usage (line 180) but NOT the second (line 210 — weight
loading always uses the hardcoded constant).

### Options to support Giant

| Option | Approach | Modifies existing code? | Effort |
|--------|----------|------------------------|--------|
| **A. New module** | Create `training_multi.py` with a `create_model_v2(backbone_size=...)` function | No | New file |
| **B. Add parameter** | Add `out_features` param to `create_model()` with default=DINOV2_LARGE | Yes (backward-compatible) | 3 lines |
| **C. Notebook-only** | Build model directly in notebook, skip `create_model()` | No | Inline code in notebook |

**Recommended: Option A** — create a new module that wraps model creation with a backbone
size selector. This keeps `training.py` untouched and provides a clean API:

```python
# New file: src/histological_image_analysis/model_config.py
BACKBONE_CONFIGS = {
    "small":  {"out_features": ["stage3", "stage6", "stage9", "stage12"], "repo_id": "facebook/dinov2-small"},
    "base":   {"out_features": ["stage3", "stage6", "stage9", "stage12"], "repo_id": "facebook/dinov2-base"},
    "large":  {"out_features": ["stage6", "stage12", "stage18", "stage24"], "repo_id": "facebook/dinov2-large"},
    "giant":  {"out_features": ["stage10", "stage20", "stage30", "stage40"], "repo_id": "facebook/dinov2-giant"},
}

def create_model_for_backbone(backbone_size: str, num_labels: int, ...):
    """Create UperNet model with the specified DINOv2 backbone size."""
    config = BACKBONE_CONFIGS[backbone_size]
    # Build Dinov2Config with correct out_features
    # Load pretrained weights with correct out_features
    # Return configured UperNetForSemanticSegmentation
```

---

## Would Giant Actually Improve Results?

### Analysis of current bottlenecks

The full mapping run (60.3% mIoU, 503/1,328 classes with valid IoU) has specific bottlenecks
that a larger backbone may or may not address:

| Bottleneck | Root Cause | Would Giant Help? |
|-----------|-----------|-------------------|
| 825 classes have 0 val pixels | Rare structures absent from val slices | **No** — backbone size doesn't create data |
| Many classes at 0.0% IoU | Tiny structures (few pixels per slice) | **Maybe** — richer features could disambiguate small regions |
| Frozen backbone | Features not adapted to Nissl staining domain | **No** — features are still frozen regardless of size |
| 50 epochs | May not converge for 1,328-class head | **No** — training duration is independent of backbone |
| Class imbalance | Background/Cerebrum dominate, rare classes starved | **No** — this is a loss/sampling problem, not a feature problem |

### Recommendation: try these BEFORE switching to Giant

These interventions address the actual bottlenecks and require less compute:

1. **Unfreeze backbone with differential LR** (highest expected impact)
   - Backbone LR: 1e-5, Head LR: 1e-4
   - Lets DINOv2 features adapt to Nissl staining domain
   - Risk: overfitting (1,016 training samples). Mitigate with early stopping + weight decay
   - Implementation: Already possible via `get_training_args(**kwargs)` — no code changes

2. **Class-weighted cross-entropy loss** (medium impact)
   - Weight rare classes inversely proportional to pixel frequency
   - Requires modifying `create_model()` to pass class weights to loss function
   - Or: use focal loss via custom trainer

3. **More epochs for full mapping** (low cost, low risk)
   - 50 epochs may not be enough for 1,328-class head on 1,016 samples
   - Try 100-200 epochs with early stopping via `load_best_model_at_end=True`

4. **Two-stage curriculum** (medium effort, high expected impact)
   - Stage 1: Train coarse head (frozen backbone) — already done, 88% mIoU
   - Stage 2: Unfreeze backbone + switch to fine head, warmup from coarse features
   - Rationale: coarse training learns domain-adapted features, fine training refines them

### When Giant IS worth it

Giant would be worth pursuing if:
- After unfreezing the backbone, you see that feature quality (not training dynamics) is the bottleneck
- The top-performing classes plateau below expected levels even with unfrozen backbone
- You want to push state-of-the-art on the larger structures (where data is plentiful)

Giant's 1536-dim features vs Large's 1024-dim provide ~50% more representation capacity per patch,
which helps distinguish visually similar structures. This matters most for fine-grained classes
where pixel appearance differences are subtle.

---

## JFrog Availability

Current pipeline downloads `facebook/dinov2-large` from JFrog Artifactory mirror. Before using
Giant, verify that `facebook/dinov2-giant` is cached in the Artifactory mirror:

```bash
# Test from Databricks cluster
curl -s "https://graingerreadonly.jfrog.io/artifactory/api/huggingfaceml/huggingfaceml-remote/facebook/dinov2-giant/config.json" | head -c 100
```

If not cached, the first download will pull from HuggingFace Hub (~4.4 GB for Giant weights)
and cache in Artifactory. Subsequent downloads will use the cache.

---

## Summary

| Question | Answer |
|----------|--------|
| Does Giant fit on L40S 48 GB? | **Yes** — ~22-28 GB, ~16-22 GB headroom |
| Does it require code changes? | **Yes** — `out_features` hardcoded for Large. Recommend new file (Option A) |
| Would it improve results? | **Unlikely to be the highest-impact intervention** |
| What should we try first? | **Unfreeze backbone** (1e-5 LR), then class-weighted loss, then more epochs |
| When to try Giant? | After unfreezing backbone, if feature quality is still the bottleneck |
