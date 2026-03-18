# Step 7: Paper Write-Up Plan

## Paper Title (Working)

**"Transfer Learning for Ultra-Fine-Grained Brain Region Segmentation: An Ablation Study with DINOv2 + UperNet on 1,328 Allen Mouse Brain Atlas Structures"**

## Target

Write a technical report / paper draft to `docs/paper_draft.md` covering mouse model experiments only (Runs 1-9). Human model is being handled separately.

## Proposed Structure

### 1. Abstract (~200 words)
- Problem: automated segmentation of 1,328 brain structures from Nissl-stained histology
- Approach: DINOv2-Large + UperNet transfer learning
- Key result: 79.1% mIoU (sliding window), 96.9% accuracy, 671 valid classes
- Key findings: backbone unfreezing (+8.5%) and extended training (+6.0%) are the dominant factors; weighted loss, aggressive augmentation, class pruning, and TTA all fail

### 2. Introduction (~500 words)
- Brain atlas segmentation challenge
- Why 1,328 classes is extreme (vs typical 20-150 class segmentation)
- Transfer learning from natural images to histology
- Contributions: systematic ablation study, negative results, sliding window evaluation methodology

### 3. Related Work (~400 words)
- Brain atlas segmentation (Allen Brain Institute tools, registration-based approaches)
- Vision transformers for semantic segmentation (SegFormer, Mask2Former, UperNet)
- DINOv2 self-supervised learning and dense prediction
- Transfer learning for histological image analysis

### 4. Methods (~800 words)
- **4.1 Dataset**: Allen CCFv3 ara_nissl_10.nrrd, 1,320 coronal slices, 1,328 classes, 657 zero-pixel classes
- **4.2 Data Splitting**: Interleaved vs spatial split strategy and why it matters
- **4.3 Model Architecture**: DINOv2-Large backbone (304M params, 24 blocks) + UperNet decode head
- **4.4 Training Configuration**: Differential LR (1e-5/1e-4), AdamW, batch 4 effective, fp16, 200 epochs
- **4.5 Evaluation Protocol**: Center-crop (518×518) and sliding window (518×518 tiles, stride 259, 50% overlap)

### 5. Experiments (~1200 words)
- **5.1 Baseline Establishment** (Runs 1-4): Coarse → depth-2 → full classes, spatial vs interleaved split
- **5.2 Backbone Unfreezing** (Run 4→5): +8.5% mIoU, differential learning rate strategy
- **5.3 Loss Function Ablation** (Run 5→6): Weighted Dice+CE → −1.6% (negative result)
- **5.4 Augmentation Ablation** (Run 5→7): Extended augmentation → −6.5% (negative result)
- **5.5 Test-Time Augmentation** (TTA on Run 5): −24% catastrophic failure
- **5.6 Class Pruning** (Run 5→8a): 1,328→673 classes → +0.2% (neutral)
- **5.7 Extended Training** (Run 5→9): 100→200 epochs → +6.0% mIoU
- **5.8 Sliding Window Evaluation** (Run 9): +4.4% over center-crop, +168 valid classes

### 6. Analysis (~600 words)
- **6.1 Ablation Summary Table**: All interventions ranked by impact
- **6.2 Per-Class Performance Predictors**: Pixel count correlation (r=0.794), IoU distribution
- **6.3 Cross-Axis Generalization**: Coronal-only model fails on axial (3.2%) and sagittal (0.5%)
- **6.4 Convergence Analysis**: 200 epochs vs 100, diminishing returns assessment

### 7. Discussion (~400 words)
- What works: backbone unfreezing, extended training, sliding window eval
- What doesn't: weighted loss, aggressive augmentation, TTA, class pruning
- The performance ceiling is data-driven (pixel counts), not model-driven
- Limitations: single orientation, single stain type, single species
- Implications for human brain segmentation

### 8. Conclusion (~200 words)
- Summary of proven recipe
- Final metrics
- Complete training history table

## Source Data

| Source File | Content Used |
|-------------|-------------|
| `docs/mouse_model_findings.md` | Comprehensive findings, recommendations, Run 9 results |
| `docs/experimental_results.md` | Per-class IoU tables, training curves, hyperparameter comparison |
| `docs/finetuning_recommendations.md` | Architecture rationale, recommendation context |
| `docs/progress.md` | Project timeline, decision log |
| `docs/step12_plan.md` | Ablation rationale |

## Tables to Include

1. **Summary results table** — All runs with mIoU, accuracy, eval loss, runtime
2. **Ablation impact table** — Each intervention ranked by delta mIoU
3. **Hyperparameter comparison table** — Key config differences across runs
4. **Training curves** — Epoch-by-epoch data for Runs 4, 5, 7, 8a (text table format)
5. **Per-class IoU top/bottom 10** — Run 9 final model
6. **IoU bracket distribution** — How many classes at each quality level
7. **Cross-axis evaluation table** — Coronal vs axial vs sagittal
8. **Sliding window vs center-crop comparison** — Run 9

## Status

- [x] Plan written
- [x] Plan approved by user (via iMessage, 2026-03-17)
- [x] Paper draft written to `docs/paper_draft.md`
- [ ] User review complete
