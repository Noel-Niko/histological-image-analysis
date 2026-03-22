 # Plan: Step 10 — Fine Granularity Training (Depth-2 → Full Mapping)

## Context

Coarse 6-class model achieved **88% mIoU** on 10µm Nissl CCFv3 (Run 2, interleaved split). 124/124 tests pass. Now scaling to finer granularity: **depth-2 (19 classes including background)** first, then **full mapping (1,328 classes including background)**.

**Key finding: the pipeline is already generic.** `build_depth_mapping()`, `build_full_mapping()`, `get_num_labels()`, `BrainSegmentationDataset`, `create_model(num_labels=N)`, and `get_training_args(**kwargs)` all work with arbitrary class counts. The only code fix needed is a bug in `get_class_names()`.

**Available clusters:**
- Single GPU: g6e.16xlarge, 1x L40S 48GB *(fallback only)*
- Multi-GPU: g6e.48xlarge, 8x L40S 48GB, 1,536 GB RAM *(primary - use for speed)*

**Note:** Cluster IDs are ephemeral and will differ at execution time.

---

## Multi-GPU Strategy: DDP for Speed (not FSDP)

**Recommendation: Use multi-GPU DDP from the start for faster iteration. FSDP is unnecessary.**

| Factor | Analysis |
|--------|----------|
| Model size | DINOv2-Large + UperNet ≈ 350M params ≈ 700 MB (fp16) — fits easily on single GPU |
| Memory bottleneck | Logits, not params. Depth-2: ~77 MB/batch. Full: ~2.7 GB/batch (fp16, bs=4) |
| FSDP overhead | Shards params across GPUs — negligible savings for 700 MB model, adds allgather communication |
| DDP simplicity | HF Trainer auto-detects multi-GPU, uses DDP. Zero code changes |
| DDP speedup | 8 GPUs → ~6-7x faster training (not perfect 8x due to communication overhead) |
| If OOM | Reduce `per_device_train_batch_size` or add `gradient_accumulation_steps` via existing `**kwargs` |

**FSDP is only needed if the model doesn't fit on one GPU** (e.g., >20B params). Our model is 350M. The proven FSDP templates in `ml-workflow-tool/training_templates/` remain a reference if we ever switch to larger backbones.

**How to enable multi-GPU DDP**: Just run the notebook on the multi-GPU cluster (g6e.48xlarge, 8x L40S). HF Trainer detects `WORLD_SIZE > 1` and wraps the model in DDP automatically. Expect ~6-7x speedup (depth-2: 4-5 min, full: 5-10 min vs 23-30 min on single GPU).

**How to enable FSDP (if ever needed)**: Pass via kwargs — `get_training_args(fsdp="full_shard auto_wrap", fsdp_config={"fsdp_transformer_layer_cls_to_wrap": "Dinov2Layer"})`. The existing `**kwargs` passthrough handles this.

---

## Phase A: Depth-2 (19 classes) — Multi-GPU

### 1. TDD: Fix `get_class_names()` bug + depth-2 tests

**Bug:** `get_class_names()` picks an arbitrary structure_id per class for non-coarse mappings (depends on set iteration order). For depth-2, class 4 could be named "Cortical plate" instead of "Cerebrum".

**Fix in `src/histological_image_analysis/ontology.py`** (lines 181-186):
Replace "first wins" with "shallowest depth wins":

```python
# Current (buggy):
class_to_sid: dict[int, int] = {}
for sid, cid in mapping.items():
    if cid not in class_to_sid:
        class_to_sid[cid] = sid

# Fixed:
class_to_sid: dict[int, int] = {}
for sid, cid in mapping.items():
    if cid == 0:
        continue  # skip background (already set to "Background" before this loop)
    current = class_to_sid.get(cid)
    if current is None:
        class_to_sid[cid] = sid
    else:
        # Prefer shallowest depth ancestor for naming
        if self._node_by_id[sid]["depth"] < self._node_by_id[current]["depth"]:
            class_to_sid[cid] = sid
```

**Tests first in `tests/test_ontology.py`** — new class `TestGetClassNamesDepthMapping`:

| Test | Asserts |
|------|---------|
| `test_depth2_names_use_ancestor_not_descendant` | `names[mapping[567]] == "Cerebrum"` (not "Cerebral cortex" or "Cortical plate") |
| `test_depth2_names_brainstem` | `names[mapping[343]] == "Brain stem"` (not "Midbrain") |
| `test_depth2_background_is_named` | `names[0] == "Background"` |
| `test_full_mapping_each_structure_named` | Each class gets its own structure's name |
| `test_real_depth2_names` (skip if no real data) | Real ontology: Cerebrum, Brain stem, Cerebellum at depth 2 |

### 2. TDD: Depth-2 integration tests

**New test class `TestDepthMappingIntegration`** in `tests/test_ontology.py`:

| Test | Asserts |
|------|---------|
| `test_depth2_class_count_minimal` | Minimal fixture: 6 depth-2 ancestors + background = 7 total |
| `test_depth2_get_num_labels` | `get_num_labels(mapping) == 7` for minimal fixture |
| `test_real_depth2_class_count` (skip) | Real ontology: 18 depth-2 structures + background = 19 total classes |
| `test_depth2_all_structures_mapped` | Every structure ID has a mapping entry |

### 3. Create notebook `notebooks/step10_finetune_depth2.ipynb`

Clone step9 structure (8 cells), change only:

| Cell | Change |
|------|--------|
| 1 (Config) | `MAPPING_TYPE = "depth2"`, `NUM_LABELS` computed dynamically, output/model dirs → `depth2`, MLflow run name updated |
| 3 (Data) | `mapping = mapper.build_depth_mapping(target_depth=2)` instead of `build_coarse_mapping()`, `NUM_LABELS = mapper.get_num_labels(mapping)` |
| 4 (Model) | `create_model(num_labels=NUM_LABELS, ...)` — already parameterized |
| 6 (Eval) | Use `class_names = mapper.get_class_names(mapping)` for per-class reporting |

**Cluster:** g6e.48xlarge (8x L40S) for speed. Memory: 8 × 19 × 518 × 518 × 2 bytes ≈ 82 MB logits per GPU — trivial. Expected runtime: ~4-5 minutes.

### 4. Deploy and train depth-2

- `make deploy` (wheel + step10 notebook)
- Staged validation: cells 0-4
- Full training: cells 5-7 (expect ~4-5 min on 8x L40S, ~6-7x faster than coarse single-GPU)
- Evaluate: compare mIoU with coarse 88% baseline

**Expected:** mIoU 50-70% (more classes = harder, frozen backbone may need unfreezing for fine distinctions)

---

## Phase B: Full Mapping (1,328 classes) — Multi-GPU

### 5. Create notebook `notebooks/step10_finetune_full.ipynb`

Clone depth-2 notebook, change:

| Cell | Change |
|------|--------|
| 1 (Config) | `MAPPING_TYPE = "full"`, `per_device_train_batch_size = 4` (conservative; could use 8), output dirs → `full` |
| 3 (Data) | `mapping = mapper.build_full_mapping()` |
| 4 (Model) | `NUM_LABELS` will be 1,328 — sanity check logits shape |

**Memory budget per GPU (fp16, batch=4, frozen backbone):**
- Model (fp16): 0.68 GB
- Logits: 2.65 GB
- Gradients (head, fp16): 0.09 GB
- Optimizer (head, fp32): 0.56 GB
- Activations (est): 1.69 GB
- **Total: ~5.7 GB per GPU (12% utilization on L40S 48 GB)**

**Safety margin: 42 GB. No OOM risk.**

**Cluster strategy:**
1. **Default: g6e.48xlarge (8x L40S DDP)** for speed. Each GPU: batch=4, effective batch=32. Training ~6-7x faster (~5-10 min vs ~30+ min on single GPU).
2. **Single GPU fallback:** g6e.16xlarge if multi-GPU cluster unavailable. Still fits comfortably (5.7 GB / 48 GB = 12% utilization).
3. **If OOM (unlikely):** Reduce to batch=2. This would give ~4.3 GB per GPU (9% utilization).

### 6. Deploy and train full mapping

- Deploy wheel + notebook to workspace
- Run on multi-GPU cluster (g6e.48xlarge, 8x L40S) for speed
- Run staged validation, then full training
- Expected runtime: ~5-10 minutes on 8 GPUs
- Expect lower mIoU than coarse (many rare classes with few pixels — this is normal)

### 7. Update Makefile

Add deploy targets for new notebooks:

```makefile
NOTEBOOK_D2_SRC    := notebooks/step10_finetune_depth2.ipynb
NOTEBOOK_D2_DEST   := $(WORKSPACE_BASE)/notebooks/step10_depth2
NOTEBOOK_FULL_SRC  := notebooks/step10_finetune_full.ipynb
NOTEBOOK_FULL_DEST := $(WORKSPACE_BASE)/notebooks/step10_full

deploy-notebook-depth2: ## Upload depth-2 training notebook
	databricks workspace mkdirs $(dir $(NOTEBOOK_D2_DEST)) ...
	databricks workspace import $(NOTEBOOK_D2_DEST) --file $(NOTEBOOK_D2_SRC) ...

deploy-notebook-full: ## Upload full-mapping training notebook
	# same pattern

deploy: deploy-wheel deploy-notebook deploy-notebook-depth2 deploy-notebook-full
```

### 8. Update docs

- `docs/progress.md` — Step 10 entries, results
- `docs/joyful-popping-planet.md` — Update plan tracker (or create new plan doc for step 10)

---

## Files to Create/Modify

| File | Action | What Changes |
|------|--------|--------------|
| `tests/test_ontology.py` | Modify | +9 tests: `TestGetClassNamesDepthMapping`, `TestDepthMappingIntegration` |
| `src/histological_image_analysis/ontology.py` | Modify | Fix `get_class_names()` lines 181-186 (prefer shallowest depth) |
| `notebooks/step10_finetune_depth2.ipynb` | Create | 8 cells, depth-2 mapping (19 classes) |
| `notebooks/step10_finetune_full.ipynb` | Create | 8 cells, full mapping (1,328 classes), conservative batch size |
| `Makefile` | Modify | Add deploy targets for new notebooks |
| `docs/progress.md` | Modify | Step 10 status and results |

**Files NOT modified:** `training.py`, `dataset.py`, `ccfv3_slicer.py`, `pyproject.toml` — all already generic.

---

## Implementation Status

| Step | Task | Status |
|------|------|--------|
| 0 | Update CLAUDE.md with plan-file-location rule | DONE |
| 1 | Write tests for `get_class_names()` fix (5 tests) | DONE — 3 failed as expected (TDD red) |
| 2 | Fix `get_class_names()` in ontology.py | DONE — all 5 tests pass (TDD green) |
| 3 | Write depth-2 integration tests (4 tests) | DONE — all 4 pass |
| 4 | Create depth-2 notebook | DONE — 8 cells (renamed to `finetune_depth2.ipynb`) |
| 5 | Create full-mapping notebook | DONE — 8 cells, batch_size=4 (renamed to `finetune_full.ipynb`) |
| 6 | Update Makefile (deploy targets, help regex) | DONE — 12 targets |
| 7 | Update README.md | DONE — current with all 3 notebooks |
| 8 | `make test` passes | DONE — **133/133 tests pass** |
| 9 | Deploy + train depth-2 | DONE — **69.6% mIoU**, 95.5% accuracy, 25 min on 1x L40S |
| 10 | Review depth-2 results, update docs | DONE |
| 11 | Deploy + train full mapping | DONE — **60.3% mIoU**, 88.9% accuracy, 5.5 hrs on 1x L40S |
| 12 | Review full results, update docs | DONE |
| 13 | Rename notebooks (remove step prefixes) | DONE — `finetune_coarse`, `finetune_depth2`, `finetune_full` |
| 14 | DINOv2 model research | DONE — see `docs/dinov2_model_research.md` |

## Training Results

### Depth-2 (19 classes) — Run 3

- **mIoU: 69.6%**, Overall accuracy: 95.5%
- Training loss: 0.305, Eval loss: 0.194
- Runtime: 25 min (1x L40S, batch=8)
- 15 of 19 classes with valid IoU (4 absent from val: interventricular foramen, Interpeduncular fossa, grooves of cerebral/cerebellar cortex)
- Top: Cerebrum 95.8%, Brain stem 94.5%, Cerebellum 93.0%, Background 91.2%
- Lowest valid: extrapyramidal fiber systems 44.4%, central canal 0.0% (only 44 val pixels)

### Full Mapping (1,328 classes) — Run 4

- **mIoU: 60.3%**, Overall accuracy: 88.9%
- Training loss: 0.994, Eval loss: 0.496
- Runtime: 5.5 hrs (1x L40S, batch=4)
- 503 of 1,328 classes with valid IoU (825 absent from val set)
- Top: Caudoputamen 95.2%, Background 94.3%, Main olfactory bulb 93.8%, Nodulus 92.9%
- Bottom: many tiny structures at 0.0% (Accessory supraoptic group, Frontal pole layer 6b, etc.)
- Multi-GPU OOM'd due to per-GPU DDP overhead — see `docs/step10_gpu_memory_review.md`

### GPU Memory Lessons

Three attempts at full mapping training on different clusters:
1. **Single GPU (batch=4)** — succeeded with ~20 GB headroom
2. **Multi-GPU DDP (batch=4)** — OOM from DDP/Spark process overhead on GPU 0
3. **Single GPU (batch=2 + grad ckpt)** — ValueError (UperNet doesn't support gradient checkpointing)

Key takeaways documented in `docs/step10_gpu_memory_review.md`:
- Single GPU is sufficient for this workload
- Multi-GPU DDP requires reduced batch size (batch=2 + grad_accum=2)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation
- Always `torch.cuda.empty_cache()` after sanity-check forward passes

## Next Steps (for future sessions)

Step 10 training is complete. Recommended next steps (see `docs/dinov2_model_research.md`):

1. **Unfreeze backbone with differential LR** (highest expected impact)
   - Backbone LR: 1e-5, Head LR: 1e-4
   - Already possible via `get_training_args(**kwargs)` — no code changes
   - Risk: overfitting on 1,016 samples. Mitigate with early stopping

2. **Class-weighted cross-entropy loss** (medium impact)
   - Weight rare classes inversely proportional to pixel frequency
   - Addresses 825 classes absent from val and many at 0.0% IoU

3. **More epochs for full mapping** (low cost)
   - 50 epochs may not converge for 1,328-class head
   - Try 100-200 with early stopping

4. **Larger backbone (DINOv2-Giant)** — try AFTER unfreezing backbone
   - 1.1B params, fits on L40S 48 GB (~22-28 GB)
   - Requires new file (`model_config.py`) — `out_features` hardcoded for Large
   - See `docs/dinov2_model_research.md` for full analysis

## Verification

1. `make test` — all existing 124 tests + 9 new tests = **133 tests pass**
2. `get_class_names(build_depth_mapping(2))` returns "Cerebrum" (not descendant name)
3. Depth-2: mIoU 69.6%, 15/19 classes valid — VERIFIED
4. Full mapping: mIoU 60.3%, 503/1,328 classes valid, no OOM on single GPU — VERIFIED
5. Models saved to `/dbfs/FileStore/allen_brain_data/models/{depth2,full}` — VERIFIED
6. MLflow experiments logged with correct metrics — VERIFIED

---

## Reference: LESSONS_LEARNED.md Patterns Applied

From `ml-workflow-tool/training_templates/LESSONS_LEARNED.md`:
- Retry pattern on `snapshot_download()` (already in step9 notebook)
- `etag_timeout=86400` (already applied)
- JFrog Artifactory URL as default (already applied)
- Single `mlflow.start_run()` spanning training+eval+save (already applied)
- `mlflow.end_run()` at notebook end (already applied)
- Checkpoints to `/tmp/`, final model to `/dbfs/` (already applied, avoids 500MB workspace limit)
- If FSDP ever needed: `model.state_dict()` is collective (all ranks must call), unwrap before MLflow export