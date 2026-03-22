# ── Configuration ──────────────────────────────────────────────────────
-include .env
export

DATABRICKS_PROFILE ?= dev
WORKSPACE_BASE     ?= /Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology
DBFS_WHEEL_DIR     ?= dbfs:/FileStore/wheels
WHEEL_NAME         := histological_image_analysis-0.1.0-py3-none-any.whl
WHEEL_PATH         := dist/$(WHEEL_NAME)
NOTEBOOK_SRC       := notebooks/finetune_coarse.ipynb
NOTEBOOK_DEST      := $(WORKSPACE_BASE)/notebooks/finetune_coarse
NOTEBOOK_D2_SRC    := notebooks/finetune_depth2.ipynb
NOTEBOOK_D2_DEST   := $(WORKSPACE_BASE)/notebooks/finetune_depth2
NOTEBOOK_FULL_SRC  := notebooks/finetune_full.ipynb
NOTEBOOK_FULL_DEST := $(WORKSPACE_BASE)/notebooks/finetune_full
NOTEBOOK_UNFROZEN_SRC  := notebooks/finetune_unfrozen.ipynb
NOTEBOOK_UNFROZEN_DEST := $(WORKSPACE_BASE)/notebooks/finetune_unfrozen
NOTEBOOK_WEIGHTED_SRC  := notebooks/finetune_weighted_loss.ipynb
NOTEBOOK_WEIGHTED_DEST := $(WORKSPACE_BASE)/notebooks/finetune_weighted_loss
NOTEBOOK_AUGMENTED_SRC  := notebooks/finetune_augmented.ipynb
NOTEBOOK_AUGMENTED_DEST := $(WORKSPACE_BASE)/notebooks/finetune_augmented
NOTEBOOK_TTA_SRC        := notebooks/eval_tta.ipynb
NOTEBOOK_TTA_DEST       := $(WORKSPACE_BASE)/notebooks/eval_tta
NOTEBOOK_PRUNED_MA_SRC  := notebooks/finetune_pruned_multiaxis.ipynb
NOTEBOOK_PRUNED_MA_DEST := $(WORKSPACE_BASE)/notebooks/finetune_pruned_multiaxis
NOTEBOOK_ABLATION_SRC   := notebooks/finetune_pruned_ablation.ipynb
NOTEBOOK_ABLATION_DEST  := $(WORKSPACE_BASE)/notebooks/finetune_pruned_ablation
NOTEBOOK_FINAL_SRC      := notebooks/finetune_final_200ep.ipynb
NOTEBOOK_FINAL_DEST     := $(WORKSPACE_BASE)/notebooks/finetune_final_200ep
NOTEBOOK_HUMAN_ALLEN_SRC  := notebooks/finetune_human_allen.ipynb
NOTEBOOK_HUMAN_ALLEN_DEST := $(WORKSPACE_BASE)/notebooks/finetune_human_allen
NOTEBOOK_HUMAN_BB_SRC     := notebooks/finetune_human_bigbrain.ipynb
NOTEBOOK_HUMAN_BB_DEST    := $(WORKSPACE_BASE)/notebooks/finetune_human_bigbrain
NOTEBOOK_HUMAN_ALLEN_D3_SRC  := notebooks/finetune_human_allen_depth3.ipynb
NOTEBOOK_HUMAN_ALLEN_D3_DEST := $(WORKSPACE_BASE)/notebooks/finetune_human_allen_depth3
NOTEBOOK_EVAL_D3_TEST_SRC    := notebooks/eval_human_depth3_test.ipynb
NOTEBOOK_EVAL_D3_TEST_DEST   := $(WORKSPACE_BASE)/notebooks/eval_human_depth3_test
NOTEBOOK_HUMAN_FIGURES_SRC   := notebooks/generate_human_paper_figures.ipynb
NOTEBOOK_HUMAN_FIGURES_DEST  := $(WORKSPACE_BASE)/notebooks/generate_human_paper_figures

.PHONY: install test lint build clean deploy-wheel deploy-notebook deploy-notebook-depth2 deploy-notebook-full deploy-notebook-unfrozen deploy-notebook-weighted-loss deploy-notebook-augmented deploy-notebook-eval-tta deploy-notebook-pruned-multiaxis deploy-notebook-pruned-ablation deploy-notebook-final deploy-notebook-human-allen deploy-notebook-human-allen-depth3 deploy-notebook-human-bigbrain deploy-notebook-eval-depth3-test deploy-notebook-human-figures deploy deploy-human-annotations validate help download-models download-models-mouse download-models-human-allen annotate-mouse annotate-human-allen annotate-human-bigbrain annotate-mouse-sliding annotate-human-allen-sliding annotate-human-bigbrain-sliding upload-models fetch-models-from-dbfs fetch-mouse-from-dbfs fetch-human-from-dbfs

# ── End-User Workflow ─────────────────────────────────────────────────

download-models: ## Download all trained models from HuggingFace Hub (~2.5 GB)
	uv run python scripts/download_models.py --species all

download-models-mouse: ## Download mouse brain model only (~1.2 GB)
	uv run python scripts/download_models.py --species mouse

download-models-human-allen: ## Download human Allen depth-3 brain model — 44 regions (~1.2 GB)
	uv run python scripts/download_models.py --species human

download-models-human-bigbrain: ## Download human BigBrain model — 10 tissue types (~1.2 GB)
	uv run python scripts/download_models.py --species human-bigbrain

annotate-mouse: ## Annotate mouse brain images (guided mode, or set IMAGES=/path/to/folder)
	@if [ -z "$(IMAGES)" ]; then \
		uv run python scripts/annotate.py; \
	else \
		uv run python scripts/annotate.py $(IMAGES); \
	fi

annotate-human-allen: ## Annotate human tissue — Allen depth-3, 44 brain regions (guided mode, or set IMAGES=/path)
	@if [ -z "$(IMAGES)" ]; then \
		uv run python scripts/annotate.py --species human; \
	else \
		uv run python scripts/annotate.py $(IMAGES) --species human; \
	fi

annotate-human-bigbrain: ## Annotate human tissue — 10 tissue types (guided mode, or set IMAGES=/path)
	@if [ -z "$(IMAGES)" ]; then \
		uv run python scripts/annotate.py --species human-bigbrain; \
	else \
		uv run python scripts/annotate.py $(IMAGES) --species human-bigbrain; \
	fi

annotate-mouse-sliding: ## Annotate mouse with sliding window — slower, more accurate (guided or IMAGES=/path)
	@if [ -z "$(IMAGES)" ]; then \
		uv run python scripts/annotate.py --sliding-window; \
	else \
		uv run python scripts/annotate.py $(IMAGES) --sliding-window; \
	fi

annotate-human-allen-sliding: ## Annotate human Allen (44 regions) with sliding window (guided or IMAGES=/path)
	@if [ -z "$(IMAGES)" ]; then \
		uv run python scripts/annotate.py --species human --sliding-window; \
	else \
		uv run python scripts/annotate.py $(IMAGES) --species human --sliding-window; \
	fi

annotate-human-bigbrain-sliding: ## Annotate human (10 tissue types) with sliding window (guided or IMAGES=/path)
	@if [ -z "$(IMAGES)" ]; then \
		uv run python scripts/annotate.py --species human-bigbrain --sliding-window; \
	else \
		uv run python scripts/annotate.py $(IMAGES) --species human-bigbrain --sliding-window; \
	fi

upload-models: ## Upload models to HuggingFace Hub (one-time, requires HUGGING_FACE_TOKEN)
	uv run python scripts/upload_to_hf.py

# ── Databricks Model Download (developer only) ───────────────────────

DBFS_MOUSE_MODEL          ?= dbfs:/FileStore/allen_brain_data/models/final-200ep
DBFS_HUMAN_MODEL          ?= dbfs:/FileStore/allen_brain_data/models/human-allen-depth3
DBFS_HUMAN_BIGBRAIN_MODEL ?= dbfs:/FileStore/allen_brain_data/models/human-bigbrain
LOCAL_MOUSE_MODEL          := ./models/dinov2-upernet-final
LOCAL_HUMAN_MODEL          := ./models/human-depth3
LOCAL_HUMAN_BIGBRAIN_MODEL := ./models/human-bigbrain

fetch-models-from-dbfs: ## Download all models from Databricks DBFS to local (requires Databricks CLI)
	@echo "=== Downloading models from Databricks DBFS ==="
	mkdir -p $(LOCAL_MOUSE_MODEL) $(LOCAL_HUMAN_MODEL) $(LOCAL_HUMAN_BIGBRAIN_MODEL)
	@echo ""
	@echo "--- Mouse model (final-200ep, 1,328 classes) ---"
	databricks fs cp -r $(DBFS_MOUSE_MODEL) $(LOCAL_MOUSE_MODEL) --profile $(DATABRICKS_PROFILE) --overwrite
	@echo ""
	@echo "--- Human model (Allen depth-3, 44 classes) ---"
	databricks fs cp -r $(DBFS_HUMAN_MODEL) $(LOCAL_HUMAN_MODEL) --profile $(DATABRICKS_PROFILE) --overwrite
	@echo ""
	@echo "--- Human BigBrain model (tissue classification, 10 classes) ---"
	databricks fs cp -r $(DBFS_HUMAN_BIGBRAIN_MODEL) $(LOCAL_HUMAN_BIGBRAIN_MODEL) --profile $(DATABRICKS_PROFILE) --overwrite
	@echo ""
	@echo "=== Download complete ==="
	@echo "  Mouse:          $(LOCAL_MOUSE_MODEL)"
	@echo "  Human:          $(LOCAL_HUMAN_MODEL)"
	@echo "  Human BigBrain: $(LOCAL_HUMAN_BIGBRAIN_MODEL)"
	@echo ""
	@echo "Next: make upload-models"

fetch-mouse-from-dbfs: ## Download mouse model from Databricks DBFS
	mkdir -p $(LOCAL_MOUSE_MODEL)
	databricks fs cp -r $(DBFS_MOUSE_MODEL) $(LOCAL_MOUSE_MODEL) --profile $(DATABRICKS_PROFILE) --overwrite
	@echo "Mouse model downloaded to $(LOCAL_MOUSE_MODEL)"

fetch-human-from-dbfs: ## Download human (Allen depth-3) model from Databricks DBFS
	mkdir -p $(LOCAL_HUMAN_MODEL)
	databricks fs cp -r $(DBFS_HUMAN_MODEL) $(LOCAL_HUMAN_MODEL) --profile $(DATABRICKS_PROFILE) --overwrite
	@echo "Human model downloaded to $(LOCAL_HUMAN_MODEL)"

fetch-human-bigbrain-from-dbfs: ## Download human (BigBrain tissue) model from Databricks DBFS
	mkdir -p $(LOCAL_HUMAN_BIGBRAIN_MODEL)
	databricks fs cp -r $(DBFS_HUMAN_BIGBRAIN_MODEL) $(LOCAL_HUMAN_BIGBRAIN_MODEL) --profile $(DATABRICKS_PROFILE) --overwrite
	@echo "Human BigBrain model downloaded to $(LOCAL_HUMAN_BIGBRAIN_MODEL)"

# ── Local Development ──────────────────────────────────────────────────

install: ## Install all dependencies (including dev)
	uv sync --all-extras

test: ## Run full test suite
	uv run pytest tests/ -v

lint: ## Run linter
	uv run ruff check src/ tests/

build: clean ## Build wheel
	uv build
	@echo "Built: $(WHEEL_PATH)"

clean: ## Remove build artifacts
	rm -rf dist/ build/ src/*.egg-info

# ── Databricks Deployment ──────────────────────────────────────────────

deploy-wheel: build ## Build and upload wheel to DBFS
	databricks fs mkdirs $(DBFS_WHEEL_DIR) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks fs cp $(WHEEL_PATH) $(DBFS_WHEEL_DIR)/$(WHEEL_NAME) --overwrite --profile $(DATABRICKS_PROFILE)
	@echo "Wheel uploaded to $(DBFS_WHEEL_DIR)/$(WHEEL_NAME)"

deploy-notebook: ## Upload training notebook to Databricks workspace
	databricks workspace mkdirs $(dir $(NOTEBOOK_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_DEST) \
		--file $(NOTEBOOK_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_DEST)"

deploy-notebook-depth2: ## Upload depth-2 training notebook to workspace
	databricks workspace mkdirs $(dir $(NOTEBOOK_D2_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_D2_DEST) \
		--file $(NOTEBOOK_D2_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_D2_DEST)"

deploy-notebook-full: ## Upload full-mapping training notebook to workspace
	databricks workspace mkdirs $(dir $(NOTEBOOK_FULL_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_FULL_DEST) \
		--file $(NOTEBOOK_FULL_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_FULL_DEST)"

deploy-notebook-unfrozen: ## Upload unfrozen-backbone training notebook to workspace
	databricks workspace mkdirs $(dir $(NOTEBOOK_UNFROZEN_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_UNFROZEN_DEST) \
		--file $(NOTEBOOK_UNFROZEN_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_UNFROZEN_DEST)"

deploy-notebook-weighted-loss: ## Upload weighted-loss training notebook to workspace
	databricks workspace mkdirs $(dir $(NOTEBOOK_WEIGHTED_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_WEIGHTED_DEST) \
		--file $(NOTEBOOK_WEIGHTED_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_WEIGHTED_DEST)"

deploy-notebook-augmented: ## Upload augmented training notebook to workspace
	databricks workspace mkdirs $(dir $(NOTEBOOK_AUGMENTED_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_AUGMENTED_DEST) \
		--file $(NOTEBOOK_AUGMENTED_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_AUGMENTED_DEST)"

deploy-notebook-eval-tta: ## Upload TTA evaluation notebook to workspace
	databricks workspace mkdirs $(dir $(NOTEBOOK_TTA_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_TTA_DEST) \
		--file $(NOTEBOOK_TTA_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_TTA_DEST)"

deploy-notebook-pruned-multiaxis: ## Upload pruned+multi-axis training notebook (Run 8)
	databricks workspace mkdirs $(dir $(NOTEBOOK_PRUNED_MA_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_PRUNED_MA_DEST) \
		--file $(NOTEBOOK_PRUNED_MA_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_PRUNED_MA_DEST)"

deploy-notebook-pruned-ablation: ## Upload pruned ablation notebook (Run 8a/8b with diagnostics)
	databricks workspace mkdirs $(dir $(NOTEBOOK_ABLATION_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_ABLATION_DEST) \
		--file $(NOTEBOOK_ABLATION_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_ABLATION_DEST)"

deploy-notebook-final: ## Upload final 200-epoch training notebook (Run 9)
	databricks workspace mkdirs $(dir $(NOTEBOOK_FINAL_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_FINAL_DEST) \
		--file $(NOTEBOOK_FINAL_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_FINAL_DEST)"

deploy-notebook-human-allen: ## Upload Allen Human training notebook (Track A)
	databricks workspace mkdirs $(dir $(NOTEBOOK_HUMAN_ALLEN_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_HUMAN_ALLEN_DEST) \
		--file $(NOTEBOOK_HUMAN_ALLEN_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_HUMAN_ALLEN_DEST)"

deploy-notebook-human-bigbrain: ## Upload BigBrain 9-class training notebook (Track B)
	databricks workspace mkdirs $(dir $(NOTEBOOK_HUMAN_BB_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_HUMAN_BB_DEST) \
		--file $(NOTEBOOK_HUMAN_BB_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_HUMAN_BB_DEST)"

deploy-notebook-human-allen-depth3: ## Upload Allen Human depth-3 training notebook (Track A-depth3)
	databricks workspace mkdirs $(dir $(NOTEBOOK_HUMAN_ALLEN_D3_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_HUMAN_ALLEN_D3_DEST) \
		--file $(NOTEBOOK_HUMAN_ALLEN_D3_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_HUMAN_ALLEN_D3_DEST)"

deploy-notebook-eval-depth3-test: ## Upload depth-3 test evaluation notebook
	databricks workspace mkdirs $(dir $(NOTEBOOK_EVAL_D3_TEST_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_EVAL_D3_TEST_DEST) \
		--file $(NOTEBOOK_EVAL_D3_TEST_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_EVAL_D3_TEST_DEST)"

deploy-notebook-human-figures: ## Upload human paper figure generation notebook
	databricks workspace mkdirs $(dir $(NOTEBOOK_HUMAN_FIGURES_DEST)) --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(NOTEBOOK_HUMAN_FIGURES_DEST) \
		--file $(NOTEBOOK_HUMAN_FIGURES_SRC) \
		--format JUPYTER \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo "Notebook uploaded to $(NOTEBOOK_HUMAN_FIGURES_DEST)"

deploy-human-annotations: ## Upload human annotation data to Databricks workspace
	@echo "=== Uploading human annotation data ==="
	@echo ""
	@echo "--- Human SVG annotations (4,463 files) ---"
	databricks workspace mkdirs $(WORKSPACE_BASE)/human_atlas/svgs --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import-dir data/allen_brain_data/human_atlas/svgs/ \
		$(WORKSPACE_BASE)/human_atlas/svgs/ \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo ""
	@echo "--- Human ontologies ---"
	databricks workspace mkdirs $(WORKSPACE_BASE)/ontology --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import $(WORKSPACE_BASE)/ontology/structure_graph_10.json \
		--file data/allen_brain_data/ontology/structure_graph_10.json \
		--format AUTO --overwrite --profile $(DATABRICKS_PROFILE)
	databricks workspace import $(WORKSPACE_BASE)/ontology/structure_graph_16.json \
		--file data/allen_brain_data/ontology/structure_graph_16.json \
		--format AUTO --overwrite --profile $(DATABRICKS_PROFILE)
	@echo ""
	@echo "--- Developing human atlas (169 images + SVGs) ---"
	databricks workspace mkdirs $(WORKSPACE_BASE)/developing_human_atlas --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import-dir data/allen_brain_data/developing_human_atlas/ \
		$(WORKSPACE_BASE)/developing_human_atlas/ \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	databricks workspace import $(WORKSPACE_BASE)/metadata/developing_human_atlas_metadata.json \
		--file data/allen_brain_data/metadata/developing_human_atlas_metadata.json \
		--format AUTO --overwrite --profile $(DATABRICKS_PROFILE)
	@echo ""
	@echo "--- BigBrain volumes ---"
	databricks workspace mkdirs $(WORKSPACE_BASE)/bigbrain --profile $(DATABRICKS_PROFILE) 2>/dev/null || true
	databricks workspace import-dir data/bigbrain/ \
		$(WORKSPACE_BASE)/bigbrain/ \
		--overwrite \
		--profile $(DATABRICKS_PROFILE)
	@echo ""
	@echo "=== Human annotation upload complete ==="
	@echo "Uploaded to: $(WORKSPACE_BASE)"
	@echo "  - human_atlas/svgs/          (4,463 SVG annotations)"
	@echo "  - ontology/structure_graph_10.json (human adult, 1,839 structures)"
	@echo "  - ontology/structure_graph_16.json (developing human, 3,317 structures)"
	@echo "  - developing_human_atlas/    (169 images + 169 SVGs)"
	@echo "  - bigbrain/                  (NIfTI volumes + layer segmentation)"

deploy: deploy-wheel deploy-notebook deploy-notebook-depth2 deploy-notebook-full deploy-notebook-unfrozen deploy-notebook-weighted-loss deploy-notebook-augmented deploy-notebook-eval-tta deploy-notebook-pruned-multiaxis deploy-notebook-pruned-ablation deploy-notebook-final deploy-notebook-human-allen deploy-notebook-human-allen-depth3 deploy-notebook-human-bigbrain deploy-notebook-eval-depth3-test deploy-notebook-human-figures ## Full deployment (wheel + all notebooks)
	@echo ""
	@echo "=== Deployment complete ==="
	@echo ""
	@echo "Notebooks deployed:"
	@echo "  - Coarse (6 classes):           $(NOTEBOOK_DEST)"
	@echo "  - Depth-2 (19 classes):         $(NOTEBOOK_D2_DEST)"
	@echo "  - Full (1,328 classes, frozen):  $(NOTEBOOK_FULL_DEST)"
	@echo "  - Full (1,328 classes, unfrozen): $(NOTEBOOK_UNFROZEN_DEST)"
	@echo "  - Full (weighted Dice+CE loss):   $(NOTEBOOK_WEIGHTED_DEST)"
	@echo "  - Full (extended augmentation):   $(NOTEBOOK_AUGMENTED_DEST)"
	@echo "  - TTA evaluation (Run 5):         $(NOTEBOOK_TTA_DEST)"
	@echo "  - Pruned + multi-axis (Run 8):    $(NOTEBOOK_PRUNED_MA_DEST)"
	@echo "  - Pruned ablation (Run 8a/8b):    $(NOTEBOOK_ABLATION_DEST)"
	@echo "  - Final 200-epoch (Run 9):        $(NOTEBOOK_FINAL_DEST)"
	@echo "  - Human Allen (Track A):          $(NOTEBOOK_HUMAN_ALLEN_DEST)"
	@echo "  - Human Allen depth-3 (Track A):  $(NOTEBOOK_HUMAN_ALLEN_D3_DEST)"
	@echo "  - Human BigBrain (Track B):       $(NOTEBOOK_HUMAN_BB_DEST)"
	@echo "  - Depth-3 Test Eval:              $(NOTEBOOK_EVAL_D3_TEST_DEST)"
	@echo "  - Human Paper Figures:            $(NOTEBOOK_HUMAN_FIGURES_DEST)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Open a notebook on Databricks"
	@echo "  2. Attach to a single-GPU cluster (L40S 48 GB recommended)"
	@echo "  3. Run cells 0-4 for staged validation"
	@echo "  4. Run cells 5-7 for full training"
	@echo ""
	@echo "Note: For multi-GPU DDP, use batch=2 + gradient_accumulation_steps=2"
	@echo "      to avoid OOM from per-GPU overhead. See docs/step10_gpu_memory_review.md"

validate: ## Print staged validation checklist
	@echo "=== Staged Validation Checklist ==="
	@echo ""
	@echo "Cluster: Single-GPU (L40S 48 GB) recommended"
	@echo ""
	@echo "Notebooks:"
	@echo "  - Coarse:          $(NOTEBOOK_DEST)"
	@echo "  - Depth-2:         $(NOTEBOOK_D2_DEST)"
	@echo "  - Full (frozen):   $(NOTEBOOK_FULL_DEST)"
	@echo "  - Full (unfrozen): $(NOTEBOOK_UNFROZEN_DEST)"
	@echo ""
	@echo "Run cells one at a time. Stop if any cell fails."
	@echo ""
	@echo "  Cell 0 - Install wheel:"
	@echo "    [ ] %pip install succeeds (no dependency errors)"
	@echo "    [ ] Python kernel restarts cleanly"
	@echo ""
	@echo "  Cell 1 - Configuration:"
	@echo "    [ ] All paths print correctly"
	@echo "    [ ] HF_ENDPOINT shows expected value"
	@echo "    [ ] Mapping type and batch size shown"
	@echo ""
	@echo "  Cell 2 - Model download:"
	@echo "    [ ] snapshot_download completes (JFrog or HuggingFace)"
	@echo "    [ ] Model path printed and non-empty"
	@echo ""
	@echo "  Cell 3 - Data pipeline:"
	@echo "    [ ] OntologyMapper loads correct number of classes"
	@echo "    [ ] CCFv3Slicer.load_volumes() succeeds"
	@echo "    [ ] Train/val split numbers printed"
	@echo "    [ ] Sample pixel_values shape: (3, 518, 518)"
	@echo "    [ ] Sample labels shape: (518, 518)"
	@echo ""
	@echo "  Cell 4 - Model creation + forward pass:"
	@echo "    [ ] create_model() succeeds"
	@echo "    [ ] torch.cuda.is_available() = True"
	@echo "    [ ] Logits shape: (1, NUM_LABELS, 518, 518)"
	@echo "    [ ] 'Forward pass OK' printed"
	@echo ""
	@echo "All 4 cells pass => environment is ready."
	@echo "Proceed to cells 5-7 for full training."

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
