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

.PHONY: install test lint build clean deploy-wheel deploy-notebook deploy-notebook-depth2 deploy-notebook-full deploy-notebook-unfrozen deploy-notebook-weighted-loss deploy-notebook-augmented deploy validate help

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

deploy: deploy-wheel deploy-notebook deploy-notebook-depth2 deploy-notebook-full deploy-notebook-unfrozen deploy-notebook-weighted-loss deploy-notebook-augmented ## Full deployment (wheel + all notebooks)
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
