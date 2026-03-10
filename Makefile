# ── Configuration ──────────────────────────────────────────────────────
-include .env
export

DATABRICKS_PROFILE ?= dev
WORKSPACE_BASE     ?= /Workspace/Users/noel.nosse@grainger.com/visual-model-ft/histology
DBFS_WHEEL_DIR     ?= dbfs:/FileStore/wheels
WHEEL_NAME         := histological_image_analysis-0.1.0-py3-none-any.whl
WHEEL_PATH         := dist/$(WHEEL_NAME)
NOTEBOOK_SRC       := notebooks/step9_finetune_coarse.ipynb
NOTEBOOK_DEST      := $(WORKSPACE_BASE)/notebooks/step9_finetune_coarse

.PHONY: install test lint build clean deploy-wheel deploy-notebook deploy validate help

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

deploy: deploy-wheel deploy-notebook ## Full deployment (wheel + notebook)
	@echo ""
	@echo "=== Deployment complete ==="
	@echo ""
	@echo "Next steps:"
	@echo "  1. Open notebook at $(NOTEBOOK_DEST) on Databricks"
	@echo "  2. Attach to cluster 0306-215929-ai2l0t8w"
	@echo "  3. Run cells 0-4 for staged validation"
	@echo "  4. Run 'make validate' for the full checklist"

validate: ## Print staged validation checklist
	@echo "=== Staged Validation Checklist ==="
	@echo ""
	@echo "Cluster: 0306-215929-ai2l0t8w (g6e.16xlarge, L40S 48GB)"
	@echo "Notebook: $(NOTEBOOK_DEST)"
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
	@echo "    [ ] ONTOLOGY_PATH = .../ontology/structure_graph_1.json"
	@echo ""
	@echo "  Cell 2 - Model download:"
	@echo "    [ ] snapshot_download completes (JFrog or HuggingFace)"
	@echo "    [ ] Model path printed and non-empty"
	@echo ""
	@echo "  Cell 3 - Data pipeline:"
	@echo "    [ ] OntologyMapper loads (6 coarse classes)"
	@echo "    [ ] CCFv3Slicer.load_volumes() succeeds"
	@echo "    [ ] Train/val split numbers printed"
	@echo "    [ ] Sample pixel_values shape: (3, 518, 518)"
	@echo "    [ ] Sample labels shape: (518, 518)"
	@echo ""
	@echo "  Cell 4 - Model creation + forward pass:"
	@echo "    [ ] create_model() succeeds"
	@echo "    [ ] torch.cuda.is_available() = True"
	@echo "    [ ] Logits shape: (1, 6, 518, 518)"
	@echo "    [ ] 'Forward pass OK' printed"
	@echo ""
	@echo "All 4 cells pass => environment is ready."
	@echo "Proceed to cells 5-7 for full training."

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
