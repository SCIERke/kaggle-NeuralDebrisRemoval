.PHONY: install install-dev install-detectron2 lint format check visualize notebook zip upload-stage upload upload-new clean help

DATASET_ID := peerapatsetsuk/Neural-Debris-Removal
STAGE_DIR  := .kaggle_upload

# kaggle's CLI has no ignore-file support — it zips everything under -p, so we
# stage a clean copy first to keep secrets (cred/), .git, and .venv out of the dataset.
UPLOAD_EXCLUDES := --exclude=.git --exclude=.venv --exclude=cred --exclude=tmp \
	--exclude=__pycache__ --exclude=.DS_Store --exclude='*.zip' --exclude=$(STAGE_DIR) \
	--exclude=ndr-trial1.ipynb

PYTHON  := uv run python
PIP     := uv pip

install:
	$(PIP) install setuptools==80.9.0
	$(PIP) install torch torchvision --reinstall
	$(PIP) install -r requirements.txt
	$(PIP) install git+https://github.com/facebookresearch/detectron2.git --no-build-isolation

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m black . && $(PYTHON) -m ruff check --fix .

check:
	@for f in poisoned_model/poisoned_model.pth unlearn_set/annotations_coco.json sample_submission.csv ndr_unlearn.ipynb; do \
		test -f $$f && echo "  ✓ $$f" || echo "  ✗ $$f MISSING"; \
	done
	@echo "  Unlearn images: $$(ls unlearn_set/*.png 2>/dev/null | wc -l | tr -d ' ') PNG files"

zip:
	zip -r kaggle_model.zip poisoned_model/
	zip -r kaggle_unlearn.zip unlearn_set/

upload-stage:
	rm -rf $(STAGE_DIR)
	mkdir -p $(STAGE_DIR)
	rsync -a $(UPLOAD_EXCLUDES) ./ $(STAGE_DIR)/

upload: upload-stage
	uv run kaggle datasets version -p $(STAGE_DIR) --dir-mode zip -m "$(or $(MSG),update)"
	rm -rf $(STAGE_DIR)

upload-new: upload-stage
	uv run kaggle datasets create -p $(STAGE_DIR) --dir-mode zip
	rm -rf $(STAGE_DIR)

clean:
	find . \( -type d -name __pycache__ -o -name "*.pyc" \) -delete 2>/dev/null || true
	rm -f kaggle_model.zip kaggle_unlearn.zip

help:
	@echo "install            Install requirements.txt"
	@echo "install-dev        Install + ruff, black, jupyterlab"
	@echo "install-detectron2 Install detectron2 (auto-detects torch/CUDA)"
	@echo "lint               Ruff linter"
	@echo "format             black + ruff --fix"
	@echo "check              Verify required files exist"
	@echo "visualize          Show unlearn set with GT boxes"
	@echo "visualize-save     Same + save to visualize_output.png"
	@echo "notebook           Open notebook in JupyterLab"
	@echo "zip                Package for Kaggle upload"
	@echo "upload             Push new version to Kaggle (MSG=... for commit message)"
	@echo "upload-new         Create dataset on Kaggle for the first time"
	@echo "clean              Remove cache and zip artifacts"
