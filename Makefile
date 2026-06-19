.PHONY: install install-dev install-detectron2 lint format check visualize notebook zip clean help

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
	@echo "clean              Remove cache and zip artifacts"
