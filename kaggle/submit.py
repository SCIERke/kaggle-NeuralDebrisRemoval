"""
Kaggle submission script for Neural Debris Removal (ESA unlearning competition).

See kaggle/README.md for the full run-on-Kaggle instructions (upload, path
setup, detectron2 install, and the recommended pre-submission k sweep).
"""

import json
import os
import sys
from pathlib import Path

import torch

# --- resolve code root so relative imports work on Kaggle ---
# When run via exec(open(...).read()) (as on Kaggle), __file__ is undefined,
# so fall back to CODE_PATH set by the caller before the exec().
if "__file__" in globals():
    _HERE = Path(__file__).resolve().parent.parent
else:
    _HERE = Path(globals()["CODE_PATH"])
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config.config import settings, UNLEARN_SET_PATH, TEST_SET_PATH, BEST_K_PATH

try:
    settings.validate_paths()
except ValueError as exc:
    raise SystemExit(f"Environment validation failed — fix paths before running:\n{exc}") from exc

print("=== Environment validated ===")
print(f"  POISONED_MODEL_PATH = {settings.poisoned_model_path}")
print(f"  UNLEARN_SET_PATH    = {settings.unlearn_set_path}")
print(f"  TEST_SET_PATH       = {settings.test_set_path}")
print(f"  CUDA available      = {torch.cuda.is_available()}\n")
from utils.loader import build_cfg, build_predictor, load_image
from helpers.diagnostics import sample_images, detection_stats
from helpers.submission import write_submission
from approach.optimal_grow_prune import (
    optimal_top_grow_indexes_kmean,
    optimal_top_grow_indexes_kfrequency,
    top_grow_indexes_kmean,
    top_grow_indexes_kfrequency,
    inference_model_with_grow_indexes,
    inject_channel_prune_hook,
    eval as unlearn_eval,
)

OUTPUT_CSV = "/kaggle/working/submission.csv"
COMPARISON_SAMPLE_SIZE = 500
MIN_RETENTION = 0.5  # warn if pruned detect_rate drops below this fraction of baseline


def predict_all(test_dir: str, prune_indices: list[int]) -> dict[int, str]:
    """Run the pruned model on every image in test_dir, return {image_id: prediction_string}."""
    cfg = build_cfg()
    predictor = build_predictor(cfg)
    inject_channel_prune_hook(predictor.model, layer_idx=6, grow_indexes=prune_indices)

    image_paths = sorted(Path(test_dir).rglob("*.png"))
    print(f"Found {len(image_paths)} test images under {test_dir}")

    results = {}
    for path in image_paths:
        image_id = int(path.stem)
        image    = load_image(str(path))
        output   = predictor(image)

        instances = output["instances"]
        if len(instances) == 0:
            results[image_id] = ""
            continue

        boxes  = instances.pred_boxes.tensor.cpu()
        scores = instances.scores.cpu()
        parts  = []
        for score, (x1, y1, x2, y2) in zip(scores, boxes):
            w = x2 - x1
            h = y2 - y1
            parts.append(f"{score:.6f} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f}")

        results[image_id] = " ".join(parts)

    return results


# ── Step 1: pick prune channels — use the sweep's answer if one exists,
#    otherwise fall back to the gradient-based optimizer. ────────────────────

print("=== Finding prune channels ===")

if Path(BEST_K_PATH).is_file():
    with open(BEST_K_PATH) as f:
        best = json.load(f)["best"]
    best_label = best["method"]
    k = best["k"]
    print(f"Found sweep result at {BEST_K_PATH}: method={best_label}  k={k}  "
          f"unlearn_silence={best['unlearn_silence']:.2%}  "
          f"retention={best['retention_vs_baseline']:.2%}")
    ranker = top_grow_indexes_kmean if best_label == "kmean" else top_grow_indexes_kfrequency
    best_indices = ranker(k)
else:
    print(f"No sweep result found at {BEST_K_PATH} — run approach/sweep_prune_k.py "
          f"first to pick k without spending a submission. Falling back to the "
          f"gradient-based optimizer for now.\n")

    kmean_indices = optimal_top_grow_indexes_kmean()
    kmean_score   = unlearn_eval(inference_model_with_grow_indexes(kmean_indices.tolist()))
    print(f"KMean   unlearn score: {kmean_score:.2f}")

    kfreq_indices = optimal_top_grow_indexes_kfrequency()
    kfreq_score   = unlearn_eval(inference_model_with_grow_indexes(kfreq_indices.tolist()))
    print(f"KFreq   unlearn score: {kfreq_score:.2f}")

    best_indices = kmean_indices if kmean_score >= kfreq_score else kfreq_indices
    best_label   = "kmean" if kmean_score >= kfreq_score else "kfreq"

print(f"\nUsing {best_label} channels for submission")

# ── Step 1.5: sanity-check the chosen channels against real (unlabeled) test
#    images before spending a submission — catches full-model collapse, which
#    is what a high unlearn score can hide (see approach/sweep_prune_k.py). ──

print("\n=== Comparing normal vs pruned model on a test-set sample ===")

_cfg = build_cfg()
_predictor = build_predictor(_cfg)
_sample = sample_images(TEST_SET_PATH, COMPARISON_SAMPLE_SIZE)

baseline_stats = detection_stats(_predictor, _sample)
print(f"  normal  detect_rate={baseline_stats['detect_rate']:.2%}  "
      f"mean_conf={baseline_stats['mean_conf']:.3f}  "
      f"mean_count={baseline_stats['mean_count']:.2f}")

_handle = inject_channel_prune_hook(_predictor.model, layer_idx=6, grow_indexes=best_indices.tolist())
try:
    pruned_stats = detection_stats(_predictor, _sample)
finally:
    _handle.remove()

retention = (
    pruned_stats["detect_rate"] / baseline_stats["detect_rate"]
    if baseline_stats["detect_rate"] > 0 else 0.0
)
print(f"  pruned  detect_rate={pruned_stats['detect_rate']:.2%}  "
      f"mean_conf={pruned_stats['mean_conf']:.3f}  "
      f"mean_count={pruned_stats['mean_count']:.2f}")
print(f"  retention vs normal: {retention:.2%}")

if retention < MIN_RETENTION:
    print(
        f"\n  WARNING: pruned model retains only {retention:.2%} of the normal "
        f"model's detection rate on real test images — this looks like model "
        f"collapse, not targeted unlearning. Consider a smaller k (see "
        f"approach/sweep_prune_k.py) before submitting.\n"
    )

# ── Step 2: run pruned model on full test set and write submission ───────────

print("\n=== Running inference on test set ===")
predictions = predict_all(TEST_SET_PATH, best_indices.tolist())
write_submission(predictions, OUTPUT_CSV)
