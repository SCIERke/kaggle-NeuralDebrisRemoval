"""
Full pipeline: prune -> EWC fine-tune -> demotion-based inference -> submission,
adapting ideas from a stronger public reference solution — EWC-anchored
fine-tuning (see approach/prune_and_finetune.py) instead of distillation, and
confidence-drop + geometry-based demotion post-processing (see
approach/postprocess.py) instead of trusting the de-poisoned model's raw
output directly.

Also runs a local sanity check (helpers/macadd.py) on the 20 unlearn images
before writing the submission, assuming an empty reference there (the
correct target for those images) — not the same computation the real
leaderboard uses (that needs the hidden clean model), but enough to
sanity-check that de-poisoning had a real effect before spending one of the
2 daily submissions.

See kaggle/README.md for how to run this on Kaggle (same setup as submit.py).
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch

if "__file__" in globals():
    _HERE = Path(__file__).resolve().parent.parent
else:
    _HERE = Path(globals()["CODE_PATH"])
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config.config import settings, UNLEARN_SET_PATH, TEST_SET_PATH, FINETUNED_STATE_PATH

try:
    settings.validate_paths()
except ValueError as exc:
    raise SystemExit(f"Environment validation failed — fix paths before running:\n{exc}") from exc

from utils.loader import build_cfg, build_predictor, load_image
from helpers.macadd import macadd
from approach.optimal_grow_prune import inject_channel_prune_hook
from approach.prune_and_finetune import prune_and_finetune, _load_grow_indexes_from_sweep
from approach.postprocess import fit_poison_geometry, demote_predictions, DEFAULT_REMAP

OUTPUT_CSV = "/kaggle/working/submission.csv"
CANDIDATE_SCORE_THRESH = 0.05  # low threshold for high-recall candidates, per the reference solution
LAYER_IDX = 6


def build_low_threshold_predictor(
    grow_indexes: list[int], finetuned_state_path: str = FINETUNED_STATE_PATH, layer_idx: int = LAYER_IDX
):
    """Rebuild a predictor at a low score threshold (needed so demotion can
    see and compare low-confidence candidates), with the pruning hook and
    saved fine-tuned weights applied."""
    cfg = build_cfg(score_thresh=CANDIDATE_SCORE_THRESH)
    predictor = build_predictor(cfg)
    state = torch.load(finetuned_state_path, map_location="cpu")
    predictor.model.head.cls_subnet[layer_idx].load_state_dict(state)
    inject_channel_prune_hook(predictor.model, layer_idx=layer_idx, grow_indexes=grow_indexes)
    return predictor


def run_pipeline():
    grow_indexes = _load_grow_indexes_from_sweep()

    print("=== Step 1: prune + EWC fine-tune ===")
    prune_and_finetune(grow_indexes=grow_indexes)

    print("\n=== Step 2: local sanity check on unlearn_set (empty-reference) ===")
    poisoned_predictor = build_predictor(build_cfg(score_thresh=CANDIDATE_SCORE_THRESH))
    depois_predictor = build_low_threshold_predictor(grow_indexes)

    unlearn_paths = sorted(Path(UNLEARN_SET_PATH).glob("*.png"))
    poisoned_preds, depois_preds = {}, {}
    for path in unlearn_paths:
        image = load_image(str(path))
        out_p = poisoned_predictor(image)["instances"]
        out_d = depois_predictor(image)["instances"]
        poisoned_preds[path.stem] = (out_p.pred_boxes.tensor.cpu().numpy(), out_p.scores.cpu().numpy())
        depois_preds[path.stem] = (out_d.pred_boxes.tensor.cpu().numpy(), out_d.scores.cpu().numpy())

    empty_ref = {k: (np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)) for k in poisoned_preds}
    score_raw = macadd(empty_ref, poisoned_preds)
    score_dep = macadd(empty_ref, depois_preds)
    print(f"  local unlearning maCADD — poisoned model:    {score_raw:.4f}")
    print(f"  local unlearning maCADD — de-poisoned model: {score_dep:.4f}")
    if score_raw > 0:
        print(f"  reduction: {100 * (score_raw - score_dep) / score_raw:.2f}%")
    print("  (this uses an empty reference on unlearn_set only — NOT the real "
          "leaderboard computation, which needs the hidden clean model)")

    print("\n=== Step 3: demotion-based inference on full test set ===")
    mu, cov_inv = fit_poison_geometry(UNLEARN_SET_PATH)
    test_paths = sorted(Path(TEST_SET_PATH).rglob("*.png"))
    print(f"  running on {len(test_paths)} test images")

    predictions = {}
    for path in test_paths:
        image_id = int(path.stem)
        image = load_image(str(path))

        cand_out = poisoned_predictor(image)["instances"]
        cand_boxes = cand_out.pred_boxes.tensor.cpu().numpy()
        cand_scores = cand_out.scores.cpu().numpy()

        if len(cand_boxes) == 0:
            predictions[image_id] = ""
            continue

        dep_out = depois_predictor(image)["instances"]
        dep_boxes = dep_out.pred_boxes.tensor.cpu().numpy()
        dep_scores = dep_out.scores.cpu().numpy()

        final_boxes, final_scores = demote_predictions(
            cand_boxes, cand_scores, dep_boxes, dep_scores, mu, cov_inv, DEFAULT_REMAP
        )

        parts = []
        for score, (x1, y1, x2, y2) in zip(final_scores, final_boxes):
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0 and score > 0:
                parts.append(f"{score:.6f} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f}")
        predictions[image_id] = " ".join(parts)

    sorted_ids = sorted(predictions.keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "image_id", "prediction_string"])
        for row_id, image_id in enumerate(sorted_ids):
            # competition requires a literal " " for no-detection rows —
            # an empty string is treated as null by Kaggle's csv parser.
            writer.writerow([row_id, image_id, predictions[image_id] or " "])
    total_boxes = sum(len(p.split()) // 5 for p in predictions.values())
    print(f"\nSubmission written -> {OUTPUT_CSV} ({len(sorted_ids)} rows, {total_boxes} predictions total)")


if __name__ == "__main__":
    run_pipeline()
