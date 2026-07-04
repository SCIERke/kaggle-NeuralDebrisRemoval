"""
Diagnostic: do the channels that respond to the poison also respond to real
streak detections, or are they separable?

If the top channels ranked by "fires inside the poison box" (unlearn_set)
heavily overlap with the top channels ranked by "fires inside the model's
own detected box" on real test images, then poison and legitimate detection
are using the same circuitry — a static per-channel mask can never tell them
apart, no matter how the ranking formula is tweaked, and channel pruning has
a hard ceiling for this task.

If they don't overlap much, the ranking CAN likely be sharpened to target
poison-only channels (see the "compare against real test images" idea from
the k-sweep discussion).

No ground truth needed for the real-detection side — it uses the model's
own predicted box as the region of interest; we only need to know "did it
detect something and where", not whether that detection is correct.

See kaggle/README.md for how to run this on Kaggle (same setup as submit.py).
"""

import math
import sys
from pathlib import Path

import torch

if "__file__" in globals():
    _HERE = Path(__file__).resolve().parent.parent
else:
    _HERE = Path(globals()["CODE_PATH"])
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config.config import TEST_SET_PATH
from utils.loader import build_cfg, build_predictor, load_image
from utils.mapper import annotation_to_layer_corr
from helpers.get_score_matrix import get_score_matrix
from helpers.diagnostics import sample_images

LAYER_IDX = 6
N_TEST_SAMPLE = 100
K_VALUES = [8, 16, 32, 48, 64, 96, 128]


def real_detection_score_matrix(
    predictor, model, image_paths: list[Path], layer_idx: int = LAYER_IDX, return_paths: bool = False
):
    """Same inside-vs-outside per-channel scoring as get_score_matrix(), but
    using the model's own top predicted box on real test images instead of
    the poison annotation. Images with zero detections are skipped.

    return_paths=True also returns the subset of image_paths that had a
    detection, aligned row-for-row with the returned score matrix — needed
    to attribute a score back to a specific image (e.g. for flagging
    suspected-poisoned test images)."""
    activations = []

    def hook(module, inp, out):
        activations.append(out.detach())

    handle = model.head.cls_subnet[layer_idx].register_forward_hook(hook)
    scores = []
    used_paths = []
    try:
        for path in image_paths:
            activations.clear()
            image = load_image(str(path))
            output = predictor(image)
            instances = output["instances"]
            if len(instances) == 0:
                continue
            p3 = activations[0]

            top_idx = instances.scores.argmax().item()
            x1, y1, x2, y2 = instances.pred_boxes.tensor[top_idx].tolist()
            fake_annotation = [{"bbox": [x1, y1, x2 - x1, y2 - y1]}]
            x, y, w, h = annotation_to_layer_corr(fake_annotation, p3, image)

            inside = p3[0, :, math.floor(y):math.ceil(y + h), math.floor(x):math.ceil(x + w)]
            all_sum = p3[0].sum(dim=(1, 2))
            inside_sum = inside.sum(dim=(1, 2))
            diff_count = (p3.shape[3] * p3.shape[2]) - (inside.shape[1] * inside.shape[2])
            outside_mean = (all_sum - inside_sum) / diff_count
            inside_mean = inside.mean(dim=(1, 2))
            scores.append(inside_mean - outside_mean)
            used_paths.append(path)
    finally:
        handle.remove()

    if not scores:
        raise RuntimeError(
            "No detections found on any sampled test image — can't compute "
            "the real-detection score matrix. Try a larger sample size."
        )
    score_matrix = torch.stack(scores, dim=0)
    if return_paths:
        return score_matrix, used_paths
    return score_matrix


def channel_overlap(poison_ranking: torch.Tensor, real_ranking: torch.Tensor, k: int) -> tuple[float, list[int]]:
    poison_top = set(torch.topk(poison_ranking, k).indices.tolist())
    real_top = set(torch.topk(real_ranking, k).indices.tolist())
    shared = poison_top & real_top
    return len(shared) / k, sorted(shared)


def run_diagnosis(n_test_sample: int = N_TEST_SAMPLE, k_values: list[int] = K_VALUES):
    cfg = build_cfg()
    predictor = build_predictor(cfg)
    model = predictor.model

    print("=== Ranking channels by poison-box activation (unlearn_set) ===")
    poison_ranking = torch.mean(get_score_matrix(), dim=0)

    print("=== Ranking channels by real-detection activation (test_set, model's own boxes) ===")
    test_images = sample_images(TEST_SET_PATH, n_test_sample)
    real_scores = real_detection_score_matrix(predictor, model, test_images)
    real_ranking = torch.mean(real_scores, dim=0)
    print(f"  used {real_scores.shape[0]}/{len(test_images)} sampled test images that had a detection\n")

    print(f"{'k':>5}{'overlap':>10}   shared channels")
    overlaps = []
    for k in k_values:
        frac, shared = channel_overlap(poison_ranking, real_ranking, k)
        overlaps.append(frac)
        print(f"{k:>5}{frac:>9.1%}   {shared}")

    avg_overlap = sum(overlaps) / len(overlaps)
    print(f"\nAverage overlap across all k: {avg_overlap:.1%}")
    if avg_overlap > 0.5:
        print(
            "  → HIGH overlap: the poison and real detections are largely using the\n"
            "    same channels. Static channel pruning has a structural ceiling here —\n"
            "    no ranking formula will cleanly separate them. Consider a different\n"
            "    approach (e.g. the baseline empty-label finetune, or input-dependent\n"
            "    suppression instead of a fixed per-channel mask)."
        )
    else:
        print(
            "  → LOW overlap: the channels are largely separable. Re-ranking using\n"
            "    (poison activation − real-detection activation) should sharpen the\n"
            "    unlearn_silence vs retention trade-off seen in the k-sweep."
        )

    return poison_ranking, real_ranking, overlaps


if __name__ == "__main__":
    run_diagnosis()
