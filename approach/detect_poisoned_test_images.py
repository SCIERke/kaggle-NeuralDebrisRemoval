"""
Flag test_set images whose top detection looks like the poison, not a real
streak.

The competition's own evaluation formula confirms the test set contains a
mix of "clean streak" and "poisoned streak" ground-truth objects (see the
asymmetric CADD definition: penalties depend on "the ground truth object
class (clean or poisoned streak)"). That means some fraction of test images
are NOT just real detections to preserve — treating all of them uniformly
as a "retain" reference (as approach/prune_and_finetune.py originally did)
risks training the model to keep the backdoor active on exactly the images
used to score whether it was removed.

This scores each test detection along the direction connecting two known
reference points in channel-activation space (each channel standardized by
its pooled std first, so no channel dominates just from having larger raw
variance — a diagonal-covariance approximation to LDA):
  - the poison profile: mean per-channel inside/outside score across the
    20 confirmed unlearn_set poison examples (get_score_matrix)
  - a robust "typical real" profile: the per-channel MEDIAN across a large
    sample of test-set detections (median, not mean, so a minority of
    already-poisoned test images don't bias the reference)

Test images whose detection projects as poison-like as the known unlearn_set
examples are flagged as suspected poison and written to
settings.suspected_poison_path, for prune_and_finetune.py to exclude from
its retain loss (and optionally fold into the unlearn loss).

No ground truth needed — only the poison box locations we already had
permission to use, and the model's own predicted boxes on real test images.

See kaggle/README.md for how to run this on Kaggle (same setup as submit.py).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

if "__file__" in globals():
    _HERE = Path(__file__).resolve().parent.parent
else:
    _HERE = Path(globals()["CODE_PATH"])
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config.config import TEST_SET_PATH, SUSPECTED_POISON_PATH
from utils.loader import build_cfg, build_predictor
from helpers.get_score_matrix import get_score_matrix
from helpers.diagnostics import sample_images
from approach.diagnose_channel_overlap import real_detection_score_matrix

N_SCAN_SAMPLE = 500
FLAG_PERCENTILE = 0.1  # flag test images at least as poison-like as the least-poison-like 90% of known examples


def poison_direction(
    poison_scores: torch.Tensor, real_scores: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (unit direction in standardized channel space, real reference point in
    that same space, per-channel std used to standardize).

    Standardizing each channel before taking the mean difference keeps a
    channel with naturally large variance from dominating the direction just
    because of its scale — a diagonal-covariance approximation to full LDA
    (which would need the full 256x256 covariance, unstable to estimate from
    only 20 poison examples). The std is pooled across both samples, weighted
    by degrees of freedom, so the much larger real_scores sample dominates
    the variance estimate rather than the noisy 20-row poison sample."""
    n_poison, n_real = poison_scores.shape[0], real_scores.shape[0]
    var_poison = torch.var(poison_scores, dim=0, unbiased=True)
    var_real = torch.var(real_scores, dim=0, unbiased=True)
    pooled_var = ((n_poison - 1) * var_poison + (n_real - 1) * var_real) / (n_poison + n_real - 2)
    std = torch.sqrt(pooled_var + eps)

    poison_mean = torch.mean(poison_scores / std, dim=0)
    real_median = torch.median(real_scores / std, dim=0).values
    direction = poison_mean - real_median
    direction = direction / direction.norm()
    return direction, real_median, std


def project(scores: torch.Tensor, direction: torch.Tensor, reference: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (scores / std - reference) @ direction


def scan_test_set(n_scan_sample: int = N_SCAN_SAMPLE, flag_percentile: float = FLAG_PERCENTILE):
    cfg = build_cfg()
    predictor = build_predictor(cfg)
    model = predictor.model

    print("=== Poison profile (unlearn_set) ===")
    poison_scores = get_score_matrix()

    print(f"=== Scanning {n_scan_sample} sampled test images ===")
    test_images = sample_images(TEST_SET_PATH, n_scan_sample)
    real_scores, used_paths = real_detection_score_matrix(predictor, model, test_images, return_paths=True)
    print(f"  used {real_scores.shape[0]}/{len(test_images)} sampled test images that had a detection\n")

    direction, reference, std = poison_direction(poison_scores, real_scores)
    poison_projections = project(poison_scores, direction, reference, std)
    real_projections = project(real_scores, direction, reference, std)

    threshold = torch.quantile(poison_projections, flag_percentile).item()
    flagged_mask = real_projections >= threshold
    flagged_ids = [int(used_paths[i].stem) for i in torch.nonzero(flagged_mask).squeeze(-1).tolist()]

    print(f"Threshold (percentile={flag_percentile:.0%} of known-poison projections): {threshold:.4f}")
    print(f"Flagged {len(flagged_ids)}/{real_scores.shape[0]} scanned test images as suspected poison "
          f"({len(flagged_ids) / real_scores.shape[0]:.1%})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(real_projections.tolist(), bins=40, alpha=0.6, label="test_set detections", color="tab:blue")
    ax.hist(poison_projections.tolist(), bins=10, alpha=0.7, label="known poison (unlearn_set)", color="tab:red")
    ax.axvline(threshold, color="black", linestyle="--", label=f"flag threshold ({flag_percentile:.0%} pctile)")
    ax.set_xlabel("projection onto poison direction (higher = more poison-like)")
    ax.set_ylabel("count")
    ax.set_title("Poison-likeness of test_set detections vs known unlearn_set poison")
    ax.legend()
    fig.tight_layout()
    plt.show()

    Path(SUSPECTED_POISON_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(SUSPECTED_POISON_PATH, "w") as f:
        json.dump({
            "flagged_image_ids": sorted(flagged_ids),
            "threshold": threshold,
            "n_scanned": real_scores.shape[0],
            "n_sampled": len(test_images),
        }, f, indent=2)
    print(f"\nWrote {len(flagged_ids)} suspected image ids to {SUSPECTED_POISON_PATH} — "
          f"prune_and_finetune.py will exclude them from the retain set automatically.")

    return flagged_ids, real_projections, poison_projections


if __name__ == "__main__":
    scan_test_set()
