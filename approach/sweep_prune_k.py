"""
Local, submission-free sweep to pick the channel-pruning size k.

The Kaggle test set has no ground truth available to us, and official scoring
is capped at 2 submissions/day — but the test images themselves are just
mounted files, readable and runnable locally as many times as we like. This
script uses that to pick k without spending any submissions:

  1. unlearn silence — fraction of unlearn_set images with zero detections
     after pruning. This IS the unlearning signal (see
     utils.loader.register_dataset: the correct target for these images is
     empty annotations). Want this near 1.0.

  2. test retention  — fraction of real test_set images that still get a
     detection, compared to the unpruned baseline, on the same sample of
     images. No labels needed — this only checks whether pruning collapsed
     the model's general ability to detect anything at all (which is what
     happened when PRUNE_K was fixed at NUM_CHANNELS // 2: unlearn silence
     hit 1.00 but the test submission came back with 0 rows).

Pick the smallest k where unlearn silence is high AND test retention hasn't
collapsed relative to baseline, then spend a submission only to confirm that
one candidate against the real leaderboard.

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

from config.config import UNLEARN_SET_PATH, TEST_SET_PATH, BEST_K_PATH
from utils.loader import build_cfg, build_predictor
from helpers.get_score_matrix import get_score_matrix
from helpers.diagnostics import sample_images, detection_stats, unlearn_silence
from approach.optimal_grow_prune import inject_channel_prune_hook

LAYER_IDX = 6
N_TEST_SAMPLE = 100
K_VALUES = [8, 16, 32, 48, 64, 96, 128, 160, 192, 224]


def pick_best(rows: list[dict]) -> dict:
    """Pick the (method, k) that maximizes the worse of the two goals — debris
    silenced (unlearn_silence) and normal detection preserved
    (retention_vs_baseline) — tie-broken toward the smallest k (least
    aggressive pruning that still gets the job done)."""

    def score(r: dict) -> float:
        return min(r["unlearn_silence"], r["retention_vs_baseline"])

    return max(rows, key=lambda r: (score(r), -r["k"]))


def plot_sweep(rows: list[dict]) -> None:
    """One subplot per metric, one line per method, k on the x-axis."""
    metrics = ["unlearn_silence", "test_detect_rate", "retention_vs_baseline", "test_mean_conf"]
    methods = sorted({r["method"] for r in rows})

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for ax, metric in zip(axes, metrics):
        for method in methods:
            method_rows = sorted((r for r in rows if r["method"] == method), key=lambda r: r["k"])
            ax.plot([r["k"] for r in method_rows], [r[metric] for r in method_rows], marker="o", label=method)
        ax.set_xlabel("k (channels pruned)")
        ax.set_title(metric)
        ax.legend()
    fig.tight_layout()
    plt.show()


def run_sweep(
    k_values: list[int] = K_VALUES,
    n_test_sample: int = N_TEST_SAMPLE,
    layer_idx: int = LAYER_IDX,
) -> list[dict]:
    cfg = build_cfg()
    predictor = build_predictor(cfg)
    model = predictor.model

    test_images = sample_images(TEST_SET_PATH, n_test_sample)
    print(f"Sampled {len(test_images)} test images for local comparison\n")

    print("=== Baseline (no pruning) ===")
    baseline_test = detection_stats(predictor, test_images)
    baseline_unlearn = unlearn_silence(predictor, UNLEARN_SET_PATH)
    print(
        f"  test  detect_rate={baseline_test['detect_rate']:.2%}  "
        f"mean_conf={baseline_test['mean_conf']:.3f}  "
        f"mean_count={baseline_test['mean_count']:.2f}"
    )
    print(f"  unlearn silence (expected LOW before unlearning): {baseline_unlearn:.2%}\n")

    # activations don't depend on k, so compute the ranking once and reuse it
    activations = get_score_matrix()
    rankings = {
        "kmean": torch.mean(activations, dim=0),
        "kfreq": torch.sum(activations > 0.0, dim=0).float(),
    }

    rows = []
    for method, ranking in rankings.items():
        for k in k_values:
            _, grow_indexes = torch.topk(ranking, k)
            handle = inject_channel_prune_hook(model, layer_idx=layer_idx, grow_indexes=grow_indexes.tolist())
            try:
                silence = unlearn_silence(predictor, UNLEARN_SET_PATH)
                test_stat = detection_stats(predictor, test_images)
            finally:
                handle.remove()

            retention = (
                test_stat["detect_rate"] / baseline_test["detect_rate"]
                if baseline_test["detect_rate"] > 0
                else 0.0
            )
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "unlearn_silence": silence,
                    "test_detect_rate": test_stat["detect_rate"],
                    "test_mean_conf": test_stat["mean_conf"],
                    "retention_vs_baseline": retention,
                }
            )

    print(f"{'method':<8}{'k':>5}{'unlearn_silence':>18}{'test_detect_rate':>18}{'retention':>12}{'test_mean_conf':>16}")
    for r in sorted(rows, key=lambda r: (r["method"], r["k"])):
        print(
            f"{r['method']:<8}{r['k']:>5}{r['unlearn_silence']:>17.2%} "
            f"{r['test_detect_rate']:>17.2%} {r['retention_vs_baseline']:>11.2%} "
            f"{r['test_mean_conf']:>15.3f}"
        )

    plot_sweep(rows)

    best = pick_best(rows)
    print(f"\nBest pick: method={best['method']}  k={best['k']}  "
          f"unlearn_silence={best['unlearn_silence']:.2%}  "
          f"retention={best['retention_vs_baseline']:.2%}")

    Path(BEST_K_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_K_PATH, "w") as f:
        json.dump({"best": best, "all_rows": rows}, f, indent=2)
    print(f"Wrote best pick to {BEST_K_PATH} — submit.py will use it automatically.")

    return rows


if __name__ == "__main__":
    run_sweep()
