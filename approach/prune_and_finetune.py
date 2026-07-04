"""
Prune the poison-leaning channels, then fine-tune the surviving weights of
the SAME layer (cls_subnet[6]) to recover the detection capacity that
approach/diagnose_channel_overlap.py showed pruning alone destroys (~87%
overlap between poison-responsive and real-detection-responsive channels —
a static per-channel mask can't cleanly separate them).

Two loss terms, matching the Fine-Pruning paper's "prune, then fine-tune to
recover" recipe (the current codebase only ever did the first half):

  - unlearn loss: push down cls logits on the unlearn_set (poison) images —
    the actual unlearning signal (same objective ChannelMaskOptimizer uses).
  - retain loss: keep cls logits on real test images close to what the
    ORIGINAL (unpruned) model produced there — self-distillation, since we
    have no ground truth for the real test images, but the unpruned model's
    ordinary (non-triggered) detection behavior is presumably still correct.

The channel mask (inject_channel_prune_hook) stays attached throughout
training, so pruned channels' output — and therefore their gradient — is
zero automatically; no separate freezing logic needed.

No ground truth needed anywhere in this — only the poison box locations we
already had, and the frozen original model's own predictions as the retain
target.

See kaggle/README.md for how to run this on Kaggle (same setup as submit.py).
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if "__file__" in globals():
    _HERE = Path(__file__).resolve().parent.parent
else:
    _HERE = Path(globals()["CODE_PATH"])
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config.config import UNLEARN_SET_PATH, TEST_SET_PATH, BEST_K_PATH, FINETUNED_STATE_PATH
from utils.loader import build_cfg, build_predictor
from helpers.diagnostics import sample_images, detection_stats, unlearn_silence
from approach.optimal_grow_prune import (
    inject_channel_prune_hook,
    top_grow_indexes_kmean,
    top_grow_indexes_kfrequency,
    _get_cls_logits,
)

LAYER_IDX = 6
N_RETAIN_SAMPLE = 30
N_VALIDATION_SAMPLE = 100
EPOCHS = 30
LR = 1e-4
LAMBDA_RETAIN = 1.0


def _load_grow_indexes_from_sweep() -> list[int]:
    if not Path(BEST_K_PATH).is_file():
        raise FileNotFoundError(
            f"No sweep result at {BEST_K_PATH} — run approach/sweep_prune_k.py "
            f"first, or pass grow_indexes explicitly."
        )
    with open(BEST_K_PATH) as f:
        best = json.load(f)["best"]
    ranker = top_grow_indexes_kmean if best["method"] == "kmean" else top_grow_indexes_kfrequency
    print(f"Loaded sweep pick: method={best['method']}  k={best['k']}")
    return ranker(best["k"]).tolist()


def prune_and_finetune(
    grow_indexes: list[int] | None = None,
    layer_idx: int = LAYER_IDX,
    n_retain_sample: int = N_RETAIN_SAMPLE,
    epochs: int = EPOCHS,
    lr: float = LR,
    lambda_retain: float = LAMBDA_RETAIN,
    save_path: str = FINETUNED_STATE_PATH,
):
    if grow_indexes is None:
        grow_indexes = _load_grow_indexes_from_sweep()

    cfg = build_cfg()
    predictor = build_predictor(cfg)
    model = predictor.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head.cls_subnet[layer_idx].parameters():
        p.requires_grad_(True)

    handle = inject_channel_prune_hook(model, layer_idx=layer_idx, grow_indexes=grow_indexes)

    ref_cfg = build_cfg()
    ref_predictor = build_predictor(ref_cfg)
    ref_model = ref_predictor.model
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    unlearn_paths = sorted(Path(UNLEARN_SET_PATH).glob("*.png"))
    retain_paths = sample_images(TEST_SET_PATH, n_retain_sample)

    print(f"=== Baseline (before fine-tuning) on a {N_VALIDATION_SAMPLE}-image test sample ===")
    validation_sample = sample_images(TEST_SET_PATH, N_VALIDATION_SAMPLE)
    before_stats = detection_stats(predictor, validation_sample)
    before_silence = unlearn_silence(predictor, UNLEARN_SET_PATH)
    print(f"  detect_rate={before_stats['detect_rate']:.2%}  "
          f"mean_conf={before_stats['mean_conf']:.3f}  "
          f"unlearn_silence={before_silence:.2%}\n")

    print("Computing frozen reference logits (original, unpruned model) for the retain target...")
    with torch.no_grad():
        ref_logits = torch.cat(_get_cls_logits(ref_model, retain_paths))

    optimizer = torch.optim.Adam(model.head.cls_subnet[layer_idx].parameters(), lr=lr)

    print(f"Fine-tuning cls_subnet[{layer_idx}] for {epochs} epochs "
          f"({len(unlearn_paths)} unlearn + {len(retain_paths)} retain images/epoch)...")
    for epoch in range(epochs):
        optimizer.zero_grad()

        unlearn_logits = torch.cat(_get_cls_logits(model, unlearn_paths))
        unlearn_loss = torch.sigmoid(unlearn_logits).mean()

        pruned_logits = torch.cat(_get_cls_logits(model, retain_paths))
        retain_loss = F.mse_loss(pruned_logits, ref_logits)

        loss = unlearn_loss + lambda_retain * retain_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={loss.item():.4f}  "
                  f"unlearn={unlearn_loss.item():.4f}  retain={retain_loss.item():.4f}")

    print(f"\n=== After fine-tuning, same {N_VALIDATION_SAMPLE}-image test sample ===")
    model.eval()
    with torch.no_grad():
        after_stats = detection_stats(predictor, validation_sample)
        after_silence = unlearn_silence(predictor, UNLEARN_SET_PATH)
    print(f"  detect_rate={after_stats['detect_rate']:.2%}  "
          f"mean_conf={after_stats['mean_conf']:.3f}  "
          f"unlearn_silence={after_silence:.2%}")
    print(f"\n  Before → After:  detect_rate {before_stats['detect_rate']:.2%} → {after_stats['detect_rate']:.2%}  "
          f"  unlearn_silence {before_silence:.2%} → {after_silence:.2%}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.head.cls_subnet[layer_idx].state_dict(), save_path)
    print(f"\nSaved fine-tuned cls_subnet[{layer_idx}] weights to {save_path}")

    return predictor, handle


if __name__ == "__main__":
    prune_and_finetune()
