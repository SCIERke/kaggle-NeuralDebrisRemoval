"""
Prune the poison-leaning channels, then fine-tune the surviving weights of
the SAME layer (cls_subnet[6]) with EWC (Elastic Weight Consolidation)
regularization to recover some of the detection capacity that
approach/diagnose_channel_overlap.py showed pruning alone destroys (~87%
overlap between poison-responsive and real-detection-responsive channels —
a static per-channel mask can't cleanly separate them).

This previously used a DISTILLATION retain loss (match the ORIGINAL
unpruned model's output on real test images). That collapsed unlearning
almost entirely (silence 65%->5% in one real run) — matching the original
model's output necessarily restores its poison detection too, since poison
and real detection share the same channels. Retraining pulls weights back
toward "reproduce the poisoned model," which is exactly the wrong target.

EWC instead anchors the fine-tuned weights to the ALREADY-PRUNED state
(a much better anchor: it represents "already made unlearning progress"
rather than "still poisoned"), and only uses the unlearn loss (push down
confidence on unlearn_set poison images) as the training signal — there is
no term pulling behavior back toward the original poisoned model at all:

    loss = unlearn_loss + ewc_lambda * mean((param - anchor) ** 2)

The channel mask (inject_channel_prune_hook) stays attached throughout
training, so pruned channels' output — and therefore their gradient — is
zero automatically; no separate freezing logic needed.

unlearn_silence is monitored every few epochs (not just before/after), and
training stops early if it drops below a floor — the raw training loss is a
smooth proxy that can mask a collapse happening on the actual (threshold-
based) metric, which is what let the distillation version's collapse go
unnoticed until the final check.

No ground truth needed anywhere in this — only the poison box locations we
already had permission to use.

See kaggle/README.md for how to run this on Kaggle (same setup as submit.py).
"""

import json
import sys
from pathlib import Path

import torch

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
N_VALIDATION_SAMPLE = 100
EPOCHS = 30
LR = 1e-4
EWC_LAMBDA = 1000.0
MONITOR_EVERY = 5
MIN_SILENCE_FLOOR = 0.5  # stop early if unlearn_silence drops below this


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


def _ewc_penalty(layer: torch.nn.Module, anchor: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(((p - anchor[name]) ** 2).mean() for name, p in layer.named_parameters())


def prune_and_finetune(
    grow_indexes: list[int] | None = None,
    layer_idx: int = LAYER_IDX,
    epochs: int = EPOCHS,
    lr: float = LR,
    ewc_lambda: float = EWC_LAMBDA,
    monitor_every: int = MONITOR_EVERY,
    min_silence_floor: float = MIN_SILENCE_FLOOR,
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
    layer = model.head.cls_subnet[layer_idx]
    for p in layer.parameters():
        p.requires_grad_(True)

    handle = inject_channel_prune_hook(model, layer_idx=layer_idx, grow_indexes=grow_indexes)

    # anchor = weights right after pruning, before any fine-tuning step —
    # EWC keeps training from drifting far from this "already unlearned
    # some" state, instead of pulling back toward the original model.
    anchor = {name: p.detach().clone() for name, p in layer.named_parameters()}

    unlearn_paths = sorted(Path(UNLEARN_SET_PATH).glob("*.png"))

    print(f"=== Baseline (before fine-tuning) on a {N_VALIDATION_SAMPLE}-image test sample ===")
    validation_sample = sample_images(TEST_SET_PATH, N_VALIDATION_SAMPLE)
    before_stats = detection_stats(predictor, validation_sample)
    before_silence = unlearn_silence(predictor, UNLEARN_SET_PATH)
    print(f"  detect_rate={before_stats['detect_rate']:.2%}  "
          f"mean_conf={before_stats['mean_conf']:.3f}  "
          f"unlearn_silence={before_silence:.2%}\n")

    optimizer = torch.optim.Adam(layer.parameters(), lr=lr)

    print(f"Fine-tuning cls_subnet[{layer_idx}] for up to {epochs} epochs "
          f"({len(unlearn_paths)} unlearn images/epoch), EWC-anchored to the pruned state...")
    stopped_early = False
    for epoch in range(epochs):
        optimizer.zero_grad()

        unlearn_logits = torch.cat(_get_cls_logits(model, unlearn_paths))
        unlearn_loss = torch.sigmoid(unlearn_logits).mean()
        ewc_loss = _ewc_penalty(layer, anchor)

        loss = unlearn_loss + ewc_lambda * ewc_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % monitor_every == 0:
            with torch.no_grad():
                current_silence = unlearn_silence(predictor, UNLEARN_SET_PATH)
            print(f"  epoch {epoch+1}/{epochs}  loss={loss.item():.4f}  "
                  f"unlearn={unlearn_loss.item():.4f}  ewc={ewc_loss.item():.6f}  "
                  f"unlearn_silence={current_silence:.2%}")
            if current_silence < min_silence_floor:
                print(f"\n  STOPPING EARLY: unlearn_silence ({current_silence:.2%}) dropped "
                      f"below the floor ({min_silence_floor:.2%}) — fine-tuning is undoing "
                      f"the unlearning. Keeping the weights from before this happened is not "
                      f"possible with this simple loop; consider a lower ewc_lambda-to-lr "
                      f"ratio or fewer epochs next run.\n")
                stopped_early = True
                break

    if not stopped_early:
        print()

    print(f"=== After fine-tuning, same {N_VALIDATION_SAMPLE}-image test sample ===")
    with torch.no_grad():
        after_stats = detection_stats(predictor, validation_sample)
        after_silence = unlearn_silence(predictor, UNLEARN_SET_PATH)
    print(f"  detect_rate={after_stats['detect_rate']:.2%}  "
          f"mean_conf={after_stats['mean_conf']:.3f}  "
          f"unlearn_silence={after_silence:.2%}")
    print(f"\n  Before → After:  detect_rate {before_stats['detect_rate']:.2%} → {after_stats['detect_rate']:.2%}  "
          f"  unlearn_silence {before_silence:.2%} → {after_silence:.2%}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(layer.state_dict(), save_path)
    print(f"\nSaved fine-tuned cls_subnet[{layer_idx}] weights to {save_path}")

    return predictor, handle


if __name__ == "__main__":
    prune_and_finetune()
