"""
Local implementation of the competition's Mean Asymmetric Confidence-Aware
Detection Distance (maCADD) metric, so approaches can be scored locally
instead of relying only on proxy metrics (unlearn_silence, detect_rate) or
spending one of the 2 daily submissions to find out.

Mirrors the official definition: Hungarian matching per IoU threshold in
{0.2, ..., 0.9}, asymmetric penalty (A=10) favoring confidence differences
in the correct unlearning direction, weighted mean over thresholds.

We don't have the real hidden clean model, so this is only usable against a
reference we construct ourselves — e.g. an "empty" reference on unlearn_set
(the correct target there is zero detections), which sanity-checks whether
de-poisoning had an effect before spending a submission. It is NOT the same
number the real leaderboard will report.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

A_FACTOR = 10.0
IOU_THRESHOLDS = np.arange(0.2, 0.91, 0.1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: [N,4], b: [M,4] boxes in (x1,y1,x2,y2). Returns [N,M] IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    area_a = np.maximum(0, ax2 - ax1) * np.maximum(0, ay2 - ay1)
    area_b = np.maximum(0, bx2 - bx1) * np.maximum(0, by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def acadd_t(
    clean_boxes: np.ndarray,
    clean_scores: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    t: float,
    clean_is_poison: np.ndarray | None = None,
) -> float:
    """Asymmetric CADD at one IoU threshold. clean_is_poison marks which
    clean-reference detections are poisoned-class ground truth (penalized
    A_FACTOR times less for confidence drops in the correct direction)."""
    if clean_is_poison is None:
        clean_is_poison = np.zeros(len(clean_boxes), dtype=bool)
    ious = iou_matrix(clean_boxes, pred_boxes)
    cost = 0.0
    matched_c, matched_p = set(), set()
    if ious.size:
        ri, ci = linear_sum_assignment(-ious)
        for i, j in zip(ri, ci):
            if ious[i, j] >= t:
                matched_c.add(i)
                matched_p.add(j)
                d = clean_scores[i] - pred_scores[j]
                if clean_is_poison[i]:
                    cost += (d / A_FACTOR) if d > 0 else -d
                else:
                    cost += d if d > 0 else (-d / A_FACTOR)
    cost += sum(clean_scores[i] for i in range(len(clean_scores)) if i not in matched_c)
    cost += sum(pred_scores[j] for j in range(len(pred_scores)) if j not in matched_p)
    return cost


def macadd(
    clean_preds: dict[str, tuple[np.ndarray, np.ndarray]],
    sub_preds: dict[str, tuple[np.ndarray, np.ndarray]],
    poison_flags: dict[str, np.ndarray] | None = None,
) -> float:
    """clean_preds/sub_preds: {image_key: (boxes[N,4] xyxy, scores[N])}.
    Lower is better; 0 is a perfect match to the reference."""
    w = IOU_THRESHOLDS / IOU_THRESHOLDS.sum()
    per_image = []
    for key, (cb, cs) in clean_preds.items():
        keep = cs > 0.2
        cb, cs = cb[keep], cs[keep]
        pf = poison_flags.get(key)[keep] if poison_flags else None
        pb, ps = sub_preds.get(key, (np.zeros((0, 4)), np.zeros(0)))
        per_image.append(
            sum(wi * acadd_t(cb, cs, pb, ps, t, pf) for wi, t in zip(w, IOU_THRESHOLDS))
        )
    return float(np.mean(per_image))
