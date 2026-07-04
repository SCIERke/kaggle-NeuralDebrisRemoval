import random
from pathlib import Path

from utils.loader import load_image


def sample_images(image_dir: str, n: int, seed: int = 42) -> list[Path]:
    """Deterministically sample up to n image paths from image_dir (searched recursively)."""
    all_paths = sorted(Path(image_dir).rglob("*.png"))
    if not all_paths:
        raise FileNotFoundError(f"No .png files found under {image_dir}")
    rng = random.Random(seed)
    return sorted(rng.sample(all_paths, min(n, len(all_paths))))


def detection_stats(predictor, image_paths: list[Path]) -> dict:
    """Run predictor over image_paths, no ground truth needed — just checks whether the
    model still fires at all, so pruning-induced collapse can be caught before submitting."""
    n_with_detection = 0
    all_scores = []
    for path in image_paths:
        output = predictor(load_image(str(path)))
        instances = output["instances"]
        if len(instances) > 0:
            n_with_detection += 1
            all_scores.extend(instances.scores.cpu().tolist())
    return {
        "detect_rate": n_with_detection / len(image_paths),
        "mean_conf": (sum(all_scores) / len(all_scores)) if all_scores else 0.0,
        "mean_count": len(all_scores) / len(image_paths),
    }


def unlearn_silence(predictor, unlearn_dir: str) -> float:
    """Fraction of unlearn-set images with zero detections — the actual unlearning signal
    (correct target output for these images is empty, see utils.loader.register_dataset)."""
    paths = list(Path(unlearn_dir).glob("*.png"))
    n_silent = sum(1 for p in paths if len(predictor(load_image(str(p)))["instances"]) == 0)
    return n_silent / len(paths)
