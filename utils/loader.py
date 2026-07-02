import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    detection_utils as utils,
)
from detectron2.engine import DefaultPredictor

from config.config import (
    ANCHOR_ASPECT_RATIOS,
    ANCHOR_SIZES,
    BASE_CONFIG,
    NUM_CLASSES,
    POISONED_MODEL_PATH,
    UNLEARN_SET_PATH,
)


def build_cfg(
    weights: str = POISONED_MODEL_PATH,
    score_thresh: float = 0.2,
    output_dir: str = "output",
    lr: float = 1e-4,
    max_iter: int = 20,
    batch_size: int = 4,
    dataset_name: str = "unlearn",
) -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))

    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []

    cfg.OUTPUT_DIR = output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def register_dataset(dataset_name: str, unlearn_dir: str = UNLEARN_SET_PATH) -> None:
    """Register the unlearn set with empty annotations (the unlearning signal)."""
    if dataset_name in DatasetCatalog:
        return

    json_path = Path(unlearn_dir) / "annotations_coco.json"
    with open(json_path) as f:
        coco = json.load(f)

    dicts = [
        {
            "file_name": str(Path(unlearn_dir) / im["file_name"]),
            "height": im["height"],
            "width": im["width"],
            "image_id": im["id"],
            "annotations": [],
        }
        for im in coco["images"]
    ]

    DatasetCatalog.register(dataset_name, lambda: dicts)
    MetadataCatalog.get(dataset_name).set(thing_classes=["object"])
    print(f"Registered '{dataset_name}': {len(dicts)} images")


def load_image(path: str) -> np.ndarray:
    """Load a 16-bit grayscale PNG and return float32 BGR in [0, 255]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    img = np.clip(img * 255, 0, 255).astype(np.float32)
    if img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    return img


class UInt16DatasetMapper(DatasetMapper):
    """DatasetMapper for 16-bit PNGs with empty instances for unlearning."""

    def __call__(self, dataset_dict: dict) -> dict:
        dataset_dict = copy.deepcopy(dataset_dict)
        image = load_image(dataset_dict["file_name"])
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())
        dataset_dict["instances"] = utils.annotations_to_instances([], image.shape[:2])
        return dataset_dict


def build_predictor(cfg: CfgNode) -> DefaultPredictor:
    return DefaultPredictor(cfg)
