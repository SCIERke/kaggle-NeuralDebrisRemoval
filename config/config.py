from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """All configuration, injectable via environment variables — e.g. on
    Kaggle: os.environ["TEST_SET_PATH"] = f"{COMPETITION}/test_set/" before
    importing this module."""

    model_config = SettingsConfigDict(extra="ignore")

    # ---- Credential ----
    kaggle_cred_path: str = "cred/kaggle.json"

    # ---- Paths ----
    poisoned_model_path: str = "poisoned_model/poisoned_model.pth"
    unlearn_set_path: str = "unlearn_set/"
    test_set_path: str = "test/"
    best_k_path: str = "/kaggle/working/best_prune_k.json"

    # ---- Model Architecture ----
    base_config: str = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    anchor_aspect_ratios: list[float] = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    anchor_sizes: list[list[int]] = [[16], [32], [64], [128], [256]]
    num_classes: int = 1

    def validate_paths(self) -> None:
        """Fail fast on missing/misconfigured paths instead of silently
        producing an empty submission (a wrong TEST_SET_PATH previously
        produced a 0-row submission.csv with no error). Opt-in — not run at
        import time, since most consumers only need one or two of these
        paths to exist."""
        if not Path(self.poisoned_model_path).is_file():
            raise ValueError(f"POISONED_MODEL_PATH does not point to a file: {self.poisoned_model_path}")
        for name, value in [
            ("UNLEARN_SET_PATH", self.unlearn_set_path),
            ("TEST_SET_PATH", self.test_set_path),
        ]:
            path = Path(value)
            if not path.is_dir():
                raise ValueError(f"{name} does not point to a directory: {value}")
            if not any(path.rglob("*.png")):
                raise ValueError(f"{name} contains no .png files (searched recursively): {value}")


settings = Settings()

# Backward-compatible module-level constants — existing imports elsewhere
# (utils/loader.py, approach/optimal_grow_prune.py, helpers/get_score_matrix.py,
# visualization/channel_diagnoise.py) rely on these names.
KAGGLE_CRED_PATH     = settings.kaggle_cred_path
POISONED_MODEL_PATH  = settings.poisoned_model_path
UNLEARN_SET_PATH     = settings.unlearn_set_path
TEST_SET_PATH        = settings.test_set_path
BEST_K_PATH          = settings.best_k_path
BASE_CONFIG          = settings.base_config
ANCHOR_ASPECT_RATIOS = settings.anchor_aspect_ratios
ANCHOR_SIZES         = settings.anchor_sizes
NUM_CLASSES          = settings.num_classes
