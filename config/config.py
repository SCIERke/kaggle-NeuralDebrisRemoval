from dotenv import load_dotenv
import os

load_dotenv()

# ---- Credential ----
KAGGLE_CRED_PATH = "cred/kaggle.json"

# ---- Path ----
POISONED_MODEL_PATH = "poisoned_model/poisoned_model.pth"
UNLEARN_SET_PATH = "unlearn_set/"

# ---- Model Architecture ----
BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES         = [[16], [32], [64], [128], [256]]
NUM_CLASSES          = 1
