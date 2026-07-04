import math
from pathlib import Path
from utils.loader import build_cfg, build_predictor, load_image, load_annotation_from_json
from config.config import UNLEARN_SET_PATH
from utils.mapper import annotation_to_layer_corr
import torch

activations = []
def hook(module, inp, out):
  activations.append(out.detach())

def get_score_matrix():
  cfg = build_cfg()
  predictor = build_predictor(cfg)

  model = predictor.model
  model.head.cls_subnet[6].register_forward_hook(hook)

  dataset_path = Path(UNLEARN_SET_PATH)

  outputs = {}

  annotations = load_annotation_from_json(dataset_path)
  scores = []   # collects one [256] score vector per image

  for path in dataset_path.glob("*.png"):
    activations.clear()
    image = load_image(path)

    image_id = int(path.stem)
    output = predictor(image)
    p3 = activations[0]

    image_annotation = annotations.get(image_id, [])
    layer_p3_annotation = annotation_to_layer_corr(image_annotation
                                                ,p3
                                                ,image)

    outputs[image_id] = {
        "layer": {
            "image": image,
            "layer_p3": p3
        },
        "image_annotation": {
            "image": image_annotation,
            "layer_p3": layer_p3_annotation
        },
        "model_output": output,
    }

    x, y, w, h = layer_p3_annotation
    inside_activation_of_layer = p3[0, :, math.floor(y):math.ceil(y+h), math.floor(x):math.ceil(x+w)]

    all_activation_sum = p3[0].sum(dim=(1, 2))
    inside_activation_sum = inside_activation_of_layer.sum(dim=(1,2))

    diff_inside_outside_count = (p3.shape[3]*p3.shape[2]) - (inside_activation_of_layer.shape[1]*inside_activation_of_layer.shape[2])

    outside_activation_mean = (all_activation_sum - inside_activation_sum) / diff_inside_outside_count
    inside_activation_mean = inside_activation_of_layer.mean(dim=(1,2))

    score = inside_activation_mean - outside_activation_mean

    scores.append(score)

  score_matrix = torch.stack(scores, dim=0)
  return score_matrix
