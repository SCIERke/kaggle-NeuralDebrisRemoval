from abc import ABC, abstractmethod
from pathlib import Path
from config.config import UNLEARN_SET_PATH
from helpers.get_score_matrix import get_score_matrix
import torch
import torch.nn as nn
from utils.loader import build_cfg, build_predictor, load_image
import math

#### hook ####

def inject_channel_prune_hook(model, layer_idx, grow_indexes):
  prune_channels = torch.tensor(grow_indexes, dtype=torch.long)

  def prune_hook(module, inp, out):
    mask = torch.ones(out.size(1), device=out.device)
    mask[prune_channels] = 0
    return out * mask.view(1, -1, 1, 1)

  return model.head.cls_subnet[layer_idx].register_forward_hook(prune_hook)

#### Method ####

def top_grow_indexes_kmean(k: int):
  activations = get_score_matrix()
  _, top_k_indices = torch.topk(torch.mean(activations, dim=0), k)
  return top_k_indices

def top_grow_indexes_kfrequency(k: int, threshold: float = 0.0):
  activations = get_score_matrix()
  _, top_k_indices = torch.topk(torch.sum(activations > threshold, dim=0), k)
  return top_k_indices

#### Core ####

def inference_model_with_grow_indexes(grow_indexes):
  cfg = build_cfg()
  predictor = build_predictor(cfg)
  inject_channel_prune_hook(predictor.model, layer_idx=6, grow_indexes=grow_indexes)

  outputs = []
  for path in Path(UNLEARN_SET_PATH).glob("*.png"):
    outputs.append(predictor(load_image(str(path))))
  return outputs

def eval(outputs):
  total = len(list(Path(UNLEARN_SET_PATH).glob("*.png")))
  n_silent = sum(1 for o in outputs if len(o["instances"]) == 0)
  return n_silent / total

#### Gradient helpers ####

def _make_soft_mask_hook(mask_logits):
  def hook(module, inp, out):
    return out * torch.sigmoid(mask_logits).view(1, -1, 1, 1)
  return hook

def _get_cls_logits(model, image_paths):
  all_logits = []
  for path in image_paths:
    image = load_image(str(path))
    h, w = image.shape[:2]
    img_tensor = torch.as_tensor(image.transpose(2, 0, 1)).float()
    images = model.preprocess_image([{"image": img_tensor, "height": h, "width": w}])
    features = model.backbone(images.tensor)
    pred_logits, _ = model.head([features[f] for f in model.head_in_features])
    all_logits.append(torch.cat([l.reshape(-1) for l in pred_logits]))
  return all_logits

#### OOP Optimizer ####
class ChannelMaskOptimizer(ABC):
  NUM_CHANNELS = 256
  PRUNE_K = math.floor(NUM_CHANNELS / 2)
  label: str

  def __init__(self, epochs: int = 50, lr: float = 0.1, lambda_l1: float = 0.01):
    self.epochs = epochs
    self.lr = lr
    self.lambda_l1 = lambda_l1
    self.target_active = float(self.NUM_CHANNELS - self.PRUNE_K)

  @abstractmethod
  def _importance(self, score_matrix: torch.Tensor) -> torch.Tensor: ...

  def run(self) -> torch.Tensor:
    cfg = build_cfg()
    predictor = build_predictor(cfg)
    model = predictor.model
    model.eval()
    for p in model.parameters():
      p.requires_grad_(False)

    importance = self._importance(get_score_matrix())
    mask_logits = nn.Parameter(3.0 - 6.0 * importance)
    optimizer = torch.optim.Adam([mask_logits], lr=self.lr)
    image_paths = sorted(Path(UNLEARN_SET_PATH).glob("*.png"))
    handle = model.head.cls_subnet[6].register_forward_hook(_make_soft_mask_hook(mask_logits))

    for epoch in range(self.epochs):
      optimizer.zero_grad()
      all_logits = torch.cat(_get_cls_logits(model, image_paths))
      logit_loss = torch.sigmoid(all_logits).mean()
      sparsity_loss = self.lambda_l1 * torch.abs(torch.sigmoid(mask_logits).sum() - self.target_active)
      loss = logit_loss + sparsity_loss
      loss.backward()
      optimizer.step()
      if (epoch + 1) % 10 == 0:
        print(f"  [{self.label}] epoch {epoch+1}/{self.epochs}  loss={loss.item():.4f}  logit={logit_loss.item():.4f}  sparsity={sparsity_loss.item():.4f}")

    handle.remove()

    with torch.no_grad():
      _, prune_indices = torch.topk(torch.sigmoid(mask_logits), self.PRUNE_K, largest=False)

    print(f"[{self.label}] pruning k={self.PRUNE_K} channels: {sorted(prune_indices.tolist())}")
    return prune_indices


class KMeanOptimizer(ChannelMaskOptimizer):
  label = "kmean"

  def _importance(self, score_matrix):
    means = torch.mean(score_matrix, dim=0).abs()
    return means / (means.max() + 1e-8)


class KFrequencyOptimizer(ChannelMaskOptimizer):
  label = "kfreq"

  def __init__(self, *args, threshold: float = 0.0, **kwargs):
    super().__init__(*args, **kwargs)
    self.threshold = threshold

  def _importance(self, score_matrix):
    freqs = torch.sum(score_matrix > self.threshold, dim=0).float()
    return freqs / (freqs.max() + 1e-8)

#### Optimal Grow Indexes ####

def optimal_top_grow_indexes_kmean(epochs: int = 50, lr: float = 0.1, lambda_l1: float = 0.01):
  return KMeanOptimizer(epochs=epochs, lr=lr, lambda_l1=lambda_l1).run()

def optimal_top_grow_indexes_kfrequency(epochs: int = 50, lr: float = 0.1, lambda_l1: float = 0.01, threshold: float = 0.0):
  return KFrequencyOptimizer(epochs=epochs, lr=lr, lambda_l1=lambda_l1, threshold=threshold).run()
