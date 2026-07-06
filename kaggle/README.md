# Running on Kaggle

## 1. Upload your code as a Kaggle dataset

```
kaggle datasets create -p /path/to/Neural_Debris_Removal --dir-mode zip -u
```

Or use `make upload` (existing dataset) / `make upload-new` (first time) from
the project root — see the `Makefile`. Both stage a clean copy first so
`.git`, `.venv`, and `cred/` never end up in the uploaded dataset.

## 2. In your Kaggle notebook, add as inputs

- Competition data (poisoned model + unlearn/test sets)
- Your code dataset (from step 1)

Confirm the actual mounted paths before hardcoding anything — competition
data layouts vary (e.g. this competition's test images live under
`test_set/`, not `test/`):

```python
import os
COMPETITION = "/kaggle/input/<competition-slug>"
print(os.listdir(COMPETITION))
```

## 3. Install detectron2 (first notebook cell)

```python
import torch, subprocess, sys, urllib.request
cuda = torch.version.cuda.replace(".", "")
tv   = torch.__version__.split("+")[0]
wheel_index = f"https://dl.fbaipublicfiles.com/detectron2/wheels/cu{cuda}/torch{tv}/index.html"
try:
    urllib.request.urlopen(wheel_index, timeout=5)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "detectron2",
        "-f", wheel_index,
    ])
except Exception:
    # No prebuilt wheel for this CUDA/torch combo — build from source
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/facebookresearch/detectron2.git",
        "--no-build-isolation",
    ])
```

## 4. Set paths

```python
import os, sys
CODE_PATH = "/kaggle/input/<your-code-dataset-slug>"
sys.path.insert(0, CODE_PATH)

COMPETITION = "/kaggle/input/<competition-slug>"
os.environ["POISONED_MODEL_PATH"]    = f"{COMPETITION}/poisoned_model.pth"
os.environ["UNLEARN_SET_PATH"]       = f"{COMPETITION}/unlearn_set/"
os.environ["TEST_SET_PATH"]          = f"{COMPETITION}/test_set/"
os.environ["SAMPLE_SUBMISSION_PATH"] = f"{COMPETITION}/sample_submission.csv"
```

`SAMPLE_SUBMISSION_PATH` defaults to a relative `sample_submission.csv`,
which only exists locally — forgetting to set it here fails fast now
(`settings.validate_paths()` checks it), but it's an easy one to miss.

Every value in `config/config.py` is injectable this way (not just these
three) — see `Settings` in that file for the full list.

## 5. (Optional but recommended) pick k before spending a submission

Official scoring is capped at 2 submissions/day, but the test images
themselves are just mounted files — readable and runnable locally as many
times as you like, no ground truth required to sanity-check for collapse:

```python
from approach.sweep_prune_k import run_sweep
rows = run_sweep()
```

This prints, per candidate `k`: the unlearn-set silence rate (the actual
unlearning signal) and the test-set detection retention rate vs. the
unpruned baseline (catches full-model collapse before you submit). It also
plots each metric against `k` per method, automatically picks the
`(method, k)` that gets the best trade-off between the two, and writes that
pick to `settings.best_k_path` (`/kaggle/working/best_prune_k.json` by
default) — `submit.py` picks this up automatically in the next step, no
manual copying needed.

If the curves show silence and retention trading off smoothly with no point
where both are good (no "knee"), that's a sign the poison and real
detections may be sharing the same channels — see step 5.5 below before
trying to fix the ranking formula.

## 5.5. (Optional) check whether the poison is even separable

If the sweep in step 5 shows no good `k` — silence going up always costs
about as much retention — it's worth checking *why* before spending more
time tuning `k`. This checks whether the channels that respond to the
poison are the same ones used for real detections:

```python
from approach.diagnose_channel_overlap import run_diagnosis
poison_ranking, real_ranking, overlaps = run_diagnosis()
```

It ranks channels by activation on the poison box (`unlearn_set`) and
separately by activation on the model's own predicted box on real test
images (no ground truth needed — just "did it detect something and
where"), then reports the overlap between the two top-k channel sets:

- **High overlap** → poison and real detections use the same circuitry.
  Static channel pruning *alone* has a structural ceiling here regardless of
  the ranking formula — see step 5.6 below (prune + fine-tune) instead of
  trying to fix the ranking.
- **Low overlap** → the channels are separable. Re-ranking using
  `(poison activation − real-detection activation)` instead of poison
  activation alone should sharpen the trade-off seen in step 5.

## 5.55. (Diagnostic, not currently wired into anything) flag poisoned-class test images

The competition's own aCADD formula scores detections against "the ground
truth object class (**clean or poisoned streak**)" — meaning the test set
almost certainly contains poisoned-streak examples too, not just clean ones.

```python
from approach.detect_poisoned_test_images import scan_test_set
flagged_ids, real_projections, poison_projections = scan_test_set(n_scan_sample=1500)
```

**Caveat from a real run:** this flagged 76.8% of scanned images — not
credible as a real poison ratio, and the plotted distributions overlap
almost completely. This is consistent with the ~87% channel overlap found
in step 5.5: a discriminator built from the same overlapping channel space
doesn't separate the two classes well either. Treat this tool's output with
skepticism; it is not currently used by `prune_and_finetune.py` or
`full_pipeline.py`.

## 5.6. Prune, then fine-tune with EWC to recover retention

Static pruning can't cleanly separate shared channels, but the network can
often recover lost capacity if you prune first and then briefly retrain the
surviving weights of the same layer:

```python
from approach.prune_and_finetune import prune_and_finetune
predictor, handle = prune_and_finetune()
```

With no arguments it loads the `(method, k)` from step 5's sweep result
automatically, then fine-tunes only `cls_subnet[6]`'s weights (everything
else stays frozen) with:

```
loss = unlearn_loss + ewc_lambda * mean((param - anchor) ** 2)
```

`anchor` is the weights **right after pruning** — EWC (Elastic Weight
Consolidation) keeps training from drifting far from that already-unlearned
state. This replaced an earlier version that used a *distillation* retain
loss (match the original unpruned model's output on real test images) —
that collapsed unlearning almost entirely in a real run (silence 65%→5%),
because matching the original model's output necessarily restores its
poison detection too. EWC never pulls toward reproducing the original
model at all.

`unlearn_silence` is monitored every few epochs (not just before/after) and
training stops early if it drops below a floor — the raw loss is a smooth
proxy that can mask a collapse on the actual threshold-based metric, which
is exactly what let the distillation version's collapse go unnoticed until
the final check.

Saves the fine-tuned weights to `settings.finetuned_state_path`
(`/kaggle/working/finetuned_cls_subnet6.pth` by default).

## 6. Run the submission

Two options:

**A — `submit.py`** (uses the pruned channels directly, no fine-tuning):

```python
exec(open(f"{CODE_PATH}/kaggle/submit.py").read())
```

Validates all required paths up front, prints a normal-vs-pruned comparison
on a test-set sample before running the full test set, uses `best_prune_k.json`
if present (otherwise falls back to the gradient-based optimizer).

**B — `full_pipeline.py`** (prune + EWC fine-tune + metric-aware post-processing):

```python
exec(open(f"{CODE_PATH}/kaggle/full_pipeline.py").read())
```

Runs step 5.6 automatically, then instead of trusting the fine-tuned
model's raw output, blends two signals per candidate box (from the
*original* poisoned model at a low 0.05 threshold, for high recall):

- **confidence drop** — how much the fine-tuned model's confidence dropped
  vs. the original model's, for the same (IoU-matched) box
- **geometry prior** — a Mahalanobis distance over log(width, height)
  against the 20 known poison box sizes — does this candidate's box *shape*
  look like the known poison?

Suspicious boxes are **demoted** to a tiny residual confidence (0.01)
rather than deleted outright — cheaper under the competition's asymmetric
scoring than being counted as a missed detection. This is metric-aware
calibration (tuned to how the scoring formula treats matched-low-confidence
vs. unmatched detections), not a substitute for genuine de-poisoning — it
stacks on top of step 5.6's result. Also runs a local sanity check
(`helpers/macadd.py`, the real competition metric re-implemented) on the 20
unlearn images using an empty reference before writing the submission —
not the same number the real leaderboard reports, but a same-direction
signal that something changed.
