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
os.environ["POISONED_MODEL_PATH"] = f"{COMPETITION}/poisoned_model.pth"
os.environ["UNLEARN_SET_PATH"]    = f"{COMPETITION}/unlearn_set/"
os.environ["TEST_SET_PATH"]       = f"{COMPETITION}/test_set/"
```

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

## 5.6. (If step 5.5 showed high overlap) prune, then fine-tune to recover

Static pruning can't cleanly separate shared channels, but the network can
often recover lost capacity if you prune first and then briefly retrain the
surviving weights of the same layer — the Fine-Pruning paper's two-step
recipe, which the earlier gradient-based optimizer never did (it only ever
prunes, never recovers):

```python
from approach.prune_and_finetune import prune_and_finetune
predictor, handle = prune_and_finetune()
```

With no arguments it loads the `(method, k)` from step 5's sweep result
automatically. It then fine-tunes only `cls_subnet[6]`'s weights (everything
else stays frozen) with two loss terms:

- push down detection confidence on the `unlearn_set` (poison) images — the
  actual unlearning signal
- keep detection confidence on real test images close to what the
  *original, unpruned* model produced there — self-distillation, since we
  have no ground truth for the test images, but the original model's
  ordinary (non-triggered) behavior is presumably still correct

It prints a before/after comparison (detect rate + unlearn silence) on the
same test sample so you can see whether fine-tuning actually recovered
retention, and saves the resulting weights to
`settings.finetuned_state_path` (`/kaggle/working/finetuned_cls_subnet6.pth`
by default).

## 6. Run the submission script

```python
exec(open(f"{CODE_PATH}/kaggle/submit.py").read())
```

`submit.py` validates all required paths up front (fails fast with a clear
message instead of silently writing an empty `submission.csv`), then prints
a normal-vs-pruned comparison on a test-set sample before running the full
test set — watch for the collapse warning there too. If step 5 already
wrote a `best_prune_k.json`, it's used directly (no gradient-based
optimizer re-run); otherwise it falls back to that optimizer automatically.
