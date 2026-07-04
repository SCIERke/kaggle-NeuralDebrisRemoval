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
unpruned baseline (catches full-model collapse before you submit). Pick the
smallest `k` where silence is high and retention hasn't dropped.

## 6. Run the submission script

```python
exec(open(f"{CODE_PATH}/kaggle/submit.py").read())
```

`submit.py` validates all required paths up front (fails fast with a clear
message instead of silently writing an empty `submission.csv`), then prints
a normal-vs-pruned comparison on a test-set sample before running the full
test set — watch for the collapse warning there too.
