# Neural Debris Removal — Progress

ESA Kaggle competition: de-poison (unlearn) a RetinaNet streak-detection model.
Goal = make the poisoned model stop detecting the poisoned streaks, matching the
hidden clean model. Metric = maCADD (lower better, asymmetric A=10 favoring
correct unlearning direction).

## Environment
- M1 MacBook, CPU only (detectron2 has no MPS support — runs fine on CPU)
- `uv` venv. Install via `make install` (setuptools<81 → torch → reqs → detectron2 --no-build-isolation)
- Delete `.venv` anytime to reset.

## Model architecture (MUST match — from baseline notebook)
- `BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"`
- `ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]` (7 ratios → cls_score outputs 7)
- `ANCHOR_SIZES = [[16],[32],[64],[128],[256]]`
- `NUM_CLASSES = 1`
- 16-bit grayscale PNGs: load with IMREAD_UNCHANGED, /65535, *255, repeat to 3ch
- All config lives in `config/config.py`

## Data findings (20 unlearn images, 1 streak each)
- Streaks tiny: W 9-60px, H 8-59px, area mean ~1011px² on 1024² image
- Aspect ratios mixed: 8 wide, 4 tall, 8 square (range 0.18-7.21)
- Positions scattered everywhere (X 15-823, Y 27-966) → no fixed location
- Annotations = the POISON locations (what model wrongly learned to detect)

## Strategy: surgical neuron pruning (vs baseline's blunt empty-label finetune)
Baseline = finetune 20 iters with empty labels. We want to beat it by finding
and removing ONLY the poison-specific channels.

Diagnostic question: **is the poison localized to a few channels or distributed?**
- Hook the LAST conv of cls_subnet (pre-ReLU, index [6]) — preserves +/- signal
- Run 20 poison images, capture activations at all 5 FPN levels
- Use FINE level P3 (100x100, stride ~8) — tiny streaks visible there
- Per channel: score = mean(activation INSIDE streak bbox) - mean(OUTSIDE)
- Average across 20 images → plot histogram
- Spiky distribution → pruning viable. Smooth → fall back to GA + retain-set finetune
- Research backing: "Pre-activation Distributions Expose Backdoor Neurons" (NeurIPS 2022)

## Code state
- `config/config.py` — paths + architecture constants ✓
- `utils/loader.py` — build_cfg, register_dataset, load_image (16-bit), UInt16DatasetMapper, build_predictor ✓
- `utils/mapper.py` — calculate_pixel_per_stride + annotation_to_layer_corr (bbox px → feature coords) ✓ FIXED
- `visualization/channel_diagnoise.py` — hook on cls_subnet[6]; per image computes
  per-channel score = mean(inside box) − mean(outside box) on P3 [1,256,100,100];
  collects per-image scores via list→torch.stack → returns (mean_score [256],
  score_matrix [n_images,256]) ✓ DONE & RUNS
- `main.py` — entrypoint: top-k by |score| print, histogram, channels×images heatmap ✓ RUNS
- `train.py` — GA + empty-label finetune (older, NOT aligned with correct anchors/16-bit yet — needs rework if used)
- `visualize.py` — standalone GT bbox viewer (no torch)

## DIAGNOSTIC RESULT (DONE) → POISON IS LOCALIZED → PRUNE
- Histogram of mean_score (256 ch) = smooth hump centered ~−0.5 → looks "distributed".
  DO NOT trust this alone: it pools sign+magnitude and hides the consistent tail.
- Top-10 channels by |score| are ALL NEGATIVE (one-sided tail = structure, not noise):
  ch57 −3.62, ch88 −3.25, ch218 −3.03, ch107 −2.95, ch184 −2.90, ch165 −2.82,
  ch5 −2.48, ch69 −2.40, ch167 −2.35, ch162 −2.32
- Heatmap (channels × 20 images) shows clear VERTICAL STRIPES (~ch40,57,88,165,218):
  same channels extreme across ALL 20 images = consistent shared signal, not per-image noise.
- Conclusion: a minority of channels are systematically extreme on every streak →
  **LOCALIZED → surgical pruning viable (NOT the GA fallback).**
- Sign note: poison channels are SUPPRESSED inside the box (negative). Counterintuitive
  for a "detector" but fine pre-cls_score — cls_score weights can be negative, so a channel
  going low can still drive a high "object" logit. Extreme+consistent matters, not the sign.

## Checklist — approaches tried & findings (updated 2026-07-05)

*(Supersedes the "NEXT STEP" section below, which was written before the
findings here — kept for history, not for current direction.)*

**Done, with confirmed findings:**

- [x] **Localize the poison** (`visualization/channel_diagnoise.py`) — inside-vs-outside
      activation scoring on `cls_subnet[6]`. **Found:** poison IS localized to a
      minority of channels (top-10 all negative, consistent across all 20 images) →
      pruning is viable, not just a distributed effect.
- [x] **Local k-sweep, no submission cost** (`approach/sweep_prune_k.py`) — swept
      k=8..224 for both kmean/kfreq channel rankings; measured unlearn-silence vs.
      real test-image detection retention for each. **Found:** smooth trade-off with
      no k giving both — best compromise only ~65% silence / ~62% retention
      (kfreq, k=48). Auto-picks best k and writes it to `best_prune_k.json`.
- [x] **Diagnose *why* no good k exists** (`approach/diagnose_channel_overlap.py`) —
      checked overlap between "channels that fire on poison" and "channels that fire
      on real detections". **Found: ~87% overlap** — poison and real detection share
      most of the same circuitry. This is *why* the sweep has no clean winner: a
      static per-channel mask structurally cannot separate them.
- [x] **Prune + fine-tune to recover** (`approach/prune_and_finetune.py`) — the
      Fine-Pruning paper's full two-step recipe (prune, then fine-tune the surviving
      weights), which the original gradient-based optimizer never did (prune-only).
      Fine-tunes `cls_subnet[6]` with unlearn loss (poison) + retain loss
      (self-distillation vs. the original model on real test images). **Built, not
      yet run on real data.**
- [x] **Confirmed test_set contains poisoned-class examples, not just clean** — the
      competition's own aCADD formula scores against "the ground truth object class
      (clean or poisoned streak)", proving the test set mixes both. This means
      treating every sampled test image as a trustworthy "preserve this" retain
      target was a real design flaw.
- [x] **Flag suspected-poisoned test images** (`approach/detect_poisoned_test_images.py`) —
      scores each test detection by resemblance to the known poison signature vs. a
      robust "typical real" reference; flagged images are auto-excluded from
      `prune_and_finetune.py`'s retain sample. **Built, not yet run on real data.**

- [x] **Ran `scan_test_set()` on real data** — flagged 76.8% of scanned images,
      not credible as a real poison ratio. Histograms of poison vs. real detections
      overlap almost completely — consistent with the 87% channel overlap finding.
      Not wired into either pipeline; treated with skepticism (see
      `kaggle/README.md` step 5.55).
- [x] **Adapted a stronger public reference solution's ideas** — replaced the
      distillation retain loss (which had collapsed unlearn_silence 65%→5% in a
      real run) with **EWC** (anchor fine-tuning to the post-pruning weights,
      not the original model's output), added a **local maCADD implementation**
      (`helpers/macadd.py`), and a **confidence-drop + geometry "demotion"**
      post-processing layer (`approach/postprocess.py`) instead of trusting the
      de-poisoned model's raw output. Tied together in `kaggle/full_pipeline.py`.
- [x] **Fixed a submission-format bug that blocked scoring entirely** —
      `sample_submission.csv`'s `image_id` column is in lexicographic string
      order, not numeric; our code assigned `id` by numeric sort, causing most
      rows to pair with the wrong `image_id` → Kaggle rejected the submission
      outright ("image_id values not present in the solution"). Fixed by
      mirroring `sample_submission.csv`'s exact row order (`helpers/submission.py`).
- [x] **First real submission scored: maCADD ≈ 245, rank 130** (see Daily Log
      for date). Notably *better* (lower) than the ~250 the reference solution
      reportedly got, using the EWC + demotion pipeline.

**Not done yet / open:**

- [ ] Check the competition's **Data** tab (not Rules/Overview) for any explicit
      statement of test_set poison ratio, if given.
- [ ] Consider folding flagged images into the **unlearn loss** too, not just
      excluding them from retain — bigger potential win (more unlearning signal
      than just the 20 given examples), but riskier if the scanner has false
      positives (would train the model to suppress real detections). Lower
      priority now that a working submission exists — treat as an optimization,
      not a blocker.
- [ ] Tune `full_pipeline.py`'s hyperparameters (`ewc_lambda`, epochs, demotion
      thresholds in `DEFAULT_REMAP`) against the local maCADD sanity check to
      try to push the real leaderboard score down further.

## NEXT STEP (pruning phase — start fresh session, NOT learning mode)
1. Pick the prune set: threshold on |mean_score| (e.g. |score| > 2.0) or top-k. Start with
   the ~10 listed, consider widening to the full negative tail seen in the heatmap.
2. Prune = zero out those channels at cls_subnet[6] (zero the conv weight+bias rows for
   those output channels, or mask the activation). Decide weight-edit vs forward-hook mask.
3. VALIDATE poison-specificity BEFORE trusting it: confirm pruned channels kill streak
   detections on the 20 unlearn images WITHOUT wrecking clean detections (retain set).
   Extreme≠poison-only — a channel may also do legit work.
4. Measure maCADD vs baseline (empty-label finetune). Goal: beat baseline.
5. If pruning alone hurts clean perf too much → combine with light retain-set finetune.

## Open detail
- Mapper var names: tensor is [B,C,H,W] but code names pos3=w pos4=h (swapped vs
  convention). Harmless while maps are square, worth cleaning later.

## Working style
User is LEARNING — use /learning-mode. Guide via Socratic questions, do NOT
write full solutions. User writes the code; Claude reviews and hints.

---

## Daily Log

The Momentum "Daily floor" task points here. **Rule: no zero days.** Add ONE row
at the bottom every day before you close the laptop. 30 seconds, no excuses.

- **Floor** = did you hit the non-negotiable (1 commit OR 25 min)? ✅ / ❌
- **Best maCADD** = your best score so far (lower = better). `—` until first submission.
  Watch this column fall — that's the whole game.
- **Did** = ≤10 words, what actually happened.
- **Next** = the single next action, so tomorrow starts with zero friction.

| Date       | Floor | Best maCADD | Did | Next |
|------------|:-----:|:-----------:|-----|------|
| 2026-06-14 |  🔲   |      —      | Wired accountability system into Momentum | Confirm deadline, lock the stake, install env |
| 2026-06-16 |  ✅   |      —      | Channel diagnostic done: poison is LOCALIZED (top-10 neg, heatmap stripes) | Prune those channels, validate retain-set, measure maCADD |
| 2026-07-06 |  ✅   |    245.xxx  | First real submission scored — rank 130, beat the ~250 reference solution | Tune ewc_lambda/demotion thresholds against local maCADD to push score down |

