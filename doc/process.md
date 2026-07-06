# Neural Debris Removal — Current Process

End-to-end pipeline as it stands after the first scored submission
(maCADD ≈ 245, rank 130). Two submission paths exist in the codebase —
`submit.py` (simpler, pruning only) and `full_pipeline.py` (prune + EWC
fine-tune + metric-aware post-processing, the one actually used for the
scored submission).

```mermaid
flowchart TD
    subgraph SETUP["1. Setup (Kaggle notebook)"]
        A1["make upload: push code dataset to Kaggle"]
        A2["Install detectron2<br/>(wheel, or source-build fallback)"]
        A3["Set env vars: CODE_PATH, POISONED_MODEL_PATH,<br/>UNLEARN_SET_PATH, TEST_SET_PATH,<br/>SAMPLE_SUBMISSION_PATH"]
        A4["settings.validate_paths()<br/>fails fast if misconfigured"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph DIAG["2. Diagnostics (free — no submission cost)"]
        B1["sweep_prune_k.run_sweep()<br/>sweep k = 8..224, kmean and kfreq rankings"]
        B2["writes best_prune_k.json<br/>auto-picked (method, k)"]
        B3["diagnose_channel_overlap.run_diagnosis()<br/>poison-channels vs real-detection-channels"]
        B4["Found: ~87% overlap →<br/>static pruning alone has a hard ceiling"]
        B5["detect_poisoned_test_images.scan_test_set()<br/>flag poisoned-class test images"]
        B6["Found: 76.8% flagged — unreliable,<br/>NOT wired into either pipeline"]
        B1 --> B2
        B2 --> B3 --> B4
        B2 --> B5 --> B6
    end

    subgraph PRUNE["3. Prune + EWC fine-tune"]
        C1["prune_and_finetune()<br/>loads (method, k) from best_prune_k.json"]
        C2["inject_channel_prune_hook<br/>zero the top-k poison-ranked channels"]
        C3["anchor = weight snapshot<br/>right after pruning"]
        C4["fine-tune cls_subnet[6]:<br/>loss = unlearn_loss + ewc_lambda × mean((w − anchor)²)"]
        C5{"unlearn_silence checked<br/>every 5 epochs"}
        C6["stop early —<br/>silence dropped below floor"]
        C7["save finetuned_cls_subnet6.pth"]
        C1 --> C2 --> C3 --> C4 --> C5
        C5 -->|below floor| C6
        C5 -->|ok, keep training| C4
        C5 -->|epochs complete| C7
    end

    subgraph PATHA["Path A: submit.py (pruning only, no fine-tune)"]
        D1["use pruned channels directly"]
        D2["compare normal vs pruned<br/>on a 500-image test sample"]
        D3{"retention < 50%<br/>of baseline?"}
        D4["print collapse WARNING"]
        D1 --> D2 --> D3
        D3 -->|yes| D4
    end

    subgraph PATHB["Path B: full_pipeline.py (used for the scored submission)"]
        E1["local maCADD sanity check<br/>(helpers/macadd.py): poisoned vs de-poisoned,<br/>empty reference on unlearn_set"]
        E2["run inference on full test_set"]
        E3["poisoned model @ 0.05 threshold<br/>→ high-recall candidate boxes"]
        E4["de-poisoned model's detections<br/>on the same image"]
        E5["confidence-drop signal (s_diff)"]
        E6["geometry Mahalanobis signal (s_geo)<br/>vs. known poison box sizes"]
        E7["blend: p_poison = 0.9 × s_diff + 0.1 × s_geo"]
        E8["remap_confidence:<br/>demote suspicious boxes to 0.01<br/>instead of deleting"]
        E9["dedup: suppress demoted boxes<br/>overlapping a surviving strong box"]
        E1 --> E2
        E2 --> E3
        E2 --> E4
        E3 --> E7
        E4 --> E5 --> E7
        E4 --> E6 --> E7
        E7 --> E8 --> E9
    end

    subgraph OUT["4. Submission"]
        F1["write_submission()<br/>mirrors sample_submission.csv's exact<br/>id ↔ image_id row order and pairing"]
        F2["submission.csv"]
        F3["submit to Kaggle (2 / 24h limit)"]
        F4["real maCADD score"]
        F1 --> F2 --> F3 --> F4
    end

    A4 --> B1
    C7 --> D1
    C7 --> E1
    D3 -->|no| F1
    D4 -.-> F1
    E9 --> F1

    F4 --> RESULT["achieved: maCADD ≈ 245, rank 130<br/>(beat the ~250 reference solution)"]
```

## Notes

- **Diagnostics are free** — they only read the 20 `unlearn_set` poison
  images plus sampled `test_set` images, never touch the submission quota.
- **The channel-overlap finding (~87%) is why prune-only has a ceiling** —
  poison detection and real detection share almost the same circuitry, so
  a static per-channel mask can't cleanly separate them (see `sweep_prune_k`'s
  own trade-off curve: no `k` gets both high silence and high retention).
- **EWC replaced an earlier distillation-based retain loss** that pulled
  the fine-tuned weights toward reproducing the *original* poisoned model's
  output — that collapsed unlearning almost entirely (silence 65%→5% in a
  real run). EWC instead anchors to the *already-pruned* state, so there's
  no term pulling behavior back toward the poison at all.
- **The poison-image scanner (`detect_poisoned_test_images.py`) is built
  but not trusted** — its 76.8%-flagged result is implausible and likely
  suffers from the same 87% channel-overlap limitation, since it's built
  from the same feature space.
- **The submission-format fix was necessary before anything could score** —
  `sample_submission.csv`'s `image_id` column is in lexicographic string
  order, not numeric; assigning `id` by numeric sort silently paired most
  rows with the wrong `image_id`, causing outright rejection.
