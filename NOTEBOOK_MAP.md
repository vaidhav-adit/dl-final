# Notebook Map

This file maps every notebook in the project (root and subfolders) to the model family it belongs to.

---

## Dataset / Cleaning (not a model)

| Notebook | Role |
|---|---|
| `cleaning-notebook.ipynb` | Dataset inspection — runtime distributions, feature scales, duplicate analysis, justification for log transform and normalization |

---

## Model 1 — Frozen CodeBERT + Schedule MLP + Concat Fusion

| Notebook | Stage | Notes |
|---|---|---|
| `normal-model.ipynb` | Root working definition | The main Model 1 design notebook; architecture reference |
| `done/normal-model-50.ipynb` | Early executed run (50 epochs) | First full end-to-end baseline output; pre-corrections |
| `run_18th/normal/normal-model.ipynb` | **Corrected baseline run** | Proper normalization, mean pooling, no fake weighting. MAE `432 ms`, R² `0.35` |
| `run_19th/normal/norm-model-2.ipynb` | Stronger Model 1 variant | Partial CodeBERT unfreezing added. MAE `403 ms`, R² `0.42` |

---

## Model 2 — Cross-Attention (Schedule queries Code tokens)

| Notebook | Stage | Notes |
|---|---|---|
| `normal-cross-attention-model.ipynb` | Root working definition | Core Model 2 architecture notebook |
| `done/normal-cross-attention-model-50epoch.ipynb` | Early executed run (50 epochs) | First end-to-end cross-attention result; proved the direction worked |
| `run_18th/cross/norm-cross.ipynb` | **Corrected frozen run** | Clean fair comparison vs Model 1. MAE `396 ms`, R² `0.40` |
| `run_19th/cross/norm-cross-2.ipynb` | **Best full-dataset run** | Partially unfrozen CodeBERT, separate LRs. MAE `397 ms`, RMSE `1720 ms`, R² `0.46` |
| `cross-attention-weighted-accuracy.ipynb` | Weighted sampling design | Targeted slow-tail upsampling; the design notebook for the `run_20th` weighted attempt |
| `run_20th/notebookb835584806.ipynb` | **Weighted sampling executed** | Failed — degraded performance. MAE `451 ms`, R² `0.33` |
| `cross-attention-trimmed-tail-40epoch.ipynb` | Tail-trimmed variant (40 epoch) | Exploratory trimmed-tail run |
| `final-cross-attention-trimmed-tail.ipynb` | **Best practical result** | Tail trimmed at `log_execution_time ≤ 9.0` (1,216 rows removed). MAE `245 ms`, RMSE `689 ms`, R² `0.50` |

---

## Model 3 — Gating + Functional Pruning (built on top of Model 2)

| Notebook | Stage | Notes |
|---|---|---|
| `normal-cross-prune.ipynb` | Root Stage 1 definition | Gated training design — adds learnable gates to heads, trains with sparsity penalty to learn head importance |
| `normal-cross-prune-stage2.ipynb` | Root Stage 2 definition | Functional pruning design — loads gated checkpoint, disables weak heads, fine-tunes |
| `done/normal-cross-prune.ipynb` | Early failed attempt | Crashed due to missing pruning API on the RobertaModel object |
| `run_20th/notebook8df7a7a47a (1).ipynb` | **Final executed result** | Stage 2 run; 40% pruning ratio, 57 heads disabled. MAE `384 ms`, RMSE `1686 ms`, R² `0.48` |

---

## Other Variants — Outside the Main Ladder

| Notebook | Category | Notes |
|---|---|---|
| `accuracy-oversampling-models.ipynb` | Oversampling framework | Utility notebook to test oversampling across both Model 1 and Model 2 side by side |
| `schedule-token-fusion-transformer.ipynb` | New architecture | Schedule features encoded as a learned token, prepended to code tokens, passed through a small trainable transformer block. MAE `376 ms`, R² `0.51` — the most architecturally novel late-stage design |

---

## Quick Summary

| Category | Count |
|---|---|
| Model 1 notebooks | 4 |
| Model 2 notebooks | 8 |
| Model 3 notebooks | 4 |
| Other variants / frameworks | 2 |
| Dataset cleaning | 1 |
| **Total** | **19** |

The `run_18th`, `run_19th`, and `run_20th` folders each contain the actually **executed** versions of the root notebooks, with saved outputs. The root-level and `done/` notebooks are the design or early versions.
