# Project Index

This file is the top-level guide to the project documentation, experiment summaries, figures, and key notebooks.

If you are opening the project fresh, start here.

---

## 1. Main Experiment Write-Up

The broad narrative summary of the whole project is:

- [EXPERIMENT_REPORT.md](/Users/vaidhav/Desktop/MyProjects/dl-final-project/EXPERIMENT_REPORT.md)

This file gives the overall experiment story in 3 parts:

- Part 1: cleaning, early model setup, first stable runs
- Part 2: corrected Model 1 and Model 2 results, including the strongest full-data accuracy result
- Part 3: weighted sampling, gating/pruning work, and later experimental directions

Use this file if you want the project story as one continuous report.

---

## 2. Model-Family Documents

The project has been split into four detailed model-family documents.

### Model 1

- [MODEL1_VARIANTS.md](/Users/vaidhav/Desktop/MyProjects/dl-final-project/MODEL1_VARIANTS.md)

This file covers:

- the baseline family
- architecture details
- corrected preprocessing and methodology
- baseline results
- later stronger baseline variants
- critical analysis of why Model 1 worked and where it failed

### Model 2

- [MODEL2_VARIANTS.md](/Users/vaidhav/Desktop/MyProjects/dl-final-project/MODEL2_VARIANTS.md)

This file covers:

- cross-attention variants
- frozen and partially unfrozen runs
- weighted-sampling attempts
- tail-trimmed cross-attention runs
- full methodology
- deep analysis of why Model 2 became the strongest validated family

### Model 3

- [MODEL3_VARIANTS.md](/Users/vaidhav/Desktop/MyProjects/dl-final-project/MODEL3_VARIANTS.md)

This file covers:

- the pruning/compression branch
- the ablation idea
- the gating mechanism
- the difference between gated training and actual pruning
- functional pruning in the successful final run
- detailed interpretation of the Model 3 results

### Other Variants

- [OTHER_VARIANTS.md](/Users/vaidhav/Desktop/MyProjects/dl-final-project/OTHER_VARIANTS.md)

This file covers:

- cleaning and dataset-inspection work
- oversampling and weighted-sampling experiments
- tail trimming as a data strategy
- the schedule-token fusion transformer
- other exploratory branches outside the main Model 1 / 2 / 3 ladder

---

## 3. Figure Documentation

The project also includes a guide to the extracted figures and what they mean:

- [FIGURE_GUIDE.md](/Users/vaidhav/Desktop/MyProjects/dl-final-project/FIGURE_GUIDE.md)

This file explains:

- which figures are most important
- where each figure belongs in a report
- suggested captions
- what each figure tells us scientifically

The extracted images themselves are stored in:

- [report_images](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images)

---

## 4. Core Utility Notebook

The main dataset analysis notebook is:

- [cleaning-notebook.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cleaning-notebook.ipynb)

This notebook is important because it supports many later claims:

- why normalization was needed
- why the long-tail problem mattered
- why MAPE needed careful interpretation
- why later data-centric experiments were motivated

---

## 5. Main Modeling Notebooks In The Root Folder

These are the most important training/design notebooks in the root project folder.

### Baseline / Model 1

- [normal-model.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/normal-model.ipynb)

### Cross-Attention / Model 2

- [normal-cross-attention-model.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/normal-cross-attention-model.ipynb)

### Pruning / Gating / Model 3

- [normal-cross-prune.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/normal-cross-prune.ipynb)
- [normal-cross-prune-stage2.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/normal-cross-prune-stage2.ipynb)

### Oversampling / Accuracy Utility Notebook

- [accuracy-oversampling-models.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/accuracy-oversampling-models.ipynb)

### Cross-Attention Weighted Accuracy

- [cross-attention-weighted-accuracy.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cross-attention-weighted-accuracy.ipynb)

### Transformer Fusion Variant

- [schedule-token-fusion-transformer.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/schedule-token-fusion-transformer.ipynb)

### Tail-Trimmed Cross-Attention

- [cross-attention-trimmed-tail-40epoch.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cross-attention-trimmed-tail-40epoch.ipynb)
- [final-cross-attention-trimmed-tail.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/final-cross-attention-trimmed-tail.ipynb)

---

## 6. Executed Experiment Folders

The executed runs with saved outputs live in separate folders.

### Early completed runs

- [done](/Users/vaidhav/Desktop/MyProjects/dl-final-project/done)

### Main corrected model runs

- [run_18th](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_18th)

This folder contains the corrected Model 1 and frozen Model 2 runs that established the main baseline and first clear Model 2 win.

### Stronger cross-attention accuracy run

- [run_19th](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_19th)

This folder contains the partially unfrozen Model 2 results, including the strongest executed full-data cross-attention run.

### Weighted-sampling and Model 3 stage-2 outputs

- [run_20th](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_20th)

This folder contains:

- the weighted-sampling cross-attention run
- the successful Model 3 functional-pruning stage-2 run

---

## 7. Best Results To Remember

If you only need the headline results:

### Best full-data Model 2 result

From `run_19th` cross-attention:

- MAE: `397.43 ms`
- RMSE: `1720.46 ms`
- R²: `0.4604`

### Best Model 3 result

From `run_20th` functional pruning:

- MAE: `384.13 ms`
- RMSE: `1686.06 ms`
- R²: `0.4817`
- `57` heads functionally disabled at the best pruning ratio

### Best practical trimmed-tail result

From the trimmed-tail cross-attention run:

- MAE: `245.01 ms`
- RMSE: `689.26 ms`
- R²: `0.5044`

This is the strongest practical result if trimming the extreme tail is acceptable.

---

## 8. Suggested Reading Order

If you want the clearest reading order, follow this:

1. [EXPERIMENT_REPORT.md](/Users/vaidhav/Desktop/MyProjects/dl-final-project/EXPERIMENT_REPORT.md)
2. [MODEL1_VARIANTS.md](/Users/vaidhav/Desktop/MyProjects/dl-final-project/MODEL1_VARIANTS.md)
3. [MODEL2_VARIANTS.md](/Users/vaidhav/Desktop/MyProjects/dl-final-project/MODEL2_VARIANTS.md)
4. [MODEL3_VARIANTS.md](/Users/vaidhav/Desktop/MyProjects/dl-final-project/MODEL3_VARIANTS.md)
5. [OTHER_VARIANTS.md](/Users/vaidhav/Desktop/MyProjects/dl-final-project/OTHER_VARIANTS.md)
6. [FIGURE_GUIDE.md](/Users/vaidhav/Desktop/MyProjects/dl-final-project/FIGURE_GUIDE.md)

That order gives:

- the overall story first
- then each model family in depth
- then the side branches
- then the figure interpretation

---

## 9. Final Purpose Of This Documentation

Together, these files are meant to give:

- a complete experiment history
- clear model-family separation
- honest discussion of failures and successes
- enough context to support a final report or presentation

This index is the central map connecting all of that material.
