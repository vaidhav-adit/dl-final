# Figure Guide

This guide explains which exported images are the most useful, where they fit in the report, and what each one tells us.

## 1. Dataset / Cleaning Section

### Figure
- [cleaning_dataset_distributions.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/cleaning_dataset_distributions.png)

### Recommended placement
- Put this in the dataset or preprocessing section, right after describing the cleaned CSV and the engineered feature set.

### Suggested caption
- `Dataset inspection after cleaning: runtime distributions, feature-scale spread, and runtime bucket counts.`

### What it tells us
- The runtime values still span a very large range even after cleaning.
- The `log_execution_time` distribution is much more manageable than raw `execution_time`, which justifies training in log space.
- The feature-range panel shows why numeric feature normalization was necessary.
- The runtime bucket panel shows that slower programs are much less common, which explains why the models struggled on the slow tail.

### Why it matters
- This figure supports the claim that preprocessing was not cosmetic.
- It explains later modeling behavior before any model is trained.

## 2. Baseline Model 1 Section

### Figure
- [run18_normal_training_diagnostics.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run18_normal_training_diagnostics.png)

### Recommended placement
- Put this in the Model 1 results section before the final metrics table.

### Suggested caption
- `Training diagnostics for the corrected baseline model (Model 1).`

### What it tells us
- The train and validation losses both decrease, so the baseline is learning meaningful signal.
- Validation MAE decreases over epochs, showing improvement in prediction error measured in milliseconds.
- Validation R² improves, meaning the baseline starts to explain some of the variance in runtime.
- The training is relatively stable after the corrected preprocessing and loss fixes.

### Why it matters
- This figure shows that the baseline is a valid reference model, not a failed run.
- It justifies using Model 1 as the comparison point for Model 2.

### Figure
- [run18_normal_test_plots.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run18_normal_test_plots.png)

### Recommended placement
- Put this immediately after the baseline metrics.

### Suggested caption
- `Prediction-quality plots for Model 1 on the test set.`

### What it tells us
- The predicted-vs-actual scatter shows that the baseline captures the broad trend but still has large spread away from the diagonal.
- The residual plot shows that error grows for larger actual runtimes.
- The error histogram shows a wide error distribution with a long tail.

### Why it matters
- This figure makes the weakness of Model 1 visible.
- It supports the later claim that the baseline especially struggles on harder examples and slow programs.

## 3. Corrected Cross-Attention Model 2 Section

### Figure
- [run18_cross_training_diagnostics.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run18_cross_training_diagnostics.png)

### Recommended placement
- Put this at the beginning of the Model 2 results subsection.

### Suggested caption
- `Training diagnostics for the corrected cross-attention model (Model 2).`

### What it tells us
- Training and validation losses improve more strongly than in the baseline.
- Validation MAE is lower than in Model 1.
- Validation R² reaches a better level than the baseline.
- Overall convergence is better, which suggests that schedule-aware code attention helps optimization and representation quality.

### Why it matters
- This is the first figure that visually supports the main architectural claim of the project: cross-attention is better than shallow concatenation.

### Figure
- [run18_cross_test_plots.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run18_cross_test_plots.png)

### Recommended placement
- Put this right after the Model 2 metric table.

### Suggested caption
- `Prediction-quality plots for Model 2 on the test set.`

### What it tells us
- The scatter is tighter than the baseline, showing better alignment between prediction and ground truth.
- The residual spread is still significant, but the fit is better overall.
- The histogram still shows heavy-tail error behavior, which means Model 2 is better but not perfect.

### Why it matters
- This figure gives the visual evidence that Model 2 improves on Model 1.
- It also shows that the remaining challenge is not solved yet, especially on difficult samples.

### Figure
- [run18_cross_attention_heatmap.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run18_cross_attention_heatmap.png)

### Recommended placement
- Put this in an interpretability or “why Model 2 helps” subsection.

### Suggested caption
- `Cross-attention heatmap showing which code tokens the schedule embedding focuses on.`

### What it tells us
- The schedule embedding is not treated as just another numeric vector. It actively attends to specific code tokens.
- This supports the idea that the model is using token-level structure rather than just a pooled code embedding.
- Depending on the actual heat concentration, you may see stronger attention around loop constructs, indices, or computational expressions.

### Why it matters
- This is one of the most visually persuasive figures in the project.
- It helps explain *why* Model 2 is more expressive than Model 1.

### Figure
- [run18_cross_schedule_variation_attention.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run18_cross_schedule_variation_attention.png)

### Recommended placement
- Put this immediately after the single heatmap.

### Suggested caption
- `Cross-attention changes as the schedule parameters change for the same code snippet.`

### What it tells us
- The same code can produce different attention patterns under different schedules.
- This is strong evidence that the cross-attention mechanism is schedule-conditioned rather than static.

### Why it matters
- This supports the project’s central interpretation of Model 2: the model is learning interactions between code structure and optimization settings.

## 4. Best Accuracy-Focused Run Section

### Figure
- [run19_cross_training_diagnostics.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run19_cross_training_diagnostics.png)

### Recommended placement
- Put this in the section describing the partially unfrozen cross-attention experiment.

### Suggested caption
- `Training diagnostics for the partially unfrozen cross-attention run.`

### What it tells us
- The model still improves, but validation is more unstable than the fully frozen Model 2 case.
- This reflects the tradeoff of partial backbone adaptation: potentially better fit, but harder optimization.
- It shows that this run pushed the architecture further than the earlier stable frozen setup.

### Why it matters
- This figure supports the claim that the best overall executed run came from a slightly more adaptive Model 2.

### Figure
- [run19_cross_test_plots.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run19_cross_test_plots.png)

### Recommended placement
- Put this right after the metrics for the `run_19th` cross-attention model.

### Suggested caption
- `Prediction-quality plots for the best executed cross-attention run.`

### What it tells us
- This is the strongest visual result among the executed accuracy-focused cross-attention runs.
- The fit is still imperfect, but overall variance explanation is better than earlier versions.
- The remaining scatter and residual structure reinforce that the hardest problem is still the slow-runtime tail.

### Why it matters
- This is probably the most important model-performance figure in the project if you want to show your strongest executed result.

### Figure
- [run19_cross_attention_heatmap.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run19_cross_attention_heatmap.png)

### Recommended placement
- Put this in the same interpretability subsection as the earlier cross-attention heatmap, or use this one instead if it looks cleaner.

### Suggested caption
- `Attention heatmap from the stronger cross-attention run.`

### What it tells us
- Same interpretability story as the earlier heatmap, but from the stronger run.
- If visually clearer, it is the better candidate for the final report.

### Figure
- [run19_cross_schedule_variation_attention.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run19_cross_schedule_variation_attention.png)

### Recommended placement
- Put this next to the `run19` attention heatmap if you want one consistent interpretability pair from the same run.

### Suggested caption
- `Schedule-conditioned attention variation in the stronger cross-attention run.`

### What it tells us
- The attention mechanism remains sensitive to schedule changes even in the stronger setup.
- This supports the interpretation that the model is learning schedule-aware code reasoning rather than a fixed text embedding.

## 5. Failure / Negative Result Section

### Figure
- [run20_weighted_training_diagnostics.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run20_weighted_training_diagnostics.png)

### Recommended placement
- Put this in a short “negative experiment” or “what did not work” section.

### Suggested caption
- `Training diagnostics for the weighted-sampling experiment focused on slow programs.`

### What it tells us
- Training is noticeably less stable.
- Validation metrics remain poor for much of training.
- The model only partially recovers late, which suggests the weighting scheme distorted optimization too much.

### Why it matters
- This figure is useful because it shows that not every apparently reasonable intervention improves the model.
- It supports a careful experimental narrative rather than a cherry-picked one.

### Figure
- [run20_weighted_test_plots.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run20_weighted_test_plots.png)

### Recommended placement
- Put this next to the negative-result discussion if you want to show why the weighted-sampling approach was not adopted.

### Suggested caption
- `Prediction-quality plots for the weighted-sampling run, which degraded overall performance.`

### What it tells us
- Predictions worsened overall despite targeting the slow tail.
- The model did not gain enough on slow programs to compensate for broader quality loss.
- The residual and scatter plots make that regression visible.

### Why it matters
- This figure helps explain why the project did not continue in that direction.

## 6. Recommended Final Figure Set

If you want a concise but strong report, use this set:

1. [cleaning_dataset_distributions.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/cleaning_dataset_distributions.png)
2. [run18_normal_test_plots.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run18_normal_test_plots.png)
3. [run18_cross_test_plots.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run18_cross_test_plots.png)
4. [run19_cross_test_plots.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run19_cross_test_plots.png)
5. [run19_cross_attention_heatmap.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run19_cross_attention_heatmap.png)
6. [run20_weighted_test_plots.png](/Users/vaidhav/Desktop/MyProjects/dl-final-project/report_images/run20_weighted_test_plots.png)

This set gives you:

- data understanding
- baseline performance
- Model 2 improvement
- best executed result
- interpretability
- one clear failure case

That is enough to tell a complete and honest story without overwhelming the reader.
