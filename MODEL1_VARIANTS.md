# Model 1 Variants

This document covers the full Model 1 line of work in detail. It explains what Model 1 was supposed to do, how the pipeline evolved, what methodology was used in each phase, what results were observed, and what those results actually mean.

Model 1 matters because it is the baseline family for the entire project. Even though later models became stronger, Model 1 is still the reference point that makes the rest of the experimental story understandable.

---

## 1. What Model 1 Was Trying To Do

Model 1 was the simplest serious runtime-prediction model in the project.

Its purpose was not to be the most sophisticated architecture. Its purpose was to answer a basic but important question:

> If we combine a pretrained code encoder with numeric schedule features and train a regression head, can we predict execution time with any meaningful accuracy?

That question matters because if the answer had been “no,” then there would have been no reason to build Model 2 or Model 3.

At a high level, Model 1 does four things:

1. It reads the code using CodeBERT.
2. It reads the schedule/hardware-related numeric features using a small MLP.
3. It fuses the code representation and schedule representation by concatenating them.
4. It predicts a single scalar target: `log_execution_time`.

So Model 1 is fundamentally a **two-tower regression architecture**:

- Tower A: code understanding
- Tower B: schedule-feature understanding

Then both towers are fused for runtime prediction.

The model is “simple” only relative to the later variants. It is still already a multimodal regression system combining text/code and structured numeric inputs.

---

## 2. Why Model 1 Was Needed

There were several reasons Model 1 was necessary.

### 2.1 It established whether the cleaned dataset was learnable

Before running a stronger architecture, the project needed to answer whether the cleaned dataset had enough signal to support runtime learning at all.

If even a simple fusion model could not learn:

- either the dataset was too noisy
- or the feature design was too weak
- or the problem setup was fundamentally flawed

Model 1 was the first real test of that.

### 2.2 It gave a baseline for judging later improvements

Without Model 1, Model 2 would have had no proper comparison point.

If Model 2 later performed better, that improvement would only mean something if there was a stable baseline to beat.

So Model 1 was the anchor for the rest of the project.

### 2.3 It exposed pipeline mistakes early

A simpler model makes pipeline bugs easier to notice:

- wrong feature shapes
- bad loss design
- poor target scaling
- broken inference transformations
- mismatch between train and eval behavior

Several of those issues were indeed discovered and corrected while working through the Model 1 family.

---

## 3. Input Data Used By Model 1

The later corrected Model 1 variants used the cleaned dataset:

- [Cleaned_LOOPerSet_GEM.csv](/Users/vaidhav/Desktop/MyProjects/dl-final-project/Cleaned_LOOPerSet_GEM.csv)

The important columns for Model 1 were:

- `Code_String`
- `T1`
- `T2`
- `Unroll`
- `locality`
- `working_set_size`
- `arithmetic_intensity`
- `T1_T2_inter`
- `log_T1`
- `log_T2`
- `T1_div_T2`
- `log_execution_time`

### 3.1 The target variable

Model 1 does **not** train on raw `execution_time`.

It trains on:

- `log_execution_time`

This choice was essential because raw execution time had a very long and highly skewed distribution.

Training directly on raw milliseconds would have caused several problems:

- unstable gradients
- huge domination of extreme slow programs
- poor sensitivity to smaller but still meaningful differences among the faster programs

Using `log_execution_time` made the target easier to model and made loss values much more manageable.

During inference and evaluation, predictions were converted back into milliseconds using the inverse transform.

### 3.2 Numeric feature space

The numeric input to Model 1 was a 10-dimensional vector:

1. `T1`
2. `T2`
3. `Unroll`
4. `locality`
5. `working_set_size`
6. `arithmetic_intensity`
7. `T1_T2_inter`
8. `log_T1`
9. `log_T2`
10. `T1_div_T2`

This feature space already contained both:

- raw schedule settings
- hand-engineered interaction terms

That matters, because Model 1 was never just “CodeBERT plus a few compiler flags.” Even the baseline had a nontrivial schedule-feature design.

---

## 4. The Original Model 1 Architecture

The earliest Model 1 notebook in the root folder was:

- [normal-model.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/normal-model.ipynb)

The original baseline architecture was:

- frozen CodeBERT backbone
- schedule-feature MLP
- concatenation-based fusion
- regression head

Conceptually:

```text
Code_String -> CodeBERT -> code embedding
Schedule features -> MLP -> schedule embedding
[code embedding ; schedule embedding] -> regressor -> log runtime
```

### 4.1 Code side

The code tower used:

- `microsoft/codebert-base`

This is an encoder-only pretrained transformer.

In the earliest versions, the code representation was sometimes taken from the `CLS` token or an equivalent single pooled token representation.

This was simple, but limited. It compressed the entire code string into one vector immediately, which meant a lot of token-level structure was lost before the schedule information was fused.

### 4.2 Schedule side

The schedule tower was a small MLP.

Its job was:

- take the 10 numeric schedule features
- convert them into a learned dense embedding
- make them compatible with the code representation before fusion

This tower was always important because runtime prediction is not driven by code alone. In this task, schedule and memory-related numeric factors are major determinants of runtime.

### 4.3 Fusion

The fusion step was simple concatenation:

- `[code_embedding, schedule_embedding]`

That is the defining limitation of Model 1.

Concatenation gives the regressor access to both modalities, but it does **not** create a dynamic interaction between them. It simply places them side by side and leaves the regression head to figure out the relationship.

This is exactly why Model 2 later mattered so much.

---

## 5. Problems Discovered In Early Model 1 Versions

Before the corrected Model 1 runs, several issues were found.

These issues are important because they explain why some earlier notebooks were weaker than they should have been.

### 5.1 The fake unroll weighting problem

One training loop used logic like:

```python
weights = torch.where(unroll_factor > 0, 5.0, 1.0)
```

This was supposed to emphasize examples with unrolling.

But after cleaning the dataset, `Unroll` was always positive. That means:

- every sample received the same weight
- the loss was just globally scaled
- no subset was truly emphasized

So the intended weighting effect did not exist anymore.

This mattered because it created the illusion of task-specific weighting while actually doing nothing useful.

### 5.2 Numeric features were not originally normalized

The schedule features had very different scales:

- some were small binaries or near-binaries like `locality`
- some were moderate like `arithmetic_intensity`
- some were much larger like `working_set_size`
- some engineered terms such as `T1_T2_inter` could be much larger still

Without normalization, the MLP had to learn under badly mismatched feature scales.

That makes optimization harder and can bias which features dominate early learning.

### 5.3 CLS-only representation was weak

When a single pooled code vector is used too early, the model has fewer ways to align code structure with schedule effects.

That does not make the model invalid, but it limits expressive power.

This is why the corrected Model 1 switched to mean pooling across valid tokens rather than relying on one special token summary.

### 5.4 Train/eval mismatch in frozen CodeBERT

At one point, CodeBERT was frozen in terms of gradients but was still left in training mode.

That meant dropout could remain active during training even though the backbone was supposed to act as a stable feature extractor.

That creates noise in the code representation and also creates a mismatch between training behavior and validation/inference behavior.

This issue was later corrected by forcing the frozen CodeBERT module into eval mode during training.

---

## 6. Corrected Model 1 Methodology

The most important corrected Model 1 run was:

- [run_18th/normal/normal-model.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_18th/normal/normal-model.ipynb)

This run is the best reference version of Model 1 as a clean baseline.

### 6.1 Train/validation/test split

The corrected notebook used an explicit train/validation/test split.

This matters because it allowed:

- model selection on validation data
- honest reporting on held-out test data

### 6.2 Feature normalization

The 10 numeric schedule features were normalized using:

- train-set mean
- train-set standard deviation

Then the same train-derived statistics were used on:

- validation set
- test set
- inference-time feature vectors

This is the correct way to normalize and prevents leakage from validation/test into training preprocessing.

### 6.3 Mean pooling instead of raw CLS

Instead of taking only one token representation, the corrected baseline pooled across valid code tokens using the attention mask.

This gave a more stable whole-code representation and reduced dependence on a single token embedding.

### 6.4 Loss and optimization

The corrected baseline used:

- standard regression loss in log space
- no fake unroll weighting
- frozen CodeBERT
- trainable schedule MLP and regression head

### 6.5 Evaluation methodology

The model was evaluated using:

- MAE
- RMSE
- median absolute error
- MAPE
- R²

Predictions were converted back to milliseconds using `expm1`.

This was important because it separated:

- training space: log scale
- evaluation space: physical runtime scale

That makes the reported results interpretable.

---

## 7. Main Results For Corrected Model 1

### 7.1 `run_18th` corrected baseline results

Observed test metrics:

- MAE: `432.08 ms`
- RMSE: `1885.54 ms`
- median AE: `25.02 ms`
- MAPE: `324.20%`
- R²: `0.3518`

### 7.2 What these numbers mean

This result is mixed, and it needs to be interpreted carefully.

#### What is good about it

- The model clearly learned nontrivial signal.
- R² above `0.35` means it explained some meaningful portion of runtime variance.
- Median absolute error around `25 ms` means that many predictions were actually fairly reasonable.

#### What is not good about it

- RMSE is very high.
- MAE is much larger than the median error.
- MAPE is extremely high.

The important insight is this:

- the model was not uniformly bad
- it was decent on a large fraction of samples
- but it made some very large mistakes

That is why:

- median AE looks much better
- while RMSE remains very large

So the core weakness was not that the baseline could not learn anything. The weakness was that it was still fragile on difficult examples.

---

## 8. Later Stronger Model 1 Variant

Another important Model 1 result came from:

- [run_19th/normal/norm-model-2.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_19th/normal/norm-model-2.ipynb)

This run tested a stronger version of the baseline.

### 8.1 Method changes

Relative to the earlier corrected baseline, this variant introduced:

- limited CodeBERT unfreezing
- better LR separation between backbone and head
- more task adaptation capacity

So this was still Model 1 conceptually, but it was no longer a purely frozen feature-extractor baseline.

### 8.2 Observed results

Test metrics:

- MAE: `403.34 ms`
- RMSE: `1778.31 ms`
- median AE: `20.81 ms`
- MAPE: `151.08%`
- R²: `0.4235`

### 8.3 Interpretation

This was a real improvement over the corrected frozen baseline.

That suggests:

- the baseline family did benefit from limited backbone adaptation
- some of the earlier weakness came from a too-rigid code representation

But even after that improvement, Model 1 was still not the best family in the project.

That is important. It means the limitation of Model 1 was not just optimization setup. Part of the limitation was architectural.

---

## 9. Critical Analysis Of Model 1 Results

This is the part that matters most scientifically.

### 9.1 Why Model 1 worked at all

Model 1 worked because the task has real signal in both modalities:

- code structure matters
- schedule features matter

CodeBERT gave a reasonable code prior, and the numeric feature tower gave the model enough schedule context to make meaningful runtime predictions.

### 9.2 Why Model 1 plateaued

The deeper reason Model 1 plateaued is that its fusion mechanism was too shallow.

Concatenation is simple, but it does not allow the schedule to dynamically inspect the code tokens.

That means the model must rely on the regression head to infer interactions like:

- “this loop structure behaves differently under this tile size”
- “this memory pattern becomes expensive under this schedule”

That is a hard burden to place on a shallow fusion head.

### 9.3 Why the outlier problem remained severe

The outlier-heavy error profile suggests that Model 1 could fit the center of the distribution better than the extremes.

This makes sense because:

- the model has limited schedule-conditioned expressiveness
- the data distribution itself is highly imbalanced
- extreme slow-runtime examples are harder and rarer

So Model 1 naturally learned the easier, more common regions first and struggled with the hardest tail cases.

### 9.4 Why partial unfreezing helped but did not solve everything

Partially unfreezing CodeBERT helped because it allowed the code representation to adapt somewhat to the runtime-prediction task.

But it did not fully solve the problem because:

- the fusion mechanism was still concatenation-based
- the model still lacked a stronger interaction mechanism between code tokens and schedule information

In other words:

- better optimization helped
- but architecture still mattered

---

## 10. What Model 1 Contributed To The Project

Model 1 was not the final best architecture, but it contributed several essential things:

### 10.1 It validated the task setup

It showed the cleaned data and target setup were workable.

### 10.2 It exposed preprocessing and training issues

Without Model 1, it would have been harder to notice:

- the useless weighting logic
- the need for feature normalization
- the importance of using a better pooled code representation

### 10.3 It gave a fair baseline for Model 2

Model 2’s gains only mean something because Model 1 existed first.

### 10.4 It showed that architecture matters

Model 1’s limitations helped reveal that:

- preprocessing alone would not be enough
- a deeper schedule-conditioned fusion mechanism was needed

---

## 11. Final Summary Of Model 1

The Model 1 family was the baseline family of the project.

It started as:

- frozen CodeBERT
- schedule MLP
- concatenation fusion
- regression head

It later became stronger through:

- better preprocessing
- mean pooling
- feature normalization
- limited backbone adaptation

Its best later result showed that Model 1 could become respectable, but not dominant.

### Best baseline-style result

From `run_19th`:

- MAE: `403.34 ms`
- RMSE: `1778.31 ms`
- R²: `0.4235`

### Most important scientific conclusion

Model 1 proved that the problem was learnable, but it also proved that shallow fusion was not enough.

That is the key reason the project had to move into the Model 2 family.
