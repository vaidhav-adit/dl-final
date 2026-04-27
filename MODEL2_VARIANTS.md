# Model 2 Variants

This document covers the full Model 2 family in detail. Model 2 is the most important architecture family in the project because this is where the first clear and consistent performance gains happened.

Model 2 was the point where the project moved from:

- “Can a simple code-plus-feature model learn anything?”

to:

- “Can schedule-aware token-level interaction between code and schedule features improve runtime prediction?”

That question defined the real research direction of the project, and most of the important model-level progress happened inside this family.

---

## 1. What Model 2 Was Trying To Do

Model 1 had already shown that:

- CodeBERT features contain useful signal
- numeric schedule features contain useful signal
- a fused regression model can learn something meaningful

But Model 1 also revealed a limitation:

- code and schedule information were being fused too shallowly

Concatenation gives the model access to both modalities, but it does not force or encourage a dynamic interaction between them.

That is where Model 2 came in.

The core idea of Model 2 was:

> Let the schedule embedding directly attend over the code tokens so the model can ask: “given this schedule, which parts of the code matter most?”

That is the defining conceptual leap of Model 2.

Instead of just combining one code vector and one schedule vector side by side, Model 2 tries to create **schedule-conditioned code understanding**.

---

## 2. What The Architecture Actually Is

The main notebook defining Model 2 is:

- [normal-cross-attention-model.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/normal-cross-attention-model.ipynb)

At a high level, the architecture works like this:

1. The code string is tokenized and passed through CodeBERT.
2. CodeBERT outputs token embeddings for the full token sequence.
3. The 10 schedule features are passed through an MLP to produce a schedule embedding.
4. That schedule embedding is used as a **query**.
5. The code token embeddings are used as **keys** and **values**.
6. Cross-attention produces a schedule-conditioned code summary.
7. The attended code summary is fused with the schedule embedding.
8. A regression head predicts `log_execution_time`.

In simplified symbolic form:

```text
Code_String -> CodeBERT -> code token embeddings
Schedule features -> MLP -> schedule embedding
schedule embedding queries code tokens -> attended code summary
[attended code summary ; schedule embedding] -> regressor -> log runtime
```

This matters because now the schedule has a meaningful mechanism to interact with the code representation before prediction.

---

## 3. Why Model 2 Was A Meaningful Change

The difference between Model 1 and Model 2 is not cosmetic.

### Model 1

Model 1 does:

- encode code
- encode schedule
- concatenate
- regress

This is still useful, but it leaves the interaction burden mostly to the final MLP/regression head.

### Model 2

Model 2 does:

- encode code into tokens
- encode schedule into a query
- let the schedule inspect the code tokens directly
- create a schedule-conditioned code summary
- then regress

This means:

- the model can focus on different code tokens depending on the schedule
- the fusion happens before final regression
- the code representation used for prediction is no longer static with respect to the schedule

That is why Model 2 is a genuine architectural step up.

---

## 4. Shared Data Pipeline Used By Corrected Model 2 Variants

The corrected Model 2 family used the same cleaned dataset and feature space as the corrected Model 1 family:

- dataset: [Cleaned_LOOPerSet_GEM.csv](/Users/vaidhav/Desktop/MyProjects/dl-final-project/Cleaned_LOOPerSet_GEM.csv)
- target: `log_execution_time`
- schedule feature count: 10

The feature vector was:

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

Important preprocessing carried into Model 2:

- feature normalization using train-set mean/std
- removal of the old `Vec` feature
- removal of the broken unroll weighting
- prediction and evaluation still reported in milliseconds after inverse transform

This is important because it means Model 2’s improvements were not caused by a cleaner input pipeline alone. The pipeline was shared across the corrected models. The gains in Model 2 are therefore more meaningfully attributable to the architecture.

---

## 5. Main Methodology Used In Model 2 Runs

Across the main Model 2 variants, the overall training methodology was:

1. Load the cleaned CSV.
2. Split into train/validation/test.
3. Normalize the numeric schedule features using train-set statistics only.
4. Tokenize `Code_String` using the CodeBERT tokenizer.
5. Encode full code tokens with CodeBERT.
6. Encode the schedule vector with an MLP.
7. Run cross-attention from schedule to code tokens.
8. Predict `log_execution_time`.
9. Save best checkpoint based on validation performance.
10. Evaluate on test data in milliseconds.

The evaluation stack included:

- MAE
- RMSE
- median absolute error
- MAPE
- R²

Model 2 also introduced an important interpretability layer:

- cross-attention heatmaps
- attention changes under different schedule settings

This made Model 2 much easier to defend conceptually. It was not just “better metrics,” it was also “a better mechanism with interpretable schedule-conditioned token behavior.”

---

## 6. Early Executed Model 2 Result

One of the earliest executed full Model 2 notebooks was:

- [done/normal-cross-attention-model-50epoch.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/done/normal-cross-attention-model-50epoch.ipynb)

This run was important because it provided the first real end-to-end evidence that the cross-attention idea worked.

What this early stage established:

- Model 2 trained successfully
- it improved over the corresponding baseline
- the project’s architectural direction was not a dead end

This was the first real sign that moving from shallow fusion to schedule-aware token interaction was worthwhile.

Even though later corrected runs became more important, this early executed Model 2 run was historically significant because it justified continuing with the architecture.

---

## 7. Corrected Frozen Model 2: `run_18th`

The main corrected frozen Model 2 run was:

- [run_18th/cross/norm-cross.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_18th/cross/norm-cross.ipynb)

This is the cleanest reference version of Model 2 before more aggressive optimization changes were tried.

### 7.1 Architecture and training setup

This version used:

- frozen CodeBERT
- corrected 10-feature normalization
- cross-attention fusion
- standard log-space regression loss
- no fake unroll weighting

So this was the right controlled comparison against the corrected Model 1 baseline.

### 7.2 Observed results

Test metrics:

- MAE: `396.27 ms`
- RMSE: `1810.33 ms`
- median AE: `18.71 ms`
- MAPE: `174.66%`
- R²: `0.4025`

### 7.3 Why this run mattered

This run was a major project milestone because it achieved what Model 2 was supposed to achieve:

- it beat Model 1 clearly
- it did so under the same corrected preprocessing pipeline
- the gain was not just one metric getting lucky; it improved MAE, RMSE, median AE, and R²

This means the architectural change was real.

### 7.4 Critical analysis

Why did this run improve?

Because the schedule-conditioned query mechanism gave the model a way to emphasize different code tokens for different schedule settings.

That is something Model 1 could not do well with simple concatenation.

But why was it still not “great”?

Because:

- CodeBERT was still frozen
- the interaction happened only once
- the dataset still had a very difficult long-runtime tail

So this run proved the architecture direction was correct, but also showed that stronger schedule-aware fusion alone would not magically eliminate the hardest error cases.

---

## 8. Stronger Model 2: Partially Unfrozen Cross-Attention (`run_19th`)

The strongest executed full-dataset Model 2 result came from:

- [run_19th/cross/norm-cross-2.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_19th/cross/norm-cross-2.ipynb)

This run tried to improve Model 2 by giving the pretrained backbone limited task-specific flexibility.

### 8.1 What changed relative to the frozen Model 2

This run introduced:

- partial CodeBERT unfreezing
- a stronger optimization setup
- separate learning rates for backbone and head

The purpose was to allow:

- the code representation to adapt slightly to runtime prediction
- without paying the cost of fully fine-tuning the entire backbone

### 8.2 Observed test results

Test metrics:

- MAE: `397.43 ms`
- RMSE: `1720.46 ms`
- median AE: `20.75 ms`
- MAPE: `159.46%`
- R²: `0.4604`

### 8.3 How to interpret this correctly

At first glance, the MAE looks almost unchanged relative to `run_18th`.

That might suggest the run was not useful. But that would be the wrong conclusion.

The more important changes are:

- RMSE improved
- R² improved
- MAPE improved

This means:

- the model explained overall variance better
- some large-error behavior improved
- the fit was stronger in a more global sense than MAE alone suggests

### 8.4 Why this run mattered so much

This run produced one of the most important lessons of the project:

> the model is not uniformly bad; its biggest weakness is the slow-runtime tail

The error analysis for this run showed:

- small and medium runtime ranges were much more manageable
- the severe failures were concentrated in the `1–10 s` and `>10 s` buckets

That changed the project’s diagnosis.

The problem was no longer:

- “the model doesn’t learn enough”

It became:

- “the model learns the common regions reasonably well, but under-handles the slow tail”

That is a much more actionable and scientifically useful conclusion.

### 8.5 Critical analysis

Why did partial unfreezing help?

Because it gave the code encoder limited freedom to adapt from generic code semantics toward task-specific runtime-relevant semantics.

Why did instability increase?

Because once the pretrained backbone starts moving, training becomes more sensitive:

- the code representation changes over time
- the cross-attention head must adapt with it
- the model is more powerful, but also easier to destabilize

So this run sat in an important middle ground:

- more powerful than the frozen version
- still tractable
- but noticeably more volatile

Despite the instability, this run became the best full-dataset executed Model 2 result.

---

## 9. Weighted-Sampling Model 2 Attempt

Once the slow tail had been identified as the major weakness, the next idea was to fight that imbalance directly using weighted sampling.

This was captured in:

- [cross-attention-weighted-accuracy.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cross-attention-weighted-accuracy.ipynb)
- executed as [run_20th/notebookb835584806.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_20th/notebookb835584806.ipynb)

### 9.1 Methodology

The logic was:

- give rare slow examples more sampling weight
- train the cross-attention model to care more about those difficult samples

This sounds reasonable given the error analysis, and that is exactly why the experiment was worth running.

### 9.2 Observed results

Test metrics:

- MAE: `450.69 ms`
- RMSE: `1921.88 ms`
- median AE: `23.06 ms`
- MAPE: `192.69%`
- R²: `0.3266`

### 9.3 Critical analysis

This experiment failed.

Why?

The weights used for the slow buckets were too aggressive relative to the overall distribution.

Instead of gently correcting the imbalance, the training process became overly distorted:

- validation was unstable
- medium-runtime performance worsened
- the slow tail did not improve enough to compensate

This is an important scientific result even though it is negative.

It shows that:

- identifying the problem correctly is not enough
- the intervention must also be calibrated properly

In this case:

- the diagnosis “slow tail is the issue” was right
- the solution “strong weighted sampling” was wrong

That is a valuable distinction.

---

## 10. Tail-Trimming Model 2

The most practically successful later Model 2 direction was tail trimming.

Relevant notebooks:

- [cross-attention-trimmed-tail-40epoch.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cross-attention-trimmed-tail-40epoch.ipynb)
- [final-cross-attention-trimmed-tail.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/final-cross-attention-trimmed-tail.ipynb)

### 10.1 Why trimming was tried

After repeated evidence that the hardest extreme tail was driving instability and large-error behavior, the idea was:

- remove only the most pathological runtime cases
- keep the rest of the dataset
- retrain the strongest validated architecture

This was not “cheating” in the casual sense. It was a deliberate reframing of the task:

- not “predict every extreme case no matter how pathological”
- but “train a strong model on the meaningful bulk of the distribution”

### 10.2 Observed trimming setup

For the main trimmed-tail run:

- original rows: `83,973`
- filtered rows: `82,757`
- removed rows: `1,216`
- trimming rule: `log_execution_time <= 9.0`

So only a small fraction of rows were removed, but they were disproportionately harmful.

### 10.3 Observed results

Test metrics:

- MAE: `245.01 ms`
- RMSE: `689.26 ms`
- median AE: `22.73 ms`
- MAPE: `186.98%`
- R²: `0.5044`

### 10.4 Why this result is so important

This is probably the strongest practical Model 2 result in the project.

The RMSE drop is especially dramatic.

That tells us something very important:

- the extreme tail was not just a minor nuisance
- it was one of the main reasons earlier full-dataset models looked so bad in absolute terms

### 10.5 Critical analysis

Why did trimming help so much?

Because the removed examples were exactly the cases that produced huge residuals and destabilized training.

By removing a relatively small but highly damaging tail:

- the learning problem became much better conditioned
- the model could fit the bulk distribution more effectively
- aggregate error metrics improved sharply

Why is this still a nuanced result?

Because it changes the problem definition.

The model is now strong on:

- the trimmed distribution

not on:

- the full original long-tail distribution

So the result is highly useful, but it must be described honestly:

- it is the best practical model on the trimmed task

---

## 11. What Model 2 Contributed To The Project

Model 2 is the family that carried the project scientifically.

It contributed several major things:

### 11.1 It proved schedule-aware fusion matters

This is the central architectural contribution of the project.

### 11.2 It gave the strongest executed full-dataset result

The `run_19th` cross-attention model became the best full-data executed result.

### 11.3 It revealed the real bottleneck

Through error analysis, Model 2 showed that:

- the model learns the common regions reasonably well
- the slow-runtime tail is the major unresolved difficulty

### 11.4 It responded best to task-specific architectural improvements

Unlike Model 1, Model 2 clearly benefited from better schedule-aware structure.

That makes it the best validated model family in the project.

---

## 12. Final Critical Assessment Of Model 2

If we step back and look at the whole Model 2 family, the overall lesson is very strong.

### What worked

- cross-attention itself
- corrected preprocessing
- normalized numeric features
- full-token code access
- limited backbone adaptation
- tail trimming

### What did not work well

- aggressive weighted sampling
- assuming the slow tail could be fixed only by data-loader reweighting

### What Model 2 teaches overall

The project improved most when:

- the model architecture respected the structure of the task
- the schedule was given a meaningful way to interact with the code
- the data distribution problem was addressed honestly

Model 2 was not just “slightly better than Model 1.”
It was the first family that really aligned the modeling method with the actual scientific problem.

---

## 13. Final Summary Of Model 2

Model 2 is the strongest validated architecture family in the project.

Its main arc was:

1. start with frozen cross-attention
2. prove it beats the baseline
3. improve it with limited backbone adaptation
4. diagnose the slow tail as the major issue
5. show that aggressive weighted sampling fails
6. show that trimming the extreme tail gives a very strong practical model

### Best full-dataset executed Model 2 result

From `run_19th`:

- MAE: `397.43 ms`
- RMSE: `1720.46 ms`
- R²: `0.4604`

### Best practical trimmed-tail Model 2 result

From the trimmed-tail run:

- MAE: `245.01 ms`
- RMSE: `689.26 ms`
- R²: `0.5044`

### Most important scientific conclusion

Model 2 showed that schedule-aware token-level interaction is essential for this task, and that the hardest remaining obstacle was not basic learning, but the extreme runtime tail.
