# Other Variant Experiments

This document covers the experimental directions that do not belong cleanly inside the main Model 1, Model 2, or Model 3 family structure.

These variants are important because they show the broader search process of the project. Not every useful experiment was a clean “Model 1 / Model 2 / Model 3” instance. Some were:

- data-centric strategies
- optimization experiments
- exploratory architecture directions
- convenience notebooks for testing alternative training setups

These side experiments mattered because they helped answer questions like:

- Is the slow tail the real problem?
- Can weighting or oversampling fix it?
- Is a different transformer fusion strategy better than standard cross-attention?
- Is the strongest practical model the same as the most architecturally interesting one?

So even though these variants sit outside the clean three-model storyline, they were crucial in shaping the final conclusions.

---

## 1. What Counts As “Other Variants”

For this document, “other variants” includes four main categories:

1. **Cleaning and diagnostic notebooks**
2. **Oversampling / weighted sampling experiments**
3. **Tail-trimming experiments**
4. **New architecture experiments outside the standard cross-attention/pruning path**

The main notebooks in this category are:

- [cleaning-notebook.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cleaning-notebook.ipynb)
- [accuracy-oversampling-models.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/accuracy-oversampling-models.ipynb)
- [cross-attention-weighted-accuracy.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cross-attention-weighted-accuracy.ipynb)
- [schedule-token-fusion-transformer.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/schedule-token-fusion-transformer.ipynb)
- [cross-attention-trimmed-tail-40epoch.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cross-attention-trimmed-tail-40epoch.ipynb)
- [final-cross-attention-trimmed-tail.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/final-cross-attention-trimmed-tail.ipynb)

This category therefore includes both:

- experiments that failed
- experiments that clarified the problem
- and experiments that produced some of the best practical outcomes

---

## 2. Cleaning Notebook As An Experimental Foundation

The first important non-mainline experiment was not a model at all, but the cleaning and dataset-inspection work captured in:

- [cleaning-notebook.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cleaning-notebook.ipynb)

This notebook deserves to be in the “other variants” document because it influenced many later decisions that did not belong to any one model family.

### 2.1 What this notebook did

It inspected the cleaned dataset and summarized:

- dataset size
- missing values
- duplicate patterns
- runtime distribution
- feature-scale spread
- runtime bucket counts

It also connected those observations to later modeling decisions.

### 2.2 Key findings

The notebook showed:

- the dataset had no missing values
- full duplicate rows were rare
- duplicate `Code_String` values were extremely common
- the runtime distribution remained extremely broad
- the numeric features had very different scales

These were not small observations. They directly shaped later modeling and training choices.

### 2.3 Why this notebook mattered scientifically

This notebook provided the first evidence for several later hypotheses:

- training should use `log_execution_time`
- numeric features must be normalized
- the old loss weighting logic was no longer meaningful
- the hardest difficulty lies in the long-runtime tail

Without this notebook, later experiments such as weighted sampling or tail trimming would have looked arbitrary. With this notebook, those experiments had a clear data-driven motivation.

### 2.4 Critical analysis

The cleaning notebook did not solve the prediction problem, but it prevented the project from continuing under false assumptions.

In particular, it showed that:

- the project’s difficulty was not simply “the model is bad”
- the problem was structurally hard because the data distribution itself was highly imbalanced and highly multi-scale

That is a major conceptual contribution.

---

## 3. Oversampling And Weighted-Sampling Experiments

One natural idea after diagnosing the slow-runtime tail as the main difficulty was:

> if slow programs are rare and hard, maybe the model just is not seeing them often enough during training

That idea led to the oversampling and weighted-sampling experiments.

### 3.1 Why this seemed promising

The earlier cross-attention results had shown that:

- performance on small and medium runtimes was much better
- performance on `1–10 s` and `>10 s` programs was much worse

This strongly suggested a tail-learning problem.

A reasonable hypothesis was therefore:

- reweight the training distribution
- make the model see slow programs more often
- improve tail accuracy

Scientifically, this was a sound question to test.

### 3.2 The general oversampling notebook

The first flexible utility notebook for this direction was:

- [accuracy-oversampling-models.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/accuracy-oversampling-models.ipynb)

This notebook was not primarily a final model notebook. It was a framework notebook for trying oversampling strategies across multiple model families.

It supported:

- baseline-style runs
- cross-attention-style runs

The reason it belongs in this document is that it was more of an experimentation harness than a core final model definition.

### 3.3 Cross-attention weighted-accuracy notebook

The more targeted notebook for this idea was:

- [cross-attention-weighted-accuracy.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cross-attention-weighted-accuracy.ipynb)

This notebook focused on the stronger architecture family only:

- cross-attention
- partial unfreezing
- lower backbone LR
- higher head LR
- weighted sampling for slow programs

That made it a much cleaner tail-focused experiment.

### 3.4 Executed weighted-sampling run

The actual executed run was:

- [run_20th/notebookb835584806.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_20th/notebookb835584806.ipynb)

The weighted sampling setup emphasized:

- `1s–10s` programs
- `>10s` programs

Observed test metrics:

- MAE: `450.69 ms`
- RMSE: `1921.88 ms`
- median AE: `23.06 ms`
- MAPE: `192.69%`
- R²: `0.3266`

### 3.5 What this result means

This experiment did not help. It made the model worse overall.

That result is important because it shows the distinction between:

- identifying the right problem
- finding the right intervention

The problem diagnosis was correct:

- slow-tail difficulty was real

But the intervention was too aggressive:

- the training distribution was distorted too strongly
- medium-runtime performance was damaged
- the tail was not improved enough to compensate

### 3.6 Critical analysis of oversampling

Why did this fail?

Because the weighting likely over-corrected.

Instead of giving the model a slightly better exposure to difficult examples, it pushed training too far away from the natural distribution.

That caused:

- instability
- poorer general fit
- degradation in buckets that had previously been modeled better

So the critical lesson from these experiments is:

- the slow-tail problem is real
- but aggressive oversampling is not the right answer in this setting

This is a useful negative result, not a wasted one.

---

## 4. Tail-Trimming As A Data Strategy

After oversampling failed, a different data-centric strategy became more attractive:

- instead of over-emphasizing the hardest tail
- remove the most extreme tail cases and train a cleaner model on the rest

This is one of the most important “other variants” because it changed the problem definition and produced one of the best practical outcomes in the entire project.

### 4.1 Why tail trimming was considered

The earlier runs repeatedly showed:

- a small number of extreme slow programs caused extremely large residuals
- these cases inflated RMSE and destabilized training
- the model was much more competent on the bulk of the distribution

That naturally suggested:

- maybe the extreme tail is closer to a pathological regime than to the core task

If so, trimming it would allow the model to focus on the main body of the data.

### 4.2 Tail-trimmed cross-attention notebook

The key notebook was:

- [cross-attention-trimmed-tail-40epoch.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cross-attention-trimmed-tail-40epoch.ipynb)

And later its cleaner final form:

- [final-cross-attention-trimmed-tail.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/final-cross-attention-trimmed-tail.ipynb)

The idea was:

- keep the strongest validated architecture direction
- trim only the most extreme runtime tail
- retrain

### 4.3 Actual trimming setup

The executed trimmed-tail run used:

- trimming on `log_execution_time`
- cutoff: `<= 9.0`

Observed row counts:

- original rows: `83,973`
- filtered rows: `82,757`
- removed rows: `1,216`

So the dataset was not heavily reduced overall. A relatively small subset of extreme cases was removed.

### 4.4 Observed results

Observed test metrics:

- MAE: `245.01 ms`
- RMSE: `689.26 ms`
- median AE: `22.73 ms`
- MAPE: `186.98%`
- R²: `0.5044`

### 4.5 Why these results are so important

This was one of the strongest practical results in the project.

Compared to the earlier full-data cross-attention runs:

- MAE dropped dramatically
- RMSE dropped dramatically
- R² improved

The improvement was so large that it strongly confirms the earlier suspicion:

- the extreme slow-runtime tail was one of the major reasons the earlier full-data models looked bad

### 4.6 Critical analysis of tail trimming

This strategy worked because it removed precisely the most damaging cases:

- rare
- extremely slow
- disproportionately harmful to loss and aggregate error

But the result must be interpreted honestly:

- this is now a strong model for the **trimmed problem**
- not for the full original long-tail distribution

That is not a weakness if the project presents it clearly. In fact, it can be a strength:

- the experiment shows that the difficulty was not evenly spread across the dataset
- the model is actually much more competent once the pathological tail is removed

This is one of the clearest data-centric findings of the entire project.

---

## 5. Schedule-Token Fusion Transformer

This was the most architecturally novel “other variant” explored in the project:

- [schedule-token-fusion-transformer.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/schedule-token-fusion-transformer.ipynb)

It is placed in “other variants” because it does not fit cleanly into the original baseline / cross-attention / pruning ladder. It was a more exploratory architectural leap.

### 5.1 Why this model was proposed

By the time this notebook was designed, the project had already learned:

- shallow fusion in Model 1 was too weak
- one-shot cross-attention in Model 2 was clearly better
- but even Model 2 still had limits

The next natural question became:

> Can schedule and code interact through a small transformer block rather than a single cross-attention step?

That led to the schedule-token fusion transformer idea.

### 5.2 What the architecture does

The main concept is:

1. Encode code with frozen CodeBERT and keep all token embeddings.
2. Encode the numeric schedule features into a learned **schedule token**.
3. Concatenate the schedule token with the code token sequence.
4. Pass the full sequence through a small trainable transformer encoder layer.
5. Use the updated schedule token to predict runtime.

So unlike Model 2, where the schedule gets one cross-attention interaction with the code, this model lets:

- schedule token and code tokens

interact together inside a trainable transformer block.

### 5.3 Why it is architecturally meaningful

This design is important because it is not just another optimization trick.

It changes the fusion mechanism itself.

Instead of:

- single-step fusion

it tries:

- multi-step transformer-style fusion

That makes it one of the most conceptually interesting late-stage architectures in the project.

### 5.4 Observed result

The executed result for this notebook showed:

- MAE: `376.37 ms`
- RMSE: `1646.05 ms`
- median AE: `20.17 ms`
- MAPE: `155.32%`
- R²: `0.5060`

### 5.5 What this result means

This is a genuinely interesting result.

It suggests:

- the schedule-token fusion idea is not nonsense
- it can model full-data variance reasonably well
- it is competitive with the better full-dataset runs

At the same time, it did not become the obvious best practical final model because:

- the full-data tail still hurt it
- its advantage over the strongest trimmed-tail cross-attention result was not enough in practical error terms

### 5.6 Critical analysis

This experiment was high-value scientifically because it tested a real architectural hypothesis.

Its strengths:

- meaningful new transformer usage
- stronger schedule-code fusion concept
- respectable final performance

Its limitation:

- it was still being judged on the full difficult distribution
- so the persistent extreme-tail issue still limited the apparent gain

This means the architecture remains interesting, but it was not the safest final choice under time pressure.

---

## 6. Why These “Other Variants” Matter

These experiments did something the main model-family experiments could not do alone.

They helped answer:

- whether the problem was mostly architectural
- mostly data-distribution-driven
- or mostly optimization-driven

The answer turned out to be:

- all three mattered, but the long-tail data difficulty was especially important

This is exactly why this category is so valuable.

### What these variants revealed

From cleaning:

- preprocessing assumptions had to be fixed

From oversampling:

- the tail problem was real, but aggressive weighting was not the right fix

From tail trimming:

- the extreme tail was disproportionately harmful

From the schedule-token fusion transformer:

- there was still room to improve the fusion mechanism itself in a more transformer-native way

So these variants together gave the project a broader scientific understanding than the clean three-model storyline alone could provide.

---

## 7. Final Critical Summary Of Other Variants

### 7.1 Most useful data-centric variant

Tail trimming was the strongest data-centric intervention.

It gave the clearest evidence that:

- the extreme tail was one of the central obstacles in the full problem

### 7.2 Most useful negative experiment

Weighted sampling was the most useful negative experiment.

It showed that:

- not every reasonable-sounding tail-focused idea helps
- aggressive reweighting can damage performance more than it helps

### 7.3 Most interesting new architecture

The schedule-token fusion transformer was the most interesting new architecture.

It gave a plausible next-step direction beyond standard cross-attention, even if it did not become the safest final model choice.

### 7.4 What this whole category contributed

This category showed that the project was not just trying one architecture after another blindly.

It systematically explored:

- data shaping
- sampling strategy
- alternative transformer fusion design

and used the results to refine the final modeling direction.

---

## 8. Bottom Line For Other Variants

The “other variants” experiments were the project’s broad exploratory branch.

They contributed four major lessons:

1. The cleaned dataset still had a severe tail problem.
2. Aggressive oversampling was not the right fix.
3. Tail trimming transformed the practical quality of the strongest architecture.
4. The schedule-token fusion transformer was a meaningful architectural direction, even if it remained a riskier final choice.

If the main model-family documents tell the structured story of the project, this document tells the experimental story of how the project’s assumptions were tested and refined outside the main ladder.
