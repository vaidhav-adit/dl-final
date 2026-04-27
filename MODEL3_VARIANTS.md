# Model 3 Variants

This document covers the full Model 3 family in detail. Model 3 is the most complicated line of work in the project because it is not just “another predictor.” It is the branch where the project tried to understand and reduce transformer head usage through gating and pruning.

Model 3 has to be explained carefully because there are really **two different stages** inside it:

1. **Learn head importance** using gates
2. **Use that learned importance to disable or prune heads**

This distinction caused confusion during experimentation, so in this document I will make it explicit and precise.

---

## 1. What Model 3 Was Trying To Do

Model 3 did **not** start as an accuracy-first experiment.

Its original purpose was closer to:

> Can we identify which CodeBERT attention heads matter for runtime prediction, and can we remove the less useful ones without ruining performance?

So Model 3 was about:

- head importance
- pruning/compression
- efficiency and interpretability

rather than just:

- raw accuracy improvement

That said, because the pruning/fine-tuning stage still used validation performance to guide decisions, Model 3 could still affect predictive quality. In fact, in your final executed Model 3-style result, the functional pruning stage ended up slightly improving some metrics.

But conceptually, the starting motivation for Model 3 was:

- take the stronger cross-attention Model 2 backbone
- analyze the internal transformer heads
- keep the important ones
- suppress the less important ones

So Model 3 is built on top of Model 2, not independently of it.

---

## 2. Why Model 3 Needed A Different Method

If you want to prune transformer heads, you need some way to decide:

- which heads are useful
- which heads are redundant

There are several possible strategies:

1. **Ablation-based importance**
   - remove one head at a time
   - measure the loss increase

2. **Magnitude-based heuristics**
   - use parameter size or norm-style heuristics

3. **Learned gating**
   - give each head a gate
   - train the gates
   - treat the learned gate values as importance signals

The project explored both the ablation idea and the gating idea, but gating became the more important route.

Why?

Because ablation is expensive and also depends heavily on stable pruning hooks being available in the transformer implementation. The environment kept causing API issues for that route.

Gating, by contrast, allows the model to learn head importance during training itself.

---

## 3. What Gating Means In This Project

This is the most important conceptual section in the Model 3 story.

### 3.1 The basic intuition

A transformer layer has multiple attention heads. Normally, all of them contribute to the output of that layer.

The gating idea introduces a learnable scalar gate per head.

Conceptually:

- if a head’s gate is high, that head is considered active and useful
- if a head’s gate is low, that head contributes very little and is a candidate for later pruning

So a gate is basically a learned “importance controller” for a head.

### 3.2 What the gate is doing during training

During gated Model 3 training:

- the model still learns the main runtime-prediction task
- but it also learns gate values attached to heads

This means the training process is not only learning:

- “how to predict runtime”

It is also learning:

- “which heads does this task seem to need?”

That is why the gates are so important. They are the mechanism used to estimate head importance.

### 3.3 What the gated training phase is **not**

This is the part that caused confusion during the project.

The long gated training run was **not yet the pruning phase**.

It was the **importance-learning phase**.

In other words:

- when you trained the gated model for many epochs, you were training a model that could later be used for pruning
- but you were not yet actually removing or disabling heads in the final sense

This is why the later stage-2 notebook was needed.

So the correct mental model is:

- Stage 1 = learn which heads seem important
- Stage 2 = use that learned information to disable/prune heads and fine-tune

### 3.4 Why gating was useful

Without gating, the alternative would have been expensive brute-force head ablation:

- remove one head
- run validation
- restore it
- repeat for every head

That is conceptually clean, but computationally expensive and, in your environment, implementation-fragile.

Gating gave a way to:

- estimate importance during ordinary task training
- then use those learned scores later

So the gates were not an extra cosmetic mechanism. They were the head-importance estimator for Model 3.

---

## 4. The Original Model 3 Plan

The original Model 3 plan was:

1. Start from the cross-attention architecture.
2. Measure or learn head importance.
3. Prune weak heads.
4. Retrain or fine-tune the remaining model.
5. Compare the pruned model against Model 2.

At the plan level, this made sense.

The trouble came in implementation.

The early pruning notebook variants encountered several library-specific issues:

- missing pruning methods on objects where they were expected
- mismatches between notebook assumptions and the actual HuggingFace `Roberta` implementation
- difficulty doing structural head removal cleanly in the environment

That meant the project had to adapt its execution strategy while keeping the scientific intent as intact as possible.

---

## 5. Main Model 3 Notebooks

The core Model 3 notebooks were:

- [normal-cross-prune.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/normal-cross-prune.ipynb)
- [normal-cross-prune-stage2.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/normal-cross-prune-stage2.ipynb)
- [done/normal-cross-prune.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/done/normal-cross-prune.ipynb)
- [run_20th/notebook8df7a7a47a (1).ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_20th/notebook8df7a7a47a%20(1).ipynb)

These notebooks represent different phases of the same Model 3 line of work:

- earlier planning/implementation
- gated training
- stage-2 disabling/pruning and fine-tuning
- final executed result

---

## 6. The Early Pruning Attempts

Before the gating-based setup became the main usable approach, the project attempted more direct pruning.

### 6.1 Ablation-style importance

The idea here was:

- compute baseline validation loss
- remove one head at a time
- observe how much validation loss worsens
- define importance as the loss increase

This is a valid and often defensible method.

The reason it is attractive is that it gives a direct answer to:

> If this head disappears, how much worse does the model get?

The larger the loss increase, the more important the head appears to be.

### 6.2 Why this route did not become the final one

There were two main problems:

1. It is expensive
   - many repeated validation passes
   - lots of bookkeeping

2. The environment did not support the pruning APIs cleanly
   - methods expected in one place did not exist
   - the exact `Roberta`/CodeBERT object layout was more restrictive than assumed

So the ablation route remained part of the project history, but it was not the final successful Model 3 path.

This is important to say clearly:

- the project did explore ablation-style pruning
- but the final successful result did **not** come from clean structural ablation pruning

---

## 7. Gated Model 3 Training Phase

The main usable Model 3 path began by training a gated model.

### 7.1 Architecture used

The gated model reused the cross-attention Model 2 backbone:

- CodeBERT encoder
- schedule-feature MLP
- cross-attention fusion
- regression head

On top of that, it added:

- learnable gates for transformer heads

The backbone idea remained the same as Model 2, but now the model could also estimate internal head importance.

### 7.2 Training setup

The gated model was trained to:

- perform runtime regression
- while also learning gate values that reflect head usefulness

The implementation included:

- task loss on runtime prediction
- a gate/sparsity-related penalty

The purpose of that additional term was not to destroy the model. It was to encourage the model to reveal which heads were less necessary.

### 7.3 What happened during gated training

The gated model trained for many epochs and did produce a best checkpoint.

Observed characteristics:

- validation improved early and then fluctuated
- best checkpoint appeared before the full nominal epoch budget was completed
- the number of effectively active heads moved only gradually rather than collapsing dramatically

This matters because it tells us something about the gate strength:

- the gating mechanism was active
- but it was not pushing the model into an extremely sparse regime

### 7.4 Critical analysis of the gated training stage

This stage was scientifically useful, but it also created confusion because it looked like “the pruning run” while actually being only the preparation phase.

The strengths of this stage:

- it learned meaningful head-importance structure
- it avoided brute-force ablation cost
- it produced the checkpoint needed for stage 2

The weaknesses:

- it was long and expensive
- it did not by itself produce a final pruned model
- it was easy to misinterpret as the pruning result itself

So the gating stage should be described honestly as:

- the **importance-learning stage**

not:

- the final pruning result

---

## 8. Stage 2: Turning Learned Importance Into Pruning

After the gated model had been trained, the project moved to stage 2.

This is where heads actually began to be suppressed based on the gate information.

### 8.1 Intended structural pruning

The original intent was true structural pruning:

- identify weak heads
- physically remove them from the transformer architecture
- fine-tune the pruned model

That is the cleanest conceptual version of pruning.

### 8.2 Why structural pruning did not fully work

In practice, the environment blocked a clean structural path because:

- expected pruning methods were missing
- pruning hooks were not available where assumed
- `Roberta` internals did not cooperate with the notebook’s structural-removal logic

These were not conceptual mistakes in the research idea. They were implementation and library compatibility problems.

### 8.3 Functional pruning fallback

To keep the project moving, the practical fallback was:

- identify the low-importance heads
- force those heads “off” by their gate settings
- freeze that gating choice
- fine-tune the rest of the model

This is best described as:

- **functional pruning**
- or **functional gate masking**

It is not the same as physically deleting parameters from the transformer.

But it is also not meaningless. If a head is forced off and can no longer contribute, then functionally it has been removed from the decision process.

So this fallback still preserved the scientific core of the experiment:

- weak heads are suppressed
- the remaining model is fine-tuned
- performance is re-measured

---

## 9. The Successful Executed Model 3 Result

The main successful Model 3-style result came from:

- [run_20th/notebook8df7a7a47a (1).ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_20th/notebook8df7a7a47a%20(1).ipynb)

This notebook was the stage-2 process after the gated checkpoint had already been trained.

### 9.1 What stage 2 actually did

It:

- loaded the trained gated checkpoint
- used gate values to decide which heads were weak
- tested multiple pruning ratios
- fine-tuned each pruned/disabled configuration
- selected the best one by validation performance

This is the first place where Model 3 produced a real final result.

### 9.2 Pruning ratios tested

The run evaluated:

- `10%`
- `20%`
- `30%`
- `40%`
- `50%`

The notebook reported how many heads were targeted at each level and fine-tuned each version.

### 9.3 Fine-tuning behavior

For each pruning ratio, the model was fine-tuned for `10` epochs.

This was important because once weak heads are disabled, the downstream trainable parts need time to adapt to the changed internal information flow.

This fine-tuning stage is what converted the raw pruning idea into a usable final model.

### 9.4 Best pruning ratio

The best result came from:

- pruning ratio: `40%`

This corresponded to:

- `57` heads functionally pruned/disabled

### 9.5 Final test metrics

Observed final metrics:

- MAE: `384.13 ms`
- RMSE: `1686.06 ms`
- median AE: `20.52 ms`
- MAPE: `193.24%`
- R²: `0.4817`
- pruning mode: `functional_gate_masking`

---

## 10. What These Final Model 3 Results Mean

This is the key interpretation section.

### 10.1 Was Model 3 successful?

Yes, with an important qualification.

If by “successful” we mean:

- did the project produce a meaningful final stage-2 Model 3 result?

then yes.

If by “successful” we mean:

- did it achieve a pure structural parameter-deletion pruning result?

then no, not in the strictest sense.

So the honest answer is:

- successful **functional pruning**
- not clean full structural compression

### 10.2 Did Model 3 help performance?

Yes, somewhat.

Compared to the strongest earlier full-dataset Model 2 run (`run_19th` cross):

`run_19th` cross:

- MAE: `397.43 ms`
- RMSE: `1720.46 ms`
- R²: `0.4604`

Model 3 functional pruning result:

- MAE: `384.13 ms`
- RMSE: `1686.06 ms`
- R²: `0.4817`

So the Model 3 result was actually slightly better on these main metrics.

That is important, because it means the gating + functional pruning workflow was not just a compression exercise. It also produced a competitive predictive model.

### 10.3 Why could disabling heads improve performance?

This is not impossible at all.

If some heads are:

- redundant
- noisy
- weakly aligned with the task

then suppressing them can:

- simplify the effective representation
- reduce internal noise
- let the model rely more strongly on the most useful heads

So slight performance gains after pruning are plausible and scientifically meaningful.

---

## 11. Critical Analysis Of The Whole Model 3 Family

Model 3 is the most complicated family to judge because it involved both strong ideas and strong implementation friction.

### 11.1 What was strong about Model 3

- It asked a more structural research question than Models 1 and 2.
- It attempted to understand the internal head structure of the transformer rather than only optimize end metrics.
- It introduced a meaningful interpretability angle through head importance.
- It produced a final result that was actually competitive.

### 11.2 What was weak or difficult about Model 3

- It was by far the most technically fragile branch.
- Structural pruning was repeatedly blocked by environment/library limitations.
- It required a two-stage workflow that was easy to misunderstand.
- The gating phase was long and expensive.

### 11.3 What the project learned from Model 3

The biggest lesson from Model 3 was this:

> not all transformer heads are equally useful for this runtime-prediction task

The successful functional-pruning result showed that a substantial fraction of heads could be disabled without hurting the model, and with slight improvement in some cases.

That is a meaningful research finding.

### 11.4 What Model 3 did not fully achieve

It did not produce the cleanest possible “compression” story because the final successful version was functional rather than structural pruning.

That does not erase the result, but it does mean the final report should describe Model 3 carefully and honestly.

The safest wording is:

- **head-importance learning and functional pruning**

rather than:

- “full structural transformer compression”

---

## 12. Final Summary Of Model 3

Model 3 began as a pruning/compression idea on top of Model 2.

Its key mechanism was gating:

- gates were added to transformer heads
- gated training learned which heads seemed important
- stage 2 used those learned gate values to disable weaker heads
- the resulting model was fine-tuned and evaluated

The most important conceptual distinction is:

- gated training was the **importance-learning stage**
- stage 2 was the **actual disabling/pruning stage**

### Best final Model 3 result

From [run_20th/notebook8df7a7a47a (1).ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_20th/notebook8df7a7a47a%20(1).ipynb):

- pruning ratio: `40%`
- heads disabled: `57`
- pruning mode: `functional_gate_masking`
- MAE: `384.13 ms`
- RMSE: `1686.06 ms`
- R²: `0.4817`

### Most important scientific conclusion

Model 3 showed that the cross-attention runtime-prediction system does not need all heads equally. A substantial fraction of heads could be effectively turned off while keeping, and even slightly improving, predictive performance.
