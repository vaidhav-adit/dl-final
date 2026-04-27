# DL Final Project Experiment Report

This report summarizes the experiment history for the project, including the dataset cleaning phase, the main model runs, the later accuracy-focused variations, and the pruning/gating work. It is organized into 3 parts so the progression is easy to follow.

## Part 1: Data Cleaning and the First Stable Model Runs

### 1. Cleaning and dataset preparation

The project started with the creation and inspection of the cleaned dataset in [cleaning-notebook.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cleaning-notebook.ipynb), using [Cleaned_LOOPerSet_GEM.csv](/Users/vaidhav/Desktop/MyProjects/dl-final-project/Cleaned_LOOPerSet_GEM.csv).

The goal of this stage was to take a noisy runtime dataset and make it trainable for a regression model. The preprocessing decisions were important because they shaped every later result:

- `execution_time` was transformed into `log_execution_time` using `log1p`
- the feature set was aligned to the cleaned GEMM version
- the old `Vec` feature was removed
- the remaining 10 numeric schedule features were:
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

The dataset inspection showed several key facts:

- total rows: about `83,973`
- missing values: none
- duplicate full rows: only `66`
- duplicate `Code_String` rows: extremely high
- unique `Code_String` values: about `1,078`

That immediately told us that the task was not really “learn arbitrary code from scratch.” Instead, it was much closer to “learn runtime behavior over a limited family of matrix-operation kernels under many schedule settings.”

The runtime distribution was still very broad even after cleaning:

- minimum runtime: around `0.000366 ms`
- median runtime: about `41 ms`
- maximum runtime: about `56,699 ms`

This mattered for two reasons:

1. `MAPE` was always going to look harsh because tiny actual runtimes make percentage error explode.
2. The model would need help with the long-runtime tail because slow examples were relatively rare.

This cleaning stage also exposed two concrete problems in the original training logic:

- the numeric features had very different scales, so normalization was necessary
- the original unroll-weighted loss was effectively broken for the cleaned dataset, because `Unroll > 0` was always true

That second issue was especially important. The old code did:

```python
weights = torch.where(unroll_factor > 0, 5.0, 1.0)
```

But after cleaning, `Unroll` was always positive. That meant every sample got the same weight, so the weighting did not actually prioritize any subset.

The output of this phase was not a model, but a much clearer problem definition:

- train on `log_execution_time`
- normalize the 10 numeric schedule features
- expect repeated code strings
- treat slow-program error as a major challenge

Critical analysis:

- The cleaning stage improved stability, but it did not remove the core difficulty of the problem. The target remained highly multi-scale even after log transformation.
- The heavy repetition of `Code_String` values meant the dataset had limited true semantic diversity, so gains from code understanding alone were always likely to be capped.
- The severe imbalance between common fast/medium programs and rarer slow programs created a built-in tendency for the models to optimize the majority region first.
- This stage revealed that many later “bad results” were not simply modeling failures. Some of them were consequences of the dataset structure itself.

### 2. Very early notebook implementations and the initial plan

The earliest model notebooks were:

- [normal-model.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/normal-model.ipynb)
- [normal-cross-attention-model.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/normal-cross-attention-model.ipynb)

These were built from the original 3-model plan:

- Model 1: frozen CodeBERT + schedule MLP + simple fusion
- Model 2: frozen CodeBERT + schedule-conditioned cross-attention
- Model 3: pruning/compression on top of Model 2

Before the corrected runs, the original notebooks had some weaknesses:

- older versions still used the unroll-weighted loss
- earlier variants did not normalize numeric features
- some versions used only the `CLS` token in the baseline

These early notebooks were still useful because they established the architectural story, but they were not yet the final corrected experimental setup.

### 3. The first executed “done” runs

The `done` folder captured the first major executed notebook versions:

- [done/normal-model-50.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/done/normal-model-50.ipynb)
- [done/normal-cross-attention-model-50epoch.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/done/normal-cross-attention-model-50epoch.ipynb)
- [done/normal-cross-prune.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/done/normal-cross-prune.ipynb)

These runs were important because they gave the first full end-to-end outputs.

#### Model 1 in `done`

This was the baseline-style model:

- frozen CodeBERT
- schedule MLP
- concat-style fusion
- regression in log space

It trained successfully and converged, but the quality was only moderate.

Observed behavior:

- train loss decreased
- validation loss improved
- the model learned something meaningful
- but it still had large outlier errors

What it taught us:

- the basic training pipeline worked
- the project was not blocked by an obvious code bug
- but the baseline architecture alone was not enough

Critical analysis:

- This baseline was useful mostly as a proof that the system could learn something, not as a strong final model.
- At this stage, the fusion between code and schedule information was too shallow, so the model was probably relying more on broad feature correlations than on fine-grained schedule-conditioned code understanding.
- The fact that it converged but still had weak quality suggested underfitting of the interaction between code semantics and optimization parameters, not a total optimization failure.

#### Model 2 in `done`

This was the first executed cross-attention version:

- frozen CodeBERT
- all token embeddings retained
- schedule embedding used as a query over code tokens
- attended code summary fused with the schedule embedding

This run was important because it showed the key design idea was directionally right: Model 2 beat Model 1.

That result gave the first strong evidence that schedule-aware token interaction was useful.

Critical analysis:

- The improvement over Model 1 indicated that code-token-level access mattered, but it was still only a first step.
- The architecture was stronger, but not yet enough to fully solve the hard tail because the backbone was still frozen and the schedule-to-code interaction happened only once.
- This was the point where the project moved from “can cross-attention help?” to “cross-attention helps, but the remaining bottleneck is somewhere else.”

#### Model 3 in `done`

This pruning notebook did not complete successfully.

It crashed when trying to prune heads because the pruning logic called a method that did not exist on the specific `RobertaModel`/CodeBERT object in that environment.

That failure mattered because it showed a pattern that continued later:

- pruning was always more implementation-fragile than the baseline and cross-attention models
- accuracy gains were not coming from pruning anyway

Critical analysis:

- This failure was mostly technical rather than scientific.
- It did, however, force a useful realization: pruning and compression are not good first tools when the main problem is still predictive accuracy.
- In hindsight, this result helped narrow the focus back to Model 2, which was where the real accuracy progress was happening.

So Part 1 ended with a clear conclusion:

- the cleaned data pipeline was necessary
- Model 1 established a working baseline
- Model 2 was the first model that clearly improved on the baseline
- pruning was not yet reliable and not yet useful for accuracy

## Part 2: Corrected Runs and the Strongest Accuracy Result

### 1. Corrected Model 1 run: `run_18th/normal`

The first major corrected run was:

- [run_18th/normal/normal-model.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_18th/normal/normal-model.ipynb)

This run used the cleaned training logic rather than the earlier rougher versions.

Methodology:

- 10-feature normalized schedule input
- no fake unroll weighting
- mean pooling over CodeBERT tokens instead of raw `CLS`
- frozen CodeBERT
- simple concat-based fusion
- evaluation in milliseconds after inverse-transforming from log space

This made it a much fairer baseline than the older notebooks.

Test results:

- MAE: `432.08 ms`
- RMSE: `1885.54 ms`
- Median AE: `25.02 ms`
- MAPE: `324.20%`
- R²: `0.3518`

Interpretation:

- the model was clearly learning
- but the gap between `median AE` and `RMSE` showed that outliers were a major issue
- the baseline was not failing everywhere, but it was weak on the hard tail

This run was important because it became the first stable corrected baseline.

Critical analysis:

- The corrected baseline was clearly better-founded than the earlier versions because feature normalization and removal of the bogus weighting made optimization more honest.
- The model still left a lot of performance on the table because mean-pooled code embeddings plus simple concatenation can only express limited schedule-aware interactions.
- The high RMSE relative to median error showed that a minority of catastrophic misses dominated the aggregate quality, which meant average metrics alone were hiding a more nuanced picture.

### 2. Corrected Model 2 run: `run_18th/cross`

The next crucial run was:

- [run_18th/cross/norm-cross.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_18th/cross/norm-cross.ipynb)

Methodology:

- same corrected data pipeline as Model 1
- frozen CodeBERT
- all code tokens retained
- schedule MLP
- one cross-attention interaction from schedule to code tokens
- regression from fused schedule-plus-attended-code representation

This was the first strong corrected version of the intended Model 2 idea.

Test results:

- MAE: `396.27 ms`
- RMSE: `1810.33 ms`
- Median AE: `18.71 ms`
- MAPE: `174.66%`
- R²: `0.4025`

Interpretation:

- Model 2 clearly beat Model 1
- all major regression metrics improved
- the cross-attention idea was validated

This run gave the first strong project result:

- schedule-conditioned attention over code tokens helps more than shallow fusion

Critical analysis:

- The gain here was meaningful because it came without changing the basic data split or target setup. That made it a fair architectural improvement.
- The improvement across MAE, RMSE, and R² suggested that cross-attention was not just fitting noise. It was actually extracting more task-relevant information.
- Still, the fact that the model remained only moderately accurate means that single-step cross-attention was helpful but not sufficient. It improved fusion, but it did not solve the long-tail runtime problem.

At this point the project had a believable narrative:

- cleaning helped
- normalization helped
- corrected Model 2 beat corrected Model 1

### 3. Partially unfrozen cross-attention run: `run_19th`

The next major experiment was a more ambitious accuracy-focused cross-attention run:

- [run_19th/cross/norm-cross-2.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_19th/cross/norm-cross-2.ipynb)

This run added limited backbone adaptation while trying to keep runtime manageable.

Methodology:

- cross-attention model
- normalized 10-feature schedule vector
- `MAX_LENGTH = 128`
- partial CodeBERT unfreezing
- separate learning rates for backbone and head

This was the first serious attempt to improve Model 2 rather than just clean it up.

Observed test metrics:

- MAE: `397.43 ms`
- RMSE: `1720.46 ms`
- Median AE: `20.75 ms`
- MAPE: `159.46%`
- R²: `0.4604`

Interpretation:

- MAE stayed about the same as `run_18th` cross
- RMSE improved noticeably
- R² improved clearly
- MAPE improved

So this run did not produce a big headline MAE drop, but it did improve overall fit quality and tail behavior more than earlier runs.

The error analysis from this run was one of the most useful outputs in the whole project. It showed the model was not uniformly bad:

- `<=1 ms` bucket was actually okay
- `1–10 ms` and `10–100 ms` were much more manageable
- the real collapse happened in the slow buckets:
- `1–10 s`
- `>10 s`

This changed the project’s diagnosis. After `run_19th`, the main problem was no longer “the model cannot learn.” The problem became:

- the model under-learns the slow tail

That was the strongest accuracy result obtained among the executed cross-attention runs, and it became the reference point for later comparisons.

Critical analysis:

- This run was important because it showed that some task-specific adaptation of the pretrained code encoder was helpful.
- The fact that RMSE and R² improved more clearly than MAE suggests that the model got better at handling variance and some harder examples, even if average absolute error did not drop much.
- The instability during validation showed that this setup was close to the practical edge: stronger than the frozen model, but also more sensitive.
- Most importantly, this run gave the clearest diagnosis of the project: the major unresolved issue was not general learning failure, but poor handling of the slowest programs.

## Part 3: Later Accuracy Attempts, Pruning/Gating Work, and the Final Notebook Direction

### 1. Weighted slow-program sampling attempt: `run_20th`

After identifying the slow-runtime tail as the main weakness, the next experiment tried to attack that problem directly with weighted sampling:

- [run_20th/notebookb835584806.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/run_20th/notebookb835584806.ipynb)

This run was based on the cross-attention model and used:

- `MAX_LENGTH = 128`
- last CodeBERT layer unfrozen
- backbone LR `5e-6`
- head LR `1e-4`
- weighted sampling:
  - `1s–10s` programs got weight `3.0`
  - `>10s` programs got weight `6.0`

The idea was reasonable: the slow tail was underrepresented, so oversample it.

But the results were worse:

- MAE: `450.69 ms`
- RMSE: `1921.88 ms`
- Median AE: `23.06 ms`
- MAPE: `192.69%`
- R²: `0.3266`

Compared with `run_19th`, this was a clear regression.

Why this mattered:

- it showed that the idea “target the slow tail” was directionally right
- but the specific implementation via aggressive weighted sampling distorted training too much

Observed behavior:

- validation was unstable for much of training
- the model only partially recovered late
- medium-runtime performance worsened
- the tail problem was not solved enough to justify the tradeoff

This experiment was useful precisely because it failed clearly. It told us:

- not every tail-focused intervention improves the model
- aggressive weighted sampling was not the right answer in this form

Critical analysis:

- The logic behind the experiment was sound: if slow programs are underrepresented, emphasize them more.
- The problem was that the weighting strength was too aggressive relative to the rest of the dataset.
- Instead of gently correcting the training distribution, it appears to have distorted optimization and damaged performance on the larger middle portion of the runtime range.
- This run showed that data imbalance was real, but naive oversampling was not the right remedy.

### 2. Gated head-importance training for Model 3

After the baseline and cross-attention work, attention turned to Model 3.

The original plan had been pruning through post-hoc head ablation, but that proved brittle and expensive. That led to a second idea: learn head importance with gates first, then prune based on those learned gates.

The relevant notebooks became:

- [normal-cross-prune.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/normal-cross-prune.ipynb)
- [normal-cross-prune-stage2.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/normal-cross-prune-stage2.ipynb)

The methodology changed significantly here:

- keep the cross-attention model backbone
- insert learnable gates into attention heads
- train those gates with a sparsity penalty
- learn which heads matter
- then move to a second stage where low-importance heads are functionally turned off and the model is fine-tuned

One very important conceptual clarification came out of this work:

- the long gated training run was not yet “pruning”
- it was the head-importance learning phase

In other words:

- stage 1: learn which heads appear important
- stage 2: disable the weak heads and fine-tune

This distinction caused confusion during execution, but it was methodologically correct.

What the gated training results suggested:

- the model did improve during training
- best checkpoint appeared before the full epoch budget finished
- sparsity pressure was fairly mild
- active-head count did not collapse dramatically, meaning the gate penalty was conservative

So the gating mechanism was not obviously broken, but it also was not producing dramatic pruning pressure by itself.

Critical analysis:

- This experiment was methodologically interesting, but it was answering the wrong question for the immediate project need.
- The project needed better predictive accuracy, while the gated setup was mainly useful for importance analysis and compression.
- The mild movement in active-head count suggested that the regularization strength was not aggressive enough to force strong pruning, but making it stronger would likely have increased instability further.
- So even when the gating worked, it was not the most efficient path to the goal that mattered most.

### 3. Model 3 implementation problems and the practical fallback

This stage was the messiest part of the project technically.

Several issues happened:

- checkpoint architecture mismatches
- HuggingFace `Roberta` forward-signature mismatches
- head-pruning API mismatches
- environment-specific missing methods for structural pruning

These were not modeling failures so much as library-compatibility problems.

Eventually, the practical fallback was:

- do not rely on true structural deletion of heads
- instead do functional pruning by forcing selected heads “off” through their learned gates
- freeze those gate choices
- fine-tune the rest

This meant the experiment still had scientific meaning:

- low-importance heads were effectively removed from contributing
- the model could be evaluated afterward

But it also meant the result was not a clean “physical compression” result in the pure structural sense.

That is important for interpretation:

- Model 3 became more of an importance-analysis and functional-pruning experiment
- not a clean final accuracy-improvement method

Critical analysis:

- The practical fallback kept the experiment alive, but it weakened the purity of the pruning claim.
- Functionally disabling heads is still meaningful, but it is not the same as a clean structural compression result.
- This stage reinforced a recurring lesson: library and implementation constraints can shape the experiment almost as much as the modeling idea itself.
- Scientifically, Model 3 remained interesting, but it became even less likely to be the best final answer for accuracy.

### 4. Final model-design notebooks prepared after the main runs

Several later notebooks were prepared to make one last strong attempt at improving accuracy:

- [cross-attention-weighted-accuracy.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/cross-attention-weighted-accuracy.ipynb)
- [accuracy-oversampling-models.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/accuracy-oversampling-models.ipynb)
- [schedule-token-fusion-transformer.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/schedule-token-fusion-transformer.ipynb)

These represented the next design directions after the earlier lessons:

#### Cross-attention weighted-accuracy notebook

This notebook encoded the “fix the slow tail” hypothesis in a simpler and more reusable form:

- cross-attention only
- weighted sampling
- last 1 CodeBERT layer unfrozen
- lighter learning-rate split

It was essentially the structured version of the `run_20th` experiment idea.

Critical analysis:

- This notebook was valuable as a controlled tool for retesting the imbalance idea, but the underlying hypothesis had already shown weakness in execution.
- It remained useful as an experiment definition, though not as the strongest final modeling direction after the `run_20th` outcome.

#### Accuracy oversampling models notebook

This notebook was a convenience notebook to compare:

- normal model
- cross-attention model

under the same oversampling/weighted-sampling scheme.

It was useful as a framework, but it was not the main final direction once the project focus settled on the stronger cross-attention family.

Critical analysis:

- This notebook was more of an organizational convenience than a core scientific contribution.
- Once the cross-attention family had clearly separated itself from the baseline family, splitting attention across both branches became less valuable than concentrating effort on the better architecture.

#### Schedule token fusion transformer notebook

This notebook was the most ambitious final-model design prepared in the root folder:

- [schedule-token-fusion-transformer.ipynb](/Users/vaidhav/Desktop/MyProjects/dl-final-project/schedule-token-fusion-transformer.ipynb)

This design tried to do something more meaningful with transformer fusion itself:

- CodeBERT stays frozen
- numeric schedule features are encoded into a learned schedule token
- that schedule token is concatenated with all code tokens
- a small trainable transformer encoder layer lets schedule and code interact jointly
- prediction is made from the updated schedule token

This was different from the earlier cross-attention model because the schedule and code could interact through a full mini-transformer block rather than one single cross-attention step.

Conceptually:

- old cross-attention model: one question from schedule to code
- schedule-token fusion model: a small transformer discussion between schedule and code before prediction

This notebook was prepared as the most “architecturally meaningful” final attempt, especially if the goal was not just to tweak sampling or pruning but to improve the fusion mechanism itself.

Critical analysis:

- This direction addresses one of the real limitations of the current Model 2: the interaction between schedule and code is still shallow.
- Unlike the weighted-sampling experiments, this approach tries to improve the model’s reasoning capacity rather than distort the training distribution.
- It is also a more defensible “final attempt” scientifically, because it makes a genuine architectural claim instead of just another optimization tweak.
- The tradeoff is that it is a fresh model idea, so it carries risk. But among the later options explored, it is the one with the clearest chance of producing a meaningful accuracy gain while still fitting the project story.

### Final overall conclusions from the full experiment series

Across all experiments, several clear lessons emerged:

1. Cleaning and feature normalization were essential.
2. Model 2 consistently beat Model 1 when both were implemented correctly.
3. The best executed accuracy result came from the partially unfrozen cross-attention run in `run_19th`.
4. The main persistent weakness was the slow-runtime tail.
5. Aggressive weighted sampling did not fix that problem and hurt the broader fit.
6. Pruning/gating work was methodologically interesting but not the path that improved accuracy.
7. If one final model was to be pushed as the strongest architectural attempt, the schedule-token fusion transformer was the most meaningful next step.

Critical analysis of the overall project arc:

- The project improved most when the changes attacked genuine bottlenecks: bad preprocessing assumptions, weak fusion, and insufficient task adaptation.
- The project improved least when the changes attacked secondary issues too early: pruning before accuracy was strong enough, or aggressive reweighting before understanding the stability tradeoff.
- The strongest executed empirical result came from the partially unfrozen cross-attention model, but the strongest final architectural idea prepared was the schedule-token fusion transformer.
- So the project’s lesson is not “more tricks always help.” It is that the right improvements were the ones that respected the structure of the task: schedule-conditioned code understanding under severe runtime-scale imbalance.

In short, the project evolved from:

- “can we train a cost model at all?”

to:

- “does schedule-aware code attention help?”

to:

- “how do we improve the hard slow-program tail without destabilizing the model?”

That progression is the real experimental story of the project.
