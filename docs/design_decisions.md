# Design decisions

Short notes on choices that a reviewer might question. Each says what we did, why, and what it costs.

### Judge score = expected value over label probabilities, not a sampled label
The judge reads the next-token probability of each rubric label ("0"/"1"/"2" for scaffolding, "1"-"5" for helpfulness) after a pre-filled `SCAFFOLDING: ` and reports the expectation (G-Eval, Liu et al. 2023). This gives a continuous, deterministic score in [-1, 1]: smoother as an RL reward, and more sensitive to bias (a dialect can shift probability mass without flipping the top label). The discrete label (argmax) is used for agreement with humans, so E1 has no thresholds fit to the labels. Cost: needs open weights or an API with log-probs; the API sanity judge falls back to parsing generated text.

### Helpfulness is conditioned on the judge's top scaffolding label; rationale comes last
Prompt order is SCAFFOLDING, HELPFULNESS, REASON on one greedy path. The rationale is generated *after* the scores, so it is a post-hoc explanation of exactly those scores. That is what E4 needs (does the stated reason match what actually moves the score?). Asking for the reason first would make it chain-of-thought and change the scores.

### The rubric never mentions dialect or names
We audit the judge as a practitioner would naturally write it. Debiased variants come later (stage 4) and are separate rubric/method versions.

### Rubric versions are files; scores record a hash
`src/tutorbias/judge/prompts/rubric_v*.txt`. Every score row carries `rubric_version` and `rubric_hash`, so scores from different rubric iterations can't be mixed by accident.

### Labels split into dev and test
Iterating on a rubric while measuring agreement on the same labels overfits the rubric to those labels. We iterate on dev (50%) and report test once. The gate (kappa >= 0.6, Spearman >= 0.5) is in the config before any labels exist.

### Per-group agreement reuses the MAE labels
Labels are given on the MAE message. We assume the correct label for a reply doesn't depend on the student's dialect or name (that is the counterfactual-fairness definition the project uses). Under that assumption, the same label is ground truth for every variant, so judge-human agreement can be computed per group with no extra labeling. If someone argues a reply *should* be judged differently for a different student, this assumption is where to push.

### Three disjoint item splits
`rl_train` (2000, GSM8K train), `audit` (300, GSM8K train, disjoint), `test` (300, GSM8K test). The judge is calibrated and audited on questions the tutor never trains on, and tutor evaluation uses GSM8K's own test split.

### Item filtering
Integer non-negative answers, 2-6 solution steps, and the answer must not already appear in the question (otherwise answer-leak can't be measured). Counts of dropped items are in `data/processed/items_stats.json`.

### Stuck messages: 4 types x 3 phrasings, base phrasing varies by item
Types: tried a partial step, wrong answer (the second-to-last step's result, i.e. "forgot the last step"), "I don't get it", and "just tell me the answer". A message never contains the correct answer, and a message type is skipped for an item if any of its numbers equals the answer. The paraphrase control is a different MAE phrasing of the same type.

### Rule-based dialects via Multi-VALUE, in a separate environment
LLM rewrites "in AAE" tend to produce caricature; Multi-VALUE applies documented eWAVE features and records which ones fired, which stage 3 needs for feature ablation. Multi-VALUE pins spaCy 3.7.4 / nltk 3.5 / stanza 1.8.2, so it lives in `.venv-data` (Python 3.10), separate from the GPU env. Variants that change any number are flagged (`numbers_ok`) and excluded from the audit; variants that change nothing are recorded (`changed`), and the audit reports results with and without them.

### Dialect features are restricted, blocklisted, and validated
Only eWAVE features rated A (pervasive or obligatory) for each variety are used; a blocklist removes features that are wrong in context; each feature is used with probability 0.5 per message; Multi-VALUE's untracked slang substitutions are disabled; and native speakers rate a sample before results are reported. Full reasoning: [ethics.md](ethics.md). Feature lists: [dialect_features.md](dialect_features.md).

### Learner (L2) English is a feature list, not a variety
Multi-VALUE has no learner-English variety. We use the grammatical morphemes L2 learners acquire late and often omit across first languages (Dulay & Burt 1974; Goldschneider & DeKeyser 2001): articles, plural -s, third-person -s, past -ed, auxiliaries in questions. Several overlap with AAE features (e.g. 3sg -s absence), so the L2 and AAE groups are not independent. Stage 3 compares features rather than only groups.

### Names
Black/white names are from Bertrand & Mullainathan (2004). Hispanic and South Asian lists are placeholders marked `needs_check` in `data/names/names.yaml` and should be verified (Rosenman et al. 2023 for Hispanic) before the final audit. The name appears as "Hi, I'm {name}." before the MAE message.

### Math only, no reading-comprehension items (for now)
The design doc mentions short reading items. They have no numeric answer, so answer-leak detection and the step-based stuck messages don't apply, and the scaffolding rubric gets fuzzier. We keep the core tier to GSM8K and add reading items only if time allows.

### Reply pool by style prompting
To span "gives the answer" to "pure hint", replies are generated with 8 style instructions (including vague and misleading ones) from Qwen2.5-1.5B and 7B, optionally an API model. The style tag is only a sampling device; human labels and the judge decide what a reply actually is. Pool: 5 replies per audit question, styles balanced, 1500 total.

### W&B is optional
Every stage writes its outputs and stats locally. W&B adds run tracking and dataset versioning when credentials exist and is a no-op otherwise (tests and CI run with `WANDB_MODE=disabled`).

### Compute: Kaggle's free 2x T4, not a cluster (Nov 14)
DCC access never arrived and the budget is $0, so every GPU stage runs on Kaggle (about 30 free GPU-hours a week, 12 h sessions). Consequences: all models run in **float16** (T4s have no bfloat16), the 7B judge and 8B reward models are split across two 16 GB GPUs, and every stage checkpoints in batches and resumes from a previous session's output (`python -m tutorbias.kaggle restore`). P100 sessions are refused up front: vLLM needs compute capability >= 7.0. If a vLLM release misbehaves on T4, `TUTORBIAS_BACKEND=hf` runs the same model blocks through transformers. Cost: fp16 can differ slightly from bf16; we report "fp16 on T4" with every number.

### Reward models are audited unquantized
Two 8B RMs fit in fp16 across both T4s with `device_map="auto"`, so we audit the published weights exactly. 4-bit loading exists only as a fallback, and any score produced that way carries "(nf4)" in its model name.

### The GRPO reward judge is 4-bit, and it is audited too
A 7B fp16 judge (~15 GB) can't share a 16 GB T4 with anything, and the policy needs the other GPU. So the training reward uses the same model and rubric in 4-bit NF4 (~5.5 GB) on GPU 1, in its own process (this also keeps the HF Trainer from seeing two GPUs and wrapping the policy in DataParallel). Rather than assume the 4-bit judge behaves like the fp16 one, the audit scores it as a separate scorer (`judge_reward`): the bias that can propagate into the tutor is measured on the exact reward used.

### GRPO sized for a T4
Qwen2.5-1.5B-Instruct + LoRA (r=16, all linear layers), fp16 mixed precision with the LoRA weights in fp32 (the grad scaler refuses fp16 gradients), 4 samples per prompt, 4 prompts per step, 300 steps, completions up to 256 tokens, KL 0.04 to the base model (the adapter switched off, so no second model in memory). No gradient checkpointing by default: with it on, HF generation can't use its KV cache during training and sampling gets much slower. Fallback: Qwen2.5-0.5B. TRL's config changes between releases, so unknown keys are dropped with a warning instead of crashing (tested with TRL 1.18 on CPU with a tiny model: training, checkpoint resume, adapter merge).

### Training prompts: one dialect per question, balanced
Each rl_train question appears once, in one of MAE / AAE / Indian English / L2, assigned round-robin. Bias in the judge can only reach the policy if the policy sees dialect prompts in training; balancing keeps groups equally weighted. Rewards are logged per sample with the group, so per-group reward curves during training are a direct view of propagation.

### Tutor B = canonicalized judge (stretch)
The cheapest debiasing that removes dialect from the reward by construction: the judge scores the reply against the MAE version of the student's message, while the tutor still sees the dialect version. One config line (`judge_sees: mae`). It doesn't need a new judge or new labels, and E7's difference-in-differences then compares Tutor A and Tutor B directly. Cost: in deployment you need the canonical message, which is easy here (we generated it) and not in real use, where the analogue is running the judge on a normalized rewrite.

### E3 attribution uses randomized feature co-occurrence
Each allowed dialect feature is applied with p = 0.5 independently, so which features a message got is close to randomized. Regressing paired score gaps on feature indicators therefore estimates per-feature effects without re-generating text. A permutation null (features shuffled within group) sets the bar and handles multiple comparisons. Leave-one-feature-out regeneration would be cleaner and is stretch.

## Second reward model: Tulu-3-8B-RM instead of ArmoRM (Nov 18)
ArmoRM's remote modeling code imports `LLAMA_INPUTS_DOCSTRING`, which current transformers removed, and
the GPU env can't pin an old transformers because vLLM needs a current one. The audit now uses AI2's
`allenai/Llama-3.1-Tulu-3-8B-RM` (on RewardBench, a standard sequence classifier, different lab and
training data from Skywork). We lose ArmoRM's per-objective breakdown, which the pipeline never used.

## Power check and bias regression gate (Nov 20)

**Power.** Many E2 results are "no penalty". A null result only means something if the audit could have
found a small effect, so `tutorbias.audit.power` turns each cluster-bootstrap CI into a minimum
detectable effect (80% power, Bonferroni over the whole test family, which is conservative for Holm).
Raw gaps: MDE 0.005 to 0.047 SD (median about 0.02). Net of the paraphrase control: 0.024 to 0.053 SD,
because subtracting the paraphrase adds its noise. So the audit can see gaps about ten times smaller than
a conventional "small" effect (0.2 SD). We also report the largest penalty the 95% CI still allows, which
is the honest form of "no penalty": for our grader, no dialect penalty larger than 0.026 SD net.

**Gate.** `configs/bias_gate.yaml` sets a limit (0.05 SD) on the pessimistic end of the 95% CI, for the
dialect groups vs MAE (raw and net of paraphrase) and for name groups vs a white name. Graders that train
the tutor must pass; off-the-shelf RMs are report-only. The limit is the smallest one the audit can check
reliably per the power check, and is a policy choice. On the committed results the training graders
pass, and the gate flags one off-the-shelf result: Tulu-3-RM on Indian English (CI down to -0.058 SD).
CI runs the gate, so a rerun or code change that makes the training grader biased breaks the build.
