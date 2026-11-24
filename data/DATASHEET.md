# Datasheet

Following Gebru et al., "Datasheets for Datasets" (2021). Numbers come from the `*_stats.json` files in `data/processed/`, where the full dataset is committed.

## Motivation
Built to test whether reward models score identical tutoring help differently depending on how a student writes (dialect) or their name, and whether that bias carries into a tutor trained with RL. Not intended for training production tutors or for drawing conclusions about real students.

## Composition
- **Questions:** GSM8K (Cobbe et al., 2021; MIT license). 2000 RL-train and 300 audit questions from GSM8K train (disjoint), 300 test questions from GSM8K test. Filter: integer non-negative answer, 2-6 solution steps, answer not printed in the question. From 7473 train / 1319 test, 6435 / 1125 pass.
- **Stuck-student messages:** one per question, from 4 types x 3 phrasings of hand-written templates. Synthetic. Types are balanced (500/75/75 per type for rl_train/audit/test).
- **Counterfactual variants:** each message in MAE, African American English, Indian English (Multi-VALUE eWAVE varieties), a learner (L2) English feature list, and an MAE paraphrase control. Dialect variants use only eWAVE features rated pervasive/obligatory for the variety, minus a reasoned blocklist (docs/dialect_features.md); audit and test messages also get 16 name variants (4 groups x 2 genders x 2 names). Every variant records the Multi-VALUE features applied. 22,600 variants in total (2,600 per dialect group, 2,400 per name group). Share of messages actually changed by the dialect rules: AAE 84%, Indian English 98%, learner English 67% (some sentences have nothing a rule applies to). Numbers stayed intact in 100% of variants.
- **Tutor replies:** ~1500 model-generated replies to audit messages (Qwen2.5-1.5B/7B, optional API model), 8 prompted styles.
- **Human labels:** 40 replies labeled for scaffolding (-1/0/+1) and helpfulness (1-5) by me (`data/labels/labels_anvita.csv`). A further 300 reference labels written by a model (`labels_model_ref.csv`) were used only during development and are never reported as human labels.

## Collection and synthetic-data notes
No real students, no minors, no personal data. All student text is template-generated and transformed by rules. **Every variant is synthetic and is labeled as such in this release.** Dialect transforms are cleaner and less natural than real speech; native speakers rate 40 variants per group for naturalness, stereotyping, and meaning (docs/dialect_validation.md); the rating sheets are in `data/validation/`; ratings have not been collected yet, so dialect results are reported as preliminary.

## Known limitations
- Rule-based dialect text is a proxy for dialect, not real speech; L2-English overlaps with AAE features.
- Name lists for Hispanic and South Asian groups are unvalidated placeholders (see `names/names.yaml`).
- Names were drawn from adult-era sources; a name is a cue, not an identity.
- Math only; single-turn messages.

## Uses to avoid
Don't use the variants to "pass" a fairness check by tuning against them (fairness washing). Publish per-feature results with any summary number.

## Known transform breakages (found Nov 18, filtered)
Reading the built variants showed two Multi-VALUE bugs in the AAE set: `aint_before_main` turns a
positive do/did into a bare "ai" ("what I ai wrong", ~20% of AAE variants), and
`past_for_past_participle` produces "I am did with it" after a form of be (~7%). No speaker writes
these, so a score gap on them would measure a reaction to broken text. `textutil.garbled_reason`
detects both; the audit pairs, the analysis (also for pairs built earlier), GRPO prompts and eval
prompts drop matching variants and report how many. The Indian English set uses "amn't" and wh-word
reduplication ("what-what") often; those are kept but go to the native-speaker check, and the
per-feature analysis (E3) shows whether they drive any gap.
