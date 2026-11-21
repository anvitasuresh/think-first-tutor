# Explaining the gaps (E3, E4)

Files: `feature_effects.csv/.png` (E3 features), `probes.csv/.md/.png` (E3 probes),
`faithfulness.csv/.md`, `faithfulness_rows.csv` (E4).

## E3. Which dialect features move the score?

Each pair's score change was regressed on the grammar features applied, with a permutation null (features
shuffled across pairs). Out of 22 features x 5 scorers, one effect beats the null, and it *raises* a
score (Skywork, `definite_abstract`, +0.044 SD, 65 pairs). The Indian English penalty in Tulu is spread
over several small effects (dropped pronouns -0.036, repeated question words -0.028, "-ing" verbs -0.023 SD),
each inside the null band. Reading: the penalty is diffuse, not one feature the model dislikes.

## E3. What the grader sees inside (linear probes on Qwen2.5-7B hidden states, 200 pairs per dialect)

- **Dialect is easy to read out.** A linear probe tells a dialect message from its MAE twin with up to
  84% accuracy (Indian English, layer 16), 77% (AAE) and 69% (learner English). Shuffled labels: about 50%.
- **But it barely points toward the score.** The cosine between the dialect direction and the grader's
  score direction is at most 0.13 (layers 24-28) and near 0 in the middle layers where dialect is clearest.
- In the last layers, how far a message moves along the score direction tracks its actual score gap
  (Spearman 0.53 to 0.74), which is expected: those layers are where the score is formed.

So the grader notices how the student writes, and mostly keeps that out of its scoring. This matches E2:
no dialect penalty for our grader.

## E4. Does the grader's written reason admit it?

For 800 pairs, the grader re-scored the reply with a written reason. A lexicon flags reasons that mention
language (grammar, dialect, wording, "proper English", etc.).

- **Never.** In the 30 dialect pairs where the hint score moved by 0.2 or more, 0 reasons mentioned the
  student's language (0 of 41 for the rewording control too; 1 mention in 800 overall).
- **It explains the change with the reply instead.** The reply is identical in each pair, yet the reason
  changes more when the score changes more: reason drift 0.42 to 0.72 for large gaps vs 0.12 to 0.27 when
  the score did not move. The reasons describe the tutor's reply differently ("does the work for the
  student" vs "guides the next step") for the same reply.
- 24 of those 30 large dialect gaps went *up* for the dialect version, so they are noise in both
  directions, not a hidden penalty.
- Rewording alone produced more large gaps (41) than any dialect (AAE 11, Indian 17, learner 2).

Reading: the written reasons are not a faithful account of why a score moved. They should not be used to
audit the grader; the counterfactual test is what catches (or rules out) the bias.
