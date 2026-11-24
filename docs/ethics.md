# Ethics: dialects, identity, and how the variants are built

This project changes how a synthetic student writes and checks whether AI graders treat the same help differently. Done carelessly, that kind of work can reproduce the stereotypes it is meant to measure. These are the rules I followed, and why.

## What I claim and what I don't

- **A dialect is not a race or a nationality.** African American English (AAE) is a rule-governed variety of English used by many, but not all, Black Americans, and by some people who are not Black. Indian English is a national variety used by speakers of many backgrounds. My groups are defined by *grammatical features*, and every result is reported as "the grader responds to these features", never as "the grader is biased against Black or Indian students".
- **Synthetic text is not real speech.** Rule-based variants are cleaner and more uniform than how anyone actually writes. A score gap shows sensitivity to dialect features; it doesn't estimate real-world harm to real students.
- **Learner English is not a dialect.** The `l2` group uses features common in second-language English. It describes a stage of learning, not an ethnic or national group.
- **Names are cues, not identities.** A name signals perceived race, ethnicity, or gender to a model. I measure the model's reaction to that cue only.

## Wording

| I say | I avoid | Why |
|---|---|---|
| Mainstream American English (MAE) | "Standard" or "correct" English | Implies other varieties are substandard; MAE is the term used in current sociolinguistics and NLP fairness work |
| African American English (AAE) | "AAVE" in prose, "broken English", "slang" | AAE is the broader linguistic term; the code keeps Multi-VALUE's class name `AfricanAmericanVernacular` only because that's the library's API |
| dialect, variety | "nonstandard", "incorrect grammar" | Dialect features are systematic, not errors |
| learner (L2) English | "bad English" | Learner features are a normal part of acquiring a language |

## How variants are built, with safeguards against caricature

1. **Documented features only.** Variants are produced by Multi-VALUE (Ziems et al., 2023), which implements features from eWAVE, the electronic World Atlas of Varieties of English (Kortmann, Lunkenheimer & Ehret, 2020). I never ask an LLM to "write like" a group.
2. **Only pervasive features.** For AAE and Indian English I keep only features eWAVE rates **A: pervasive or obligatory** in that variety. Features rated B or C (less common or rare) are not used. The full list, with eWAVE numbers and examples, is in [dialect_features.md](dialect_features.md).
3. **A blocklist with reasons.** Features that are wrong in my context are removed, with the reason recorded in `configs/data.yaml`. Examples: *y'all* (my student addresses one tutor, so a plural pronoun would be ungrammatical) and *fixin' to* (a Southern US form that Multi-VALUE files under the same eWAVE feature as AAE *finna*).
4. **No stacking.** Each allowed feature is used in a given message with probability 0.5, so a message doesn't pile on every feature at once, which reads as exaggerated. The features used in each variant are recorded.
5. **No untracked slang.** Multi-VALUE also rewrites "going to" to "I'ma", "have to" to "gotta", and "trying to" to "tryna" for *every* dialect, outside any eWAVE feature. I disable those substitutions and keep only its spelling-repair rules.
6. **Meaning and numbers are preserved.** A variant that changes any number is excluded from the audit.
7. **Native-speaker validation** before the variants are used for any reported result: see [dialect_validation.md](dialect_validation.md). Features that raters flag as unnatural or stereotyped are dropped by a rule fixed in advance, and the variants are rebuilt.

## Problems found in the built variants (Nov 18)

- **Garbled AAE text is filtered out.** Two Multi-VALUE features misfire. `aint_before_main` turns a positive "do/did" into a bare "ai" ("what I ai wrong", about 20% of AAE variants), and `past_for_past_participle` turns adjectival "done" into "did" after a form of *be* ("I am did with it", about 7%). No AAE speaker writes either, so a score gap on them would measure a reaction to broken text, not to the dialect. `tutorbias.textutil.garbled_reason` detects both, and the audit pairs, the analysis, GRPO prompts, eval prompts and the demo all drop matching variants and report how many.
- **Possible caricature in Indian English.** "amn't" and wh-word reduplication ("what-what") are attested in eWAVE but show up in a large share of Indian English variants, and at that rate may read as exaggerated. They are kept for now, flagged for the native-speaker check, and reported separately in the per-feature analysis (E3). If raters flag them, the pre-registered drop rule removes them. No dialect result is reported as final before that check.

## Data and people

- No real student data and no minors. Every student message is synthetic (templated stuck messages, then rule-based rewrites). Tutor replies are model-generated.
- The only human labels are the project author's own 40 spot-check ratings. Native-speaker raters are adult volunteers rating sentence naturalness; they are not research subjects and no data about them is stored beyond their ratings.

## Stakeholders

| Who | What is at stake |
|---|---|
| Students who use AAE, Indian English, or are learning English | Could receive answer-giving help that teaches less, from a tutor trained on a biased grader |
| Teachers, schools and districts | Decide which AI tutors to allow; need evidence about equal treatment |
| Builders of AI tutors and reward models | Their graders can pass bias into every model trained on them |
| Speakers of these varieties generally | Could be caricatured by carelessly generated dialect text |

## Remaining limitations (reported on the poster and in the README)

- Even pervasive features are not used by every speaker, and written messages to a tutor are a more formal register in which speakers may use fewer of them.
- AAE and learner English share some features (e.g. third-person *-s* absence), so group comparisons overlap. Stage 3 compares individual features.
- Validation uses a small number of raters (classmates or volunteers), not a representative sample of speakers.
- eWAVE ratings describe varieties at a regional level; they don't capture age, region, or individual variation.

## Misuse

The variants and audit could be used to tune a grader until it "passes" this test without being fair in general (fairness washing). I publish per-feature results and the full variant set (marked synthetic), not only a pass/fail number.

The variant generator could also be used to produce mocking dialect text. It is limited to documented features, broken output is filtered, and every variant is marked synthetic. Finally, a gap in this synthetic setting is a warning sign, not proof that a deployed tutor harms real students; results should not be cited as that.

## References

- Ziems, C., Held, W., Yang, J., Dhamala, J., Gupta, R., & Yang, D. (2023). Multi-VALUE: A framework for cross-dialectal English NLP. *ACL*.
- Ziems, C., Chen, J., Harris, C., Anderson, J., & Yang, D. (2022). VALUE: Understanding dialect disparity in NLU. *ACL*.
- Kortmann, B., Lunkenheimer, K., & Ehret, K. (Eds.) (2020). *The Electronic World Atlas of Varieties of English* (eWAVE 3.0). https://ewave-atlas.org
- Hofmann, V., Kalluri, P. R., Jurafsky, D., & King, S. (2024). AI generates covertly racist decisions about people based on their dialect. *Nature*, 633, 147-154.
- Blodgett, S. L., Barocas, S., Daumé III, H., & Wallach, H. (2020). Language (technology) is power: A critical survey of "bias" in NLP. *ACL*.
- Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). The risk of racial bias in hate speech detection. *ACL*.
- Green, L. J. (2002). *African American English: A Linguistic Introduction*. Cambridge University Press.
- Rickford, J. R. (1999). *African American Vernacular English: Features, Evolution, Educational Implications*. Blackwell.
- Dulay, H., & Burt, M. (1974). Natural sequences in child second language acquisition. *Language Learning*, 24(1).
- Goldschneider, J. M., & DeKeyser, R. M. (2001). Explaining the "natural order of L2 morpheme acquisition" in English: A meta-analysis of multiple determinants. *Language Learning*, 51(1).
- Bertrand, M., & Mullainathan, S. (2004). Are Emily and Greg more employable than Lakisha and Jamal? *American Economic Review*, 94(4).

Check each citation against the original before it goes on the poster.
