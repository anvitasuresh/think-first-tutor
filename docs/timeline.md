# Timeline (re-planned Sat Nov 14; finish Sat Nov 21, Expo Mon Nov 23, 2026)

**Compute change (Nov 14).** DCC access never came, so the plan drops DCC entirely and runs every GPU
stage on Kaggle's free 2x T4 (about 30 GPU-hours a week, 12 h sessions; see [kaggle.md](kaggle.md)).
The whole pipeline needs about 14 GPU-hours. Budget stays at $0.

**What changed for RL.** Tutor A is GRPO on Qwen2.5-1.5B-Instruct with LoRA in fp16, 4 samples per
prompt, 300 steps (~3-4 h), with the 7B judge in 4-bit as the reward on the second T4. Qwen2.5-0.5B is
the fallback. **Tutor B (debiased judge) is stretch only.**

## Tiers
- **Core (must ship):** judge validated against human labels (E1), counterfactual audit of the judge and
  two RewardBench RMs (E2), feature attribution of the gaps (E3, CPU from the audit scores). This is a
  complete poster result even if RL fails.
- **Full:** Tutor A + evaluation (E5 scaffolding, E6 reward hacking, E7 bias propagation), rationale
  faithfulness (E4), probes.
- **Stretch:** Tutor B (canonicalized judge, `RUN=tutor_b`), extra GRPO seeds, leave-one-feature-out
  re-generation, Hispanic/South Asian name lists from a published source.

## Plan (each gate is a go/no-go)

| Date | Work | GPU (Kaggle) | Gate |
|---|---|---|---|
| Sat Nov 14 | Kaggle account + phone verification. Upload `data/processed` as dataset `tutorbias-data`. Run notebook **01** (replies + judge). Meanwhile: pilot-label 20 items with the guide, line up the second labeler. After 01 finishes: run the **GRPO smoke test** (cells 1-4 of notebook 03) to check memory and seconds per step. | ~1.75 h | |
| Sun Nov 15 | Label 300 replies (second labeler: 100). Calibrate on dev; if the judge misses the bar, write rubric v2 and rescore (notebook 01, judge cell only). Then test once. Start notebook **02** (audit) in the evening; it runs overnight. | ~5 h | **Gate 1:** kappa >= 0.6, Spearman >= 0.5 on test. If missed by Sunday night, report the judge as unreliable and audit the RMs only (core still ships). |
| Mon Nov 16 | Read the audit (Gate 2). Start notebook **03** (Tutor A, ~3-4 h). Run E3 feature effects on the laptop from the audit output (or in notebook 04). | ~4 h | **Gate 2:** audit result exists (the poster's main result). |
| Tue Nov 17 | Check Tutor A: reward rising, KL bounded, replies still fluent (`rewards.jsonl`, logs). Run notebook **05** (eval) and notebook **04** (explain). | ~3 h | **Gate 3:** if Tutor A isn't stable by Tue night, rerun with the 0.5B policy overnight; if that also fails, ship the core tier with the RL failure analysed as a limitation. |
| Wed Nov 18 | E5-E7 analysis, reward-hacking review (read 50 replies by hand). Stretch, only if Tutor A trained cleanly and quota remains: Tutor B (`RUN=tutor_b`, ~3-4 h) and rerun 05. | 0-5 h | |
| Thu Nov 19 | Native-speaker validation results in (rebuild variants only if features get dropped and time allows; otherwise report as a limitation). README results, figures, demo. | | |
| Fri Nov 20 | Poster draft. | | |
| Sat Nov 21 | Done. Fresh-clone reproducibility pass. | | |
| Sun Nov 22 | Buffer. | | |

GPU total for core + full: ~14 h of the ~30 h weekly quota, leaving room for one rerun of any stage
and Tutor B. Check remaining quota on your Kaggle profile before starting a long notebook.

## Earlier plan (Nov 11, superseded)
Stages 1-2 on DCC Thu-Sat, Tutor A Mon Nov 16 on DCC, Tutor B Tue Nov 17. The steps and gates are
unchanged; only the compute moved to Kaggle and Tutor B became stretch.
