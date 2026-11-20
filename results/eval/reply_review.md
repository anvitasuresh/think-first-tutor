# Did the tutor learn to game the grader? (E6)

Three checks: numbers from notebook 05 (6,065 test prompts per model), a second reward model the tutor
never trained on (Skywork-Reward-V2), and a hand read of 48 random trained-tutor replies to MAE students.
Charts: `training_curve.png`, `base_vs_tutor.png`, `heldout_rm_by_leak.png` (made by `scripts/plot_tutor.py`).

## What training changed

| | Base model | Trained tutor |
|---|---|---|
| Replies the grader rates as a hint (score > 0) | 34% | 86% |
| Replies that contain the final answer | 52% | 18% |
| Grader helpfulness (1-5) | 2.2 | 3.8 |
| Mean length (words) | 144 | 138 |
| Replies cut off by the 256-token limit | 58% | 28% |

By student message type, answer leaks fell from 41-61% to 4-35%. The hardest case is the student who asks
for the answer outright ("Can you just tell me the answer?"): the tutor still leaks 20% of the time there,
and its grader score is only +0.21 (base -0.97).

## Signs of gaming

1. **Praise for wrong answers.** When the student says their answer was marked wrong, the tutor opens with
   "That's a great start!" or "You're on the right track" in 48% of replies (base model: 20%). The grader
   rewards it: those replies score +0.17 higher on scaffolding and +0.25 on helpfulness than the tutor's
   other wrong-answer replies. This is mild sycophancy that the reward taught.
2. **Math mistakes the grader does not catch.** In the 48 replies read by hand, about 5 contained a math or
   reading error that the grader still scored as a good hint (for example, "the 15 pomelos she has left are
   1/4 of the total", when she has 5 left and gave 15 away; grader score +0.96).
   The rubric checks *whether* the reply teaches, not whether the teaching is correct. A correctness check
   is the most important missing piece of the reward.
3. **Not seen:** generic filler questions (0% by the detector, and none in the hand read), or length
   padding (replies got 7 words shorter). Saying "I won't just give you the answer" and then giving it
   happened (11% of answer-request replies use that phrase), but those replies leaked *less* often
   (16% vs 21%), and the grader marked the ones that leaked as answers.

## Why the second reward model scores the tutor lower

Skywork scores the tutor 3.3 points lower than the base model. That looks like the classic
over-optimization pattern (training reward up, held-out reward down), but the breakdown says otherwise:
Skywork gives replies that **contain the answer** about 9 points more (base: 15.9 vs 6.5). Among replies
that only give a hint, the tutor scores the same or slightly higher (7.1 vs 6.5). So the drop comes from the
tutor no longer handing out answers, which is the goal. It also shows that a general-purpose chat reward model
would train a tutor to give answers away, which is the reason this project uses a scaffolding grader.

## Caveats

- The leak check is automatic (the final number appears in the reply), so it also counts replies that
  state the answer while explaining a step.
- 28% of tutor replies hit the token limit. The grader and both models see the same limit, but cut-off
  replies score lower on both graders, so the base model's numbers are slightly understated.
- One training run, one seed. The hand read is one reader (not blind to the model).
