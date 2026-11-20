# E2 power check

How small a gap could the audit have caught? All numbers are in SD units of each scorer's MAE scores. **MDE** = smallest true gap detected with 80% power at the family-wise alpha (Bonferroni over all tests, so conservative). **Largest penalty** = the most negative gap still inside the 95% CI (0 if the CI is above 0). For scale: 0.2 SD is a conventional "small" effect.

| Scorer | Group | vs | Gap | MDE | Largest penalty | Net gap | Net MDE | Net largest penalty |
|---|---|---|---|---|---|---|---|---|
| judge_scaffolding | aae | mae | +0.011 | 0.016 | +0.000 | +0.011 | 0.042 | -0.008 |
| judge_scaffolding | indian | mae | +0.023 | 0.021 | +0.000 | +0.024 | 0.042 | +0.000 |
| judge_scaffolding | l2 | mae | +0.012 | 0.010 | +0.000 | +0.013 | 0.042 | -0.007 |
| judge_scaffolding | name_black | mae | -0.007 | 0.019 | -0.015 |  |  |  |
| judge_scaffolding | name_black | name_white | -0.001 | 0.006 | -0.003 |  |  |  |
| judge_scaffolding | name_hispanic | mae | -0.009 | 0.018 | -0.018 |  |  |  |
| judge_scaffolding | name_hispanic | name_white | -0.003 | 0.006 | -0.006 |  |  |  |
| judge_scaffolding | name_south_asian | mae | -0.011 | 0.018 | -0.019 |  |  |  |
| judge_scaffolding | name_south_asian | name_white | -0.005 | 0.006 | -0.008 |  |  |  |
| judge_scaffolding | name_white | mae | -0.006 | 0.019 | -0.015 |  |  |  |
| judge_scaffolding | paraphrase | mae | -0.001 | 0.044 | -0.022 |  |  |  |
| judge_helpfulness | aae | mae | +0.013 | 0.026 | +0.000 | -0.001 | 0.053 | -0.026 |
| judge_helpfulness | indian | mae | +0.017 | 0.028 | +0.000 | +0.008 | 0.049 | -0.015 |
| judge_helpfulness | l2 | mae | +0.008 | 0.011 | +0.000 | -0.001 | 0.049 | -0.023 |
| judge_helpfulness | name_black | mae | -0.013 | 0.022 | -0.023 |  |  |  |
| judge_helpfulness | name_black | name_white | -0.003 | 0.010 | -0.008 |  |  |  |
| judge_helpfulness | name_hispanic | mae | -0.013 | 0.019 | -0.022 |  |  |  |
| judge_helpfulness | name_hispanic | name_white | -0.003 | 0.009 | -0.008 |  |  |  |
| judge_helpfulness | name_south_asian | mae | -0.015 | 0.022 | -0.025 |  |  |  |
| judge_helpfulness | name_south_asian | name_white | -0.006 | 0.010 | -0.010 |  |  |  |
| judge_helpfulness | name_white | mae | -0.009 | 0.022 | -0.020 |  |  |  |
| judge_helpfulness | paraphrase | mae | +0.009 | 0.047 | -0.014 |  |  |  |
| judge_reward_scaffolding | aae | mae | +0.018 | 0.017 | +0.000 | +0.026 | 0.039 | +0.000 |
| judge_reward_scaffolding | indian | mae | +0.027 | 0.022 | +0.000 | +0.024 | 0.041 | +0.000 |
| judge_reward_scaffolding | l2 | mae | +0.014 | 0.010 | +0.000 | +0.011 | 0.039 | -0.007 |
| judge_reward_scaffolding | paraphrase | mae | +0.003 | 0.039 | -0.016 |  |  |  |
| skywork_v2 | aae | mae | -0.025 | 0.009 | -0.029 | +0.021 | 0.036 | +0.000 |
| skywork_v2 | indian | mae | -0.034 | 0.017 | -0.042 | -0.021 | 0.037 | -0.038 |
| skywork_v2 | l2 | mae | -0.018 | 0.010 | -0.023 | -0.005 | 0.035 | -0.021 |
| skywork_v2 | name_black | mae | -0.099 | 0.027 | -0.111 |  |  |  |
| skywork_v2 | name_black | name_white | -0.011 | 0.003 | -0.012 |  |  |  |
| skywork_v2 | name_hispanic | mae | -0.119 | 0.026 | -0.131 |  |  |  |
| skywork_v2 | name_hispanic | name_white | -0.031 | 0.004 | -0.033 |  |  |  |
| skywork_v2 | name_south_asian | mae | -0.099 | 0.026 | -0.112 |  |  |  |
| skywork_v2 | name_south_asian | name_white | -0.011 | 0.004 | -0.013 |  |  |  |
| skywork_v2 | name_white | mae | -0.088 | 0.026 | -0.101 |  |  |  |
| skywork_v2 | paraphrase | mae | -0.013 | 0.035 | -0.030 |  |  |  |
| tulu3_rm | aae | mae | -0.020 | 0.007 | -0.023 | +0.011 | 0.027 | -0.001 |
| tulu3_rm | indian | mae | -0.053 | 0.011 | -0.058 | -0.034 | 0.024 | -0.046 |
| tulu3_rm | l2 | mae | -0.016 | 0.005 | -0.019 | +0.002 | 0.024 | -0.009 |
| tulu3_rm | name_black | mae | -0.083 | 0.017 | -0.091 |  |  |  |
| tulu3_rm | name_black | name_white | -0.006 | 0.002 | -0.007 |  |  |  |
| tulu3_rm | name_hispanic | mae | -0.081 | 0.017 | -0.089 |  |  |  |
| tulu3_rm | name_hispanic | name_white | -0.004 | 0.002 | -0.005 |  |  |  |
| tulu3_rm | name_south_asian | mae | -0.076 | 0.016 | -0.084 |  |  |  |
| tulu3_rm | name_south_asian | name_white | +0.002 | 0.002 | +0.000 |  |  |  |
| tulu3_rm | name_white | mae | -0.077 | 0.016 | -0.085 |  |  |  |
| tulu3_rm | paraphrase | mae | -0.018 | 0.023 | -0.029 |  |  |  |

## In one line

- Raw gaps: MDE 0.005 to 0.047 SD (median 0.019).
- Net of paraphrase: MDE 0.024 to 0.053 SD. The paraphrase control adds noise, so net effects are measured less precisely than raw gaps.
