# Tutor evaluation (E5-E7)

## Metrics by model and group

| model | group | scaffolding | scaffolds | helpfulness | answer_leaked | n_words | generic_question | heldout_rm |
|---|---|---|---|---|---|---|---|---|
| base | aae | -0.193 | 0.422 | 2.408 | 0.497 | 144.647 | 0.000 | 12.656 |
| base | indian | -0.363 | 0.321 | 2.125 | 0.476 | 132.426 | 0.000 | 11.103 |
| base | l2 | -0.430 | 0.286 | 2.004 | 0.556 | 125.250 | 0.000 | 10.884 |
| base | mae | -0.333 | 0.327 | 2.176 | 0.490 | 129.937 | 0.000 | 11.262 |
| base | name_black | -0.322 | 0.326 | 2.173 | 0.530 | 148.127 | 0.000 | 11.361 |
| base | name_hispanic | -0.335 | 0.321 | 2.182 | 0.540 | 146.823 | 0.000 | 11.630 |
| base | name_south_asian | -0.349 | 0.313 | 2.130 | 0.507 | 148.041 | 0.000 | 10.925 |
| base | name_white | -0.325 | 0.319 | 2.178 | 0.518 | 147.296 | 0.000 | 11.745 |
| base | paraphrase | -0.304 | 0.340 | 2.203 | 0.530 | 133.727 | 0.000 | 11.781 |
| tutor_a | aae | 0.756 | 0.884 | 3.875 | 0.156 | 136.850 | 0.000 | 9.395 |
| tutor_a | indian | 0.732 | 0.895 | 3.856 | 0.206 | 139.179 | 0.000 | 7.519 |
| tutor_a | l2 | 0.649 | 0.852 | 3.764 | 0.245 | 140.791 | 0.000 | 6.545 |
| tutor_a | mae | 0.707 | 0.863 | 3.802 | 0.190 | 141.757 | 0.000 | 7.222 |
| tutor_a | name_black | 0.668 | 0.843 | 3.780 | 0.160 | 136.318 | 0.000 | 8.399 |
| tutor_a | name_hispanic | 0.675 | 0.844 | 3.767 | 0.176 | 137.522 | 0.000 | 7.974 |
| tutor_a | name_south_asian | 0.644 | 0.823 | 3.739 | 0.177 | 135.621 | 0.000 | 8.338 |
| tutor_a | name_white | 0.657 | 0.844 | 3.733 | 0.177 | 139.044 | 0.000 | 8.029 |
| tutor_a | paraphrase | 0.674 | 0.860 | 3.822 | 0.227 | 144.100 | 0.000 | 8.861 |

## E5/E6: tutor minus base (paired by prompt)

| model | vs | metric | diff | ci_lo | ci_hi | p | n | p_holm |
|---|---|---|---|---|---|---|---|---|
| tutor_a | base | scaffolding | 1.0018 | 0.9490 | 1.0554 | 1.324e-290 | 6065 | 9.265e-290 |
| tutor_a | base | scaffolds | 0.5223 | 0.4924 | 0.5530 | 1.29e-243 | 6065 | 6.452e-243 |
| tutor_a | base | helpfulness | 1.6013 | 1.5157 | 1.6882 | 1.61e-290 | 6065 | 9.661e-290 |
| tutor_a | base | answer_leaked | -0.3406 | -0.3799 | -0.3011 | 4.627e-63 | 6065 | 1.851e-62 |
| tutor_a | base | n_words | -6.5598 | -9.5282 | -3.5771 | 1.279e-05 | 6065 | 2.559e-05 |
| tutor_a | base | generic_question | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 6065 | 1.0000 |
| tutor_a | base | heldout_rm | -3.3088 | -3.8345 | -2.7750 | 7.521e-32 | 6065 | 2.256e-31 |

## E7: difference in differences (tutor gap minus base gap)

| metric | model | vs | group | did | ci_lo | ci_hi | p | n_items | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| scaffolding | tutor_a | base | aae | -0.0665 | -0.1969 | 0.0599 | 0.3037 | 173 | 1.0000 |
| scaffolding | tutor_a | base | indian | 0.0563 | -0.0502 | 0.1646 | 0.3094 | 296 | 1.0000 |
| scaffolding | tutor_a | base | l2 | -0.0439 | -0.1671 | 0.0762 | 0.4739 | 196 | 1.0000 |
| scaffolding | tutor_a | base | name_black | -0.0500 | -0.1487 | 0.0530 | 0.3313 | 300 | 1.0000 |
| scaffolding | tutor_a | base | name_hispanic | -0.0305 | -0.1148 | 0.0566 | 0.4933 | 300 | 1.0000 |
| scaffolding | tutor_a | base | name_south_asian | -0.0463 | -0.1477 | 0.0596 | 0.3881 | 300 | 1.0000 |
| scaffolding | tutor_a | base | name_white | -0.0585 | -0.1590 | 0.0395 | 0.2482 | 300 | 1.0000 |
| scaffolding | tutor_a | base | paraphrase | -0.0623 | -0.1814 | 0.0464 | 0.2740 | 300 | 1.0000 |
| scaffolds | tutor_a | base | aae | -0.0636 | -0.1561 | 0.0289 | 0.1724 | 173 | 1.0000 |
| scaffolds | tutor_a | base | indian | 0.0372 | -0.0405 | 0.1182 | 0.3603 | 296 | 1.0000 |
| scaffolds | tutor_a | base | l2 | -0.0306 | -0.1122 | 0.0561 | 0.4685 | 196 | 1.0000 |
| scaffolds | tutor_a | base | name_black | -0.0192 | -0.0867 | 0.0492 | 0.5752 | 300 | 1.0000 |
| scaffolds | tutor_a | base | name_hispanic | -0.0133 | -0.0742 | 0.0450 | 0.6707 | 300 | 1.0000 |
| scaffolds | tutor_a | base | name_south_asian | -0.0275 | -0.0967 | 0.0417 | 0.4471 | 300 | 1.0000 |
| scaffolds | tutor_a | base | name_white | -0.0117 | -0.0817 | 0.0534 | 0.7421 | 300 | 1.0000 |
| scaffolds | tutor_a | base | paraphrase | -0.0167 | -0.0934 | 0.0600 | 0.6657 | 300 | 1.0000 |
| helpfulness | tutor_a | base | aae | -0.0720 | -0.2885 | 0.1488 | 0.5240 | 173 | 1.0000 |
| helpfulness | tutor_a | base | indian | 0.1051 | -0.0769 | 0.2831 | 0.2634 | 296 | 1.0000 |
| helpfulness | tutor_a | base | l2 | -0.0313 | -0.2220 | 0.1566 | 0.7429 | 196 | 1.0000 |
| helpfulness | tutor_a | base | name_black | -0.0180 | -0.1800 | 0.1551 | 0.8338 | 300 | 1.0000 |
| helpfulness | tutor_a | base | name_hispanic | -0.0412 | -0.1860 | 0.1050 | 0.5856 | 300 | 1.0000 |
| helpfulness | tutor_a | base | name_south_asian | -0.0164 | -0.1849 | 0.1638 | 0.8535 | 300 | 1.0000 |
| helpfulness | tutor_a | base | name_white | -0.0709 | -0.2326 | 0.0948 | 0.4069 | 300 | 1.0000 |
| helpfulness | tutor_a | base | paraphrase | -0.0074 | -0.1941 | 0.1847 | 0.9375 | 300 | 1.0000 |
| answer_leaked | tutor_a | base | aae | -0.0116 | -0.0809 | 0.0520 | 0.7394 | 173 | 1.0000 |
| answer_leaked | tutor_a | base | indian | 0.0270 | -0.0439 | 0.1047 | 0.4753 | 296 | 1.0000 |
| answer_leaked | tutor_a | base | l2 | 0.0102 | -0.0714 | 0.0918 | 0.8064 | 196 | 1.0000 |
| answer_leaked | tutor_a | base | name_black | -0.0700 | -0.1358 | -0.0067 | 0.0379 | 300 | 1.0000 |
| answer_leaked | tutor_a | base | name_hispanic | -0.0642 | -0.1292 | 0.0058 | 0.0591 | 300 | 1.0000 |
| answer_leaked | tutor_a | base | name_south_asian | -0.0292 | -0.0950 | 0.0359 | 0.3926 | 300 | 1.0000 |
| answer_leaked | tutor_a | base | name_white | -0.0408 | -0.1117 | 0.0284 | 0.2540 | 300 | 1.0000 |
| answer_leaked | tutor_a | base | paraphrase | -0.0033 | -0.0800 | 0.0733 | 0.9320 | 300 | 1.0000 |
| heldout_rm | tutor_a | base | aae | 0.1375 | -1.4192 | 1.5581 | 0.8532 | 173 | 1.0000 |
| heldout_rm | tutor_a | base | indian | 0.4065 | -0.9372 | 1.6902 | 0.5318 | 296 | 1.0000 |
| heldout_rm | tutor_a | base | l2 | 0.7950 | -0.5271 | 2.2429 | 0.2567 | 196 | 1.0000 |
| heldout_rm | tutor_a | base | name_black | 1.0784 | -0.1358 | 2.2888 | 0.0776 | 300 | 1.0000 |
| heldout_rm | tutor_a | base | name_hispanic | 0.3843 | -0.7904 | 1.4764 | 0.5164 | 300 | 1.0000 |
| heldout_rm | tutor_a | base | name_south_asian | 1.4542 | 0.2208 | 2.6681 | 0.0192 | 300 | 0.7689 |
| heldout_rm | tutor_a | base | name_white | 0.3247 | -0.8792 | 1.4816 | 0.5970 | 300 | 1.0000 |
| heldout_rm | tutor_a | base | paraphrase | 1.1205 | -0.1959 | 2.5157 | 0.1012 | 300 | 1.0000 |
