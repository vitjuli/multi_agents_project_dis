# GSM8K accuracy with bootstrap 95% CI
(seed 0, 10000 resamples with replacement over the 64 held-out test prompts; greedy decoding. Source: each experiment's eval_results/*_per_prompt.csv.)

| Model | Accuracy | 95% CI | correct/total |
|-------|----------|--------|---------------|
| base | 48.44% | [35.94%, 60.94%] | 31/64 |
| R0 baseline K2 s42 | 3.12% | [0.00%, 7.81%] | 2/64 |
| R5 baseline K2 s1 | 40.62% | [29.69%, 53.12%] | 26/64 |
| R3 Dr.GRPO K2 | 31.25% | [20.31%, 42.19%] | 20/64 |
| R1 K=4 | 48.44% | [35.94%, 60.94%] | 31/64 |
| R6 K=8 | 56.25% | [43.75%, 68.75%] | 36/64 |
| R4 beta=0 | 46.88% | [34.38%, 59.38%] | 30/64 |
| R4 beta=0.12 | 0.00% | [0.00%, 0.00%] | 0/64 |

**Reading it:** non-overlapping CIs ⇒ a real difference. E.g. R0 baseline vs R3 Dr.GRPO / K=4 are clearly separated (the collapse and the recovery are significant); base vs K=4 overlap (K=4 preserves base accuracy).
