# R4 — Cosine Length Penalty (output-length reward shaping)

One factor changed vs baseline: the reward functions. `scripts/rewards.py` adds
`cosine_length_penalty` (defined at line 83) to the `REWARD_FNS` list (line 100).
Everything else is the baseline (still K=2, standard GRPOLearner). See
`scripts/rewards.py` lines 83–100.

---

## Config

Reward functions = `[match_format_exactly, match_format_approximately, check_answer,
check_numbers, cosine_length_penalty]` (one extra term vs baseline).  
Length penalty: +0.75 for responses ≥ 50 tokens, scaled by a cosine curve that
decays to 0 at 400 tokens; −0.75 for responses shorter than 50 tokens.  
K=2, everything else stock.

```
CKPT_DIR = /tmp/content/ckpts_R4_cosine_len/
```

---

## Provenance

W&B: `R4_cosine_len_s0` · branch `R4-cosine-length-penalty` · commit `<fill in>` · 3364 steps

---

## Results (held-out GSM8K, greedy, 64 test; via `scripts/evaluate.py`)

TODO
---

## Findings 

TODO

---


The cosine curve gives maximum reward (+0.75) at exactly 50 tokens and tapers
smoothly to 0 at 400 tokens, so the model is never pushed to write infinitely
long completions. Responses below 50 tokens receive a flat penalty of −0.75,
discouraging the empty-completion collapse seen in the baseline.
