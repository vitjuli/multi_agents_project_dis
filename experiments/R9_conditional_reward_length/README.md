R5 — Correctness-Conditional Cosine Length Reward
One factor changed vs R4: the length-shaping reward is now conditioned on whether the answer is correct, and length is measured in tokenizer tokens instead of whitespace words. `scripts/rewards.py` replaces `cosine_length_penalty` with `make_cosine_length_reward(tokenizer)` — a closure that captures the tokenizer — and exposes it through a new `build_reward_fns(tokenizer)` factory (the old module-level `REWARD_FNS` is gone). `scripts/train.py` now builds the reward list at runtime by passing the tokenizer in. Everything else is identical to R4 (still K=2, standard GRPOLearner, same hyperparameters). See `scripts/rewards.py` lines 119–192.

Config
Reward functions = `[match_format_exactly, match_format_approximately, check_answer, check_numbers, cosine_length_reward]`. The cosine factor `f(n) = 0.5 * (1 + cos(π · min(n, 400) / 400))` runs from 1 at length 0 to 0 at length 400, then interpolates between correctness-dependent bounds:
  · correct  →  r = 0.3 + 0.7 · f          (short+correct = +1.0, long+correct = +0.3)
  · wrong    →  r = −0.3 − 0.7 · f         (short+wrong   = −1.0, long+wrong   = −0.3)
Length is measured with `tokenizer.encode()`, falling back to `.split()` if the call fails. K=2, everything else stock.

```
CKPT_DIR = /tmp/content/ckpts_R5_conditional_cosine/
```

Provenance
W&B: `R5_conditional_cosine_s0` · branch `R5-conditional-cosine-length` · commit `<fill in>` · 3364 steps

Results (held-out GSM8K, greedy, 64 test; via `scripts/evaluate.py`)
TODO

Findings
TODO

Where R4's length term gave the same reward to a short correct answer and a short wrong one, the conditional version splits them: when the model is right, brevity is rewarded; when it's wrong, length is rewarded so it keeps reasoning rather than committing fast to a wrong answer. The R4 discontinuity at n=50 (a ~1.5-point cliff from −0.75 to +0.72) is gone — reward is smooth across all lengths. Switching from `.split()` to tokenizer tokens aligns the length signal with what training actually optimises.
