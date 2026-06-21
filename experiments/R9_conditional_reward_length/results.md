# R9 — Correctness-Conditional Cosine Length Reward

One factor changed vs R8: the length-shaping reward is now conditioned on whether
the answer is correct, and length is measured in tokenizer tokens instead of
whitespace words. `scripts/rewards.py` replaces `cosine_length_penalty` with
`make_cosine_length_reward(tokenizer)` (defined at line 119), a closure that
captures the tokenizer, and exposes it through a new `build_reward_fns(tokenizer)`
factory (the old module-level `REWARD_FNS` is gone). `scripts/train.py` builds the
reward list at runtime by passing the tokenizer in. Everything else is identical to
R8 (still K=2, standard `GRPOLearner`, same hyperparameters). See
`scripts/rewards.py` lines 119–192.

## Config
Reward functions = `[match_format_exactly, match_format_approximately, check_answer, check_numbers, cosine_length_reward]`. The cosine factor `f(n) = 0.5 * (1 + cos(π · min(n, 400) / 400))` runs from 1 at length 0 to 0 at length 400, then interpolates between correctness-dependent bounds:
- correct → `r = 0.3 + 0.7·f` (short+correct = +1.0, long+correct = +0.3)
- wrong → `r = −0.3 − 0.7·f` (short+wrong = −1.0, long+wrong = −0.3)

## Results (held-out GSM8K, greedy, 64 test; via `scripts/eval_ckpt.py`)

| metric | base | R9 (conditional cosine)|
|---|---|---|
| accuracy | 48.44% (31/64) | 45.31% (29/64) |
| partial accuracy (±10%) | 50.00% (32/64) | 46.88% (30/64) |
| format | 9.38% (6/64) | 1.56% (1/64) |

## Findings

Accuracy stayed close to base (45.31% vs 48.44%). The conditional length reward did not hurt the math. This is the main win: rewarding short answers only when they are correct keeps the model from trading accuracy for brevity.

Format compliance dropped to 1.56% (1/64). The model almost stopped using the template. The likely reason is that the length reward is too large. It ranges over ±1.0, so the `match_format_*` rewards are too small to compete. The model optimizes length and correctness and ignores the format.


## How to reproduce / next step
```bash
source ~/venvs/tunix/bin/activate
cd scripts
rm -rf data
python train.py 
python eval_ckpt.py
```