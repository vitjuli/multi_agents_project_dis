# R8 — Cosine Length Penalty (output-length reward shaping)

One factor changed vs baseline: the reward functions. `scripts/rewards.py` adds
`cosine_length_penalty` (defined at line 83) to the `REWARD_FNS` list (line 100).
Everything else is the baseline (still K=2, standard `GRPOLearner`). See
`scripts/rewards.py` lines 83–100.

## Config
Reward functions = `[match_format_exactly, match_format_approximately, check_answer, check_numbers, cosine_length_penalty]` (one extra term vs baseline). `cosine_length_penalty` (length = `len(completion.split())`, whitespace-split word count, not tokenizer tokens): below 50 words → flat −0.75; from 50 to 400 words → ramps from 0 up to +0.75 following `0.75 * (1 + cos(π·n/400)) / 2`, capped at the n=400 value beyond that. K=2, everything else stock.

## Results (held-out GSM8K, greedy, 64 test; via `scripts/eval_ckpt.py`)

| metric | base | R8 (cosine length penalty, final) |
|---|---|---|
| accuracy | 48.44% (31/64) | 37.50% (24/64) |
| partial accuracy (±10%) | 50.00% (32/64) | 40.62% (26/64) |
| format | 9.38% (6/64) | 98.44% (63/64) |

Base numbers are the shared `base_summary.json` eval (untrained `gemma-3-1b-it`, no LoRA restored, same 64-prompt greedy harness) — the same base eval R3 and other runs in this repo compare against, not a fresh per-experiment baseline.

## Findings

Format compliance went from 9.38% (base) to 98.44%. Accuracy went from 48.44% (base) down to 37.50%. The model learned the template but got worse at the math. This looks like reward hacking.

The penalty is meant to discourage long completions. The code does the opposite: reward rises from 0 at 50 words to +0.75 at 400 words, with no decay. So this run currently rewards longer completions, not shorter ones.


## How to reproduce
```bash
source ~/venvs/tunix/bin/activate
cd scripts
rm -rf data
python train.py
python eval_ckpt.py
```