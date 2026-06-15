"""Reward functions for GRPO on GSM8K.

Reward shaping rationale
------------------------
Pure correctness reward is very sparse: the model must produce a parseable
number AND get it right. Early in training it does neither, so the gradient
signal is near-zero. We add cheap shaping rewards that are non-zero almost
immediately:

  1. match_format_exactly        — full template match  (large, sparse)
  2. match_format_approximately  — count of expected tags (dense, easy)
  3. check_answer                — exact / close / wrong numeric match
  4. check_numbers               — fallback that just extracts a number

The total per-rollout reward is the sum across all four. GRPO then normalises
this within the group of G rollouts to compute the advantage. This is the
"group-relative" part of GRPO: no value network is learned; advantages come
from comparing siblings drawn from the same prompt.
"""
import re
import math 
from data import reasoning_start, reasoning_end, solution_start, solution_end


match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags=re.MULTILINE | re.DOTALL,
)


def match_format_exactly(prompts, completions, **kwargs):
    """+3 if the whole template parses, 0 otherwise."""
    return [
        0 if match_format.search(r) is None else 3.0
        for r in completions
    ]


def match_format_approximately(prompts, completions, **kwargs):
    """Up to +2.5 for having each of the five expected tags exactly once."""
    scores = []
    for response in completions:
        s = 0.0
        s += 0.5 if response.count(reasoning_start) == 1 else -0.5
        s += 0.5 if response.find(reasoning_start) == 0 else -0.5
        s += 0.5 if response.count(reasoning_end) == 1 else -0.5
        s += 0.5 if response.count(solution_start) == 1 else -0.5
        s += 0.5 if response.count(solution_end) == 1 else -0.5
        scores.append(s)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    """Reward correctness of the bracketed answer with partial credit."""
    extracted = [
        guess.group(1) if r is not None and (guess := match_format.search(r)) is not None else None
        for r in completions
    ]
    assert len(extracted) == len(answer)

    scores = []
    for guess, true in zip(extracted, answer):
        if guess is None:
            scores.append(0)
            continue
        if guess == true:
            scores.append(3.0)
        elif guess.strip() == true.strip():
            scores.append(1.5)
        else:
            try:
                ratio = float(guess) / float(true)
                if 0.9 <= ratio <= 1.1:
                    scores.append(0.5)
                elif 0.8 <= ratio <= 1.2:
                    scores.append(0.25)
                else:
                    scores.append(-1.0)
            except Exception:
                scores.append(-0.5)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    """Fallback: extract any number after <answer> and compare numerically."""
    question = kwargs["question"]
    extracted = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in completions
    ]

    print("START ============================")
    print(f"Question:\t{question[0]}")
    print(f"Answer:\t{answer[0]}")
    print(f"Response:\t{completions[0]}")
    print(f"Extracted:\t{extracted[0]}")
    print("END ==============================")

    scores = []
    for guess, true in zip(extracted, answer):
        if guess is None:
            scores.append(0)
            continue
        try:
            scores.append(1.5 if float(guess.strip()) == float(true.strip()) else 0.0)
        except Exception:
            scores.append(0)
    return scores


def make_cosine_length_reward(tokenizer):
    """Correctness-conditional cosine length reward (Yeo et al., 2025).

    Shape: a single cosine factor f(n) = 0.5 * (1 + cos(pi * n / L_MAX))
    goes from 1 at n=0 to 0 at n=L_MAX (clamped above). We interpolate
    between two reward bounds, picked by whether the completion is correct:

      correct + short  -> +R_C_MAX  (efficient: rewarded most)
      correct + long   -> +R_C_MIN  (don't ramble when you got it right)
      wrong   + short  ->  R_W_MIN  (don't give up)
      wrong   + long   ->  R_W_MAX  (encourage trying longer when stuck)

    Closure over `tokenizer` because Tunix only passes dataset columns to
    reward fns. Token counts use the tokenizer when possible, otherwise
    fall back to whitespace splitting.
    """
    L_MAX = 400
    R_C_MAX, R_C_MIN =  1.0,  0.3   # correct: short=+1.0, long=+0.3
    R_W_MAX, R_W_MIN = -0.3, -1.0   # wrong:   long =-0.3, short=-1.0

    def _token_count(text):
        try:
            return len(tokenizer.encode(text))
        except Exception:
            return len(text.split())

    def cosine_length_reward(prompts, completions, answer, **kwargs):
        extracted = [
            guess.group(1) if (guess := match_format.search(r)) is not None else None
            for r in completions
        ]
        scores = []
        lengths = []
        for completion, guess, true in zip(completions, extracted, answer):
            n = _token_count(completion)
            lengths.append(n)
            f = 0.5 * (1 + math.cos(math.pi * min(n, L_MAX) / L_MAX))

            correct = False
            if guess is not None and true is not None:
                if guess.strip() == true.strip():
                    correct = True
                else:
                    try:
                        correct = abs(float(guess) / float(true) - 1) < 0.01
                    except Exception:
                        pass

            if correct:
                r = R_C_MIN + (R_C_MAX - R_C_MIN) * f
            else:
                r = R_W_MAX + (R_W_MIN - R_W_MAX) * f
            scores.append(r)

        print(f"[cosine_length_reward] lengths={lengths} scores={[round(s,3) for s in scores]}")
        return scores

    return cosine_length_reward


def build_reward_fns(tokenizer):
    """Assemble the reward function list. Needed because the cosine length
    reward closes over the tokenizer, which is only available at runtime."""
    return [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
        make_cosine_length_reward(tokenizer),
    ]
