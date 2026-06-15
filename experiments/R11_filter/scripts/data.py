"""GSM8K dataset loading and prompt formatting.

GSM8K is a benchmark of grade-school math word problems. Each example is
(question, answer) where the gold answer string ends with `#### <number>`.

We wrap each question in a chat template that asks the model to produce
its reasoning between <reasoning>...</reasoning> and the final numeric
answer between <answer>...</answer>. The reward functions later check
both the format and the number itself.
"""
import csv
import os
import random
import re
import shutil
from pathlib import Path

import grain
import kagglehub
import tensorflow_datasets as tfds

# Special tokens used by the policy and parsed by the reward fns.
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = (
    f"You are given a problem. First, think about the problem and provide your "
    f"reasoning. Place it between {reasoning_start} and {reasoning_end}. Then, "
    f"provide the final answer (i.e., just one numerical value) between "
    f"{solution_start} and {solution_end}."
)

TEMPLATE = (
    "<start_of_turn>user\n"
    "{system_prompt}\n\n"
    "{question}<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def extract_hash_answer(text: str) -> str | None:
    """GSM8K answers look like '...long explanation... #### 42'."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


_CALC_STEP = re.compile(r"<<.*?>>")


def reasoning_difficulty(answer_text: str) -> int:
    """Difficulty proxy = number of reasoning steps in the GSM8K gold solution.

    GSM8K annotates every arithmetic step with a calculator span like
    `<<48/2=24>>`; the count of these is the standard, label-free proxy for the
    number of reasoning steps / problem difficulty (Cobbe et al. 2021,
    arXiv:2110.14168). Falls back to the number of solution lines (before the
    `#### N` final answer) when no annotations are present, so the proxy is
    always well-defined.
    """
    n = len(_CALC_STEP.findall(answer_text))
    if n == 0:
        body = answer_text.split("####")[0]
        n = len([ln for ln in body.splitlines() if ln.strip()])
    return n


def _as_text(v):
    return v if isinstance(v, str) else v.decode("utf-8")


def _order_examples(examples: list[dict], curriculum: str) -> grain.MapDataset:
    """Sort a (materialised) list of {question, answer} examples by difficulty.

    A fixed seed-42 shuffle is applied first so that difficulty *ties* are broken
    with the same seed the baseline uses — Python's stable sort then preserves
    that random order within each difficulty level. The only factor that changes
    vs the baseline is therefore the ordering across difficulty levels.
    """
    rng = random.Random(42)
    rng.shuffle(examples)
    if curriculum == "easy2hard":
        examples.sort(key=lambda e: reasoning_difficulty(e["answer"]))
    elif curriculum == "hard2easy":
        examples.sort(key=lambda e: reasoning_difficulty(e["answer"]), reverse=True)
    else:
        raise ValueError(f"Unknown curriculum for ordering: {curriculum}")
    return grain.MapDataset.source(examples)


def _download_kaggle_dataset(target_dir: str = "./data/gsm8k") -> str:
    os.makedirs(target_dir, exist_ok=True)
    src = Path(kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a"))
    dst = Path(target_dir)
    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, dst / csv_file.name)
    return target_dir


def _load_source(data_dir: str, split: str, source: str):
    """Return the raw, index-addressable GSM8K split (native order)."""
    os.makedirs(data_dir, exist_ok=True)
    if source == "tfds":
        import tensorflow_datasets.text.gsm8k  # noqa: F401  (registers the builder)
        return tfds.data_source(
            "gsm8k",
            split=split,
            data_dir=data_dir,
            builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
            download=True,
        )
    if source == "kaggle":
        kaggle_dir = _download_kaggle_dataset(data_dir)
        csv_path = os.path.join(kaggle_dir, f"main_{split}.csv")
        with open(csv_path, newline="", encoding="utf-8") as f:
            return [{"question": row["question"], "answer": row["answer"]}
                    for row in csv.DictReader(f)]
    raise ValueError(f"Unknown source: {source}")


_LEADING_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _first_number(s) -> str | None:
    """First numeric token (drops thousands commas and trailing units like '(apples)')."""
    m = _LEADING_NUM.search(str(s).replace(",", ""))
    return m.group(0) if m else None


def _load_second_source(name: str) -> list[dict]:
    """Second math source for the R12 mix. Returns {question, answer} dicts where
    `answer` is wrapped as '#### <number>' so the shared extract_hash_answer() in
    _finalise() recovers the gold number exactly as it does for GSM8K. GRPO uses
    only that number (never the solution text), so this is all the source needs.
    """
    if name == "asdiv":
        from datasets import load_dataset
        ds = load_dataset("EleutherAI/asdiv", split="validation")  # 2305 examples
        out = []
        for ex in ds:
            q = f"{ex.get('body', '') or ''} {ex.get('question', '') or ''}".strip()
            num = _first_number(ex.get("answer", ""))   # "9 (apples)" -> "9"
            if q and num is not None:
                out.append({"question": q, "answer": f"#### {num}"})
        return out
    raise ValueError(f"Unknown second source: {name}")


def _finalise(ds: grain.MapDataset) -> grain.MapDataset:
    """Apply the shared prompt template + answer extraction map."""
    return ds.map(lambda x: {
        "prompts": TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=_as_text(x["question"]),
        ),
        "question": _as_text(x["question"]),
        "answer": extract_hash_answer(_as_text(x["answer"])),
    })


def get_dataset(data_dir: str, split: str = "train", source: str = "tfds",
                curriculum: str = "shuffle") -> grain.MapDataset:
    """Return a grain.MapDataset of {prompts, question, answer} dicts.

    `curriculum` controls the ordering of the WHOLE split:
      "shuffle"   -> random order, seed 42 (the as-shipped baseline behaviour)
      "easy2hard" -> sorted by reasoning_difficulty() ascending  (curriculum)
      "hard2easy" -> sorted by reasoning_difficulty() descending (control)
    NOTE: when a train/val split is taken downstream, use build_train_val_test,
    which splits first and reorders ONLY the train portion (so the curriculum
    does not change which examples land in train vs val).
    """
    data = _load_source(data_dir, split, source)
    if curriculum == "shuffle":
        # Original baseline path — grain's own seeded shuffle, byte-for-byte R0.
        ds = grain.MapDataset.source(data).shuffle(seed=42)
    else:
        examples = [{"question": _as_text(d["question"]), "answer": _as_text(d["answer"])}
                    for d in data]
        ds = _order_examples(examples, curriculum)
    return _finalise(ds)


def build_train_val_test(num_batches: int,
                         num_test_batches: int,
                         train_micro_batch_size: int,
                         train_fraction: float,
                         num_epochs: int,
                         train_dir: str,
                         test_dir: str,
                         source: str = "tfds",
                         curriculum: str = "shuffle",
                         difficulty_min: int | None = None,
                         difficulty_max: int | None = None,
                         second_source: str = "",
                         mix_alpha: float = 0.0,
                         mix_seed: int = 42):
    """Materialise (train, val, test) datasets with batching applied.

    Data-direction knobs (all keep the train COUNT — hence MAX_STEPS and compute —
    fixed; only the *content/order* of the train split changes):
      * curriculum        — R9/R10: reorder the train portion easy/hard.
      * difficulty_[min,max] — R11: keep only GSM8K problems in that step-count band
        (drops the trivial / impossible prompts that form degenerate groups).
      * second_source/mix_alpha — R12: replace an `mix_alpha` fraction of the train
        examples with a second math source (e.g. ASDiv).
    The held-out TEST split is always the fixed seed-42 GSM8K order, so every
    experiment evaluates on the same 64 prompts.
    """
    filtering = (difficulty_min is not None) or (difficulty_max is not None)
    mixing = bool(second_source) and mix_alpha > 0
    simple = (curriculum == "shuffle") and not filtering and not mixing

    if simple or train_fraction == 1.0:
        full = get_dataset(train_dir, "train", source, curriculum).batch(train_micro_batch_size)[:num_batches]
        if train_fraction == 1.0:
            train_ds, val_ds = full.repeat(num_epochs), None
        else:
            cut = int(len(full) * train_fraction)
            train_ds = full[:cut].repeat(num_epochs)
            val_ds = full[cut:].repeat(num_epochs)
    else:
        # Materialised pipeline. Start from the baseline seed-42 GSM8K order.
        raw = _load_source(train_dir, "train", source)
        shuffled = grain.MapDataset.source(raw).shuffle(seed=42)
        examples = [{"question": _as_text(shuffled[i]["question"]),
                     "answer": _as_text(shuffled[i]["answer"])} for i in range(len(shuffled))]
        # R11: keep only the mid-difficulty band (high within-group sigma_r).
        if filtering:
            lo = difficulty_min if difficulty_min is not None else 0
            hi = difficulty_max if difficulty_max is not None else 10**9
            examples = [e for e in examples if lo <= reasoning_difficulty(e["answer"]) <= hi]
        # Same budget as baseline: first num_batches, then the train/val cut.
        n = min(num_batches, len(examples))
        examples = examples[:n]
        cut = int(n * train_fraction)
        train_examples, val_examples = examples[:cut], examples[cut:]
        # R12: mix the second source into the TRAIN split only (val stays GSM8K).
        if mixing:
            second = _load_second_source(second_source)
            random.Random(mix_seed).shuffle(second)
            n_second = min(int(round(len(train_examples) * mix_alpha)), len(second))
            train_examples = train_examples[:len(train_examples) - n_second] + second[:n_second]
            random.Random(mix_seed).shuffle(train_examples)
        # R9/R10: optional curriculum ordering of the (possibly mixed) train split.
        if curriculum in ("easy2hard", "hard2easy"):
            train_src = _order_examples(train_examples, curriculum)
        else:
            train_src = grain.MapDataset.source(train_examples)
        train_ds = _finalise(train_src).batch(train_micro_batch_size).repeat(num_epochs)
        val_ds = (_finalise(grain.MapDataset.source(val_examples))
                  .batch(train_micro_batch_size).repeat(num_epochs))

    test_ds = get_dataset(test_dir, "test", source).batch(train_micro_batch_size)[:num_test_batches]
    return train_ds, val_ds, test_ds
