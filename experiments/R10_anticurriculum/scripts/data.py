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
                         curriculum: str = "shuffle"):
    """Materialise (train, val, test) datasets with batching applied.

    For the curriculum modes (R9) we replicate the baseline's train/val
    *membership* exactly — the seed-42 shuffle, the first `num_batches`
    examples, the `train_fraction` cut — and then reorder ONLY the train
    portion by difficulty. The data content of every split is therefore
    identical to the baseline; the sole changed factor is the order in which
    the train examples are visited. The held-out test split is always the
    fixed seed-42 order, so every experiment evaluates on the same 64 prompts.
    """
    if curriculum == "shuffle" or train_fraction == 1.0:
        full = get_dataset(train_dir, "train", source, curriculum).batch(train_micro_batch_size)[:num_batches]
        if train_fraction == 1.0:
            train_ds, val_ds = full.repeat(num_epochs), None
        else:
            cut = int(len(full) * train_fraction)
            train_ds = full[:cut].repeat(num_epochs)
            val_ds = full[cut:].repeat(num_epochs)
    else:
        # Baseline membership: seed-42 shuffle, first num_batches, then cut.
        raw = _load_source(train_dir, "train", source)
        shuffled = grain.MapDataset.source(raw).shuffle(seed=42)
        n = min(num_batches, len(shuffled))
        examples = [{"question": _as_text(shuffled[i]["question"]),
                     "answer": _as_text(shuffled[i]["answer"])} for i in range(n)]
        cut = int(n * train_fraction)
        # Reorder ONLY the train portion; val keeps the baseline order.
        train_ds = (_finalise(_order_examples(examples[:cut], curriculum))
                    .batch(train_micro_batch_size).repeat(num_epochs))
        val_ds = (_finalise(grain.MapDataset.source(examples[cut:]))
                  .batch(train_micro_batch_size).repeat(num_epochs))

    test_ds = get_dataset(test_dir, "test", source).batch(train_micro_batch_size)[:num_test_batches]
    return train_ds, val_ds, test_ds
