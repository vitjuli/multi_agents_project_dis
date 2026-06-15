"""Eval BASE or trained-LoRA CHECKPOINT on GSM8K, writing a results artifact.

The shipped evaluate.py builds a fresh LoRA (== base) and never restores the
trained adapter. This restores the Tunix/Orbax checkpoint AND writes machine
output so the numbers are reproducible artifacts, not hand-typed:
  <out>_summary.json    metrics + provenance (model, ckpt, step, timestamp)
  <out>_per_prompt.csv  one row per test prompt (gold, extracted, correct/...)

  python eval_ckpt.py --policy base --out eval_results/base
  python eval_ckpt.py --policy lora --ckpt ~/ckpts_R0_baseline/actor --out eval_results/R0_baseline
  python eval_ckpt.py --policy lora --ckpt ~/ckpts_R3_drgrpo/actor  --out eval_results/R3_drgrpo
  python eval_ckpt.py --policy lora --ckpt ~/ckpts_R1K4/actor       --out eval_results/R1_K4
"""
import argparse
import csv
import json
import os
import time

from tqdm.auto import tqdm
from flax import nnx
from tunix.generate import sampler as sampler_lib
from tunix.sft import checkpoint_manager as cm_lib

from config import (
    GENERATION_CONFIGS,
    MAX_PROMPT_LENGTH,
    NUM_TEST_BATCHES,
    TEST_DATA_DIR,
    TOTAL_GENERATION_STEPS,
    TRAIN_DATA_DIR,
    TRAIN_FRACTION,
    TRAIN_MICRO_BATCH_SIZE,
    NUM_BATCHES,
    NUM_EPOCHS,
    DATA_SOURCE,
    MODEL_ID,
)
from data import build_train_val_test
from model import (
    build_mesh,
    download_weights,
    load_base_model,
    get_lora_model,
    load_tokenizer,
)
from evaluate import generate
from rewards import match_format, match_numbers


def restore_lora(lora, ckpt_dir, step=None):
    """Restore the trained LoRA adapters (Tunix CheckpointManager). Returns (lora, step)."""
    d = os.path.expanduser(ckpt_dir).rstrip("/")
    if os.path.basename(d).isdigit():        # .../actor/3364 -> .../actor, step=3364
        if step is None:
            step = int(os.path.basename(d))
        d = os.path.dirname(d)
    cm = cm_lib.CheckpointManager(root_directory=d)
    # The checkpoint stores ONLY the trainable LoRA adapters (~312 leaves) — the
    # frozen base is not saved — so restore_only_lora_params=True makes it match.
    got, _ = cm.maybe_restore(lora, restore_only_lora_params=True, step=step)
    if got == 0:
        print(f"[eval_ckpt] WARNING: restored step 0 (NO checkpoint in {d}) -> UNTRAINED model.")
    else:
        print(f"[eval_ckpt] restored LoRA adapters from step {got} ({d})")
    return lora, got


def score_and_log(dataset, sampler, eos, gen_cfg, meta, out_prefix):
    """Greedy eval that records per-prompt outcomes and writes JSON+CSV artifacts."""
    rows = []
    corr = partial = fmt = total = 0
    for batch in tqdm(dataset):
        questions = batch["question"]
        answers = batch["answer"]
        responses = generate(questions, sampler, eos, seed=0, **gen_cfg)
        for q, r, ans in zip(questions, responses, answers):
            m = match_numbers.search(r)
            ext = m.group(1) if m is not None else "-1e9"
            is_corr = is_partial = is_fmt = False
            try:
                if float(ext.strip()) == float(ans.strip()):
                    is_corr = True
                ratio = float(ext.strip()) / float(ans.strip())
                if 0.9 <= ratio <= 1.1:
                    is_partial = True
            except Exception:
                pass
            if match_format.search(r) is not None:
                is_fmt = True
            corr += int(is_corr); partial += int(is_partial); fmt += int(is_fmt); total += 1
            rows.append({
                "idx": total,
                "gold": ans.strip(),
                "extracted": ext.strip()[:32],
                "correct": int(is_corr),
                "partial": int(is_partial),
                "format": int(is_fmt),
                "question": " ".join(q.split())[:140],
                "response": " ".join(r.split())[:240],
            })
            if total % 10 == 0:
                print(f"===> corr={corr} total={total} acc={corr/total*100:.2f}% "
                      f"partial={partial/total*100:.2f}% fmt={fmt/total*100:.2f}%")

    acc, pacc, facc = corr/total*100, partial/total*100, fmt/total*100
    summary = dict(
        meta,
        n_correct=corr, n_total=total,
        accuracy_pct=round(acc, 2), partial_pct=round(pacc, 2), format_pct=round(facc, 2),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    if out_prefix:
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        with open(out_prefix + "_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(out_prefix + "_per_prompt.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[eval_ckpt] wrote {out_prefix}_summary.json + {out_prefix}_per_prompt.csv")
    return corr, total, acc, pacc, facc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="greedy", choices=list(GENERATION_CONFIGS))
    ap.add_argument("--source", default=DATA_SOURCE, choices=["tfds", "kaggle"])
    ap.add_argument("--policy", default="lora", choices=["base", "lora"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--step", type=int, default=None,
                    help="restore a specific checkpoint step (for the collapse curve)")
    ap.add_argument("--out", default=None,
                    help="prefix for <out>_summary.json + <out>_per_prompt.csv")
    args = ap.parse_args()

    mesh = build_mesh()
    local_path, eos = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    tok, eos = load_tokenizer(eos)

    restored_step = None
    if args.policy == "base":
        model = base
        print("[eval_ckpt] BASE gemma-3-1b-it")
    else:
        lora = get_lora_model(base, mesh)
        if args.ckpt:
            lora, restored_step = restore_lora(lora, args.ckpt, args.step)
        else:
            print("[eval_ckpt] WARN: no --ckpt -> untrained adapter")
        model = lora

    _, _, test_ds = build_train_val_test(
        NUM_BATCHES, NUM_TEST_BATCHES, TRAIN_MICRO_BATCH_SIZE,
        TRAIN_FRACTION, NUM_EPOCHS, TRAIN_DATA_DIR, TEST_DATA_DIR,
        source=args.source,
    )
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tok,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        ),
    )
    meta = {
        "model": MODEL_ID,
        "policy": args.policy,
        "ckpt": args.ckpt,
        "restored_step": restored_step,
        "preset": args.preset,
        "dataset": "gsm8k:test",
        "n_prompts_requested": NUM_TEST_BATCHES,
    }
    n, t, acc, pacc, facc = score_and_log(
        test_ds, sampler, eos, GENERATION_CONFIGS[args.preset], meta, args.out
    )
    print(f"\nFINAL [{args.policy} ckpt={args.ckpt} step={restored_step}]: "
          f"acc={acc:.2f}% partial={pacc:.2f}% format={facc:.2f}% ({n}/{t})")


if __name__ == "__main__":
    main()