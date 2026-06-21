"""Self-consistency (majority-vote @ k) eval of a trained LoRA checkpoint on GSM8K.

Test-time scaling: draw k completions per prompt (temperature>0, different seed each)
and take the majority-voted final numeric answer. Same JSON+CSV artifact shape as
eval_ckpt.py, plus a `k` / `method` field.

  python eval_majk.py --ckpt ~/ckpts_R6_K8/actor --k 8 --preset standard --out ~/R6_K8_maj8
  python eval_majk.py --policy base               --k 8 --preset standard --out ~/base_maj8   # control
"""
import argparse
import csv
import json
import os
import time
from collections import Counter

from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.sft import checkpoint_manager as cm_lib

from config import (
    GENERATION_CONFIGS, MAX_PROMPT_LENGTH, NUM_TEST_BATCHES, TEST_DATA_DIR,
    TOTAL_GENERATION_STEPS, TRAIN_DATA_DIR, TRAIN_FRACTION, TRAIN_MICRO_BATCH_SIZE,
    NUM_BATCHES, NUM_EPOCHS, DATA_SOURCE, MODEL_ID,
)
from data import build_train_val_test
from model import build_mesh, download_weights, load_base_model, get_lora_model, load_tokenizer
from evaluate import generate
from rewards import match_numbers


def restore_lora(lora, ckpt_dir, step=None):
    """Restore trained LoRA adapters (Tunix CheckpointManager); same logic as eval_ckpt.py."""
    d = os.path.expanduser(ckpt_dir).rstrip("/")
    if os.path.basename(d).isdigit():
        if step is None:
            step = int(os.path.basename(d))
        d = os.path.dirname(d)
    cm = cm_lib.CheckpointManager(root_directory=d)
    got, _ = cm.maybe_restore(lora, restore_only_lora_params=True, step=step)
    if got == 0:
        print(f"[eval_majk] WARNING: restored step 0 (NO checkpoint in {d}) -> UNTRAINED model.")
    else:
        print(f"[eval_majk] restored LoRA adapters from step {got} ({d})")
    return lora, got


def _extract(r):
    m = match_numbers.search(r)
    if m is None:
        return None
    try:
        return float(m.group(1).strip())
    except Exception:
        return None


def score_majk(dataset, sampler, eos, gen_cfg, meta, out_prefix, k):
    """k samples per prompt, majority-vote the extracted answer, score the vote."""
    rows = []
    corr = total = 0
    for batch in tqdm(dataset):
        questions = batch["question"]
        answers = batch["answer"]
        # k independent samples for the whole batch (one seed per sample round)
        samples = [generate(questions, sampler, eos, seed=j, **gen_cfg) for j in range(k)]
        for qi, (q, ans) in enumerate(zip(questions, answers)):
            votes = [v for v in (_extract(samples[j][qi]) for j in range(k)) if v is not None]
            chosen = Counter(votes).most_common(1)[0][0] if votes else None
            is_corr = False
            try:
                is_corr = chosen is not None and chosen == float(ans.strip())
            except Exception:
                pass
            corr += int(is_corr)
            total += 1
            rows.append({
                "idx": total,
                "gold": ans.strip(),
                "voted": ("%g" % chosen) if chosen is not None else "-",
                "n_valid": len(votes),
                "k": k,
                "correct": int(is_corr),
                "question": " ".join(q.split())[:140],
            })
            if total % 10 == 0:
                print(f"===> corr={corr} total={total} maj@{k}_acc={corr/total*100:.2f}%")

    acc = corr / total * 100
    summary = dict(
        meta, n_correct=corr, n_total=total, k=k,
        accuracy_pct=round(acc, 2),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    if out_prefix:
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        with open(out_prefix + "_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(out_prefix + "_per_prompt.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[eval_majk] wrote {out_prefix}_summary.json + {out_prefix}_per_prompt.csv")
    return corr, total, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="standard", choices=list(GENERATION_CONFIGS))
    ap.add_argument("--source", default=DATA_SOURCE, choices=["tfds", "kaggle"])
    ap.add_argument("--policy", default="lora", choices=["base", "lora"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--k", type=int, default=8, help="samples per prompt for the majority vote")
    ap.add_argument("--out", default=None, help="prefix for <out>_summary.json + <out>_per_prompt.csv")
    args = ap.parse_args()

    mesh = build_mesh()
    local_path, eos = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    tok, eos = load_tokenizer(eos)

    restored_step = None
    if args.policy == "base":
        model = base
        print("[eval_majk] BASE gemma-3-1b-it")
    else:
        lora = get_lora_model(base, mesh)
        if args.ckpt:
            lora, restored_step = restore_lora(lora, args.ckpt, args.step)
        else:
            print("[eval_majk] WARN: no --ckpt -> untrained adapter")
        model = lora

    _, _, test_ds = build_train_val_test(
        NUM_BATCHES, NUM_TEST_BATCHES, TRAIN_MICRO_BATCH_SIZE,
        TRAIN_FRACTION, NUM_EPOCHS, TRAIN_DATA_DIR, TEST_DATA_DIR, source=args.source,
    )
    sampler = sampler_lib.Sampler(
        transformer=model, tokenizer=tok,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            num_layers=cfg.num_layers, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
        ),
    )
    meta = {
        "model": MODEL_ID, "policy": args.policy, "ckpt": args.ckpt,
        "restored_step": restored_step, "preset": args.preset,
        "dataset": "gsm8k:test", "n_prompts_requested": NUM_TEST_BATCHES,
        "method": f"self-consistency maj@{args.k}",
    }
    n, t, acc = score_majk(test_ds, sampler, eos, GENERATION_CONFIGS[args.preset], meta, args.out, args.k)
    print(f"\nFINAL [maj@{args.k} {args.policy} ckpt={args.ckpt} step={restored_step}]: acc={acc:.2f}% ({n}/{t})")


if __name__ == "__main__":
    main()
