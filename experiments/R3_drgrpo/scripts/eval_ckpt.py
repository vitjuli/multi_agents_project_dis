"""Eval BASE or trained-LoRA CHECKPOINT on GSM8K.

The shipped evaluate.py builds a fresh LoRA (== base) and never restores the
trained adapter, so it silently measures the base model. This restores the
Tunix/Orbax actor checkpoint into the LoRA before sampling.

  python eval_ckpt.py --policy base --preset greedy
  python eval_ckpt.py --policy lora --ckpt ~/ckpts_R0_baseline/actor/3364
  python eval_ckpt.py --policy lora --ckpt ~/ckpts_R3_drgrpo/actor/3364
"""
import argparse
import os

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
)
from data import build_train_val_test
from model import (
    build_mesh,
    download_weights,
    load_base_model,
    get_lora_model,
    load_tokenizer,
)
from evaluate import evaluate


def restore_lora(lora, ckpt_dir, step=None):
    # Use Tunix's own CheckpointManager (the exact inverse of how it saved).
    # ckpt_dir may be the actor dir (.../actor) or a step dir (.../actor/3364);
    # CheckpointManager wants the actor root and finds the latest step itself
    # (or a specific --step for the collapse curve).
    d = os.path.expanduser(ckpt_dir).rstrip("/")
    if os.path.basename(d).isdigit():        # .../actor/3364 -> .../actor, step=3364
        if step is None:
            step = int(os.path.basename(d))
        d = os.path.dirname(d)
    cm = cm_lib.CheckpointManager(root_directory=d)
    # The checkpoint stores ONLY the trainable LoRA adapters (~312 leaves,
    # keys like layers.*.attn.q_einsum.w_lora_a) — the frozen base is NOT saved.
    # So restore_only_lora_params=True makes the abstract match (LoRAParam only);
    # the base stays at the freshly-downloaded (correct) weights.
    got, _ = cm.maybe_restore(lora, restore_only_lora_params=True, step=step)
    if got == 0:
        print(f"[eval_ckpt] WARNING: restored step 0 (NO checkpoint found in {d}) "
              f"-> this is the UNTRAINED model. Check the path / --step.")
    else:
        print(f"[eval_ckpt] restored LoRA adapters from step {got} ({d})")
    return lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="greedy",
                    choices=list(GENERATION_CONFIGS))
    ap.add_argument("--source", default=DATA_SOURCE,
                    choices=["tfds", "kaggle"])
    ap.add_argument("--policy", default="lora",
                    choices=["base", "lora"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--step", type=int, default=None,
                    help="restore a specific checkpoint step (for the collapse curve)")
    args = ap.parse_args()

    mesh = build_mesh()
    local_path, eos = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    tok, eos = load_tokenizer(eos)

    if args.policy == "base":
        model = base
        print("[eval_ckpt] BASE gemma-3-1b-it")
    else:
        lora = get_lora_model(base, mesh)
        if args.ckpt:
            lora = restore_lora(lora, args.ckpt, args.step)
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
    n, t, acc, pacc, facc = evaluate(
        test_ds, sampler, eos, **GENERATION_CONFIGS[args.preset]
    )
    print(f"\nFINAL [{args.policy} {args.ckpt}]: "
          f"acc={acc:.2f}% partial={pacc:.2f}% "
          f"format={facc:.2f}% ({n}/{t})")


if __name__ == "__main__":
    main()
