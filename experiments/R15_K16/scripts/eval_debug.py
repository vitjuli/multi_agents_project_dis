"""Diagnostic: print the top-level state-tree keys of base vs LoRA models.

The checkpoint (model_params) has the plain gemma keys (embedder, final_norm,
layers.*, norms). The restore says nnx.state(lora) is MISSING them. This prints
what nnx.state actually contains for base and for the LoRA-wrapped model, so we
can see the exact structural difference and write a restore that matches.

  python eval_debug.py
"""
import os

from flax import nnx

from model import (
    build_mesh,
    download_weights,
    load_base_model,
    get_lora_model,
)


def top_keys(model, name):
    st = nnx.state(model)
    keys = list(st)                      # top-level keys of the State
    print(f"=== {name}: {len(keys)} top-level keys ===")
    for k in keys:
        print("   ", k)
    print()


def main():
    mesh = build_mesh()
    local_path, _ = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    top_keys(base, "BASE nnx.state")

    lora = get_lora_model(base, mesh)
    top_keys(lora, "LORA nnx.state")

    # also: how many leaves each has
    import jax
    bl = jax.tree_util.tree_leaves(nnx.state(base))
    ll = jax.tree_util.tree_leaves(nnx.state(lora))
    print(f"BASE leaves: {len(bl)}   LORA leaves: {len(ll)}")

    # what is ACTUALLY on disk: raw-restore (no target) and inspect the tree.
    import orbax.checkpoint as ocp
    p = os.path.expanduser("~/ckpts_R0_baseline/actor/3364/model_params")
    print("\n=== CHECKPOINT raw restore (no target) ===")
    raw = ocp.StandardCheckpointer().restore(p)
    print("   type:", type(raw).__name__)
    try:
        print("   top keys:", list(raw))
    except Exception as e:
        print("   (top keys err)", e)
    fp, _ = jax.tree_util.tree_flatten_with_path(raw)
    print("   leaves:", len(fp))
    print("   sample paths (first 8, last 4):")
    for path, _ in list(fp)[:8]:
        print("     ", jax.tree_util.keystr(path))
    for path, _ in list(fp)[-4:]:
        print("     ", jax.tree_util.keystr(path))


if __name__ == "__main__":
    main()
