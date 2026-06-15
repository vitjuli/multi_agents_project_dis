"""Single source of truth for hyperparameters and paths.

Everything in this file is a knob you might tune. Other scripts import from here
so a change in one place propagates everywhere.
"""
import os
import jax

# ====== Model ======
MODEL_ID = "google/gemma-3-1b-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# ====== Data ======
TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test"
TRAIN_FRACTION = 0.9
DATA_SOURCE = os.environ.get("DATA_SOURCE", "tfds")  # "tfds" or "kaggle"

# ====== Curriculum (R9: data-ordering experiment) ======
# The ONLY change vs the R0 baseline: reorder the TRAIN split by a difficulty
# proxy (# of <<...>> calculator steps in the GSM8K gold solution; Cobbe et al.
# 2021). Same prompts, same held-out eval split, same step count, same compute,
# same seed=42 tie-break -> a controlled single-factor comparison (order only).
#   "shuffle"   : baseline behaviour (random order, seed 42) -> reproduces R0
#   "easy2hard" : curriculum        — fewest reasoning steps first
#   "hard2easy" : anti-curriculum   — hardest first (control)
# Theory (I.2c / I.4 Q1): early on, easy problems are winnable, so the *correctness*
# reward varies within a group (sigma_r > 0) instead of being all-equal (a
# degenerate group, sigma_r = 0, zero advantage signal). Fewer degenerate groups
# early => real learning signal when the baseline has only format-shaping signal
# => should mitigate the seed-42 over-optimisation collapse seen in R0.
CURRICULUM = os.environ.get("CURRICULUM", "shuffle")

# ====== Difficulty filter (R11) ======
# Keep only GSM8K train problems whose difficulty (reasoning_difficulty(), the
# <<...>> step count) falls in [DIFFICULTY_MIN, DIFFICULTY_MAX]. Theory: trivial
# problems (all K rollouts correct) and impossible ones (all K fail) are both
# *degenerate* groups (sigma_r = 0). Filtering to the mid band keeps the prompts
# where rollouts disagree -> maximises within-group sigma_r -> most informative
# gradient per step. Empty/0 => no filter (whole range).
_dmin = os.environ.get("DIFFICULTY_MIN", "")
_dmax = os.environ.get("DIFFICULTY_MAX", "")
DIFFICULTY_MIN = int(_dmin) if _dmin else None
DIFFICULTY_MAX = int(_dmax) if _dmax else None

# ====== Second-source mix (R12) ======
# Blend a fraction MIX_ALPHA of the train set with a second math source.
# Only (question, gold numeric answer) are used (GRPO never sees the solution
# text), so any grade-school math QA set works. "asdiv" => EleutherAI/asdiv.
# Theory: an easier second source adds winnable problems -> fewer degenerate
# correctness groups -> higher early sigma_r (same lever as the curriculum),
# at the cost of some distribution shift away from the GSM8K test set.
SECOND_SOURCE = os.environ.get("SECOND_SOURCE", "")        # "" (off) or "asdiv"
MIX_ALPHA = float(os.environ.get("MIX_ALPHA", "0"))        # fraction of train from the 2nd source
MIX_SEED = int(os.environ.get("MIX_SEED", "42"))

# ====== LoRA (parameter-efficient finetuning) ======
# Only the LoRA adapters are trained; the base model is frozen and shared with
# the reference model. Smaller rank => fewer trainable params, smaller KL drift.
RANK = 64
ALPHA = 64.0

# ====== Sharding (TPU mesh) ======
NUM_TPUS = len(jax.devices())
if NUM_TPUS == 8:
    MESH_COUNTS = (1, 4)
elif NUM_TPUS == 4:
    MESH_COUNTS = (1, 4)
elif NUM_TPUS == 1:
    MESH_COUNTS = (1, 1)
else:
    raise ValueError(f"Unsupported number of TPUs: {NUM_TPUS}")

MESH = [MESH_COUNTS, ("fsdp", "tp")]

# ====== Generation during GRPO rollouts ======
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
TEMPERATURE = 0.9          # high enough that the G samples actually differ
TOP_P = 1.0
TOP_K = 50
NUM_GENERATIONS = 2        # G in the GRPO paper — group size for advantage norm

# ====== GRPO loss ======
NUM_ITERATIONS = 1         # mu — PPO-style inner optimisation passes per batch
BETA = 0.08                # KL penalty coefficient (anchors to reference model)
EPSILON = 0.2              # PPO-style clip range

# ====== Training ======
TRAIN_MICRO_BATCH_SIZE = 1
NUM_BATCHES = 3738
NUM_TEST_BATCHES = 64
EVAL_EVERY_N_STEPS = 64
NUM_EPOCHS = 1
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# ====== Optimiser ======
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 0.1 * MAX_STEPS
MAX_GRAD_NORM = 0.1        # tight clipping keeps KL well-behaved

# ====== Checkpointing ======
# Write under the current user's HOME (not /tmp/content, which on the shared VM
# is owned by whichever teammate ran first and is not writable by others). HOME
# is also persistent, so checkpoints survive a /tmp wipe. Override with R9_RUN_ROOT.
_RUN_ROOT = os.environ.get("R9_RUN_ROOT", os.path.join(os.path.expanduser("~"), "r9_runs"))
INTERMEDIATE_CKPT_DIR = f"{_RUN_ROOT}/intermediate_ckpt/"
# Per-run CKPT_DIR (methodology §1): a unique tag per run so runs never resume
# from each other's checkpoints. EXP_TAG can be set explicitly; otherwise it is
# derived from whichever data mode is active (mix > filter > curriculum).
EXP_TAG = os.environ.get("EXP_TAG", "")
if not EXP_TAG:
    if SECOND_SOURCE and MIX_ALPHA > 0:
        EXP_TAG = f"mix_{SECOND_SOURCE}_a{MIX_ALPHA}"
    elif DIFFICULTY_MIN is not None or DIFFICULTY_MAX is not None:
        EXP_TAG = f"filter_{DIFFICULTY_MIN}-{DIFFICULTY_MAX}"
    else:
        EXP_TAG = CURRICULUM
CKPT_DIR = f"{_RUN_ROOT}/ckpts_{EXP_TAG}/"
TENSORBOARD_DIR = f"{_RUN_ROOT}/tensorboard/grpo"
# R9: save often and keep ALL checkpoints. R9 is about early-training dynamics
# (does the curriculum delay/prevent the R0 collapse?), so we need the early
# trajectory for the accuracy-vs-step curve and to catch the held-out reward
# peak (R0 peaked ~step 800 but had already evicted it at MAX_TO_KEEP=4).
# LoRA adapters are tiny (~312 leaves), so keeping ~14 checkpoints is cheap.
SAVE_INTERVAL_STEPS = 250
MAX_TO_KEEP = 20

# ====== Inference presets ======
GENERATION_CONFIGS = {
    "greedy":   {"temperature": None, "top_k": 1,    "top_p": None},
    "standard": {"temperature": 0.7,  "top_k": 50,   "top_p": 0.95},
    "liberal":  {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

# ====== W&B ======
# Set WANDB_RUN_ID in env to resume an existing run (e.g. "bnh9ttlt").
# Project + entity must match the existing run, otherwise wandb won't find it.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "tunix")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "milindsarkaryt-iiser-mohali")
WANDB_RUN_ID = os.environ.get("WANDB_RUN_ID", None)
