"""Fetch the per-step W&B series we need for plots (b) and (c) — one shot.

Pulls reward, KL and response length vs step for R0 / R3 / R1 and writes one CSV
per run. Then plot_wandb.py (run on the Mac) makes:
  (b) mean reward + KL vs step, baseline & variants on shared axes
  (c) response length vs step  (the Dr.GRPO length finding; a listed (c) failure-mode plot)

Run on the VM (wandb logged in) in a NEW ssh window — no TPU used:
    source ~/venvs/tunix/bin/activate
    cd ~/tpu-2026/scripts && python fetch_wandb.py
Then scp wandb_*.csv back to the Mac.
"""
import csv
import wandb

PROJECT = "chfazhvi/chfazhvi-grpo"
RUNS = {                       # display name -> short tag
    "R0_baseline_s0": "R0",
    "R3_drgrpo_s0":   "R3",
    "R1_K4_s0":       "R1",
}
KEYS = [
    "_step",
    "rewards/train/sum",
    "rewards/train/mean",
    "actor/train/kl",
    "completions/train/mean_length",
    "completions/eval/mean_length",
]

api = wandb.Api()
found = {r.name: r for r in api.runs(PROJECT) if r.name in RUNS}

for name, tag in RUNS.items():
    if name not in found:
        print(f"!! run '{name}' not found — skipping")
        continue
    rows = []
    for row in found[name].scan_history(keys=KEYS):
        rows.append({k: row.get(k) for k in KEYS})
    with open(f"wandb_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEYS)
        w.writeheader()
        w.writerows(rows)
    # quick sanity: last non-null length + kl
    last_len = next((r["completions/eval/mean_length"] for r in reversed(rows)
                     if r["completions/eval/mean_length"] is not None), None)
    last_kl = next((r["actor/train/kl"] for r in reversed(rows)
                    if r["actor/train/kl"] is not None), None)
    print(f"wrote wandb_{tag}.csv  ({len(rows)} rows)  last eval_len={last_len}  last kl={last_kl}")

print("\nDone. scp wandb_R0.csv wandb_R3.csv wandb_R1.csv back to the Mac.")
