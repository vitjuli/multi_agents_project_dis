"""Plots (b) and (c) from the fetched W&B CSVs (run on the Mac).

    python3 plot_wandb.py
(b) figs/reward_kl_curves.png  — mean reward + KL vs step, R0/R3/R1 on shared axes
(c) figs/length_curves.png     — eval response length vs step (the Dr.GRPO finding)
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)
RUNS = {"R0": ("baseline K=2", "#d62728"),
        "R3": ("Dr.GRPO (K=2)", "#2a6fdb"),
        "R1": ("K=4", "#2ca02c")}


def load(tag):
    with open(os.path.join(HERE, f"wandb_{tag}.csv")) as f:
        return list(csv.DictReader(f))


def series(rows, key, drop_zero=False):
    xs, ys = [], []
    for r in rows:
        s, v = r.get("_step"), r.get(key)
        if not s or v in (None, ""):
            continue
        try:
            x, y = float(s), float(v)
        except ValueError:
            continue
        if drop_zero and abs(y) < 1e-7:   # KL/length logging-gap zeros
            continue
        xs.append(x); ys.append(y)
    return xs, ys


def smooth(ys, w=5):
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - w // 2), min(len(ys), i + w // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


# (b) reward + KL on shared x
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
for tag, (label, c) in RUNS.items():
    rows = load(tag)
    xr, yr = series(rows, "rewards/train/mean")
    ax1.plot(xr, yr, "-", color=c, lw=0.6, alpha=.25)               # raw (shows K=2 noise)
    ax1.plot(xr, smooth(yr), "-", color=c, lw=1.8, alpha=.9, label=label)  # trend
    ax2.plot(*series(rows, "actor/train/kl", drop_zero=True), "-", color=c, lw=1.4, alpha=.85, label=label)
ax1.axhline(0, color="gray", lw=.5)
ax1.set_ylabel("mean reward (train)")
ax1.set_title("(b) Mean reward and KL(π_θ‖π_ref) vs GRPO step")
ax1.legend(fontsize=8)
ax2.set_ylabel("KL(π_θ‖π_ref)")
ax2.set_xlabel("GRPO step")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "reward_kl_curves.png"), dpi=160)
print("wrote figs/reward_kl_curves.png")

# (c) response length over training
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for tag, (label, c) in RUNS.items():
    rows = load(tag)
    ax.plot(*series(rows, "completions/eval/mean_length", drop_zero=True),
            "-o", ms=3, color=c, lw=1.5, label=label)
ax.set_xlabel("GRPO step")
ax.set_ylabel("eval mean response length (tokens)")
ax.set_title("(c) Response length over training — baseline spikes (~470) then collapses (~130);\n"
             "Dr.GRPO & K=4 stay stable (~290)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "length_curves.png"), dpi=160)
print("wrote figs/length_curves.png")
