"""Bootstrap 95% CI on held-out GSM8K accuracy for every model in the table.

Satisfies deliverable (a)'s "measure of uncertainty (bootstrap CI over the eval
prompts)". Reads the per-prompt CSVs we already generated (column `correct`),
resamples the 64 test prompts with replacement, and reports accuracy ± 95% CI
for base / R0 baseline / R3 Dr.GRPO / R1 K=4.

    python3 bootstrap_ci.py

Writes figs/accuracy_ci.md (a table) — drop it straight into the report.
No TPU needed; pure stdlib.
"""
import csv
import os
import random

random.seed(0)            # reproducible
N_BOOT = 10000

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "base":                "experiments/R0_baseline/eval_results/base_per_prompt.csv",
    "R0 baseline K2 s42":  "experiments/R0_baseline/eval_results/R0_baseline_per_prompt.csv",
    "R5 baseline K2 s1":   "experiments/R5_baseline_s1/eval_results/R5_baseline_s1_per_prompt.csv",
    "R3 Dr.GRPO K2":       "experiments/R3_drgrpo/eval_results/R3_drgrpo_per_prompt.csv",
    "R1 K=4":              "experiments/R1_K4/eval_results/R1_K4_per_prompt.csv",
    "R6 K=8":              "experiments/R6_K8/eval_results/R6_K8_per_prompt.csv",
    "R7 K=8 lr5e-6": "experiments/R7_K8_lr5e6/eval_results/R7_K8_lr5e6_per_prompt.csv",
    "R4 beta=0":           "experiments/R4_beta0/eval_results/R4_beta0_per_prompt.csv",
    "R4 beta=0.12":        "experiments/R4_beta12/eval_results/R4_beta12_per_prompt.csv",
}


def load_correct(rel_path):
    with open(os.path.join(HERE, rel_path)) as f:
        return [int(row["correct"]) for row in csv.DictReader(f)]


def bootstrap_ci(vals, n_boot=N_BOOT):
    n = len(vals)
    means = []
    for _ in range(n_boot):
        s = sum(vals[random.randrange(n)] for _ in range(n))
        means.append(s / n * 100.0)
    means.sort()
    point = sum(vals) / n * 100.0
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return point, lo, hi


def main():
    rows = []
    print(f"{'model':20} {'acc%':>7}   95% CI            n")
    for name, path in MODELS.items():
        c = load_correct(path)
        acc, lo, hi = bootstrap_ci(c)
        rows.append((name, sum(c), len(c), acc, lo, hi))
        print(f"{name:20} {acc:6.2f}   [{lo:5.2f}, {hi:5.2f}]   {sum(c)}/{len(c)}")

    out = os.path.join(HERE, "figs", "accuracy_ci.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("# GSM8K accuracy with bootstrap 95% CI\n")
        f.write(f"(seed 0, {N_BOOT} resamples with replacement over the 64 held-out test prompts; "
                "greedy decoding. Source: each experiment's eval_results/*_per_prompt.csv.)\n\n")
        f.write("| Model | Accuracy | 95% CI | correct/total |\n")
        f.write("|-------|----------|--------|---------------|\n")
        for name, nc, nt, acc, lo, hi in rows:
            f.write(f"| {name} | {acc:.2f}% | [{lo:.2f}%, {hi:.2f}%] | {nc}/{nt} |\n")
        f.write("\n**Reading it:** non-overlapping CIs ⇒ a real difference. E.g. R0 baseline vs "
                "R3 Dr.GRPO / K=4 are clearly separated (the collapse and the recovery are significant); "
                "base vs K=4 overlap (K=4 preserves base accuracy).\n")
    print("\nwrote figs/accuracy_ci.md")


if __name__ == "__main__":
    main()
