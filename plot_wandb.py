import csv
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = [
    ("R0 baseline K=2 s42", "wandb_R0.csv"),
    ("R5 baseline K=2 s1", "wandb_R5.csv"),
    ("R3 Dr.GRPO K=2", "wandb_R3.csv"),
    ("R1 K=4", "wandb_R1.csv"),
    ("R6 K=8", "wandb_R6.csv"),
    ("R7 K=8 lr5e-6", "wandb_R7.csv"),
    ("R4 beta=0", "wandb_R4b0.csv"),
    ("R4 beta=0.12", "wandb_R4b12.csv"),
]


def read_series(csv_path, key):
    steps = []
    values = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = row.get("_step")
            value = row.get(key)

            if step in (None, "") or value in (None, ""):
                continue

            try:
                steps.append(float(step))
                values.append(float(value))
            except ValueError:
                continue

    return steps, values


def plot_metric(key, ylabel, title, output_path):
    plt.figure(figsize=(8, 5))

    for label, csv_path in RUNS:
        path = Path(csv_path)
        if not path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue

        steps, values = read_series(path, key)
        if not steps:
            print(f"No values for {key} in {csv_path}")
            continue

        plt.plot(steps, values, marker="o", linewidth=1.5, markersize=3, label=label)

    plt.xlabel("GRPO training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Wrote {output_path}")


def main():
    Path("figs").mkdir(exist_ok=True)

    plot_metric(
        key="rewards/train/mean",
        ylabel="Mean train reward",
        title="GRPO training mean reward",
        output_path="figs/train_mean_reward_curves.png",
    )

    plot_metric(
        key="actor/train/kl",
        ylabel="Train KL",
        title="GRPO train KL",
        output_path="figs/train_kl_curves.png",
    )

    plot_metric(
        key="completions/eval/mean_length",
        ylabel="Mean eval response length",
        title="Diagnostic: eval response length during GRPO",
        output_path="figs/diagnostic_eval_response_length_curves.png",
    )


if __name__ == "__main__":
    main()
