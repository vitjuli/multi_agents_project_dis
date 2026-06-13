"""Degenerate-group fraction vs K (deliverable c) — computed from W&B logs.

A 'degenerate group' = all K rollout rewards equal ⇒ std(r)=0 ⇒ the GRPO advantage
Â_i=(r_i−r̄)/σ_r is undefined / zero signal. With micro_batch=1, `rewards/train/min`
and `rewards/train/max` are the min/max reward *within one group*, so the group is
degenerate ⟺ min==max. Theory (I.4 Q1): larger K ⇒ fewer degenerate groups
(P(all K equal) ~ p^K+(1-p)^K falls with K). This measures it empirically.

Run on the VM (wandb is installed + logged in there) in a NEW ssh window — no TPU used:
    cd ~/tpu-2026/scripts && python degenerate_groups.py
Writes degenerate_groups.csv (overall) + degenerate_groups_binned.csv (vs training
progress, for a plot). scp both back to the Mac; plot_degenerate.py makes the figure.
"""
import csv
import wandb

PROJECT = "chfazhvi/chfazhvi-grpo"
RUNS = {                      # W&B display name -> (label, K)
    "R0_baseline_s0": ("baseline", 2),
    "R3_drgrpo_s0":   ("Dr.GRPO", 2),
    "R1_K4_s0":       ("K=4", 4),
}
NBINS = 10
TOL = 1e-9

api = wandb.Api()
found = {r.name: r for r in api.runs(PROJECT) if r.name in RUNS}

overall, binned = [], []
for name, (label, K) in RUNS.items():
    if name not in found:
        print(f"!! run '{name}' not found in {PROJECT} — skipping")
        continue
    mn, mx = [], []
    for row in found[name].scan_history(keys=["rewards/train/min", "rewards/train/max"]):
        a, b = row.get("rewards/train/min"), row.get("rewards/train/max")
        if a is not None and b is not None:
            mn.append(a); mx.append(b)
    degen = [1 if abs(b - a) < TOL else 0 for a, b in zip(mn, mx)]
    n = len(degen)
    frac = sum(degen) / n if n else float("nan")
    overall.append({"run": name, "label": label, "K": K, "n_steps": n,
                    "degenerate_fraction": round(frac, 4)})
    print(f"{label:10} (K={K}): degenerate-group fraction = {frac:.3f}   over {n} steps")
    for bidx in range(NBINS):
        lo, hi = bidx * n // NBINS, (bidx + 1) * n // NBINS
        seg = degen[lo:hi]
        binned.append({"label": label, "K": K, "bin": bidx,
                       "progress": round((bidx + 0.5) / NBINS, 3),
                       "degenerate_fraction": round(sum(seg) / len(seg), 4) if seg else ""})

if overall:
    with open("degenerate_groups.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(overall[0].keys())); w.writeheader(); w.writerows(overall)
    with open("degenerate_groups_binned.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(binned[0].keys())); w.writeheader(); w.writerows(binned)
    print("\nwrote degenerate_groups.csv + degenerate_groups_binned.csv")
    print("Expectation: K=4 fraction < K=2 fraction (theory: fewer degenerate groups at larger K).")
else:
    print("No runs found — check the display names in RUNS / the project path.")
