# R14 — Activating the PPO clip (μ=2) on K=4

## Idea
Every other run uses μ=1 (NUM_ITERATIONS=1), so the importance ratio ρ≡1 at the update and the
PPO clip never activates (pg_clipfrac=0). Setting μ=2 performs two inner optimisation passes per
rollout batch; on the second pass π_θ has moved, ρ≠1, and the clip engages. This isolates the
effect of the clip on the otherwise-stable K=4 configuration (R1).

## Method
- vs R1_K4 (same K, same rollout budget): NUM_GENERATIONS=4, **NUM_ITERATIONS=2**, β=0.08, seed 42,
  NUM_BATCHES=1869 ⇒ equal rollout budget (≈6728 rollouts) and **MAX_STEPS = 1869·2·0.9 = 3364**.
- Surrogate: L = −E[clip(ρ_i, 1±ε)·Â_i] + β·KL(π_θ‖π_ref), ρ_i = π_θ/π_old. With μ>1 the second
  pass reuses the rollouts sampled under π_old, so ρ_i≠1 and clip(ρ_i,1±ε) becomes active.
- The step count 3364 (vs 1682 at μ=1) confirms the μ=2 override took effect.
- **Mechanism.** With μ>1 the importance ratio leaves 1 on the second inner pass so the PPO clip engages, but the clip is a per-sample, one-sided, first-order trust region and does not bound the global KL, so the policy can still drift catastrophically.

## Results
Held-out GSM8K, greedy, 64 prompts (restored_step 3364):

| run | accuracy | 95% CI | format | note |
|---|---|---|---|---|
| R1_K4 (μ=1) | 48.44% | [35.9, 60.9] | 85.9% | stable |
| **R14 (μ=2)** | **0.00%** (0/64) | [0.0, 0.0] | **0.0%** | total collapse |

Completions are degenerate multilingual gibberish (extracted answer = none on all 64 prompts;
per-prompt CSV confirms). The W&B log (`wandb_results/wandb_R14.csv`) confirms the mechanism empirically: with μ=2
the clip **does** engage — `actor/train/pg_clipfrac` rises to **0.62** (mean ≈0.012), strictly >0, versus
`pg_clipfrac`≡0 in every μ=1 run — yet the reference KL **blows up** to a max of **6.65** (final 2.21, vs
≈0.05–0.5 for stable runs) and the eval-reward metrics go blank. The per-sample clip activated exactly as
intended but did **not** prevent the global-KL blow-up and collapse.

## Discussion
Activating the clip via μ>1 destroys training rather than stabilising it: the second pass reuses
stale rollouts and the extra gradient steps accelerate shaping-reward over-optimisation into
degenerate outputs, even on the K=4 configuration that is stable at μ=1. This confirms I.4 Q2: the
PPO clip is a per-sample, one-sided, first-order trust region that does **not** bound the global
KL, so the policy can drift catastrophically with the clip active. Concise statement: seed/β/K
govern *whether* the collapse occurs; μ governs *how fast*.

## Provenance & reproduction
- W&B run: `R10_mu2_K4` (folder renumbered R10→R14; the W&B run name is unchanged).
- Artifacts: `eval_results/R10_final_summary.json` (+ per-prompt CSV), `run_R10_mu2_K4.log`.
```bash
# config.py reads NUM_ITERATIONS from env (default 1)
bash ~/tpu-2026/scripts/run_mu2.sh     # NUM_ITERATIONS=2 NUM_GENERATIONS=4 NUM_BATCHES=1869
python eval_ckpt.py --ckpt=/tmp/content/ckpts_R10_mu2_K4/actor --step=3364 --out=$HOME/R14_final --policy=lora
```
