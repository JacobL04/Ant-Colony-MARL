# CLAUDE.md

Operational context for working in the **AntWorld** repository. Read this before making changes. Keep it updated as conventions solidify.

> This file is for AI coding assistants and contributors. For the full project rationale, biology, and experiment design, see `docs/AntWorld_Project_Documentation.docx`. This file is the *how*; that document is the *why*.

---

## What this project is

AntWorld is a **multi-agent reinforcement learning** study of collective foraging, inspired by real ant colonies. Independent learning agents ("ants") forage cooperatively using only **local perception** and a **shared pheromone field** — no explicit agent-to-agent communication. The central question: does **stigmergy** (coordination through a shared, decaying environment) *emerge* as a learned strategy, and does it measurably help?

Status: **early / pre-implementation.** Treat the structure below as the target layout. Not all of it exists yet.

---

## Environment & setup

- Python 3.11+, managed with conda: `conda activate antmarl`
- Install deps: `pip install -r requirements.txt`
- Core stack: `torch`, `pettingzoo`, `gymnasium`, `numpy`, `scipy`, `pygame`, `wandb`, `imageio`
- GPU is optional — the 2D grid world trains fine on CPU at this scale.

## Common commands

```bash
# Run unit tests (environment dynamics, pheromone math, reward logic)
pytest tests/ -v

# Train with a given experiment config
python train.py --config experiments/exp1_stigmergy.yaml

# Evaluate a checkpoint: render + log metrics + save video
python eval.py --checkpoint runs/<run_id>/best.pt --render --record

# Sanity-check the env with a scripted (non-learning) policy
python -m antworld.scripted_check --map open

# Launch the interactive demo locally
streamlit run demo/app.py
```

> Before assuming a command is unavailable, check whether the referenced file exists yet — this is a growing repo.

---

## Repository structure

```
antworld/
├── antworld/
│   ├── env.py          # AntWorldEnv — PettingZoo ParallelEnv (the env contract)
│   ├── pheromone.py    # PheromoneField — deposit / evaporate / diffuse
│   ├── ant.py          # per-agent state (position, carrying, energy, role)
│   ├── maps.py         # curriculum map layouts (open → obstacles → T-shape → multi-food)
│   ├── renderer.py     # Pygame visualization (tiles, ants, pheromone heatmaps)
│   └── config.py       # environment hyperparameters (dataclass)
├── train.py            # IPPO training loop (shared-parameter policy)
├── eval.py             # checkpoint → render + metrics
├── experiments/        # one YAML config per experiment (see docs section 9)
├── demo/               # Streamlit app
├── tests/              # pytest unit tests
├── notebooks/          # exploration only — not part of the pipeline
├── docs/               # project documentation (the docx lives here)
└── README.md
```

**Separation of concerns is load-bearing here:**
- `env.py` knows nothing about the learning algorithm.
- The policy/training code knows nothing about Pygame.
- Rendering reads state; it never mutates it.

If a change would couple these, stop and reconsider.

---

## Architecture quick reference

- **Env interface:** PettingZoo `ParallelEnv` (all agents step simultaneously — ants act in parallel, not in turns). Do **not** convert to `AECEnv`.
- **Observation:** flat 179-dim float32 vector per agent = 5×5×4 local terrain (100) + 5×5×3 local pheromone (75) + 4 self-state. Normalized before the network.
- **Action:** discrete, 10 actions = 8 moves + pick-up + deposit-trail.
- **Policy:** shared-parameter MLP actor-critic (`[256, 256]` hidden). One network for all agents.
- **Algorithm:** Independent PPO (IPPO) is the baseline. Only escalate to MAPPO / QMIX if IPPO demonstrably fails to coordinate — and document why.
- **Pheromone update order each step:** deposit → clip → evaporate → diffuse (Gaussian) → clip. Order matters; don't reorder casually.

---

## Project-specific rules (do NOT violate without discussion)

These encode the scientific validity of the project. Breaking them invalidates results.

1. **Never reward pheromone deposition directly.** Trail-laying must *emerge* because it improves foraging. Rewarding it would make the core claim circular. Shaping rewards are limited to: food delivered (+1, shared), pick-up (+0.01), return-while-carrying (+0.01), per-step cost (−0.001).
2. **No explicit agent-to-agent communication channel.** Coordination happens only through the shared pheromone field. Adding a message channel is a *future-work baseline*, not part of v1.
3. **Agents observe locally only.** Never give the policy global state. The centralized critic (if MAPPO is used) may see global state *during training only* — never the actor.
4. **Homogeneous population.** All agents share one policy. Roles must emerge; do not hard-code castes in v1.
5. **Seeded and reproducible.** Every run must be reproducible from a seed. No un-seeded randomness in env or training.
6. **Config-driven experiments.** All hyperparameters live in `experiments/*.yaml`. Never hard-code experiment values in logic — configs must be diffable.
7. **Biological grounding.** Design choices should trace to a documented ant behavior (see docs §4). Reference species: *Camponotus* (carpenter ant). If adding a mechanic, note its biological basis in the PR/commit.

---

## Conventions

- **Style:** PEP 8, type hints on public functions, `dataclass` for config/state.
- **Tests first for env logic:** any change to rewards, observations, or pheromone dynamics needs a test verifying behavior with a scripted policy *before* retraining.
- **Logging:** all metrics go to Weights & Biases. Primary metric = food delivered per episode. Always log the stigmergy diagnostic (trail-utilization rate) — it's the key evidence.
- **Commits:** small and descriptive. If a commit changes the env contract (obs/action/reward), say so explicitly in the message — it invalidates old checkpoints.
- **Checkpoints:** an obs/action/reward change breaks checkpoint compatibility. Note it and bump a version in `config.py`.

---

## Experiments (see docs §9 for full protocol)

| ID | Question | Config |
|----|----------|--------|
| exp1 | Does stigmergy emerge and help? (core) | `experiments/exp1_stigmergy.yaml` |
| exp2 | Optimal pheromone evaporation rate? | `experiments/exp2_evaporation.yaml` |
| exp3 | Does curriculum learning improve transfer? | `experiments/exp3_curriculum.yaml` |
| exp4 | Effect of colony size / growth? | `experiments/exp4_colony_size.yaml` |
| exp5 | Do behavioral roles emerge? | `experiments/exp5_roles.yaml` |

Run exp1 first. It's the headline result and validates the whole setup. A **null result on exp1 is still a valid finding** — report it honestly, don't tune until it "works."

---

## Common failure modes (check these before deep debugging)

- **Agents idle / don't learn:** reward too sparse → verify shaping rewards active; check obs normalization; start on the `open` map.
- **All agents cluster at nest:** nest-gradient signal dominating → reduce its magnitude.
- **Reward flat after ~200k steps:** check LR, try smaller net, confirm obs normalized.
- **NaN losses:** clip gradients (`max_grad_norm=0.5`), lower LR.
- **Pheromone field explodes:** deposit > evaporation equilibrium → reduce deposit amount or raise evaporation.

---

## What not to do

- Don't add predators, aphids, or inter-colony mechanics in v1 — they're future work and a scope-creep risk. Freeze features until exp1 succeeds.
- Don't make the world 3D or photorealistic. It's a 2D top-down grid by design.
- Don't reach for a fancier algorithm to fix a reward-shaping or env bug. Fix the env first.
- Don't commit large run artifacts, videos, or checkpoints to git. Use `.gitignore` and store them with W&B / externally.
