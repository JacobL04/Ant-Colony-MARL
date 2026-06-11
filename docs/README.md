# docs/

Design notes and the demo asset for AntWorld.

- **`demo.gif`** — a short capture of the scripted colony, used in the top-level
  `README.md`. Regenerate it by recording frames from the `rgb_array` renderer
  (see the viewer / renderer modules).
- **`AntWorld_Project_Documentation.docx`** *(optional)* — long-form rationale and
  biological grounding (the *why*). Not committed; drop it here if you keep one.

## Design at a glance (v2 — automatic deposition)

Pheromone secretion is **automatic**, as in real ants — not an action. Each step
a carrying ant lays a food/recruitment trail and a searching ant leaves a faint
exploration scent; the nest continuously emits a homing scent. The policy only
chooses **movement** (8-connected). Pickup and delivery happen on contact. The
sole channel coupling agents is the shared, decaying pheromone field (stigmergy).

## Quick map of the codebase

| Path | Role |
|------|------|
| `antworld/env.py` | `AntWorldEnv` — PettingZoo ParallelEnv; auto pickup/secrete/deliver |
| `antworld/pheromone.py` | shared field: deposit → clip → evaporate → diffuse → clip |
| `antworld/ant.py` | per-agent state (position, carrying, hunger) |
| `antworld/maps.py` | curriculum maps: open → obstacles → t_shape → multi_food |
| `antworld/scripted.py` | `ScriptedForager` — non-learning policy for the viewer |
| `antworld/renderer.py` | Pygame heatmap rendering (read-only) |
| `antworld/viewer.py` | interactive real-time app (`python -m antworld.viewer`) |
| `antworld/config.py` | `EnvConfig` + `OBS_VERSION` (env-contract version) |
| `demo/app.py` | Streamlit in-browser viewer |
| `tests/` | pheromone math / mechanics / determinism / stigmergy diagnostic |

**Next milestone:** an IPPO trainer (`train.py`) + eval/video (`eval.py`) +
experiment configs (`experiments/*.yaml`) for the stigmergy ablation. The env is
already RL-ready (normalized local obs, discrete actions, shaped cooperative reward).
