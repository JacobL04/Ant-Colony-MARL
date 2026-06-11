"""Environment configuration for AntWorld.

Everything that defines the world and its dynamics lives in one dataclass so a
whole run is specified by a single object (and so the interactive viewer can
tweak fields live). Bump ``OBS_VERSION`` if the observation/action contract
changes — it invalidates any saved RL checkpoints.

Design note (v2 — "automatic deposition"):
Real ants do not *decide* to lay pheromone; they secrete it continuously. So in
this version depositing is NOT an action. A carrying ant automatically lays a
food/recruitment trail every step; a searching ant leaves a faint exploration
scent. The policy therefore controls only *navigation* — pickup and delivery
also happen automatically on contact. The colony's only channel of coordination
remains the shared, decaying pheromone field (stigmergy).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


OBS_VERSION: int = 2  # v2 = automatic deposition, 8-move action space

# --- fixed observation/action contract ---------------------------------------
VIEW_RADIUS: int = 2                  # 2 -> 5x5 egocentric window
WINDOW: int = 2 * VIEW_RADIUS + 1     # 5
N_TERRAIN_CHANNELS: int = 4           # floor, wall, food, nest
N_PHEROMONE_CHANNELS: int = 3         # food-trail, exploration, nest-scent
N_SELF_FEATURES: int = 4              # carrying, hunger, own-row, own-col

OBS_TERRAIN_DIM: int = WINDOW * WINDOW * N_TERRAIN_CHANNELS       # 100
OBS_PHEROMONE_DIM: int = WINDOW * WINDOW * N_PHEROMONE_CHANNELS   # 75
OBS_DIM: int = OBS_TERRAIN_DIM + OBS_PHEROMONE_DIM + N_SELF_FEATURES  # 179

# Action set: 8-connected movement only. Pickup/deposit/deliver are automatic.
N_ACTIONS: int = 8

# Pheromone channel indices (semantic layout of the shared field).
#   0: food-trail   — auto-secreted while CARRYING; recruitment trail to food.
#   1: exploration  — faint auto-secretion while SEARCHING; "I was here" haze.
#   2: nest-scent   — emitted by the nest, diffuses outward; the homing gradient.
PH_FOOD_TRAIL: int = 0
PH_EXPLORE: int = 1
PH_NEST_SCENT: int = 2

PHEROMONE_NAMES: tuple[str, ...] = ("food-trail", "exploration", "nest-scent")


@dataclass
class EnvConfig:
    """All environment hyperparameters. One instance fully specifies the world.

    Biological grounding: reference species *Camponotus* (carpenter ant).
    Food-trail = recruitment pheromone; nest-scent = colony/home odour;
    exploration scent = the diffuse home-range marking left while foraging.
    """

    # --- world ---
    map_name: str = "open"            # open | obstacles | t_shape | multi_food
    grid_height: int = 32
    grid_width: int = 32
    n_agents: int = 40
    max_steps: int = 600

    # --- pheromone field dynamics (order: deposit -> clip -> evaporate -> diffuse -> clip) ---
    food_trail_deposit: float = 1.0   # secreted per step while carrying (strong)
    explore_deposit: float = 0.12     # secreted per step while searching (faint)
    nest_emission: float = 0.6        # nest-scent emitted at nest cells / step
    evaporation_rate: float = 0.03    # field *= (1 - rate) each step
    diffusion_sigma: float = 0.7      # Gaussian blur sigma each step
    pheromone_max: float = 8.0        # clip ceiling (prevents field blow-up)

    # Ablations: the stigmergy on/off switch. trail_enabled gates the agent-laid
    # channels (food-trail + exploration); nest-scent (homing) is separate so the
    # control condition differs in exactly one variable — peer coordination.
    trail_enabled: bool = True
    nest_scent_enabled: bool = True

    # --- rewards (for the RL phase; deposition is never rewarded) ---
    reward_delivery: float = 1.0      # food delivered to nest; SHARED across all
    reward_pickup: float = 0.01       # picking up food (to the picker)
    reward_return_step: float = 0.01  # per step a carrier moves closer to nest
    reward_step_cost: float = 0.001   # subtracted from every agent each step

    # --- dynamics ---
    food_per_cell: int = 12           # units of food at each source cell
    n_food_sources: int = 5
    food_min_dist_from_nest: int = 8  # min Chebyshev distance of food from nest
    food_respawn: bool = False        # if True, exhausted sources refill (endless)
    hunger_per_step: float = 1.0      # hunger clock increment (obs feature only)

    # --- reproducibility ---
    seed: int | None = None

    # --- diagnostics ---
    trail_use_threshold: float = 0.05  # min local food-trail to count a trail-following move

    def __post_init__(self) -> None:
        if self.grid_height < WINDOW or self.grid_width < WINDOW:
            raise ValueError(f"grid must be at least {WINDOW}x{WINDOW}")
        if self.n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        if not (0.0 <= self.evaporation_rate < 1.0):
            raise ValueError("evaporation_rate must be in [0, 1)")

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})
