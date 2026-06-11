"""AntWorldEnv — PettingZoo ParallelEnv for stigmergic foraging.

v2 "automatic deposition": pheromone is secreted automatically (chemically), not
chosen. Each step an ant only decides *which way to move* (8-connected). On top
of that the env handles, automatically:

  * pick-up  — stepping onto a food cell while empty-handed grabs one unit.
  * trail    — a CARRYING ant secretes food-trail at its cell every step;
               a SEARCHING ant leaves a faint exploration scent.
  * deliver  — a carrying ant on the nest drops its food (shared reward).
  * nest scent — the nest continuously emits the homing gradient.

The env knows nothing about the learning algorithm and nothing about pygame
(rendering is delegated). The only coordination channel between ants is the
shared pheromone field.

Observation (179-dim float32, normalized, LOCAL only):
    terrain   5x5x4 = 100  [floor, wall, food, nest]
    pheromone 5x5x3 =  75  [food-trail, exploration, nest-scent]
    self            =   4  [carrying, hunger, own-row, own-col]
Action: Discrete(8) — N, NE, E, SE, S, SW, W, NW.
"""

from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from antworld.ant import Ant
from antworld.config import (
    EnvConfig, OBS_DIM, N_ACTIONS, VIEW_RADIUS, WINDOW, N_TERRAIN_CHANNELS,
    PH_FOOD_TRAIL, PH_EXPLORE, PH_NEST_SCENT,
)
from antworld.maps import make_map, CELL_FLOOR, CELL_WALL, CELL_NEST
from antworld.pheromone import PheromoneField

# 8-connected moves: indices 0..7 = N, NE, E, SE, S, SW, W, NW (row, col deltas).
MOVES: tuple[tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
)


def ant_index(agent: str) -> int:
    return int(agent.rsplit("_", 1)[1])


class AntWorldEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "antworld_v2"}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.possible_agents = [f"ant_{i}" for i in range(self.cfg.n_agents)]
        self.agents: list[str] = []

        self.height = self.cfg.grid_height
        self.width = self.cfg.grid_width
        self.terrain: np.ndarray | None = None
        self.food: np.ndarray | None = None
        self.food_sources: list[tuple[int, int]] = []
        self.nest_cells: list[tuple[int, int]] = []
        self.nest_dist: np.ndarray | None = None
        self.field: PheromoneField | None = None
        self.ants: list[Ant] = []
        self.map_layout = None

        self.t = 0
        self.total_food = 0
        self.food_delivered = 0
        self.episode_pickups = 0
        self._trail_follow_num = 0
        self._trail_follow_den = 0

        self._rng = np.random.default_rng(self.cfg.seed)
        self._renderer = None

    # --- spaces -------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(N_ACTIONS)

    # --- reset --------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.t = 0
        self.food_delivered = 0
        self.episode_pickups = 0
        self._trail_follow_num = 0
        self._trail_follow_den = 0

        self.map_layout = make_map(self.cfg.map_name, self.cfg, self._rng)
        self.terrain = self.map_layout.terrain.copy()
        self.height, self.width = self.terrain.shape
        self.nest_cells = list(self.map_layout.nest_cells)
        self.food_sources = list(self.map_layout.food_sources)

        self.food = np.zeros((self.height, self.width), dtype=np.int32)
        for (r, c) in self.food_sources:
            self.food[r, c] = self.cfg.food_per_cell
        self.total_food = int(self.food.sum())

        self.nest_dist = self._compute_nest_distance()

        self.field = PheromoneField(
            self.height, self.width,
            evaporation_rate=self.cfg.evaporation_rate,
            diffusion_sigma=self.cfg.diffusion_sigma,
            max_value=self.cfg.pheromone_max,
            trail_enabled=self.cfg.trail_enabled,
            nest_scent_enabled=self.cfg.nest_scent_enabled,
        )

        spawn = self.map_layout.spawn_cells
        self.ants = []
        for i in range(self.cfg.n_agents):
            r, c = spawn[int(self._rng.integers(0, len(spawn)))]
            ant = Ant(agent_id=i, row=r, col=c)
            ant.prev_dist_to_nest = float(self.nest_dist[r, c])
            self.ants.append(ant)

        # Prime the nest scent so a homing gradient exists from the first frame.
        for _ in range(8):
            self._emit_nest_scent()
            self.field.step()

        obs = {a: self._observe(self.ants[i]) for i, a in enumerate(self.agents)}
        return obs, {a: {} for a in self.agents}

    # --- step ---------------------------------------------------------------
    def step(self, actions):
        assert self.field is not None and self.food is not None
        self.t += 1
        food_trail_pre = self.field.field[PH_FOOD_TRAIL].copy()
        rewards = {a: -self.cfg.reward_step_cost for a in self.agents}
        deliveries = 0

        # --- 1. move, auto-pickup, auto-secrete --------------------------
        for a in self.agents:
            ant = self.ants[ant_index(a)]
            action = int(actions[a])
            ant.last_action = action
            ant.bumped = False

            prev_carry = ant.carrying
            pr, pc = ant.row, ant.col
            prev_dist = float(self.nest_dist[pr, pc])

            dr, dc = MOVES[action]
            nr, nc = pr + dr, pc + dc
            if self._passable(nr, nc):
                ant.row, ant.col = nr, nc
            else:
                ant.bumped = True

            # stigmergy diagnostic: did a searching ant climb an existing trail?
            if not prev_carry:
                self._record_trail_use(food_trail_pre, pr, pc, ant.row, ant.col)

            # auto-pickup on arrival
            if not ant.carrying and self.food[ant.row, ant.col] > 0:
                self.food[ant.row, ant.col] -= 1
                ant.carrying = True
                ant.hunger = 0.0
                rewards[a] += self.cfg.reward_pickup
                self.episode_pickups += 1

            # auto-secrete pheromone (the whole point — done for you, chemically)
            if ant.carrying:
                self.field.deposit(PH_FOOD_TRAIL, ant.row, ant.col, self.cfg.food_trail_deposit)
            else:
                self.field.deposit(PH_EXPLORE, ant.row, ant.col, self.cfg.explore_deposit)

            # return-while-carrying shaping (based on carry status at step start)
            new_dist = float(self.nest_dist[ant.row, ant.col])
            if prev_carry and new_dist < prev_dist:
                rewards[a] += self.cfg.reward_return_step
            ant.prev_dist_to_nest = new_dist

        # --- 2. auto-deliver at the nest ---------------------------------
        for a in self.agents:
            ant = self.ants[ant_index(a)]
            if ant.carrying and self.terrain[ant.row, ant.col] == CELL_NEST:
                ant.carrying = False
                ant.hunger = 0.0
                deliveries += 1
                self.food_delivered += 1

        if deliveries:
            shared = deliveries * self.cfg.reward_delivery
            for a in self.agents:
                rewards[a] += shared

        # --- 3. hunger + optional food respawn ---------------------------
        for ant in self.ants:
            ant.hunger += self.cfg.hunger_per_step
        if self.cfg.food_respawn:
            for (r, c) in self.food_sources:
                if self.food[r, c] == 0:
                    self.food[r, c] = self.cfg.food_per_cell

        # --- 4. nest scent + field update --------------------------------
        self._emit_nest_scent()
        self.field.step()

        # --- 5. done flags -----------------------------------------------
        all_delivered = (not self.cfg.food_respawn) and self.food_delivered >= self.total_food
        truncated = self.t >= self.cfg.max_steps
        terminated = bool(all_delivered)

        terminations = {a: terminated for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        obs = {a: self._observe(self.ants[ant_index(a)]) for a in self.agents}
        infos = self._build_infos(done=terminated or truncated)
        if terminated or truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    # --- mechanics ----------------------------------------------------------
    def _passable(self, r, c) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width and self.terrain[r, c] != CELL_WALL

    def _emit_nest_scent(self) -> None:
        if self.cfg.nest_emission <= 0.0:
            return
        for (r, c) in self.nest_cells:
            self.field.deposit(PH_NEST_SCENT, r, c, self.cfg.nest_emission)

    def _record_trail_use(self, pre, pr, pc, nr, nc) -> None:
        if pre[pr, pc] <= self.cfg.trail_use_threshold:
            return
        self._trail_follow_den += 1
        if pre[nr, nc] > pre[pr, pc]:
            self._trail_follow_num += 1

    def _compute_nest_distance(self) -> np.ndarray:
        rows = np.arange(self.height)[:, None]
        cols = np.arange(self.width)[None, :]
        dist = np.full((self.height, self.width), np.inf, dtype=np.float32)
        for (nr, nc) in self.nest_cells:
            dist = np.minimum(dist, np.maximum(np.abs(rows - nr), np.abs(cols - nc)).astype(np.float32))
        return dist

    # --- observation --------------------------------------------------------
    def _observe(self, ant: Ant) -> np.ndarray:
        r, c, rad = ant.row, ant.col, VIEW_RADIUS
        tcodes = self._window2d(self.terrain, r, c, rad, CELL_WALL)
        fwin = self._window2d(self.food, r, c, rad, 0).astype(np.float32)

        terrain = np.zeros((N_TERRAIN_CHANNELS, WINDOW, WINDOW), dtype=np.float32)
        terrain[0] = (tcodes == CELL_FLOOR)
        terrain[1] = (tcodes == CELL_WALL)
        terrain[2] = np.clip(fwin / max(self.cfg.food_per_cell, 1), 0.0, 1.0)
        terrain[3] = (tcodes == CELL_NEST)

        pher = self.field.window(r, c, rad)
        self_feats = np.array([
            1.0 if ant.carrying else 0.0,
            min(ant.hunger / max(self.cfg.max_steps, 1), 1.0),
            r / max(self.height - 1, 1),
            c / max(self.width - 1, 1),
        ], dtype=np.float32)
        return np.concatenate([terrain.ravel(), pher.ravel(), self_feats]).astype(np.float32)

    @staticmethod
    def _window2d(arr, row, col, radius, pad_value):
        size = 2 * radius + 1
        out = np.full((size, size), pad_value, dtype=arr.dtype)
        h, w = arr.shape
        r0, c0 = row - radius, col - radius
        sr0, sr1 = max(r0, 0), min(row + radius + 1, h)
        sc0, sc1 = max(c0, 0), min(col + radius + 1, w)
        out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = arr[sr0:sr1, sc0:sc1]
        return out

    # --- diagnostics --------------------------------------------------------
    @property
    def trail_utilization(self) -> float:
        if self._trail_follow_den == 0:
            return 0.0
        return self._trail_follow_num / self._trail_follow_den

    @property
    def n_carrying(self) -> int:
        return sum(1 for a in self.ants if a.carrying)

    def _build_infos(self, done: bool):
        infos = {a: {} for a in self.agents}
        if done:
            ep = {
                "food_delivered": self.food_delivered,
                "total_food": self.total_food,
                "delivered_fraction": self.food_delivered / self.total_food if self.total_food else 0.0,
                "pickups": self.episode_pickups,
                "trail_utilization": self.trail_utilization,
                "length": self.t,
            }
            for a in infos:
                infos[a]["episode"] = ep
        return infos

    # --- rendering (delegated; env never imports pygame) --------------------
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from antworld.renderer import Renderer
            self._renderer = Renderer(self, mode=self.render_mode)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
