"""A hand-coded (non-learning) forager.

This is the policy that drives the interactive viewer before any RL is trained.
It is a *heuristic*, so it is allowed to read env state directly (it is not the
learned policy and is not bound by the local-observation rule). Its job is to
make the world *do something interesting* — stream food home, lay visible
recruitment trails, and fan out to explore — so the pheromone heatmaps come
alive.

Behaviour per ant:
  * carrying  -> head home by climbing the nest-scent gradient (which bends
                 around walls); fall back to greedy distance-to-nest descent.
  * searching -> follow a nearby food-trail up-gradient (recruitment); otherwise
                 wander toward unexplored ground (low exploration scent) with a
                 bias away from the nest, plus a little noise.
Pickup/secretion/delivery are automatic in the env — the forager only steers.
"""

from __future__ import annotations

import numpy as np

from antworld.env import MOVES
from antworld.config import PH_FOOD_TRAIL, PH_EXPLORE, PH_NEST_SCENT


class ScriptedForager:
    def __init__(self, seed: int = 0, trail_follow_prob: float = 0.85,
                 explore_outward_prob: float = 0.5, noise: float = 0.1):
        self.rng = np.random.default_rng(seed)
        self.trail_follow_prob = trail_follow_prob
        self.explore_outward_prob = explore_outward_prob
        self.noise = noise

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # --- gradient helpers (over passable neighbours) ------------------------
    def _ascend(self, env, r, c, channel, strict=True):
        here = env.field.field[channel, r, c]
        best, best_val = None, (here if strict else -np.inf)
        for i, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if not env._passable(nr, nc):
                continue
            v = env.field.field[channel, nr, nc]
            if v > best_val:
                best_val, best = v, i
        return best

    def _descend_to_nest(self, env, r, c):
        cur = env.nest_dist[r, c]
        best, best_score = None, -np.inf
        for i, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if not env._passable(nr, nc):
                continue
            score = cur - env.nest_dist[nr, nc]
            if score > best_score:
                best_score, best = score, i
        return best if best is not None else int(self.rng.integers(0, 8))

    def _explore(self, env, r, c):
        # Prefer the passable neighbour with the LEAST exploration scent (fresh
        # ground), optionally biased outward from the nest.
        outward = self.rng.random() < self.explore_outward_prob
        cur_d = env.nest_dist[r, c]
        best, best_score = None, None
        for i, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if not env._passable(nr, nc):
                continue
            score = -env.field.field[PH_EXPLORE, nr, nc]
            if outward:
                score += 0.3 * (env.nest_dist[nr, nc] - cur_d)
            if best_score is None or score > best_score:
                best_score, best = score, i
        return best if best is not None else int(self.rng.integers(0, 8))

    def _act_one(self, env, ant):
        r, c = ant.row, ant.col
        if self.rng.random() < self.noise:
            return int(self.rng.integers(0, 8))

        if ant.carrying:
            home = self._ascend(env, r, c, PH_NEST_SCENT, strict=True)
            return home if home is not None else self._descend_to_nest(env, r, c)

        # searching: follow a food trail if one is nearby
        if env.field.field[PH_FOOD_TRAIL, r, c] > env.cfg.trail_use_threshold:
            if self.rng.random() < self.trail_follow_prob:
                follow = self._ascend(env, r, c, PH_FOOD_TRAIL, strict=True)
                if follow is not None:
                    return follow
        return self._explore(env, r, c)

    def act(self, env) -> dict[str, int]:
        return {a: self._act_one(env, env.ants[int(a.rsplit('_', 1)[1])]) for a in env.agents}
