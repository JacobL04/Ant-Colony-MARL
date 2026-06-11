"""The scripted forager makes the world *work*, and the stigmergy diagnostic
measures what it should."""

from __future__ import annotations

import numpy as np
import pytest

from antworld import AntWorldEnv, EnvConfig, ScriptedForager
from antworld.env import MOVES
from antworld.config import PH_FOOD_TRAIL

EAST = MOVES.index((0, 1))
WEST = MOVES.index((0, -1))


def test_scripted_forager_forages_and_lays_trails():
    env = AntWorldEnv(EnvConfig(map_name="open", grid_height=20, grid_width=20,
                                n_agents=24, max_steps=400, n_food_sources=4,
                                food_min_dist_from_nest=5, seed=0))
    policy = ScriptedForager(seed=0)
    env.reset(seed=0)
    while env.agents:
        env.step(policy.act(env))
    # The colony picked food up, laid food-trail pheromone, and delivered some.
    assert env.episode_pickups > 0
    assert env.field.total(PH_FOOD_TRAIL) > 0.0
    assert env.food_delivered > 0


def test_trail_utilization_counts_following():
    env = AntWorldEnv(EnvConfig(n_agents=1, grid_height=11, grid_width=11, seed=0))
    env.reset(seed=0)
    # Paint a food-trail gradient increasing to the East.
    env.field.field[PH_FOOD_TRAIL] = np.tile(
        np.arange(env.width, dtype=np.float32) * 0.5, (env.height, 1))
    env.ants[0].row, env.ants[0].col, env.ants[0].carrying = 5, 4, False
    env.step({"ant_0": EAST})  # steps up-gradient
    assert env._trail_follow_den == 1
    assert env._trail_follow_num == 1
    assert env.trail_utilization == pytest.approx(1.0)


def test_trail_utilization_excludes_carrying_and_no_trail():
    env = AntWorldEnv(EnvConfig(n_agents=1, grid_height=11, grid_width=11, seed=0))
    env.reset(seed=0)
    env.field.field[PH_FOOD_TRAIL].fill(0.0)
    env.ants[0].row, env.ants[0].col, env.ants[0].carrying = 5, 5, False
    env.step({"ant_0": EAST})
    assert env._trail_follow_den == 0  # no trail present -> not counted
