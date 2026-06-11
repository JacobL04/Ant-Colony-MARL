"""Env contract + the automatic mechanics that define v2."""

from __future__ import annotations

import numpy as np
import pytest

from antworld import AntWorldEnv, EnvConfig, OBS_DIM, N_ACTIONS
from antworld.env import MOVES, ant_index
from antworld.config import PH_FOOD_TRAIL, PH_EXPLORE, PH_NEST_SCENT
from tests.conftest import make_env

EAST = MOVES.index((0, 1))
WEST = MOVES.index((0, -1))
SOUTH = MOVES.index((1, 0))


# --- contract ---------------------------------------------------------------
def test_obs_and_action_contract():
    assert OBS_DIM == 179
    assert N_ACTIONS == 8
    env = make_env()
    a = env.possible_agents[0]
    assert env.action_space(a).n == 8
    assert env.observation_space(a).shape == (OBS_DIM,)


def test_reset_obs_normalized_local():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert set(obs) == set(env.agents)
    for o in obs.values():
        assert o.shape == (OBS_DIM,) and o.dtype == np.float32
        assert o.min() >= 0.0 and o.max() <= 1.0


def test_step_returns_five_dicts():
    env = make_env()
    env.reset(seed=0)
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rew, term, trunc, info = env.step(actions)
    for d in (obs, rew, term, trunc, info):
        assert set(d) == set(env.possible_agents)
    assert all(isinstance(r, float) for r in rew.values())


def test_truncates_and_empties_agents():
    env = make_env(max_steps=8, food_per_cell=99)
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    steps = 0
    while env.agents:
        env.step({a: int(rng.integers(0, N_ACTIONS)) for a in env.agents})
        steps += 1
        assert steps <= 50
    assert steps == 8 and env.agents == []


def test_pettingzoo_parallel_api():
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test
    parallel_api_test(make_env(max_steps=40), num_cycles=200)


# --- automatic mechanics ----------------------------------------------------
def test_auto_pickup_on_stepping_onto_food():
    env = make_env(n_agents=1)
    env.reset(seed=0)
    ant = env.ants[0]
    ant.row, ant.col, ant.carrying = 5, 5, False
    env.food[5, 6] = 3
    _, rew, *_ = env.step({"ant_0": EAST})
    assert env.ants[0].carrying is True
    assert env.food[5, 6] == 2
    assert rew["ant_0"] == pytest.approx(env.cfg.reward_pickup - env.cfg.reward_step_cost)


def test_carrying_ant_secretes_food_trail():
    env = make_env(n_agents=1)
    env.reset(seed=0)
    ant = env.ants[0]
    ant.row, ant.col, ant.carrying = 5, 5, True
    assert env.field.total(PH_FOOD_TRAIL) == 0.0
    env.step({"ant_0": EAST})
    assert env.field.total(PH_FOOD_TRAIL) > 0.0   # auto-secreted, no action needed


def test_searching_ant_secretes_only_exploration():
    env = make_env(n_agents=1)
    env.reset(seed=0)
    ant = env.ants[0]
    ant.row, ant.col, ant.carrying = 5, 5, False
    env.food[:] = 0  # ensure it doesn't pick up and flip to carrying
    env.step({"ant_0": EAST})
    assert env.field.total(PH_EXPLORE) > 0.0
    assert env.field.total(PH_FOOD_TRAIL) == 0.0


def test_auto_deliver_gives_shared_reward():
    env = make_env(n_agents=2)
    env.reset(seed=0)
    nr, nc = env.nest_cells[0]
    carrier, other = env.ants[0], env.ants[1]
    carrier.row, carrier.col, carrier.carrying = nr, nc + 1, True
    other.row, other.col, other.carrying = 0, 0, False  # far away, searching
    _, rew, *_ = env.step({"ant_0": WEST, "ant_1": SOUTH})
    assert env.food_delivered == 1
    assert env.ants[0].carrying is False
    # both agents receive the +1 delivery reward (shared / cooperative)
    assert rew["ant_0"] >= env.cfg.reward_delivery - env.cfg.reward_step_cost - 1e-9
    assert rew["ant_1"] >= env.cfg.reward_delivery - env.cfg.reward_step_cost - 1e-9


def test_no_action_rewards_deposition():
    # There IS no deposit action (deposition is automatic) — sanity check the
    # action space is movement-only and an idle bump just costs a step.
    env = make_env(n_agents=1)
    env.reset(seed=0)
    ant = env.ants[0]
    ant.row, ant.col, ant.carrying = 0, 0, False  # corner; move NW = blocked
    env.food[:] = 0
    nw = MOVES.index((-1, -1))
    _, rew, *_ = env.step({"ant_0": nw})
    assert env.ants[0].bumped is True
    assert rew["ant_0"] == pytest.approx(-env.cfg.reward_step_cost)


# --- reproducibility --------------------------------------------------------
def test_same_seed_same_rollout():
    def rollout(seed):
        env = make_env(seed=seed)
        env.reset(seed=seed)
        rng = np.random.default_rng(99)
        hist = []
        for _ in range(25):
            if not env.agents:
                break
            obs, rew, *_ = env.step({a: int(rng.integers(0, N_ACTIONS)) for a in env.agents})
            hist.append(({a: o.copy() for a, o in obs.items()}, dict(rew)))
        return hist

    h1, h2 = rollout(7), rollout(7)
    assert len(h1) == len(h2)
    for (o1, r1), (o2, r2) in zip(h1, h2):
        assert r1 == r2
        for a in o1:
            assert np.array_equal(o1[a], o2[a])
