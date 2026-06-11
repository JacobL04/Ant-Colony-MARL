"""Shared pytest fixtures for AntWorld tests."""

from __future__ import annotations

import pytest

from antworld import AntWorldEnv, EnvConfig


def make_env(**overrides) -> AntWorldEnv:
    defaults = dict(
        map_name="open", grid_height=11, grid_width=11, n_agents=4,
        max_steps=50, n_food_sources=2, food_per_cell=5,
        food_min_dist_from_nest=3, seed=0,
    )
    defaults.update(overrides)
    return AntWorldEnv(EnvConfig(**defaults))


@pytest.fixture
def env():
    e = make_env()
    e.reset(seed=0)
    return e
