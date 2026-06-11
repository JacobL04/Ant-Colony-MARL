"""Curriculum maps build cleanly and satisfy structural invariants."""

from __future__ import annotations

import numpy as np
import pytest

from antworld import EnvConfig, make_map, CURRICULUM
from antworld.maps import CELL_WALL, CELL_NEST


@pytest.mark.parametrize("name", CURRICULUM)
def test_map_builds(name):
    cfg = EnvConfig(map_name=name, grid_height=20, grid_width=20, n_food_sources=3)
    m = make_map(name, cfg, np.random.default_rng(0))
    assert m.terrain.shape == (20, 20)
    assert m.nest_cells and all(m.terrain[r, c] == CELL_NEST for r, c in m.nest_cells)
    assert m.food_sources and all(m.terrain[r, c] != CELL_WALL for r, c in m.food_sources)
    assert all(m.terrain[r, c] != CELL_WALL for r, c in m.spawn_cells)


def test_unknown_map_raises():
    with pytest.raises(KeyError):
        make_map("nope", EnvConfig(), np.random.default_rng(0))


def test_open_food_respects_min_distance():
    cfg = EnvConfig(map_name="open", grid_height=24, grid_width=24,
                    n_food_sources=3, food_min_dist_from_nest=6)
    m = make_map("open", cfg, np.random.default_rng(3))
    nr, nc = m.nest_cells[0]
    assert all(max(abs(r - nr), abs(c - nc)) >= 6 for r, c in m.food_sources)
