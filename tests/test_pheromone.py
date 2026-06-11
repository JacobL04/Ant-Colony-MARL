"""Pheromone field math — the dynamics the whole stigmergy result rests on."""

from __future__ import annotations

import pytest

from antworld.pheromone import PheromoneField
from antworld.config import PH_FOOD_TRAIL, PH_EXPLORE, PH_NEST_SCENT


def test_deposit_adds_to_cell():
    f = PheromoneField(7, 7, evaporation_rate=0.0, diffusion_sigma=0.0)
    f.deposit(PH_FOOD_TRAIL, 3, 3, 2.5)
    assert f.value_at(PH_FOOD_TRAIL, 3, 3) == pytest.approx(2.5)


def test_evaporation_decays_geometrically():
    f = PheromoneField(7, 7, evaporation_rate=0.1, diffusion_sigma=0.0)
    f.deposit(PH_FOOD_TRAIL, 3, 3, 1.0)
    f.step()
    assert f.value_at(PH_FOOD_TRAIL, 3, 3) == pytest.approx(0.9)
    f.step()
    assert f.value_at(PH_FOOD_TRAIL, 3, 3) == pytest.approx(0.81)


def test_clip_caps_field():
    f = PheromoneField(7, 7, evaporation_rate=0.0, diffusion_sigma=0.0, max_value=3.0)
    f.deposit(PH_FOOD_TRAIL, 3, 3, 100.0)
    f.step()
    assert f.field.max() <= 3.0 + 1e-6


def test_step_order_clip_then_evaporate():
    # clip(50->5) -> evaporate(5*0.8=4.0) -> diffuse(noop) -> clip(noop)
    f = PheromoneField(9, 9, evaporation_rate=0.2, diffusion_sigma=0.0, max_value=5.0)
    f.deposit(PH_FOOD_TRAIL, 4, 4, 50.0)
    f.step()
    assert f.value_at(PH_FOOD_TRAIL, 4, 4) == pytest.approx(4.0)


def test_diffusion_spreads_and_loses_to_boundary():
    f = PheromoneField(9, 9, evaporation_rate=0.0, diffusion_sigma=1.0)
    f.deposit(PH_FOOD_TRAIL, 4, 4, 10.0)
    f.step()
    assert f.value_at(PH_FOOD_TRAIL, 4, 5) > 0.0
    assert f.value_at(PH_FOOD_TRAIL, 4, 4) < 10.0
    assert f.total(PH_FOOD_TRAIL) <= 10.0 + 1e-5


def test_disabled_trail_channels_stay_zero_but_nest_persists():
    f = PheromoneField(7, 7, trail_enabled=False)
    f.deposit(PH_FOOD_TRAIL, 3, 3, 5.0)
    f.deposit(PH_EXPLORE, 3, 3, 5.0)
    f.deposit(PH_NEST_SCENT, 3, 3, 5.0)
    f.step()
    assert f.total(PH_FOOD_TRAIL) == 0.0
    assert f.total(PH_EXPLORE) == 0.0
    assert f.total(PH_NEST_SCENT) > 0.0


def test_window_shape_and_normalization():
    f = PheromoneField(10, 10, evaporation_rate=0.0, diffusion_sigma=0.0, max_value=4.0)
    f.deposit(PH_FOOD_TRAIL, 0, 0, 4.0)
    w = f.window(0, 0, radius=2)
    assert w.shape == (3, 5, 5)
    assert 0.0 <= w.min() and w.max() <= 1.0
    assert w[PH_FOOD_TRAIL, 2, 2] == pytest.approx(1.0)  # center = the (0,0) cell
    assert w[PH_FOOD_TRAIL, 0, 0] == 0.0                 # padded outside grid
