"""Curriculum map layouts: open -> obstacles -> t_shape -> multi_food.

A map is static terrain (floor / wall / nest) plus the initial food-source
cells. Food *amounts* are dynamic and owned by the env. All placement
randomness is driven by the env's seeded RNG so runs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CELL_FLOOR: int = 0
CELL_WALL: int = 1
CELL_NEST: int = 2


@dataclass
class MapLayout:
    name: str
    terrain: np.ndarray              # (H, W) int8 of CELL_* codes
    nest_cells: list[tuple[int, int]]
    food_sources: list[tuple[int, int]]
    spawn_cells: list[tuple[int, int]]

    @property
    def height(self) -> int:
        return int(self.terrain.shape[0])

    @property
    def width(self) -> int:
        return int(self.terrain.shape[1])


def _neighbours(r, c, h, w):
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w
    ]


def _sample_food(terrain, nest, *, n_sources, min_dist, rng):
    h, w = terrain.shape
    nr, nc = nest
    cand = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if terrain[r, c] == CELL_FLOOR and max(abs(r - nr), abs(c - nc)) >= min_dist
    ]
    if not cand:
        cand = [(r, c) for r in range(h) for c in range(w) if terrain[r, c] == CELL_FLOOR]
    n = min(n_sources, len(cand))
    idx = rng.choice(len(cand), size=n, replace=False)
    return [cand[i] for i in idx]


def _spawn_around_nest(terrain, nest_cells):
    h, w = terrain.shape
    cells = list(nest_cells)
    for (r, c) in nest_cells:
        for (nr, nc) in _neighbours(r, c, h, w):
            if terrain[nr, nc] != CELL_WALL and (nr, nc) not in cells:
                cells.append((nr, nc))
    return cells


def _open(cfg, rng):
    h, w = cfg.grid_height, cfg.grid_width
    terrain = np.full((h, w), CELL_FLOOR, dtype=np.int8)
    nest = (h // 2, w // 2)
    terrain[nest] = CELL_NEST
    food = _sample_food(terrain, nest, n_sources=cfg.n_food_sources,
                        min_dist=cfg.food_min_dist_from_nest, rng=rng)
    return MapLayout("open", terrain, [nest], food, _spawn_around_nest(terrain, [nest]))


def _obstacles(cfg, rng):
    h, w = cfg.grid_height, cfg.grid_width
    terrain = np.full((h, w), CELL_FLOOR, dtype=np.int8)
    nest = (h // 2, w // 2)
    for _ in range(max(3, (h * w) // 110)):
        bh, bw = int(rng.integers(1, 4)), int(rng.integers(1, 5))
        r, c = int(rng.integers(0, h - bh)), int(rng.integers(0, w - bw))
        if abs(r - nest[0]) <= 2 and abs(c - nest[1]) <= 2:
            continue
        sub = terrain[r:r + bh, c:c + bw]
        sub[sub == CELL_FLOOR] = CELL_WALL
    terrain[nest] = CELL_NEST
    for (nr, nc) in _neighbours(*nest, h, w):
        terrain[nr, nc] = CELL_FLOOR
    food = _sample_food(terrain, nest, n_sources=cfg.n_food_sources,
                        min_dist=cfg.food_min_dist_from_nest, rng=rng)
    return MapLayout("obstacles", terrain, [nest], food, _spawn_around_nest(terrain, [nest]))


def _t_shape(cfg, rng):
    """T-maze: a vertical stem rising into a horizontal bar; nest at the foot,
    food at both bar ends — a binary recruitment choice."""
    h, w = cfg.grid_height, cfg.grid_width
    terrain = np.full((h, w), CELL_WALL, dtype=np.int8)
    cc = w // 2
    half = 1
    bar_row = max(2, h // 5)
    terrain[bar_row:h - 1, cc - half:cc + half + 1] = CELL_FLOOR
    terrain[bar_row - half:bar_row + half + 1, 1:w - 1] = CELL_FLOOR
    nest = (h - 2, cc)
    terrain[nest] = CELL_NEST
    food = [(bar_row, 1), (bar_row, w - 2)]
    return MapLayout("t_shape", terrain, [nest], food, _spawn_around_nest(terrain, [nest]))


def _multi_food(cfg, rng):
    h, w = cfg.grid_height, cfg.grid_width
    terrain = np.full((h, w), CELL_FLOOR, dtype=np.int8)
    nest = (h // 2, w // 2)
    terrain[nest] = CELL_NEST
    food = _sample_food(terrain, nest, n_sources=max(4, cfg.n_food_sources + 2),
                        min_dist=max(3, cfg.food_min_dist_from_nest - 2), rng=rng)
    return MapLayout("multi_food", terrain, [nest], food, _spawn_around_nest(terrain, [nest]))


_BUILDERS = {"open": _open, "obstacles": _obstacles, "t_shape": _t_shape, "multi_food": _multi_food}
CURRICULUM: list[str] = ["open", "obstacles", "t_shape", "multi_food"]


def make_map(name: str, cfg, rng: np.random.Generator) -> MapLayout:
    if name not in _BUILDERS:
        raise KeyError(f"unknown map '{name}'. Valid: {sorted(_BUILDERS)}")
    return _BUILDERS[name](cfg, rng)
