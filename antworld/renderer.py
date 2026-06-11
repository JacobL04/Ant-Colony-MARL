"""Pygame rendering for AntWorld — terrain, ants, and pheromone heatmaps.

Strictly read-only: it draws env state and never mutates it. Pygame is imported
here and only here; the env imports this lazily so training/tests never need it.

The pheromone heatmap is the star. Each channel is tinted a distinct colour and
the enabled channels are *additively blended* over a dark floor, so:
  * a single channel reads as an intensity heatmap (brighter = more pheromone),
  * overlapping channels mix (green food-trail + blue explore = cyan; all three
    = white-hot) — you can literally see where trails overlap and pile up.

Display toggles (which layers are shown, terrain/ants/grid/HUD) live here; the
interactive viewer just flips these booleans.

Modes: "human" opens a window (returns None); "rgb_array" draws offscreen and
returns an (H, W, 3) uint8 frame for video.
"""

from __future__ import annotations

import os

import numpy as np

from antworld.config import (
    PHEROMONE_NAMES, PH_FOOD_TRAIL, PH_EXPLORE, PH_NEST_SCENT, N_PHEROMONE_CHANNELS,
)
from antworld.maps import CELL_WALL, CELL_NEST, CELL_FLOOR

# Dark theme so neon pheromone glows (reads well on screen recordings).
_BG = (16, 18, 24)
_FLOOR = (30, 32, 40)
_WALL = (66, 70, 84)
_NEST = (122, 92, 56)
_FOOD = (80, 230, 120)
_GRID = (44, 47, 56)
_ANT_SEARCH = (205, 210, 220)
_ANT_CARRY = (255, 205, 70)
_HUD_BG = (10, 11, 15)
_HUD_FG = (225, 228, 235)

# Per-channel heat colours: food-trail green, exploration blue, nest-scent amber.
_LAYER_COLORS = ((40, 235, 130), (60, 130, 255), (255, 150, 40))


class Renderer:
    def __init__(self, env, mode: str = "human", cell_size: int = 22,
                 heat_gain: float = 1.7):
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"unknown render mode '{mode}'")
        if mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        import pygame
        self.pygame = pygame
        self.env = env
        self.mode = mode
        self.cell = cell_size
        self.heat_gain = heat_gain

        # display toggles (flipped by the viewer)
        self.layer_on = [True] * N_PHEROMONE_CHANNELS
        self.show_terrain = True
        self.show_ants = True
        self.show_food = True
        self.show_grid = False
        self.show_hud = True
        self.status = ""  # extra HUD line the viewer can set (speed, paused, ...)
        self.handle_quit = True  # viewer sets False when it owns the event queue

        pygame.init()
        pygame.font.init()
        self.px_w = env.width * cell_size
        self.px_h = env.height * cell_size
        self.hud_h = 92 if mode == "human" else 92

        if mode == "human":
            self.surface = pygame.display.set_mode((self.px_w, self.px_h + self.hud_h))
            pygame.display.set_caption("AntWorld — pheromone heatmap")
        else:
            self.surface = pygame.Surface((self.px_w, self.px_h + self.hud_h))

        self.font = pygame.font.SysFont("consolas,menlo,monospace", 15)
        self.small = pygame.font.SysFont("consolas,menlo,monospace", 13)
        self._terrain_surf = None
        self._terrain_token = None

    # --- terrain cache (static within an episode) ---------------------------
    def _build_terrain(self):
        pygame = self.pygame
        surf = pygame.Surface((self.px_w, self.px_h))
        surf.fill(_FLOOR)
        cell = self.cell
        for r in range(self.env.height):
            for c in range(self.env.width):
                code = self.env.terrain[r, c]
                if code == CELL_WALL:
                    color = _WALL
                elif code == CELL_NEST:
                    color = _NEST
                else:
                    continue
                pygame.draw.rect(surf, color, (c * cell, r * cell, cell, cell))
        return surf

    # --- heatmap ------------------------------------------------------------
    def _heat_surface(self):
        pygame = self.pygame
        norm = self.env.field.normalized()  # (C, H, W) in [0,1]
        rgb = np.zeros((self.env.height, self.env.width, 3), dtype=np.float32)
        for c in range(N_PHEROMONE_CHANNELS):
            if not self.layer_on[c]:
                continue
            inten = np.clip(norm[c] * self.heat_gain, 0.0, 1.0)
            rgb += inten[..., None] * np.asarray(_LAYER_COLORS[c], dtype=np.float32)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        small = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))  # (W,H,3)
        return pygame.transform.smoothscale(small, (self.px_w, self.px_h))

    # --- main draw ----------------------------------------------------------
    def render(self):
        pygame = self.pygame
        env = self.env
        cell = self.cell

        if self.mode == "human" and self.handle_quit:
            for event in pygame.event.get(pygame.QUIT):
                self.close()
                return None

        self.surface.fill(_BG)

        # terrain (cached)
        token = id(env.terrain)
        if self._terrain_surf is None or self._terrain_token != token:
            self._terrain_surf = self._build_terrain()
            self._terrain_token = token
        if self.show_terrain:
            self.surface.blit(self._terrain_surf, (0, 0))

        # pheromone heatmap, additively blended over terrain
        if any(self.layer_on) and env.field is not None:
            self.surface.blit(self._heat_surface(), (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        if self.show_grid:
            for x in range(0, self.px_w + 1, cell):
                pygame.draw.line(self.surface, _GRID, (x, 0), (x, self.px_h))
            for y in range(0, self.px_h + 1, cell):
                pygame.draw.line(self.surface, _GRID, (0, y), (self.px_w, y))

        if self.show_food:
            max_food = max(int(env.food.max()), 1)
            ys, xs = np.nonzero(env.food)
            for r, c in zip(ys, xs):
                frac = env.food[r, c] / max_food
                rad = max(2, int(0.42 * cell * (0.45 + 0.55 * frac)))
                pygame.draw.circle(self.surface, _FOOD,
                                   (c * cell + cell // 2, r * cell + cell // 2), rad)

        if self.show_ants:
            rad = max(2, cell // 3)
            for ant in env.ants:
                color = _ANT_CARRY if ant.carrying else _ANT_SEARCH
                pygame.draw.circle(self.surface, color,
                                   (ant.col * cell + cell // 2, ant.row * cell + cell // 2), rad)

        if self.show_hud:
            self._draw_hud()

        if self.mode == "human":
            pygame.display.flip()
            return None
        arr = pygame.surfarray.array3d(self.surface)
        return np.transpose(arr, (1, 0, 2)).copy()

    def _draw_hud(self):
        pygame = self.pygame
        env = self.env
        y0 = self.px_h
        pygame.draw.rect(self.surface, _HUD_BG, (0, y0, self.px_w, self.hud_h))

        delivered = env.food_delivered
        total = env.total_food
        frac = (delivered / total) if total else 0.0
        line1 = (f"map {env.cfg.map_name:9s}  step {env.t:>4}/{env.cfg.max_steps}  "
                 f"food {delivered}/{total} ({frac:4.0%})  carrying {env.n_carrying}/{len(env.ants)}")
        line2 = (f"trail-util {env.trail_utilization:4.2f}   "
                 f"coverage  food {env.field.coverage(PH_FOOD_TRAIL):3.0%}  "
                 f"explore {env.field.coverage(PH_EXPLORE):3.0%}   "
                 f"evap {env.field.evaporation_rate:.3f}  diff {env.field.diffusion_sigma:.2f}")
        self.surface.blit(self.font.render(line1, True, _HUD_FG), (8, y0 + 6))
        self.surface.blit(self.font.render(line2, True, _HUD_FG), (8, y0 + 26))

        # colour-coded layer toggles
        x = 8
        for c in range(N_PHEROMONE_CHANNELS):
            col = _LAYER_COLORS[c] if self.layer_on[c] else (90, 90, 96)
            label = f"[{c+1}] {PHEROMONE_NAMES[c]}{'•' if self.layer_on[c] else '·'}"
            surf = self.small.render(label, True, col)
            self.surface.blit(surf, (x, y0 + 48))
            x += surf.get_width() + 14

        if self.status:
            s = self.small.render(self.status, True, (255, 220, 120))
            self.surface.blit(s, (x + 6, y0 + 48))

        controls = ("1/2/3 layers  T terrain  G grid  A ants  H hud  "
                    "↑↓ evap  ←→ diff  +/- speed  Space pause  S step  R reset  Esc quit")
        self.surface.blit(self.small.render(controls, True, (140, 145, 155)), (8, y0 + 70))

    def close(self):
        try:
            self.pygame.quit()
        except Exception:
            pass
