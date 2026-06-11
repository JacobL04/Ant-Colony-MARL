"""The shared, decaying pheromone field — the only coordination medium.

Ants secrete into this field and sense it locally; there is no other channel
between them (that is what makes coordination *stigmergic*). The per-step update
order is fixed and load-bearing:

    deposit -> clip -> evaporate -> diffuse (Gaussian) -> clip

Deposits are applied during the step (as ants act) via :meth:`deposit`; the
remaining four operations run in :meth:`step`. Evaporation is what makes trails
transient — a trail only persists while ants keep reinforcing it.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from antworld.config import (
    N_PHEROMONE_CHANNELS,
    PH_FOOD_TRAIL,
    PH_EXPLORE,
    PH_NEST_SCENT,
)


class PheromoneField:
    """A multi-channel scalar field over the grid, shape ``(channels, H, W)``."""

    def __init__(
        self,
        height: int,
        width: int,
        *,
        n_channels: int = N_PHEROMONE_CHANNELS,
        evaporation_rate: float = 0.03,
        diffusion_sigma: float = 0.7,
        max_value: float = 8.0,
        trail_enabled: bool = True,
        nest_scent_enabled: bool = True,
    ) -> None:
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.evaporation_rate = float(evaporation_rate)
        self.diffusion_sigma = float(diffusion_sigma)
        self.max_value = float(max_value)

        # Per-channel enable mask (ablation support). Disabled channels never
        # receive deposits and are held identically zero.
        self.enabled = np.ones(n_channels, dtype=bool)
        if not trail_enabled:
            self.enabled[PH_FOOD_TRAIL] = False
            self.enabled[PH_EXPLORE] = False
        if not nest_scent_enabled:
            self.enabled[PH_NEST_SCENT] = False

        self.field = np.zeros((n_channels, height, width), dtype=np.float32)

    # --- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self.field.fill(0.0)

    # --- the ordered operations ---------------------------------------------
    def deposit(self, channel: int, row: int, col: int, amount: float) -> None:
        """Add pheromone to one cell (step op #1). No-op on disabled channels."""
        if amount == 0.0 or not self.enabled[channel]:
            return
        self.field[channel, row, col] += amount

    def clip(self) -> None:
        np.clip(self.field, 0.0, self.max_value, out=self.field)

    def evaporate(self) -> None:
        if self.evaporation_rate > 0.0:
            self.field *= (1.0 - self.evaporation_rate)

    def diffuse(self) -> None:
        """Gaussian blur each channel; open boundary (scent dissipates at edges)."""
        if self.diffusion_sigma <= 0.0:
            return
        for c in range(self.n_channels):
            if self.enabled[c]:
                self.field[c] = gaussian_filter(
                    self.field[c], sigma=self.diffusion_sigma, mode="constant", cval=0.0
                )

    def step(self) -> None:
        """Post-deposit update: clip -> evaporate -> diffuse -> clip."""
        self.clip()
        self.evaporate()
        self.diffuse()
        self.clip()

    # --- reading -------------------------------------------------------------
    def window(self, row: int, col: int, radius: int) -> np.ndarray:
        """Egocentric ``(channels, 2r+1, 2r+1)`` window, zero-padded, normalized to [0,1]."""
        size = 2 * radius + 1
        out = np.zeros((self.n_channels, size, size), dtype=np.float32)
        r0, c0 = row - radius, col - radius
        sr0, sr1 = max(r0, 0), min(row + radius + 1, self.height)
        sc0, sc1 = max(c0, 0), min(col + radius + 1, self.width)
        out[:, sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0] = self.field[:, sr0:sr1, sc0:sc1]
        if self.max_value > 0:
            out /= self.max_value
        return out

    def value_at(self, channel: int, row: int, col: int) -> float:
        return float(self.field[channel, row, col])

    def total(self, channel: int | None = None) -> float:
        return float(self.field.sum() if channel is None else self.field[channel].sum())

    def coverage(self, channel: int, threshold: float = 0.05) -> float:
        """Fraction of cells with pheromone above ``threshold`` (a heatmap stat)."""
        return float((self.field[channel] > threshold).mean())

    def normalized(self) -> np.ndarray:
        """Whole field scaled to [0,1] by ``max_value`` — for the heatmap renderer."""
        if self.max_value <= 0:
            return self.field.copy()
        return np.clip(self.field / self.max_value, 0.0, 1.0)
