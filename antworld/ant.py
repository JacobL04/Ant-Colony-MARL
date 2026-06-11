"""Per-agent ("ant") state.

Pure data. The env owns a list of these and mutates them during step(). The
population is homogeneous — no caste/role is stored here; any specialization
must emerge in behaviour, not be hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Ant:
    agent_id: int
    row: int
    col: int

    carrying: bool = False          # True after auto-pickup, until auto-deliver
    hunger: float = 0.0             # steps since last delivery (obs feature only)
    prev_dist_to_nest: float = 0.0  # for the return-while-carrying shaping reward
    last_action: int = -1
    bumped: bool = False            # last move was blocked by a wall/boundary

    @property
    def pos(self) -> tuple[int, int]:
        return (self.row, self.col)
