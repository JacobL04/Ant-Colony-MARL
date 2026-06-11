"""AntWorld — a multi-agent RL study of collective foraging via stigmergy.

Ant-like agents forage cooperatively using only *local* perception and a
*shared, decaying pheromone field*. Pheromone is secreted automatically (as in
real ants); the only thing an agent decides is which way to move. The question:
does coordination through the shared field (stigmergy) help the colony forage,
and does it emerge from learning?

Importing this package does NOT import pygame — rendering is pulled in lazily by
``AntWorldEnv.render`` / ``antworld.renderer`` only when actually needed.
"""

from antworld.config import (
    EnvConfig, OBS_VERSION, OBS_DIM, N_ACTIONS, PHEROMONE_NAMES,
    PH_FOOD_TRAIL, PH_EXPLORE, PH_NEST_SCENT,
)
from antworld.pheromone import PheromoneField
from antworld.ant import Ant
from antworld.maps import make_map, MapLayout, CURRICULUM
from antworld.env import AntWorldEnv, MOVES
from antworld.scripted import ScriptedForager

__all__ = [
    "EnvConfig", "OBS_VERSION", "OBS_DIM", "N_ACTIONS", "PHEROMONE_NAMES",
    "PH_FOOD_TRAIL", "PH_EXPLORE", "PH_NEST_SCENT",
    "PheromoneField", "Ant", "make_map", "MapLayout", "CURRICULUM",
    "AntWorldEnv", "MOVES", "ScriptedForager",
]
