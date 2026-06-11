"""Interactive real-time viewer for AntWorld.

Watch a colony forage and lay pheromone live, toggle the heatmap layers, and
tweak the field dynamics (evaporation / diffusion) on the fly to *see* how trail
formation responds. Driven by the scripted forager (no training required).

    python -m antworld.viewer
    python -m antworld.viewer --map t_shape --agents 60 --evaporation 0.02

Controls (also shown in the HUD):
    1 / 2 / 3   toggle food-trail / exploration / nest-scent heat layers
    T / G / A / H   toggle terrain / grid / ants / HUD
    ↑ / ↓        raise / lower evaporation rate (live)
    ← / →        lower / raise diffusion sigma (live)
    + / -        faster / slower simulation (sim-steps per frame)
    Space        pause/resume       S or .   single step (while paused)
    R            reset (new layout)  Esc / Q  quit
"""

from __future__ import annotations

import argparse

from antworld import AntWorldEnv, EnvConfig
from antworld.renderer import Renderer
from antworld.scripted import ScriptedForager


def run(cfg: EnvConfig, seed: int = 0, cell_size: int = 22, fps: int = 30,
        speed: int = 2) -> None:
    import pygame

    env = AntWorldEnv(cfg)
    policy = ScriptedForager(seed=seed)
    env.reset(seed=seed)
    policy.reset(seed)

    renderer = Renderer(env, mode="human", cell_size=cell_size)
    renderer.handle_quit = False  # the viewer owns the event queue
    clock = pygame.time.Clock()

    running, paused, ep_seed = True, False, seed

    def sim_step():
        if env.agents:
            env.step(policy.act(env))

    def reset():
        nonlocal ep_seed
        ep_seed += 1
        env.reset(seed=ep_seed)
        policy.reset(ep_seed)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_1:
                    renderer.layer_on[0] = not renderer.layer_on[0]
                elif k == pygame.K_2:
                    renderer.layer_on[1] = not renderer.layer_on[1]
                elif k == pygame.K_3:
                    renderer.layer_on[2] = not renderer.layer_on[2]
                elif k == pygame.K_t:
                    renderer.show_terrain = not renderer.show_terrain
                elif k == pygame.K_g:
                    renderer.show_grid = not renderer.show_grid
                elif k == pygame.K_a:
                    renderer.show_ants = not renderer.show_ants
                elif k == pygame.K_h:
                    renderer.show_hud = not renderer.show_hud
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k in (pygame.K_s, pygame.K_PERIOD):
                    sim_step()
                elif k == pygame.K_r:
                    reset()
                elif k == pygame.K_UP:
                    env.field.evaporation_rate = min(0.5, env.field.evaporation_rate + 0.005)
                elif k == pygame.K_DOWN:
                    env.field.evaporation_rate = max(0.0, env.field.evaporation_rate - 0.005)
                elif k == pygame.K_RIGHT:
                    env.field.diffusion_sigma = min(3.0, env.field.diffusion_sigma + 0.1)
                elif k == pygame.K_LEFT:
                    env.field.diffusion_sigma = max(0.0, env.field.diffusion_sigma - 0.1)
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(32, speed + 1)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(1, speed - 1)

        if not paused:
            for _ in range(speed):
                sim_step()
            if not env.agents:  # episode ended (only if not endless/respawn)
                reset()

        renderer.status = f"speed x{speed}" + ("  [PAUSED]" if paused else "")
        renderer.render()
        clock.tick(fps)

    renderer.close()


def build_config(args) -> EnvConfig:
    return EnvConfig(
        map_name=args.map,
        grid_height=args.grid,
        grid_width=args.grid,
        n_agents=args.agents,
        max_steps=args.steps,
        evaporation_rate=args.evaporation,
        diffusion_sigma=args.diffusion,
        food_respawn=not args.no_respawn,  # endless by default (good for a demo)
        n_food_sources=args.food_sources,
        seed=args.seed,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive AntWorld viewer.")
    p.add_argument("--map", default="open", help="open|obstacles|t_shape|multi_food")
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--agents", type=int, default=40)
    p.add_argument("--steps", type=int, default=10_000_000, help="max steps (huge => endless)")
    p.add_argument("--food-sources", type=int, default=5)
    p.add_argument("--evaporation", type=float, default=0.03)
    p.add_argument("--diffusion", type=float, default=0.7)
    p.add_argument("--no-respawn", action="store_true", help="episodic (food does not refill)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cell-size", type=int, default=22)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--speed", type=int, default=2, help="sim steps per rendered frame")
    args = p.parse_args()
    run(build_config(args), seed=args.seed, cell_size=args.cell_size, fps=args.fps, speed=args.speed)


if __name__ == "__main__":
    main()
