"""Streamlit demo for AntWorld — a shareable, in-browser pheromone viewer.

This complements the real-time pygame viewer (`python -m antworld.viewer`). It
runs an episode with the scripted forager, lets you toggle the pheromone heat
layers and tweak the field dynamics, and scrub through the resulting frames.

    streamlit run demo/app.py
"""

from __future__ import annotations

import streamlit as st

from antworld import AntWorldEnv, EnvConfig, ScriptedForager, CURRICULUM, PHEROMONE_NAMES
from antworld.renderer import Renderer

st.set_page_config(page_title="AntWorld — stigmergy viewer", layout="wide")
st.title("🐜 AntWorld — collective foraging via stigmergy")
st.caption(
    "Ants secrete pheromone automatically (like real ants) and coordinate only "
    "through the shared, decaying field. Watch recruitment trails (green) form, "
    "the exploration haze (blue) spread, and the nest scent (amber) pool at home."
)

sb = st.sidebar
sb.header("World")
map_name = sb.selectbox("Map", CURRICULUM, index=0)
grid = sb.slider("Grid size", 16, 40, 28, step=2)
agents = sb.slider("Colony size", 4, 120, 48, step=4)
food_sources = sb.slider("Food sources", 1, 10, 5)
seed = int(sb.number_input("Seed", value=0, step=1))

sb.header("Pheromone dynamics")
evaporation = sb.slider("Evaporation rate", 0.0, 0.30, 0.03, step=0.005)
diffusion = sb.slider("Diffusion sigma", 0.0, 2.0, 0.7, step=0.1)

sb.header("Stigmergy ablation")
trail_enabled = sb.checkbox(
    "Agent trails ON (stigmergy)", value=True,
    help="Turn OFF for the control condition: ants can't lay/sense recruitment "
         "or exploration trails. Compare foraging with vs without.",
)

sb.header("Heat layers")
layer_on = tuple(sb.checkbox(name, value=True) for name in PHEROMONE_NAMES)

sb.header("Run")
steps = sb.slider("Steps to simulate", 100, 1500, 600, step=50)
run = sb.button("▶ Run simulation", type="primary")

cache_key = (map_name, grid, agents, food_sources, seed, evaporation,
             diffusion, trail_enabled, steps, layer_on)


@st.cache_data(show_spinner="Simulating colony…")
def simulate(key):
    (map_name, grid, agents, food_sources, seed, evaporation, diffusion,
     trail_enabled, steps, layer_on) = key
    cfg = EnvConfig(
        map_name=map_name, grid_height=grid, grid_width=grid, n_agents=agents,
        max_steps=steps + 1, n_food_sources=food_sources,
        evaporation_rate=evaporation, diffusion_sigma=diffusion,
        trail_enabled=trail_enabled, food_respawn=True, seed=seed,
    )
    env = AntWorldEnv(cfg)
    policy = ScriptedForager(seed=seed)
    env.reset(seed=seed)
    r = Renderer(env, mode="rgb_array", cell_size=16)
    r.show_hud = False
    r.layer_on = list(layer_on)

    frames, deliveries, utils = [], [], []
    record_every = max(1, steps // 200)
    for t in range(steps):
        env.step(policy.act(env))
        if t % record_every == 0:
            frames.append(r.render())
            deliveries.append(env.food_delivered)
            utils.append(env.trail_utilization)
    r.close()
    return frames, deliveries, utils, env.food_delivered, env.trail_utilization, record_every


if run or st.session_state.get("key") == cache_key:
    frames, deliveries, utils, final_food, final_util, every = simulate(cache_key)
    st.session_state["key"] = cache_key

    c1, c2, c3 = st.columns(3)
    c1.metric("Food delivered", final_food)
    c2.metric(
        "Trail-utilization", f"{final_util:.2f}",
        help="Fraction of searching moves that climb an existing food trail — "
             "the key evidence of stigmergic coordination.",
    )
    c3.metric("Colony size", agents)

    idx = st.slider("Frame", 0, len(frames) - 1, len(frames) - 1)
    st.image(frames[idx], caption=f"step ~{idx * every}", use_container_width=True)
    st.line_chart({"food delivered": deliveries, "trail-utilization": utils})
else:
    st.info("Set the colony up in the sidebar and press **Run simulation**. "
            "Try the same settings with **Agent trails** ON vs OFF to see whether "
            "stigmergy helps the colony forage.")
