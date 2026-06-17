"""Render the hand-brace climb policy to a GIF (Mac offscreen). Starts at phase 0.42
(the reliable deterministic frontier — hands still braced on the armrests, R foot on
the footrest) and climbs to standing on the platform. The bridge-start first-foot-up
(phase 0-0.30) does not solidify on CPU, so this shows the part that DOES work: the
4-point braced rise + hand release.

    START=0.42 .venv-rl/bin/python training/g1_climb_brace_play.py
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "training")
import mujoco  # noqa: E402
from PIL import Image  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from g1_climb_brace_env import (  # noqa: E402
    G1ClimbBraceEnv, REF_J, REF_BASE, REF_QUAT, REF_N)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIF = os.path.join(REPO, "assets", "g1_climb_brace.gif")
START = float(os.environ.get("START", "0.42"))
FPS = 20

model = PPO.load("runs_climb/ppo_climb_latest.zip", device="cpu")
st = pickle.load(open("runs_climb/ppo_climb_latest_vecnorm.pkl", "rb"))
mean, var = st.obs_rms.mean, st.obs_rms.var

env = G1ClimbBraceEnv(seed=3)
env.reset()
env._k = int(START * (REF_N - 1))
rj, rb, rq = REF_J[env._k], REF_BASE[env._k], REF_QUAT[env._k]
env.d.qpos[0:3] = (0.0, rb[0], rb[1])
env.d.qpos[3:7] = rq
env.d.qpos[7:] = rj
env.d.qvel[:] = 0
env.d.ctrl[:] = rj
mujoco.mj_forward(env.m, env.d)
env._last_a[:] = 0
o = env._obs()

renderer = mujoco.Renderer(env.m, height=480, width=640)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = env.m.body("pelvis").id
cam.distance, cam.azimuth, cam.elevation = 2.7, 130, -8
frame_every = max(1, int(round(1.0 / (FPS * 0.02))))

frames = []
peak = env.d.qpos[2]
k = 0
while True:
    on = np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10)
    a, _ = model.predict(on.astype(np.float32), deterministic=True)
    o, r, term, trunc, info = env.step(a)
    peak = max(peak, env.d.qpos[2])
    k += 1
    if k % frame_every == 1:
        renderer.update_scene(env.d, camera=cam)
        frames.append(Image.fromarray(renderer.render())
                      .convert("P", palette=Image.ADAPTIVE, colors=128))
    if term or trunc:
        break

os.makedirs(os.path.dirname(GIF), exist_ok=True)
frames[0].save(GIF, save_all=True, append_images=frames[1:],
               duration=int(1000 / FPS), loop=0, optimize=True)
print(f"start={START} steps={k} success={int(info['success'])} "
      f"final_z={env.d.qpos[2]:.3f} peak_z={peak:.3f} -> {GIF} ({len(frames)} frames)")
