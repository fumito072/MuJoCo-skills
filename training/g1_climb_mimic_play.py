"""Render the mimic-env climb policy to a GIF. Starts mid-climb (default phase 0.45,
just before the lunge — the earliest point the policy reliably climbs from; the
floor->first-foot-up part, phases 0-0.40, is the unsolved single-leg-balance skill)
and rolls out to standing on the platform.

    .venv-rl/bin/python training/g1_climb_mimic_play.py            # phase 0.45
    START=0.55 .venv-rl/bin/python training/g1_climb_mimic_play.py # reliable start
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "training")
import mujoco  # noqa: E402
from PIL import Image  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from g1_climb_mimic_env import (  # noqa: E402
    G1ClimbMimicEnv, REF_BASE, REF_LEGS, REF_N, YAW_REF)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIF = os.path.join(REPO, "assets", "g1_climb_mimic.gif")
START = float(os.environ.get("START", "0.45"))
FPS = 20

model = PPO.load("runs_climb/ppo_climb_latest.zip", device="cpu")
st = pickle.load(open("runs_climb/ppo_climb_latest_vecnorm.pkl", "rb"))
mean, var = st.obs_rms.mean, st.obs_rms.var

env = G1ClimbMimicEnv(seed=7)
env.reset()
env._k = int(START * (REF_N - 1))
rl, rb = REF_LEGS[env._k], REF_BASE[env._k]
env.d.qpos[:] = env.m.key_qpos[env.key]
env.d.qpos[0:3] = (0.0, rb[0], rb[1])
env.d.qpos[3:7] = (np.cos(YAW_REF / 2), 0, 0, np.sin(YAW_REF / 2))
env.d.qpos[7:19] = rl
env.d.qvel[:] = 0
env.d.ctrl[:] = env.default_pose
env.d.ctrl[:12] = rl
mujoco.mj_forward(env.m, env.d)
env._last_a[:] = 0
o = env._obs()

renderer = mujoco.Renderer(env.m, height=480, width=640)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = env.m.body("pelvis").id
cam.distance, cam.azimuth, cam.elevation = 2.6, 125, -10
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

_, plat = env._foot_contacts()
os.makedirs(os.path.dirname(GIF), exist_ok=True)
frames[0].save(GIF, save_all=True, append_images=frames[1:],
               duration=int(1000 / FPS), loop=0, optimize=True)
print(f"start_phase={START} steps={k} success={int(info['success'])} "
      f"final_z={env.d.qpos[2]:.3f} peak_z={peak:.3f} "
      f"feet_plat={int(plat[env.lf])}/{int(plat[env.rf])} -> {GIF} ({len(frames)} frames)")
