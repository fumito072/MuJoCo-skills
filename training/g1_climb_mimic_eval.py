"""Eval the mimic-env climb policy from a deterministic FLOOR start (reference
frame 0, no RSI, no noise) — the true full floor->platform climb test. The
training reset spawns 50% mid-reference (RSI); here we force frame 0 every episode
so success means the policy climbed the WHOLE chain from the floor.

Reports peak base_z reached per episode (so a near-miss that gets one foot up at
z~0.87 is distinguishable from a full stand at z~0.97) and feet-on-platform.
"""
import glob
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, "training")
import mujoco  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from g1_climb_mimic_env import (  # noqa: E402
    G1ClimbMimicEnv, REF_BASE, REF_LEGS, REF_N, YAW_REF)

# pick the NEWEST run's final policy by mtime (step-number sort grabbed a stale
# 116M-step checkpoint from a prior run). CK env var overrides.
ck = os.environ.get("CK", "runs_climb/ppo_climb_latest.zip")
vn = ck.replace("ppo_climb_latest.zip", "ppo_climb_latest_vecnorm.pkl")
if not ck.endswith("latest.zip"):
    vn = ck.replace(".zip", ".pkl").replace("ppo_climb_", "ppo_climb_vecnormalize_")
if not os.path.exists(vn):
    vn = sorted(glob.glob("runs_climb/*vecnorm*.pkl"), key=os.path.getmtime)[-1]
print("eval:", ck, "|", vn)
model = PPO.load(ck, device="cpu")
stats = pickle.load(open(vn, "rb"))
mean, var = stats.obs_rms.mean, stats.obs_rms.var

env = G1ClimbMimicEnv(seed=42)
pelvis = env.m.body("pelvis").id
n_ok = 0
peaks = []
for ep in range(20):
    env.reset()
    # FORCE deterministic floor start = reference frame 0, no noise
    env._k = 0
    rl, rb = REF_LEGS[0], REF_BASE[0]
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
    peak = env.d.qpos[2]
    while True:
        on = np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10)
        a, _ = model.predict(on.astype(np.float32), deterministic=True)
        o, r, term, trunc, info = env.step(a)
        peak = max(peak, env.d.qpos[2])
        if term or trunc:
            d = env.d
            _, plat = env._foot_contacts()
            n_ok += info["success"]
            peaks.append(peak)
            print(f"ep{ep:2d}: steps={env._k:3d} success={int(info['success'])} "
                  f"final_z={d.qpos[2]:.3f} peak_z={peak:.3f} "
                  f"feet_plat={int(plat[env.lf])}/{int(plat[env.rf])} "
                  f"base=({d.qpos[0]:+.2f},{d.qpos[1]:+.2f})")
            break
print(f"\nFLOOR-START FULL CLIMB SUCCESS: {n_ok}/20  | peak_z mean={np.mean(peaks):.3f} "
      f"max={np.max(peaks):.3f}  (stand≈0.97, one-foot-stuck≈0.87)")
