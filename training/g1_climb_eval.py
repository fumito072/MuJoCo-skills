import re
import os, sys, glob, pickle
import os
os.environ["G1CLIMB_NO_RSI"] = "1"
import numpy as np
sys.path.insert(0, "training")
from g1_climb_env import G1ClimbEnv
from stable_baselines3 import PPO

ck = sorted(glob.glob("runs_climb/ppo_climb_*_steps.zip"),
            key=lambda f: int(re.search(r"_(\d+)_steps", f).group(1)))[-1]
vn = ck.replace(".zip", ".pkl").replace("ppo_climb_", "ppo_climb_vecnormalize_")
if not os.path.exists(vn):
    vn = sorted(glob.glob("runs_climb/*vecnormalize*.pkl"))[-1]
print("eval:", ck, "|", vn)
model = PPO.load(ck, device="cpu")
stats = pickle.load(open(vn, "rb"))
mean, var = stats.obs_rms.mean, stats.obs_rms.var

env = G1ClimbEnv(seed=42)
n_ok = 0
for ep in range(20):
    o, _ = env.reset()
    ret, n = 0.0, 0
    while True:
        on = np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10)
        a, _ = model.predict(on.astype(np.float32), deterministic=True)
        o, r, term, trunc, info = env.step(a)
        ret += r; n += 1
        if term or trunc:
            d = env.d
            _, plat = env._foot_contacts()
            n_ok += info['success']
            print(f"ep{ep}: steps={n} ret={ret:7.1f} success={info['success']} "
                  f"base=({d.qpos[0]:+.2f},{d.qpos[1]:+.2f},{d.qpos[2]:.3f}) "
                  f"feet_plat={plat[env.lf]}/{plat[env.rf]}")
            break
print(f"FLOOR-SPAWN SUCCESS: {n_ok}/20")
