"""Render the vendored G1 pretrained-policy walk to a compact GIF (tracking camera).

Offscreen via CGL on Apple Silicon (plain python3, not mjpython) -> PIL GIF. NVIDIA-free.

Usage: python render_g1_walk.py [out.gif] [--secs 6] [--vx 0.5] [--wz 0]
"""
import os
import argparse
import importlib.util
import numpy as np
import mujoco
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("g1_walk", os.path.join(HERE, "g1_walk.py"))
gw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gw)

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default=os.path.join(HERE, "..", "assets", "g1_walk.gif"))
ap.add_argument("--secs", type=float, default=6.0)
ap.add_argument("--vx", type=float, default=0.5)
ap.add_argument("--vy", type=float, default=0.0)
ap.add_argument("--wz", type=float, default=0.0)
ap.add_argument("--fps", type=int, default=12)
args = ap.parse_args()

m, d, policy = gw.make()
base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
if base_id < 0:
    base_id = 1
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = base_id
cam.distance, cam.azimuth, cam.elevation = 3.0, 130, -12

renderer = mujoco.Renderer(m, height=360, width=480)
frames = []
every = int(round(1.0 / (args.fps * gw.SIM_DT)))
state = {"k": 0}


def cap(dd):
    if state["k"] % every == 0:
        renderer.update_scene(dd, camera=cam)
        frames.append(Image.fromarray(renderer.render()).convert("P", palette=Image.ADAPTIVE, colors=64))
    state["k"] += 1


gw.walk(m, d, policy, [args.vx, args.vy, args.wz], int(args.secs / gw.SIM_DT), log=cap)

out = os.path.abspath(args.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000/args.fps), loop=0, optimize=True)
sz = os.path.getsize(out) / 1e6
print(f"saved {len(frames)} frames -> {out} ({sz:.1f} MB)  dx={d.qpos[0]:+.2f} z={d.qpos[2]:.3f}")
