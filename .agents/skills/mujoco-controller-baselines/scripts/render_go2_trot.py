"""Render the GO2 model-based trot to an animated GIF (tracking side camera).

Reuses the controller from go2_trot.py. Offscreen render via CGL on Apple Silicon
(plain python3, NOT mjpython) -> PIL GIF. NVIDIA-free.

Usage: python render_go2_trot.py [out.gif] [--secs 6] [--fps 25]
"""
import os
import sys
import argparse
import importlib.util
import mujoco
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("go2_trot", os.path.join(HERE, "go2_trot.py"))
gt = importlib.util.module_from_spec(spec)
sys.argv_backup = sys.argv
sys.argv = ["go2_trot"]
spec.loader.exec_module(gt)
sys.argv = sys.argv_backup

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default=os.path.join(HERE, "..", "assets", "go2_trot.gif"))
ap.add_argument("--scene", default="/tmp/mjm/unitree_go2/scene.xml")
ap.add_argument("--secs", type=float, default=6.0)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--freq", type=float, default=2.0)
ap.add_argument("--xamp", type=float, default=0.08)
ap.add_argument("--h0", type=float, default=0.26)
ap.add_argument("--lift", type=float, default=0.10)
ap.add_argument("--duty", type=float, default=0.5)
ap.add_argument("--kp", type=float, default=80.0)
ap.add_argument("--kd", type=float, default=3.0)
args = ap.parse_args()

m = mujoco.MjModel.from_xml_path(args.scene)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
flim = m.actuator_ctrlrange[:, 1].copy()
base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = base_id
cam.distance, cam.azimuth, cam.elevation = 1.8, 120, -12

renderer = mujoco.Renderer(m, height=360, width=480)
frames = []
render_every = int(round(1.0 / (args.fps * m.opt.timestep)))
n = int(args.secs / m.opt.timestep)

for k in range(n):
    t = k * m.opt.timestep
    g = t * args.freq
    qd = np.zeros(12)
    for i, leg in enumerate(gt.LEGS):
        x, z = gt.foot_target(g + gt.PHASE_OFFSET[leg], args.h0, args.xamp, args.lift, args.duty)
        th, ca = gt.leg_ik(x, z)
        qd[3 * i + 1] = th
        qd[3 * i + 2] = ca
    d.ctrl[:] = np.clip(args.kp * (qd - d.qpos[7:]) - args.kd * d.qvel[6:], -flim, flim)
    mujoco.mj_step(m, d)
    if k % render_every == 0:
        renderer.update_scene(d, camera=cam)
        frames.append(Image.fromarray(renderer.render()))

out = os.path.abspath(args.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
frames[0].save(out, save_all=True, append_images=frames[1:],
               duration=int(1000 / args.fps), loop=0, optimize=True)
print(f"saved {len(frames)} frames -> {out}")
print(f"forward dx={d.qpos[0]:+.3f} m, final z={d.qpos[2]:.3f}")
