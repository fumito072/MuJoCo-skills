"""Offscreen RGB / depth / segmentation rendering for a robot — plain python3, NOT mjpython.

On macOS this uses the built-in OpenGL renderer via CGL (no EGL/OSMesa, no NVIDIA). Run with
plain `python3` (NOT mjpython — combining offscreen render with the mjpython viewer crashes,
MuJoCo issue #798). Saves a PNG (and optionally a depth map). Depth via the OpenGL Z-buffer is
usable for RL/eval but precision-limited on macOS (ARB_clip_control unavailable under CGL).

Usage: python render.py [unitree_go2|unitree_g1] [--out frame.png] [--depth]
"""
import os
import sys
import argparse
import mujoco
import numpy as np
from PIL import Image

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "models")

ap = argparse.ArgumentParser()
ap.add_argument("robot", nargs="?", default="unitree_go2")
ap.add_argument("--out", default="frame.png")
ap.add_argument("--depth", action="store_true")
ap.add_argument("--seg", action="store_true")
ap.add_argument("--w", type=int, default=480)
ap.add_argument("--h", type=int, default=360)
args = ap.parse_args()

scene = os.path.join(_MODELS, args.robot, "scene.xml")
if not os.path.exists(scene):
    sys.exit(f"model not found: {scene}\nRun the mujoco-env-setup skill first.")

m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m)
if m.nkey:
    mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

r = mujoco.Renderer(m, args.h, args.w)
r.update_scene(d)
rgb = r.render()
Image.fromarray(rgb).save(args.out)
print(f"RGB  {rgb.shape} -> {os.path.abspath(args.out)}")

if args.depth:
    r.enable_depth_rendering()
    r.update_scene(d)
    dep = r.render()
    r.disable_depth_rendering()
    dn = (255 * (dep - dep.min()) / (np.ptp(dep) + 1e-9)).astype(np.uint8)
    Image.fromarray(dn).save(args.out.replace(".png", "_depth.png"))
    print(f"DEPTH {dep.shape} min={dep.min():.3f} max={dep.max():.3f} "
          f"(macOS CGL depth precision is limited)")

if args.seg:
    r.enable_segmentation_rendering()
    r.update_scene(d)
    seg = r.render()[:, :, 0]
    r.disable_segmentation_rendering()
    print(f"SEG   {seg.shape} ids {sorted(set(np.unique(seg)) - {-1})[:8]}...")
