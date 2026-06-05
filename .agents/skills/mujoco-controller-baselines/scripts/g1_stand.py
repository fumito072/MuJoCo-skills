"""Unitree G1 humanoid standing: hold the 'home' stance with the built-in position servos.

Unlike GO2 (torque <motor>), G1 uses <position> actuators (kp=500, dampratio=1), so
control = target JOINT ANGLES written to d.ctrl; the servos do the PD. We just command
the nominal stance and check the biped stays upright. NVIDIA-free, CPU-only.

Usage: python g1_stand.py [scene.xml] [--secs 3] [--key home]
"""
import sys
import argparse
import mujoco
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("scene", nargs="?", default="/tmp/mjm/unitree_g1/scene.xml")
ap.add_argument("--secs", type=float, default=4.0)
ap.add_argument("--key", default="home")
args = ap.parse_args()

m = mujoco.MjModel.from_xml_path(args.scene)
d = mujoco.MjData(m)
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, args.key)
mujoco.mj_resetDataKeyframe(m, d, max(kid, 0))

# position actuators: target angle = the nominal joint angles of the chosen keyframe
q_des = d.qpos[7:].copy()
d.ctrl[:] = q_des

n = int(args.secs / m.opt.timestep)
zs, rolls, pitches = [], [], []
for i in range(n):
    d.ctrl[:] = q_des
    mujoco.mj_step(m, d)
    zs.append(d.qpos[2])
    w, x, y, z = d.qpos[3:7]
    rolls.append(np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))))
    pitches.append(np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1))))

zs = np.array(zs)
xy = np.linalg.norm(d.qpos[0:2])
upright = (zs[-1] > 0.70) and abs(rolls[-1]) < 15 and abs(pitches[-1]) < 15
print(f"keyframe={args.key}  steps={n} ({args.secs}s @ {1/m.opt.timestep:.0f}Hz)  nu={m.nu}")
print(f"base z: start={zs[0]:.3f} end={zs[-1]:.3f} min={zs.min():.3f}")
print(f"final roll={rolls[-1]:+.1f} pitch={pitches[-1]:+.1f}  xy-drift={xy:.3f} m")
print(f"RESULT: {'STANDS UPRIGHT ✓' if upright else 'FELL ✗'}")
sys.exit(0 if upright else 1)
