"""GO2 standing controller: software PD to the 'home' stance on torque actuators.

Validates the control loop + torque-actuator model on Apple Silicon, NVIDIA-free.
GO2 Menagerie uses <motor> (torque) actuators, so PD is computed in software:
    tau = kp*(q_des - q) - kd*qd , clipped to the actuator force range.

Usage: python go2_stand.py [scene.xml] [--kp 60] [--kd 3] [--secs 3]
"""
import sys
import argparse
import mujoco
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("scene", nargs="?", default="/tmp/mjm/unitree_go2/scene.xml")
ap.add_argument("--kp", type=float, default=60.0)
ap.add_argument("--kd", type=float, default=3.0)
ap.add_argument("--secs", type=float, default=3.0)
args = ap.parse_args()

m = mujoco.MjModel.from_xml_path(args.scene)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)  # 'home'
q_des = d.qpos[7:].copy()             # nominal joint angles [0,0.9,-1.8]x4
# GO2 motors are direct-drive (gear=1): ctrl == torque. forcerange is (0,0)=disabled,
# so the real torque limit is ctrlrange (±23.7 hip/thigh, ±45.43 calf).
flim = m.actuator_ctrlrange[:, 1].copy()

def pd_control(d):
    q = d.qpos[7:]
    qd = d.qvel[6:]
    tau = args.kp * (q_des - q) - args.kd * qd
    return np.clip(tau, -flim, flim)

n = int(args.secs / m.opt.timestep)
zs, rolls, pitches = [], [], []
for i in range(n):
    d.ctrl[:] = pd_control(d)
    mujoco.mj_step(m, d)
    zs.append(d.qpos[2])
    # base orientation (quat -> roll/pitch) for tip-over check
    w, x, y, z = d.qpos[3:7]
    rolls.append(np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))))
    pitches.append(np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1))))

zs = np.array(zs)
xy_drift = np.linalg.norm(d.qpos[0:2])
joint_err = np.abs(d.qpos[7:] - q_des).max()
standing = (0.20 < d.qpos[2] < 0.35) and abs(rolls[-1]) < 20 and abs(pitches[-1]) < 20

print(f"kp={args.kp} kd={args.kd}  steps={n} ({args.secs}s @ {1/m.opt.timestep:.0f}Hz)")
print(f"base height z: start={zs[0]:.3f} end={zs[-1]:.3f} min={zs.min():.3f} max={zs.max():.3f}")
print(f"final roll={rolls[-1]:+.1f}deg pitch={pitches[-1]:+.1f}deg  xy-drift={xy_drift:.3f}m  max joint err={joint_err:.3f}rad")
print(f"RESULT: {'STANDS ✓' if standing else 'FELL ✗'}")
sys.exit(0 if standing else 1)
