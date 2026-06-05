"""Unitree G1 sit/stand (squat): smoothly lower to a deep knee-bend and rise back up.

Quasi-static posture control on G1's <position> actuators. The legs follow a symmetric
squat: hip_pitch = ankle_pitch = -knee/2 keeps the torso vertical and feet flat while the
CoM lowers. Arms/waist held at home. NVIDIA-free, CPU-only.

Phases: stand -> lower -> hold (sit) -> rise -> stand.

Usage: python g1_squat.py [scene.xml] [--knee 1.3] [--secs 6]
"""
import sys
import argparse
import mujoco
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("scene", nargs="?", default="/tmp/mjm/unitree_g1/scene.xml")
ap.add_argument("--knee", type=float, default=1.3, help="deep-squat knee angle (rad)")
ap.add_argument("--secs", type=float, default=6.0)
args = ap.parse_args()

m = mujoco.MjModel.from_xml_path(args.scene)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)            # home
home = d.qpos[7:].copy()

# actuator indices (from inspection): left leg 0-5, right leg 6-11
L = dict(hip_pitch=0, hip_roll=1, hip_yaw=2, knee=3, ankle_pitch=4, ankle_roll=5)
R = {k: v + 6 for k, v in L.items()}
KNEE_STAND = 0.3                                # home knee angle


def leg_targets(knee):
    """Symmetric squat: torso stays vertical, foot flat. Returns dict idx->angle."""
    t = {}
    for side in (L, R):
        t[side["hip_pitch"]] = -knee / 2
        t[side["knee"]] = knee
        t[side["ankle_pitch"]] = -knee / 2
    return t


def smoothstep(a, b, u):
    u = np.clip(u, 0, 1)
    return a + (b - a) * (u * u * (3 - 2 * u))


# phase schedule (fractions of total time): stand, lower, hold, rise, stand
def knee_at(frac):
    if frac < 0.15:
        return KNEE_STAND
    if frac < 0.40:
        return smoothstep(KNEE_STAND, args.knee, (frac - 0.15) / 0.25)
    if frac < 0.60:
        return args.knee
    if frac < 0.85:
        return smoothstep(args.knee, KNEE_STAND, (frac - 0.60) / 0.25)
    return KNEE_STAND


n = int(args.secs / m.opt.timestep)
zs, rolls, pitches = [], [], []
z_bottom = 1.0
for i in range(n):
    frac = i / n
    target = home.copy()
    for idx, ang in leg_targets(knee_at(frac)).items():
        target[idx] = ang
    d.ctrl[:] = target
    mujoco.mj_step(m, d)
    zs.append(d.qpos[2])
    if 0.4 < frac < 0.6:
        z_bottom = min(z_bottom, d.qpos[2])
    w, x, y, z = d.qpos[3:7]
    rolls.append(np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))))
    pitches.append(np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1))))

zs = np.array(zs)
returned = abs(zs[-1] - zs[0]) < 0.05
lowered = z_bottom < zs[0] - 0.10
upright = abs(rolls[-1]) < 15 and abs(pitches[-1]) < 20 and zs.min() > 0.4
ok = returned and lowered and upright
print(f"knee_squat={args.knee}  steps={n} ({args.secs}s)")
print(f"base z: stand={zs[0]:.3f}  sit(bottom)={z_bottom:.3f}  back={zs[-1]:.3f}  "
      f"(lowered {zs[0]-z_bottom:.3f} m)")
print(f"final roll={rolls[-1]:+.1f} pitch={pitches[-1]:+.1f}")
print(f"RESULT: {'SIT->STAND OK ✓' if ok else 'UNSTABLE ✗'}  "
      f"(lowered={lowered} returned={returned} upright={upright})")
sys.exit(0 if ok else 1)
