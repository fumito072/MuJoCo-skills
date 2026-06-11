"""Build a kinematic REFERENCE trajectory for the forward step-up (DeepMimic-style).

The 22 scripted climb attempts failed DYNAMICALLY (open-loop), but their waypoints
are kinematically sound. As a tracking-reward reference they convert the RL problem
from "discover a 0.22 m single-support transfer" (50M steps, never found) into
"track this motion and learn the feedback corrections" — the standard way such
single hard skills are solved, at zero extra compute.

Reference = 50 Hz arrays over ~4.6 s: 12 leg-joint targets + base (y, z, yaw) +
support phase flags. Base path is specified by hand (smooth interpolation between
stance keyposes); joints come from the probed kinematic waypoints (tuck -> reach ->
place -> lunge-transfer -> trail-leg-up -> stand).

Writes training/g1_climb_reference.npz; visual check with PLOT=1.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco  # noqa: E402

import g1_sit_env  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "training", "g1_climb_reference.npz")
DT = 0.02

m = g1_sit_env.build_fbx_chair_model(0.002)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
DEF = np.array(m.key_qpos[key][7:19])    # 12 leg joints, keyframe

# (t, legs{idx: val}, base_y, base_z) — forward climb facing -y (toward the chair),
# RIGHT leg leads. Joint numbers from the kinematic probes (2026-06-11).
TSCALE = 1.6   # slower = closer to quasi-static = trackable
KEY_FRAMES = [
    (0.0, {},                                                0.68, 0.755),
    (0.6, {},                                                0.68, 0.755),  # settle
    (1.2, {},                                                0.66, 0.745),  # weight L (PD does lateral)
    (1.8, {6: -1.00, 9: 1.90, 10: -0.40},                    0.64, 0.740),  # R tuck up
    (2.4, {6: -1.30, 9: 1.15, 10: -0.15},                    0.62, 0.745),  # R reach fwd onto platform
    (2.8, {6: -1.20, 9: 1.20, 10: -0.10},                    0.60, 0.750),  # R plant (foot ~y0.40 z0.22)
    (3.6, {6: -0.45, 9: 0.70, 10: 0.05, 0: -0.20, 3: 0.90, 4: 0.30},
                                                             0.50, 0.870),  # lunge: rise over R foot
    (4.2, {6: -0.31, 9: 0.63, 10: -0.30, 0: -1.10, 3: 1.90, 4: -0.40},
                                                             0.44, 0.940),  # L trail leg up
    (4.6, {0: -0.31, 3: 0.63, 4: -0.30},                     0.41, 0.975),  # L plant beside, stand
    (5.2, {},                                                0.40, 0.975),  # hold stand
]

KEY_FRAMES = [(t * TSCALE, d_, y, z) for t, d_, y, z in KEY_FRAMES]
T = KEY_FRAMES[-1][0]
N = int(T / DT) + 1
legs = np.zeros((N, 12))
base = np.zeros((N, 2))      # y, z

cur = DEF.copy()
frames = []
for t_kf, deltas, by, bz in KEY_FRAMES:
    tgt = cur.copy()
    for i, v in deltas.items():
        tgt[i] = v
    frames.append((t_kf, tgt, by, bz))
    cur = tgt

for k in range(N):
    t = k * DT
    for (t0, j0, y0, z0), (t1, j1, y1, z1) in zip(frames[:-1], frames[1:]):
        if t0 <= t <= t1:
            a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            a = a * a * (3 - 2 * a)            # smoothstep
            legs[k] = j0 + a * (j1 - j0)
            base[k] = (y0 + a * (y1 - y0), z0 + a * (z1 - z0))
            break

np.savez_compressed(OUT, legs=legs, base=base, dt=DT,
                    yaw=-np.pi / 2, duration=T)
print(f"reference written: {OUT}  ({N} frames, {T:.1f} s)")

if os.environ.get("PLOT"):
    for k in range(0, N, 15):
        print(f"t={k * DT:4.2f} base=({base[k][0]:.2f},{base[k][1]:.3f}) "
              f"Rhip={legs[k][6]:+.2f} Rkn={legs[k][9]:+.2f} "
              f"Lhip={legs[k][0]:+.2f} Lkn={legs[k][3]:+.2f}")
