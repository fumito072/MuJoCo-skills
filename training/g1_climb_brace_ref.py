"""Build a 29-DOF HAND-BRACE climb reference (DeepMimic scaffold) for Mac-CPU.

The floor->platform first-foot-up is single-leg-balance-hard; the user's idea is to
brace both hands on the chair armrests (the verified 4-point bridge,
g1_bridge_state.npz) so the hands take load while a foot steps up. The bridge is an
ACTIVELY-balanced pose (open-loop it sags), so we don't just RSI into it — we author
a full braced-climb trajectory the policy tracks with the stiff arm gains + hand
contact pairs providing the support, and RL learns only the feedback.

Channels (29 joints + base y,z; yaw fixed facing the chair, lean lives in the joints):
  legs (0-11): keyframed step-up from the bridge crouch onto the platform
  waist(12-14): hold the bridge lean, then un-lean to neutral as it stands
  arms (15-28): HOLD the bridge brace through the first-foot-up, release near the end

Writes training/g1_climb_brace_reference.npz.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco  # noqa: E402

import g1_sit_env  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "training", "g1_climb_brace_reference.npz")
DT = 0.02
TSCALE = 1.7                       # slower = closer to quasi-static = trackable

B = np.load(os.path.join(REPO, "training", "g1_bridge_state.npz"))["qpos"]
BRIDGE_J = B[7:].copy()            # 29 joint angles of the verified brace
BRIDGE_BASE = (float(B[1]), float(B[2]))   # (y, z) = (0.50, 0.705)
BRIDGE_QUAT = B[3:7].copy()        # the lean lives HERE (pitch ~34deg) + yaw -90
YAW = -np.pi / 2
UPRIGHT_QUAT = np.array([np.cos(YAW / 2), 0, 0, np.sin(YAW / 2)])  # stand: yaw only


def nlerp(q0, q1, a):
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - a) * q0 + a * q1
    return q / np.linalg.norm(q)

m = g1_sit_env.build_fbx_chair_model(0.002)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
DEF = np.array(m.key_qpos[key][7:])        # default neutral 29-dim pose

# stand-on-platform leg pose (from the verified leg climb reference end)
STAND_LEGS = np.load(os.path.join(REPO, "training", "g1_climb_reference.npz"))["legs"][-1]

# --- LEG step-up keyframes: (t, {leg_idx: value}, base_y, base_z) -----------------
# running pose starts at the BRIDGE leg crouch; deltas override specific joints.
# R leg leads onto the footrest (y~0.40, z0.22), then both extend to stand.
LEG_KF = [
    (0.0,  {},                                              0.50, 0.705),  # brace, hold
    (0.8,  {},                                              0.50, 0.705),  # settle in brace
    (1.6,  {6: -1.25, 9: 1.35, 10: -0.30},                 0.50, 0.715),  # lift R foot (tuck up)
    (2.6,  {6: -1.40, 9: 0.95, 10: -0.10},                 0.47, 0.745),  # R reach fwd onto footrest
    (3.4,  {6: -1.15, 9: 1.05, 10: -0.05},                 0.45, 0.805),  # R plant, weight forward
    (4.4,  {6: -0.55, 9: 0.78, 10: 0.0,
            0: -1.05, 3: 1.65, 4: -0.35},                  0.42, 0.900),  # rise over R, L trail up
    (5.2,  {0: -0.31, 3: 0.63, 4: -0.30},                  0.41, 0.950),  # L plant on platform
    (6.0,  dict(zip(range(12), STAND_LEGS)),               0.40, 0.975),  # stand
    (6.6,  dict(zip(range(12), STAND_LEGS)),               0.40, 0.975),  # hold
]
LEG_KF = [(t * TSCALE, d_, y, z) for t, d_, y, z in LEG_KF]
T = LEG_KF[-1][0]
N = int(T / DT) + 1

# build the running LEG pose track from the keyframes
leg_frames = []
cur = BRIDGE_J[:12].copy()
for t_kf, deltas, by, bz in LEG_KF:
    tgt = cur.copy()
    for i, v in deltas.items():
        tgt[i] = v
    leg_frames.append((t_kf, tgt, by, bz))
    cur = tgt


def smoothstep(a):
    a = np.clip(a, 0.0, 1.0)
    return a * a * (3 - 2 * a)


legs = np.zeros((N, 12))
base = np.zeros((N, 2))
quat = np.zeros((N, 4))             # base orientation: bridge lean -> upright stand
arms = np.zeros((N, 7 + 7))        # 14 arm joints (15-28)
waist = np.zeros((N, 3))           # 12-14

# RELEASE schedules (phase-based): hold the bridge brace + lean through the
# first-foot-up, then ease to neutral. The lean (base quat) un-leans with the waist;
# the hands release LAST (after the feet are up on the platform).
LEAN_REL = (0.50, 0.88)            # base un-lean window (phase)
WAIST_REL = (0.50, 0.88)
ARM_REL = (0.62, 0.95)             # hand-release window (phase)

for k in range(N):
    t = k * DT
    ph = k / (N - 1)
    for (t0, j0, y0, z0), (t1, j1, y1, z1) in zip(leg_frames[:-1], leg_frames[1:]):
        if t0 <= t <= t1:
            a = smoothstep(0.0 if t1 == t0 else (t - t0) / (t1 - t0))
            legs[k] = j0 + a * (j1 - j0)
            base[k] = (y0 + a * (y1 - y0), z0 + a * (z1 - z0))
            break
    la = smoothstep((ph - LEAN_REL[0]) / (LEAN_REL[1] - LEAN_REL[0]))
    quat[k] = nlerp(BRIDGE_QUAT, UPRIGHT_QUAT, la)
    wa = smoothstep((ph - WAIST_REL[0]) / (WAIST_REL[1] - WAIST_REL[0]))
    waist[k] = BRIDGE_J[12:15] + wa * (DEF[12:15] - BRIDGE_J[12:15])
    aa = smoothstep((ph - ARM_REL[0]) / (ARM_REL[1] - ARM_REL[0]))
    arms[k] = BRIDGE_J[15:29] + aa * (DEF[15:29] - BRIDGE_J[15:29])

# --- IK the R (lead) foot onto the footrest and keep it planted -------------------
# The hand-authored R-leg leaves the foot below the footrest top (it would hit the
# front face). R is the stance foot through the climb, so foot-target IK it: lift from
# the floor, plant on the footrest top (y~0.42,z0.24), hold, then it extends to stand.
d = mujoco.MjData(m)
RF = m.geom("right_foot").id
J6, J9, J10 = 6, 9, 10            # R hip-pitch, knee, ankle-pitch


def r_foot(fr, jvec):
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.qpos[0:3] = (0.0, base[fr][0], base[fr][1])
    d.qpos[3:7] = quat[fr]
    d.qpos[7:] = jvec
    mujoco.mj_forward(m, d)
    return d.geom_xpos[RF].copy()


def rfoot_target(ph):
    pts = [(0.15, 0.83, 0.05), (0.27, 0.62, 0.24), (0.40, 0.44, 0.25),
           (0.50, 0.42, 0.24), (0.85, 0.40, 0.23)]
    for (p0, y0, z0), (p1, y1, z1) in zip(pts[:-1], pts[1:]):
        if p0 <= ph <= p1:
            a = smoothstep((ph - p0) / (p1 - p0))
            return np.array([y0 + a * (y1 - y0), z0 + a * (z1 - z0)])
    return None


for k in range(N):
    ph = k / (N - 1)
    tgt = rfoot_target(ph)
    if tgt is None:
        continue
    jv = ref29_row = np.concatenate([legs[k], waist[k], arms[k]])
    v = np.array([jv[J6], jv[J9], jv[J10]], float)
    for _ in range(40):
        jj = jv.copy(); jj[J6], jj[J9], jj[J10] = v
        f = r_foot(k, jj)[1:3]                       # (y,z)
        r = f - tgt
        if np.linalg.norm(r) < 1e-3:
            break
        Jc = np.zeros((2, 3))
        for i in range(3):
            dv = v.copy(); dv[i] += 1e-4
            jj2 = jv.copy(); jj2[J6], jj2[J9], jj2[J10] = dv
            Jc[:, i] = (r_foot(k, jj2)[1:3] - f) / 1e-4
        dvv = -np.linalg.solve(Jc.T @ Jc + 1e-3 * np.eye(3), Jc.T @ r)
        v = v + np.clip(dvv, -0.2, 0.2)
    legs[k][J6], legs[k][J9], legs[k][J10] = v

# assemble full 29-dim joint reference
ref29 = np.zeros((N, 29))
ref29[:, 0:12] = legs
ref29[:, 12:15] = waist
ref29[:, 15:29] = arms

# light 5-tap smoothing to kill IK<->keyframe handoff jumps (max joint jump 0.20 rad/fr)
k5 = np.ones(5) / 5.0
for c in range(29):
    ref29[:, c] = np.convolve(np.pad(ref29[:, c], 2, "edge"), k5, "valid")
for c in range(2):
    base[:, c] = np.convolve(np.pad(base[:, c], 2, "edge"), k5, "valid")

np.savez_compressed(OUT, joints=ref29, base=base, quat=quat, dt=DT, yaw=YAW, duration=T)
print(f"brace reference written: {OUT}  ({N} frames, {T:.1f} s)")
print(f"  frame0 = bridge brace (leaned); end = stand on platform (base {base[-1].round(3)})")
