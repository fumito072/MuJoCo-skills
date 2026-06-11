"""BUTT-SCOOT climb exploration. VERIFIED primitive: phase A (backward stiff descent
onto the 0.22 m platform edge -> stable edge-sit, base z 0.375, pitch 0, indefinitely
stable). OPEN: phases B/C — on a 0.3 m-deep platform the seated butt occupies the
space the feet need; candidate continuations: armrest hand-press scoot (hand pairs
not yet wired), or RL from the edge-sit state. Kept as the starting point for the
next climb attempt — see docs/FULL_MISSION_DEPLOY.md §3.

Original plan: sit on the platform edge -> swing feet up -> stand up.
Phases (all stiff kp300, CoM-PD on grounded ankles like probe_climb):
  A: backward stiff descent until the pelvis box rests on the platform edge
  B: lift each foot onto the platform (sitting: no balance constraint)
  C: lean forward + squat-rise to a stand on the platform
Args via env: START_Y (floor stand), PH (run through phase A/B/C)
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/hoshinafumito/development/Colapis_project/MuJoCo-skills/training")
import mujoco
import g1_sit_env as env

m = env.build_fbx_chair_model(0.002)
d = mujoco.MjData(m)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
START_Y = float(os.environ.get("START_Y", "0.62"))
PH = os.environ.get("PH", "ABC")
DBG = bool(os.environ.get("DBG"))

yaw = np.pi / 2          # facing +y, platform behind
mujoco.mj_resetDataKeyframe(m, d, key)
d.qpos[0:3] = (0.0, START_Y, 0.755)
d.qpos[3:7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
mujoco.mj_forward(m, d)
for a in range(12):
    m.actuator_gainprm[a, 0] = 300.0
    m.actuator_biasprm[a, 1] = -300.0
    m.actuator_biasprm[a, 2] = -8.0

default_pose = np.array(m.key_qpos[key][7:])
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
LF, RF = m.geom("left_foot").id, m.geom("right_foot").id
PB = m.geom("pelvis_collision").id
rc = {i for i in range(m.ngeom)
      if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("rc_part")}

# pose snippets (ctrl indices): edge-sit descent target — deep squat-sit, feet stay fwd
EDGE_SIT = {0: -1.9, 3: 2.5, 4: 0.3, 6: -1.9, 9: 2.5, 10: 0.3}
# foot-up while seated: hip deep flex + knee mid -> foot lands on platform ~y0.37
L_UP = {0: -2.4, 3: 2.2, 4: 0.0}
L_DOWN = {0: -2.1, 3: 2.6, 4: 0.1}
R_UP = {6: -2.4, 9: 2.2, 10: 0.0}
R_DOWN = {6: -2.1, 9: 2.6, 10: 0.1}
# squat-rise target = keyframe stand
RISE = {0: float(default_pose[0]), 3: float(default_pose[3]), 4: float(default_pose[4]),
        6: float(default_pose[6]), 9: float(default_pose[9]), 10: float(default_pose[10])}

WPS = []
if "A" in PH:
    WPS += [
        (0.4, {}, "mid", 0.0),
        (1.4, dict(EDGE_SIT), "mid", 0.0),       # descend onto the edge
        (2.2, {}, "mid", 0.0),                   # settle seated
    ]
# B (brace strategy): recline the torso against the chair-seat front edge (full
# support: butt on platform + back on seat) -> tuck BOTH feet onto the platform at
# once -> push torso back upright over the tucked feet
RECLINE = {14: -0.50, 0: -2.3, 6: -2.3}
TUCK = {0: -2.5, 3: 2.85, 4: -0.6, 6: -2.5, 9: 2.85, 10: -0.6}
UNRECLINE = {14: 0.30}
if "B" in PH:
    WPS += [
        (3.4, dict(RECLINE), "mid", 0.0),
        (4.6, dict(TUCK), "mid", 0.0),
        (5.4, dict(UNRECLINE), "mid", 0.0),
        (6.0, {}, "mid", 0.0),
    ]
if "C" in PH:
    WPS += [
        (6.8, {14: 0.35}, "mid", 0.04),          # lean torso forward over the feet
        (9.0, {**RISE, 14: 0.10}, "mid", 0.02),  # rise
        (10.4, {14: 0.0}, "mid", 0.0),
    ]

KR, KP_, KD = 2.5, 2.5, 1.0
prev = default_pose.copy()
prev_t = 0.0
segs = []
for t_end, deltas, com_over, lead in WPS:
    tgt = prev.copy()
    for idx, val in deltas.items():
        tgt[int(idx)] = val
    segs.append((prev_t, t_end, prev.copy(), np.clip(tgt, lo, hi), com_over, lead))
    prev, prev_t = np.clip(tgt, lo, hi), t_end

d.ctrl[:] = default_pose
t = 0.0
fail = None
while t < segs[-1][1]:
    for t0, t1, c0, c1, com_over, lead in segs:
        if t0 <= t < t1:
            a = (t - t0) / (t1 - t0)
            ctrl = c0 + a * (c1 - c0)
            mujoco.mj_subtreeVel(m, d)
            com = d.subtree_com[1]
            cv = d.subtree_linvel[1]
            pl, pr = d.geom_xpos[LF], d.geom_xpos[RF]
            ref = (pl + pr) / 2
            qw, qx, qy, qz = d.qpos[3:7]
            cy = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            fwd = np.array([np.cos(cy), np.sin(cy)])
            left = np.array([-np.sin(cy), np.cos(cy)])
            err = np.array([ref[0] - com[0], ref[1] - com[1]])
            vel = np.array([cv[0], cv[1]])
            e_f = float(err @ fwd) + lead
            e_l = float(err @ left)
            droll = float(np.clip(KR * e_l - KD * float(vel @ left), -0.26, 0.26))
            dpitch = float(np.clip(-KP_ * e_f + KD * float(vel @ fwd), -0.45, 0.45))
            for hr, ap, ar, foot in ((1, 4, 5, pl), (7, 10, 11, pr)):
                if foot[2] < 0.06 or 0.21 < foot[2] < 0.26:
                    ctrl[ar] += droll
                    ctrl[ap] += dpitch
            d.ctrl[:] = np.clip(ctrl, lo, hi)
            break
    for _ in range(10):
        mujoco.mj_step(m, d)
    t += 0.02
    w, x_, y_, z_ = d.qpos[3:7]
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y_ - z_ * x_), -1, 1)))
    roll = np.degrees(np.arctan2(2 * (w * x_ + y_ * z_), 1 - 2 * (x_**2 + y_**2)))
    if DBG and abs(t - round(t * 5) / 5) < 1e-9:
        pl, pr = d.geom_xpos[LF], d.geom_xpos[RF]
        pb = d.geom_xpos[PB]
        print(f"  t={t:4.1f} base=({d.qpos[0]:+.3f},{d.qpos[1]:+.3f},{d.qpos[2]:.3f}) "
              f"p={pitch:+5.1f} r={roll:+5.1f} pbz={pb[2]:.2f} "
              f"L=({pl[1]:+.2f},{pl[2]:.2f}) R=({pr[1]:+.2f},{pr[2]:.2f})")
    if abs(roll) > 50 or abs(pitch) > 65:
        fail = f"fell t={t:.2f} (pitch={pitch:+.1f}, roll={roll:+.1f})"
        break

pl, pr = d.geom_xpos[LF], d.geom_xpos[RF]
print(f"FINAL: base=({d.qpos[0]:+.3f},{d.qpos[1]:+.3f},{d.qpos[2]:.3f}) "
      f"L=({pl[1]:+.2f},{pl[2]:.2f}) R=({pr[1]:+.2f},{pr[2]:.2f})")
ok = pl[2] > 0.20 and pr[2] > 0.20 and d.qpos[2] > 0.90
print("RESULT:", fail if fail else ("CLIMBED (standing on platform)" if ok else "incomplete"))
