"""Hand-braced climb probe: lean forward and PRESS the armrests (user's idea,
2026-06-13). Scripted, plain C engine, NVIDIA-free.

VERIFIED RESULT (default waypoints): from a floor stand facing the chair, the
G1 leans into a quasi-static 4-POINT BRIDGE — pitch ~34 deg, both hands pressed
on the armrest fronts at ~(±0.26, +0.13, 1.03) carrying 45-50 N each, toes on
the floor. qvel-norm 0.08 at t=7.0 (use BRIDGE_T=7.0 to dump the state). The
bridge holds ~2 s, then the point-contact hands creep inward and it collapses.

WHAT 21 ITERATIONS ESTABLISHED (telemetry, not vibes):
  * spawn: START_Y < 0.66 clips the toes into the step wall -> violent bounce.
  * arm pre-pose BEFORE the swing shifts CoM and breaks the (already marginal)
    no-hands swing; arms must deploy during/with the lean.
  * ankle pitch sign: NEGATIVE = dorsiflex = lean forward (sit recipe's +0.137
    is a squat context, not a lean). waist_pitch: NEGATIVE = bend forward.
  * the CoM-PD pitch feedback fights any deliberate lean -> per-waypoint
    `nopitch` flag disables it while the hands are the intended catcher.
  * armrest TIPS are unreachable from an upright floor stand (need ~25 deg of
    lean; unbraced ankles give out at ~18). The reachable contact is the
    armrest FRONT SLOPE -> hands slide down it unless wrists are stiff.
  * WRIST STIFFNESS IS THE KEY ENABLER: with stock kp=2 the hand flops and
    rolls off the narrow rest; kp=80 turns the hand into a rigid strut and the
    bridge becomes repeatable (this maps to a real G1 wrist gain mode).
  * hip crouch from the bridge shifts load to the hands (40->70 N) without
    breaking it; both static states are reproducible run-to-run.
  * EVERY open-loop attempt to lift a foot from the bridge fails the same way:
    the swing impulse exceeds the friction cone of the point-contact hands
    (round capsule on round rest edge, no grasp DOF) and a hand slips -> roll.
    Toe-drag instead of air-swing reduces the disturbance (roll -7 vs -29 deg)
    but the final 25 cm flick still breaks it.

CONCLUSION: "press the armrests" gives a real, reproducible braced pre-climb
stance but finishing the step-up needs contact FEEDBACK, not a fixed timeline.
The bridge qpos (training/g1_bridge_state.npz) is exported as an RSI start
state, and g1_climb_mjx_env.py now has armrests + stiff wrists so the GPU
policy can learn the brace. See docs/FULL_MISSION_DEPLOY.md.

Usage:
  .venv-rl/bin/python training/g1_handbrace_probe.py            # bridge demo
  DBG=1 ... / KIN=1 ... / WP='[...]' ... / BRIDGE_T=7.0 ...     # experiments
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco  # noqa: E402

import g1_sit_env  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_OUT = os.path.join(REPO, "training", "g1_bridge_state.npz")

# full 153-hull chair + stock leg pairs + hand-vs-chair pairs
m = g1_sit_env.build_fbx_chair_model(0.002, pair_hands=True)
d = mujoco.MjData(m)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
START_Y = float(os.environ.get("START_Y", "0.66"))   # < 0.66 = toes clip the step
YAW = -np.pi / 2


def reset():
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.qpos[0:3] = (0.0, START_Y, 0.755)
    d.qpos[3:7] = (np.cos(YAW / 2), 0, 0, np.sin(YAW / 2))
    mujoco.mj_forward(m, d)


reset()
default_pose = np.array(m.key_qpos[key][7:])
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
LH = m.geom("left_hand_collision").id
RH = m.geom("right_hand_collision").id
LF, RF = m.geom("left_foot").id, m.geom("right_foot").id

if os.environ.get("KIN"):
    # scan: (l_shoulder_pitch, l_shoulder_roll, elbow) mirrored to right
    for sp, sr, el in [(-0.6, 0.8, 0.2), (-0.7, 0.9, 0.2), (-0.5, 0.7, 0.1),
                       (-0.8, 0.8, 0.3), (-0.6, 1.0, 0.0), (-0.7, 0.7, 0.0)]:
        reset()
        d.qpos[7 + 15], d.qpos[7 + 16], d.qpos[7 + 18] = sp, sr, el     # left arm
        d.qpos[7 + 22], d.qpos[7 + 23], d.qpos[7 + 25] = sp, -sr, el    # right arm
        mujoco.mj_forward(m, d)
        pl, pr = d.geom_xpos[LH], d.geom_xpos[RH]
        print(f"sp{sp:+.1f} sr{sr:+.1f} el{el:+.1f} -> "
              f"L=({pl[0]:+.2f},{pl[1]:+.2f},{pl[2]:.2f}) R=({pr[0]:+.2f},{pr[1]:+.2f},{pr[2]:.2f})"
              f"   [armrest top: x±0.30, y -0.3..0.1, z 0.80-0.93]")
    sys.exit(0)

# mode gains for the press (mirrors real G1 mode-dependent gains)
for a in range(12):                          # legs: SIT-mode stiff
    m.actuator_gainprm[a, 0] = 300.0
    m.actuator_biasprm[a, 1] = -300.0
    m.actuator_biasprm[a, 2] = -8.0
for a in (15, 16, 17, 18, 22, 23, 24, 25):   # shoulders+elbows: stiff for the press
    m.actuator_gainprm[a, 0] = 150.0
    m.actuator_biasprm[a, 1] = -150.0
    m.actuator_biasprm[a, 2] = -4.0
for a in (19, 20, 21, 26, 27, 28):           # wrists: rigid strut, not a floppy hinge
    m.actuator_gainprm[a, 0] = 80.0
    m.actuator_biasprm[a, 1] = -80.0
    m.actuator_biasprm[a, 2] = -2.0

# waypoints: [t_end, {ctrl_idx: target}, com_over, fwd_lead, nopitch?]
# default = the VERIFIED bridge demo: settle -> arms out -> tripod lean ->
# committed lean (hands catch the rest fronts) -> ease ankles -> hip crouch
# (loads hands to ~50 N each) -> hold.
WPS = json.loads(os.environ.get("WP", "[]"))
if not WPS:
    WPS = [
        [0.5, {}, "mid", 0.0],
        [1.5, {"15": -0.8, "16": 0.95, "18": 0.1, "22": -0.8, "23": -0.95, "25": 0.1},
         "mid", 0.0],
        [3.2, {"4": -0.60, "10": -0.60, "14": -0.25}, "mid", 0.0, 1],
        [4.2, {"4": -0.95, "10": -0.95, "14": -0.50, "15": -0.55, "22": -0.55},
         "mid", 0.0, 1],
        [5.4, {"4": -0.60, "10": -0.60}, "mid", 0.0, 1],
        [6.4, {"0": -0.5, "6": -0.5, "3": 0.8, "9": 0.8, "4": -0.4, "10": -0.4,
               "15": -0.40, "22": -0.40}, "mid", 0.0, 1],
        [7.4, {}, "mid", 0.0, 1],
    ]

rc = {i for i in range(m.ngeom)
      if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("rc_part")}
KR, KP_, KD = 2.5, 2.5, 1.0
prev = default_pose.copy()
prev_t = 0.0
segs = []
for wp in WPS:
    t_end, deltas, com_over, lead = wp[:4]
    nopitch = bool(wp[4]) if len(wp) > 4 else False
    tgt = prev.copy()
    for idx, val in deltas.items():
        tgt[int(idx)] = val
    segs.append((prev_t, t_end, prev.copy(), np.clip(tgt, lo, hi), com_over, lead, nopitch))
    prev, prev_t = np.clip(tgt, lo, hi), t_end

f6 = np.zeros(6)
d.ctrl[:] = default_pose
t, fail = 0.0, None
DBG = bool(os.environ.get("DBG"))
BRIDGE_T = float(os.environ.get("BRIDGE_T", "0"))
bridge_h = (0.0, 0.0)
while t < segs[-1][1]:
    for t0, t1, c0, c1, com_over, lead, nopitch in segs:
        if t0 <= t < t1:
            a = (t - t0) / (t1 - t0)
            ctrl = c0 + a * (c1 - c0)
            mujoco.mj_subtreeVel(m, d)
            com = d.subtree_com[1]
            cv = d.subtree_linvel[1]
            pl, pr = d.geom_xpos[LF], d.geom_xpos[RF]
            ref = {"L": pl, "R": pr, "mid": (pl + pr) / 2}[com_over]
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
            if nopitch:
                dpitch = 0.0
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
    roll = np.degrees(np.arctan2(2 * (w * x_ + y_ * z_), 1 - 2 * (x_ ** 2 + y_ ** 2)))
    fh = {LH: 0.0, RH: 0.0}
    for i in range(d.ncon):
        c = d.contact[i]
        for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if g in fh and o in rc:
                mujoco.mj_contactForce(m, d, i, f6)
                fh[g] += abs(f6[0])
    bridge_h = (fh[LH], fh[RH])
    if DBG and abs(t - round(t * 5) / 5) < 1e-9:
        pl, pr = d.geom_xpos[LF], d.geom_xpos[RF]
        hl = d.geom_xpos[LH]
        print(f"  t={t:4.1f} base=({d.qpos[0]:+.2f},{d.qpos[1]:+.2f},{d.qpos[2]:.3f}) "
              f"p={pitch:+5.1f} r={roll:+5.1f} H=({fh[LH]:3.0f},{fh[RH]:3.0f})N "
              f"hL=({hl[0]:+.2f},{hl[1]:+.2f},{hl[2]:.2f}) "
              f"Lf=({pl[1]:+.2f},{pl[2]:.2f}) Rf=({pr[1]:+.2f},{pr[2]:.2f})")
    if BRIDGE_T and abs(t - BRIDGE_T) < 1e-9:
        np.savez(BRIDGE_OUT, qpos=d.qpos.copy(), qvel=d.qvel.copy())
        print(f"bridge state saved -> {BRIDGE_OUT} "
              f"(qvel norm {float(np.linalg.norm(d.qvel)):.4f})")
    if abs(roll) > 50 or pitch > 82 or pitch < -65:
        fail = f"fell t={t:.2f} (p={pitch:+.1f}, r={roll:+.1f})"
        break

w, x_, y_, z_ = d.qpos[3:7]
pitch = np.degrees(np.arcsin(np.clip(2 * (w * y_ - z_ * x_), -1, 1)))
bridged = (not fail and 25 < pitch < 45
           and bridge_h[0] > 20 and bridge_h[1] > 20 and d.qpos[2] > 0.6)
pl, pr = d.geom_xpos[LF], d.geom_xpos[RF]
climbed = pl[2] > 0.20 and pr[2] > 0.20 and d.qpos[2] > 0.90
print(f"FINAL: base=({d.qpos[0]:+.3f},{d.qpos[1]:+.3f},{d.qpos[2]:.3f}) "
      f"pitch={pitch:+.1f} H=({bridge_h[0]:.0f},{bridge_h[1]:.0f})N")
print("RESULT:", fail if fail else
      ("CLIMBED" if climbed else ("BRIDGE HELD" if bridged else "incomplete")))
raise SystemExit(0 if (not fail and (climbed or bridged)) else 1)
