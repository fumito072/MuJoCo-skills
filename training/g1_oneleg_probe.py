"""One-leg stand probe — the human curriculum step (user idea, 2026-06-13):
shift weight onto the stance leg via HIP ROLL, raise the other to a 90-deg
tuck, balance. Flat ground, no chair. Plain C engine, NVIDIA-free.

STATUS: NOT a verified primitive — best run reaches the full 90-deg tuck
(swing foot z 0.37 m) and holds one-legged for ~0.3-0.4 s, then falls.
Default waypoints reproduce that best attempt; VIDEO=path renders it
(assets/g1_oneleg_attempt.gif).

WHAT 16 ITERATIONS ESTABLISHED (all measured, docs/FULL_MISSION_DEPLOY.md 3.5):
  * hip-roll strategy moves the CoM ~12 cm laterally (ankle-only PD saturates
    at ~5 cm at its +-0.26 rad clip). Signs: hip roll NEGATIVE = lean left
    (+-0.15 -> base x -+0.22 m), waist_roll POSITIVE = torso left.
  * the lateral transfer is an AVALANCHE: the closed-chain leg push
    accelerates the CoM (~8 cm/s at center crossing); timer transitions
    cannot catch it -> state gates ("shifting" fires at 3.5 cm) + a brake
    segment land the CoM within 1 cm of the stance foot, quasi-static.
  * foot-contact threshold matters: a planted foot reads z ~0.01; using 0.06
    delayed single-support detection by ~0.3 s (the drift window).
  * from the balanced start, EVERY unload/lift still falls: four distinct
    fall modes mapped (translate-right, translate-left, rotate-right,
    reaction-coupled). Hip-roll feedback alone cannot fix position AND
    attitude in single support (the torque pair pushes one while rotating
    the other) — classic underactuated balance (capture-point territory).
    Softening the stance ankle (SOFT_ANKLE=40) does not change the outcome.
  * conclusion: exactly the feedback-coordination problem RL solves; the
    G1ClimbBox mode-D resets (g1_climb_mjx_env.py) spawn INSIDE this state
    so the GPU policy learns the balance where the climb needs it.

Experiment switches (env): WP (waypoint JSON), DBG=1, VIDEO=out.gif,
ARS (single-support ankle-roll term sign/scale), HRS (hip feedback sign),
SOFT_ANKLE (stance ankle kp in single support), KIN=1 (sign scan).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.g1 import base as g1_base

assets = g1_base.get_assets()
spec = mujoco.MjSpec.from_string(consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets)
spec.assets = assets
m = spec.compile()
m.opt.timestep = 0.002
d = mujoco.MjData(m)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
YAW = -np.pi / 2

mujoco.mj_resetDataKeyframe(m, d, key)
d.qpos[3:7] = (np.cos(YAW / 2), 0, 0, np.sin(YAW / 2))
mujoco.mj_forward(m, d)
default_pose = np.array(m.key_qpos[key][7:])
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
LF, RF = m.geom("left_foot").id, m.geom("right_foot").id

for a in range(12):
    m.actuator_gainprm[a, 0] = 300.0
    m.actuator_biasprm[a, 1] = -300.0
    m.actuator_biasprm[a, 2] = -8.0

if os.environ.get("KIN"):
    # sign scan: which hip-roll delta moves the base toward +x (left, facing -y)?
    for hr in (+0.15, -0.15):
        mujoco.mj_resetDataKeyframe(m, d, key)
        d.qpos[3:7] = (np.cos(YAW / 2), 0, 0, np.sin(YAW / 2))
        d.ctrl[:] = default_pose
        d.ctrl[1] += hr   # L hip roll
        d.ctrl[7] += hr   # R hip roll
        mujoco.mj_forward(m, d)
        for _ in range(500):   # 1.0 s
            mujoco.mj_step(m, d)
        print(f"hip_roll delta {hr:+.2f} -> base x {d.qpos[0]:+.3f}  "
              f"(want +x = over LEFT foot)")
    sys.exit(0)

WPS = json.loads(os.environ.get("WP", "[]"))
if not WPS:
    # the BEST ATTEMPT (renders assets/g1_oneleg_attempt.gif): hip-roll shift
    # -> fast lift to the 90-deg tuck with trunk counter-lean -> ~0.3 s
    # one-legged -> falls left (overcorrection; see docstring)
    WPS = [
        [0.5, {}, "mid", 0.0],
        [3.0, {"1": -0.16, "7": -0.16}, "L", 0.0],
        [3.8, {"6": -1.57, "9": 1.5, "10": -0.1, "7": 0.25, "1": -0.26, "13": 0.20},
         "L", 0.0],
        [9.0, {}, "L", 0.0],
    ]
KR, KP_, KD = 2.5, 2.5, 1.0
prev = default_pose.copy()
prev_t = 0.0
segs = []
for wp in WPS:
    t_end, deltas, com_over, lead = wp[:4]
    nopitch = bool(wp[4]) if len(wp) > 4 else False
    gate = wp[5] if len(wp) > 5 else None
    tgt = prev.copy()
    for idx, val in deltas.items():
        tgt[int(idx)] = val
    segs.append([t_end - prev_t, prev.copy(), np.clip(tgt, lo, hi), com_over, lead,
                 nopitch, gate])
    prev, prev_t = np.clip(tgt, lo, hi), t_end

VIDEO = os.environ.get("VIDEO")
rend = cam = None
frames = []
if VIDEO:
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
    rend = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance, cam.azimuth, cam.elevation = 2.4, 200, -12

d.ctrl[:] = default_pose
t, fail = 0.0, None
DBG = bool(os.environ.get("DBG"))
hold_ok = 0.0
seg_i, seg_t0 = 0, 0.0
total = sum(sg[0] for sg in segs)
while t < total + 2.0 and seg_i < len(segs):
    if True:
        if True:
            dur, c0, c1, com_over, lead, nopitch, gate = segs[seg_i]
            a = min(1.0, (t - seg_t0) / dur)
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
            grounded = [(hr, ap, ar) for hr, ap, ar, foot in
                        ((1, 4, 5, pl), (7, 10, 11, pr)) if foot[2] < 0.025]
            ARS = float(os.environ.get("ARS", "1"))
            SOFT = float(os.environ.get("SOFT_ANKLE", "0"))
            for hr, ap, ar in grounded:
                if SOFT and len(grounded) == 1:
                    m.actuator_gainprm[ar, 0] = SOFT
                    m.actuator_biasprm[ar, 1] = -SOFT
                    m.actuator_biasprm[ar, 2] = -2.0
                ctrl[ar] += droll * (ARS if len(grounded) == 1 else 1.0)
                ctrl[ap] += dpitch
                if len(grounded) == 1:
                    # single support: the stance HIP joins the balance loop
                    # (the ankle alone = +-0.26 rad clip = too little authority)
                    HRS = float(os.environ.get("HRS", "1"))
                    ctrl[hr] += HRS * float(np.clip(-1.8 * e_l + 1.2 * float(vel @ left),
                                                    -0.25, 0.25))
            d.ctrl[:] = np.clip(ctrl, lo, hi)
            adv = (t - seg_t0) >= dur
            if (gate == "shifted" and a > 0.25
                    and abs(e_l) < 0.025 and abs(float(vel @ left)) < 0.04):
                adv = True
                if DBG:
                    print(f"  [gate] shifted at t={t:.2f} e_l={e_l:+.3f}")
            if (gate == "shifting" and a > 0.25 and e_l < 0.035):
                adv = True
                if DBG:
                    print(f"  [gate] shifted at t={t:.2f} e_l={e_l:+.3f}")
            if adv and seg_i < len(segs) - 1:
                seg_i += 1
                seg_t0 = t
                segs[seg_i][1] = ctrl.copy()   # continuity from the live target
    for _ in range(10):
        mujoco.mj_step(m, d)
    t += 0.02
    w, x_, y_, z_ = d.qpos[3:7]
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y_ - z_ * x_), -1, 1)))
    roll = np.degrees(np.arctan2(2 * (w * x_ + y_ * z_), 1 - 2 * (x_**2 + y_**2)))
    pl, pr = d.geom_xpos[LF], d.geom_xpos[RF]
    com = d.subtree_com[1]
    # one-leg criterion: R foot high, L foot planted, upright
    if pr[2] > 0.15 and pl[2] < 0.06 and abs(roll) < 15 and abs(pitch) < 15:
        hold_ok += 0.02
    if DBG and abs(t - round(t * 5) / 5) < 1e-9:
        print(f"  t={t:4.1f} base=({d.qpos[0]:+.3f},{d.qpos[1]:+.3f},{d.qpos[2]:.3f}) "
              f"p={pitch:+5.1f} r={roll:+5.1f} "
              f"comx-Lfx={com[0]-pl[0]:+.3f} Rf_z={pr[2]:.3f} Lf_z={pl[2]:.3f}")
    if rend is not None and abs(t - round(t * 25) / 25) < 1e-9:
        rend.update_scene(d, camera=cam)
        from PIL import Image
        frames.append(Image.fromarray(rend.render())
                      .convert("P", palette=Image.ADAPTIVE, colors=128))
    if abs(roll) > 40 or abs(pitch) > 40:
        fail = f"fell t={t:.2f} (p={pitch:+.1f}, r={roll:+.1f})"
        break

print(f"FINAL: hold_one_leg={hold_ok:.2f}s  Rf_z={d.geom_xpos[RF][2]:.3f}")
print("RESULT:", fail if fail else (f"ONE-LEG STAND {hold_ok:.1f}s" if hold_ok > 2.0 else "incomplete"))
if VIDEO and frames:
    frames[0].save(VIDEO, save_all=True, append_images=frames[1:],
                   duration=40, loop=0, optimize=True)
    print("gif:", VIDEO, f"({len(frames)} frames)")
