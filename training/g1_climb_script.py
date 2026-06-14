"""SCRIPTED step-up demonstration (2026-06-15). After ~51 RL runs the floor->
platform CLIMB would not emerge from exploration (the robot never steps forward
onto the step). User's call: hand-script a rough climb, then use its trajectory as
RSI / imitation seeds so SAC only has to REFINE it (not discover it).

Open-loop joint-target waypoints under the SIT-mode stiff gains (kp300). Lead leg
= left (0-5), trail = right (6-11). The robot faces +x; the step is ahead in +x.
Phases: lean -> lift lead -> swing lead forward onto step -> shift weight forward
-> lift trail -> bring trail up -> settle. Tune the WP_* params, render, iterate.

  .venv-rl/bin/python training/g1_climb_script.py --step_h 0.08 --video assets/g1_climb_script.gif
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
from g1_climb_dr_env import (build_step_model, SIT_KP, SIT_KD, STEP_FRONT_X,
                             STEP_CENTER_X, CTRL_DT, SUBSTEPS)

# leg joint indices
LHP, LHR, LHY, LK, LAP, LAR = 0, 1, 2, 3, 4, 5
RHP, RHR, RHY, RK, RAP, RAR = 6, 7, 8, 9, 10, 11
WP, WR, WPI = 12, 13, 14  # waist yaw/roll/pitch


def _ramp(t, a, b, va, vb):
    if t <= a:
        return va
    if t >= b:
        return vb
    return va + (vb - va) * (t - a) / (b - a)


def climb_targets(t, dp, p):
    """Scripted step-up: lift lead -> place lead on step -> RISE (extend lead leg to
    push body up) + lift trail -> bring trail up -> settle. Lead=left, trail=right."""
    tgt = dp.copy()
    t1, t2, t3, t4, t5 = p["t_lift"], p["t_place"], p["t_rise"], p["t_trailup"], p["t_settle"]
    r = lambda a, b, va, vb: _ramp(t, a, b, va, vb)
    # gentle forward lean during lift (small — too much topples forward)
    lean = r(0, t2, 0, p["lean"]) + r(t3, t4, 0, -p["lean"])      # lean in, then back
    tgt[WPI] = dp[WPI] + lean
    # LEAD (left): lift up, place forward on step, then RISE (extend knee = push up)
    tgt[LHP] = dp[LHP] + r(0, t1, 0, p["lift_hip"]) + r(t1, t2, 0, p["place_hip"]) + r(t2, t3, 0, p["rise_hip"])
    tgt[LK] = dp[LK] + r(0, t1, 0, p["lift_knee"]) + r(t1, t2, 0, p["place_knee"]) + r(t2, t3, 0, p["rise_knee"])
    tgt[LAP] = dp[LAP] + r(t1, t2, 0, p["lead_ankle"]) + r(t2, t3, 0, p["rise_ankle"])
    # TRAIL (right): stays planted to push, then lifts during rise and comes up
    tgt[RK] = dp[RK] + r(t2, t3, 0, p["trail_lift_knee"]) + r(t3, t4, 0, p["trail_swing_knee"])
    tgt[RHP] = dp[RHP] + r(t2, t3, 0, p["trail_lift_hip"]) + r(t3, t4, 0, p["trail_swing_hip"])
    tgt[RAP] = dp[RAP] + r(t3, t4, 0, p["lead_ankle"])
    # settle both legs back toward default (standing) on the step
    for j in (LHP, LK, LAP, RHP, RK, RAP, WPI):
        tgt[j] = dp[j] + r(t4, t5, tgt[j] - dp[j], 0)
    return tgt


DEFAULT_P = dict(
    t_lift=0.6, t_place=1.2, t_rise=2.0, t_trailup=2.8, t_settle=3.4,
    lean=0.12, lift_hip=-0.55, lift_knee=0.6, place_hip=-0.15, place_knee=-0.85,
    lead_ankle=0.2, rise_hip=0.35, rise_knee=-0.45, rise_ankle=-0.1,
    trail_lift_knee=0.5, trail_lift_hip=-0.4, trail_swing_knee=-0.6, trail_swing_hip=0.1,
)


def run_climb(p, step_h=0.08, seconds=3.6, render=False, collect=False):
    m = build_step_model(step_h)
    for a in range(12):
        m.actuator_gainprm[a, 0] = SIT_KP
        m.actuator_biasprm[a, 1] = -SIT_KP
        m.actuator_biasprm[a, 2] = -SIT_KD
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    dp = np.array(m.key_qpos[key][7:])
    lo, hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
    lf, rf, stepg, floor = (m.geom("left_foot").id, m.geom("right_foot").id,
                            m.geom("step").id, m.geom("floor").id)
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.qpos[0] = 0.0
    d.ctrl[:] = dp
    mujoco.mj_forward(m, d)

    def feet_on_step():
        n = 0
        for i in range(d.ncon):
            c = d.contact[i]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in (lf, rf) and o == stepg and c.pos[2] > step_h - 0.02:
                    n += 1
                    break
        return n

    frames, traj = [], []
    rend = cam = None
    if render:
        m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
        rend = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        cam.distance, cam.azimuth, cam.elevation = 2.2, 90, -8
    from PIL import Image
    n_steps = int(seconds / CTRL_DT)
    max_feet, fell = 0, False
    for k in range(n_steps):
        t = k * CTRL_DT
        d.ctrl[:] = np.clip(climb_targets(t, dp, p), lo, hi)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        max_feet = max(max_feet, feet_on_step())
        if collect and feet_on_step() >= 1:
            traj.append((d.qpos.copy(), d.qvel.copy()))
        if render:
            rend.update_scene(d, camera=cam)
            frames.append(Image.fromarray(rend.render()).convert("RGB")
                          .resize((400, 300), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=48))
        w, x_, y_, z_ = d.qpos[3:7]
        roll = np.degrees(np.arctan2(2 * (w * x_ + y_ * z_), 1 - 2 * (x_**2 + y_**2)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (w * y_ - z_ * x_), -1, 1)))
        if d.qpos[2] < 0.45 or abs(roll) > 55 or abs(pitch) > 55:
            fell = True
            break
    final_feet = feet_on_step()
    w, x_, y_, z_ = d.qpos[3:7]
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y_ - z_ * x_), -1, 1)))
    res = dict(max_feet=max_feet, final_feet=final_feet, fell=fell,
               base_x=float(d.qpos[0]), base_y=float(d.qpos[1]), base_z=float(d.qpos[2]),
               pitch=float(pitch))
    if render and frames:
        os.makedirs(os.path.dirname(rend_path := os.environ.get("GIF", "assets/g1_climb_script.gif")), exist_ok=True)
        fr = frames[::2]
        fr[0].save(rend_path, save_all=True, append_images=fr[1:], duration=80, loop=0, optimize=True)
        res["gif"] = rend_path
    if collect:
        res["traj"] = traj
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step_h", type=float, default=0.08)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    if args.video:
        os.environ["GIF"] = args.video
    res = run_climb(DEFAULT_P, step_h=args.step_h, render=bool(args.video))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "traj"})


if __name__ == "__main__":
    raise SystemExit(main())
