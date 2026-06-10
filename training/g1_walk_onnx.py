"""Verify the OFFICIAL pretrained MuJoCo Playground 29-DOF G1 joystick policy (ONNX) on macOS,
CPU-only, in OUR chaired scene (g1_sit_env.build_plain_chair_model) — NVIDIA-free local inference.

Policy: models/policies/g1_joystick_29dof.onnx (input 'obs' [1,103], output 'continuous_actions'
[1,29]), vendored from google-deepmind/mujoco_playground experimental/sim2sim/onnx/g1_policy.onnx.

Obs recipe (EXACTLY the reference loader models/policies/play_g1_joystick_reference.py, which
matches the training env joystick.py _get_obs bit-for-bit at noise=0; same 103-dim layout as
training/g1_sit_play.py):
    [ local_linvel_pelvis(3), gyro_pelvis(3),
      gravity(3) = site_xmat('imu_in_pelvis').reshape(3,3).T @ [0,0,-1],
      command vx,vy,wz(3)  (raw units, NO scaling),
      qpos[7:] - default_pose(29), qvel[6:](29), last_action(29),
      phase [cos(p0), cos(p1), sin(p0), sin(p1)] ]   # p init [0, pi]
No external normalization — it is folded into the ONNX. Phase: gait_freq 1.5 Hz,
phase += 2*pi*1.5*0.02 AFTER each policy call, wrapped to (-pi, pi].
Action: ctrl = action * 0.5 + default_pose (knees_bent keyframe qpos[7:], POSITION actuators).
Rate: policy at 50 Hz (0.02 s), 10 mj_step substeps at 0.002 s.

COMMAND TRIM (measured, deploy-side shaping — the only deviation from the raw reference):
in plain-C MuJoCo this policy CREEPS FORWARD ~14 cm/s at command vx=0 (it must keep stepping —
any frozen phase, incl. the joystick.py NOTE's [pi,pi], topples it; that trick needs retraining).
Negative-vx commands are in-distribution (training lin_vel_x = [-1, 1]), so we cancel the bias:
    if user vx == 0:            command vx := -0.20      (calibrated creep trim)
    if user cmd == [0,0,0]:     command vy := +0.07      (calibrated lateral trim)
With the trim the robot marches in place at zero user command with ~1-2 cm/s drift.

Rollout (single episode; robot starts at (1.0, 0) heading +x so it never meets the seat at
(-0.08, 0)): (a) 4 s cmd [0.5,0,0] walk, (b) 3 s cmd [0,0,0] stand, (c) 2 s cmd [0,0,0.5] turn
in place, (d) 2 s cmd [0,0.3,0] lateral creep. Verifies zero contacts involve 'seat' or
'pelvis_collision'.

Run (plain python, NOT mjpython; offscreen CGL rendering only):
    /Users/hoshinafumito/development/Colapis_project/MuJoCo-skills/.venv-rl/bin/python \
        training/g1_walk_onnx.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402
import onnxruntime as rt  # noqa: E402
from PIL import Image  # noqa: E402

import g1_sit_env  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(REPO, "models", "policies", "g1_joystick_29dof.onnx")
GIF_PATH = os.path.join(REPO, "assets", "g1_walk_onnx.gif")

CTRL_DT = 0.02
SUBSTEPS = 10
ACTION_SCALE = 0.5
GAIT_FREQ = 1.5
FPS = 20
TRIM_VX = -0.20   # vx command at user-zero vx (cancels the measured +14 cm/s forward creep)
TRIM_VY = +0.07   # vy command at full-zero user command (cancels residual lateral creep)

# (duration_s, USER command [vx, vy, wz], tag)
PHASES = [
    (4.0, np.array([0.5, 0.0, 0.0]), "walk_fwd"),
    (3.0, np.array([0.0, 0.0, 0.0]), "stand"),
    (2.0, np.array([0.0, 0.0, 0.5]), "turn"),
    (2.0, np.array([0.0, 0.3, 0.0]), "lateral"),
]


def shape_command(user):
    """Apply the calibrated creep trim to a user (joystick) command."""
    cmd = user.astype(float).copy()
    if abs(user[0]) < 1e-6:
        cmd[0] = TRIM_VX
    if np.linalg.norm(user) < 1e-6:
        cmd[1] = TRIM_VY
    return cmd


def yaw_of(d):
    w, x, y, z = d.qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    m = g1_sit_env.build_plain_chair_model(sim_dt=0.002)
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.qpos[0:2] = (1.0, 0.0)  # away from the seat (-0.08, 0), heading +x
    mujoco.mj_forward(m, d)

    default_pose = np.array(m.key_qpos[key][7:])
    imu_site = m.site("imu_in_pelvis").id
    seat_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "seat")
    pelv_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pelvis_collision")
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]

    policy = rt.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    renderer = mujoco.Renderer(m, height=360, width=480)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance, cam.azimuth, cam.elevation = 2.6, 120, -10
    frame_every = max(1, int(round(1.0 / (FPS * CTRL_DT))))

    last_action = np.zeros(m.nu, dtype=np.float32)
    phase = np.array([0.0, np.pi])
    phase_dt = 2 * np.pi * GAIT_FREQ * CTRL_DT

    frames = []
    log = []  # t, tag, x, y, z, yaw, Lcontact, Rcontact
    chair_contacts = 0
    fell = False

    t = 0.0
    for dur, user_cmd, tag in PHASES:
        cmd = shape_command(user_cmd)
        for _ in range(int(round(dur / CTRL_DT))):
            # --- obs (reference recipe, no scaling) ---
            linvel = d.sensor("local_linvel_pelvis").data
            gyro = d.sensor("gyro_pelvis").data
            gravity = d.site_xmat[imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
            obs = np.hstack([
                linvel, gyro, gravity, cmd,
                d.qpos[7:] - default_pose, d.qvel[6:], last_action,
                np.concatenate([np.cos(phase), np.sin(phase)]),
            ]).astype(np.float32)

            action = policy.run(["continuous_actions"], {"obs": obs.reshape(1, -1)})[0][0]
            last_action = action.copy()
            d.ctrl[:] = np.clip(action * ACTION_SCALE + default_pose, lo, hi)
            phase = np.fmod(phase + phase_dt + np.pi, 2 * np.pi) - np.pi

            for _ in range(SUBSTEPS):
                mujoco.mj_step(m, d)
            t += CTRL_DT

            for i in range(d.ncon):
                c = d.contact[i]
                if seat_gid in (c.geom1, c.geom2) or pelv_gid in (c.geom1, c.geom2):
                    chair_contacts += 1

            log.append((
                t, tag, d.qpos[0], d.qpos[1], d.qpos[2], yaw_of(d),
                float(d.sensor("left_foot_floor_found").data[0] > 0),
                float(d.sensor("right_foot_floor_found").data[0] > 0),
            ))
            if d.qpos[2] < 0.35:
                fell = True

            if len(log) % frame_every == 1:
                renderer.update_scene(d, camera=cam)
                frames.append(Image.fromarray(renderer.render()))
    renderer.close()

    a = np.array([(r[0],) + tuple(r[2:]) for r in log])  # t,x,y,z,yaw,Lc,Rc
    tags = np.array([r[1] for r in log])

    def window(tag, last_s=None):
        s = a[tags == tag]
        if last_s is not None:
            s = s[s[:, 0] > s[-1, 0] - last_s]
        return s

    def body_frame_vel(s):
        """Mean displacement velocity over window s, in the mean-heading body frame."""
        yawm = np.unwrap(s[:, 4]).mean()
        dx, dy = s[-1, 1] - s[0, 1], s[-1, 2] - s[0, 2]
        T = s[-1, 0] - s[0, 0]
        return (np.cos(yawm) * dx + np.sin(yawm) * dy) / T, \
               (-np.sin(yawm) * dx + np.cos(yawm) * dy) / T

    # (a) forward walk: steady-state displacement speed over last 2 s
    w = window("walk_fwd", last_s=2.0)
    speed_fwd = float(np.hypot(w[-1, 1] - w[0, 1], w[-1, 2] - w[0, 2]) / (w[-1, 0] - w[0, 0]))

    # (b) stand: xy drift over last 2 s + stepping detection (floor-contact transitions)
    s = window("stand", last_s=2.0)
    drift_cm_s = float(np.hypot(s[-1, 1] - s[0, 1], s[-1, 2] - s[0, 2])
                       / (s[-1, 0] - s[0, 0]) * 100)
    steps_in_place = int(np.abs(np.diff(s[:, 5])).sum() + np.abs(np.diff(s[:, 6])).sum())
    stands = (not fell) and bool(np.all(window("stand")[:, 3] > 0.5))

    # (c) turn: yaw change (unwrapped)
    c = window("turn")
    dyaw_deg = float(np.degrees(np.unwrap(c[:, 4])[-1] - np.unwrap(c[:, 4])[0]))

    # (d) lateral: displacement velocity in the body frame
    d_fwd, d_lat = body_frame_vel(window("lateral"))

    final_z = float(a[-1, 3])
    walks = (not fell) and speed_fwd > 0.25 and final_z > 0.5

    os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
    frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0)

    print("=== G1 official joystick ONNX policy in OUR chaired scene ===")
    print(f"fell                = {fell}   final pelvis z = {final_z:.3f}")
    print(f"(a) fwd speed @cmd 0.5      = {speed_fwd:.3f} m/s (last 2 s)")
    print(f"(b) stand @cmd 0 (trimmed): drift = {drift_cm_s:.2f} cm/s (last 2 s), "
          f"contact transitions = {steps_in_place} "
          f"({'marches in place' if steps_in_place > 2 else 'feet planted'})")
    print(f"(c) turn @wz 0.5            = {dyaw_deg:+.1f} deg over 2 s (expect ~+57)")
    print(f"(d) lateral @vy 0.3         = {d_lat:+.3f} m/s lateral, {d_fwd:+.3f} m/s fwd")
    print(f"seat/pelvis_collision contacts during rollout = {chair_contacts}")
    print(f"gif: {GIF_PATH}")
    ok = (walks and stands and drift_cm_s < 5.0 and abs(dyaw_deg) > 25
          and d_lat > 0.05 and chair_contacts == 0)
    print(f"SUCCESS = {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
