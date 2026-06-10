"""Scripted SIT-ON-CHAIR baseline for the Unitree G1 — CPU-only, NO reinforcement learning.

A single linear ctrl interpolation (1.0 s) from the 'knees_bent' keyframe to a fixed seated
pose drops the pelvis onto the seat of the G1Sit chair task (g1_sit_env.build_plain_chair_model);
the seat physically catches the pelvis collision box and carries ~160 N (about half the robot's
327 N weight — the rest goes through the planted feet), giving a stable upright sit that holds
for the rest of the 6 s rollout. This makes RL OPTIONAL for the sit primitive.

The seated pose (found by a scripted sweep, see git history):
    hip_pitch = -1.254, knee = +1.611, ankle_pitch = +0.137, all other joints at the keyframe.

Two non-obvious ingredients, both discovered by failure analysis:
  1. ankle_pitch is commanded into PLANTARFLEXION (+0.137), NOT the feet-flat value
     -(hip+knee) = -0.357. The ankle servo is weak (kp=20 vs 75 for hip/knee); under gravity
     the knee overshoots its command and drags the weak ankle into dorsiflexion, the shank
     leans forward and the robot topples over its toes (every feet-flat candidate fell forward
     at +50..+85 deg pitch). The plantarflex bias is a torque knob: it presses the toes down
     and rocks the body BACKWARD onto the seat, which catches it.
  2. the descent is fast (1.0 s): the toe-pivot runaway needs ~0.7 s to develop, so the seat
     must catch the pelvis before that. Slow stately descents (1.5-3.0 s) all toppled.

Run (plain python, NOT mjpython; offscreen CGL rendering only):
    /Users/hoshinafumito/development/Colapis_project/MuJoCo-skills/.venv-rl/bin/python \
        training/g1_sit_scripted.py

Prints end-of-rollout metrics (pelvis z, pitch, seat contact, seat normal force, stable-hold
time) and writes a tracking-camera GIF to assets/g1_sit_scripted.gif.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

import g1_sit_env  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIF_PATH = os.path.join(REPO, "assets", "g1_sit_scripted.gif")

# --- the scripted seated pose (left/right symmetric) ---
HIP_PITCH = -1.254   # rad, thigh flexed forward/up
KNEE = +1.611        # rad, knee flexed
ANKLE_PITCH = +0.137  # rad, plantarflex BIAS (see module docstring, ingredient 1)
T_DESCENT = 1.0      # s, fast drop so the seat catches the pelvis (ingredient 2)

CTRL_DT = 0.02       # control step (10 x 0.002 s physics substeps)
SUBSTEPS = 10
TOTAL_T = 6.0        # rollout length
FPS = 20             # GIF frame rate

# ctrl indices (actuator order == joint order after the free joint)
L_HIP, L_KNEE, L_ANK = 0, 3, 4
R_HIP, R_KNEE, R_ANK = 6, 9, 10


def pitch_deg(d):
    w, x, y, z = d.qpos[3:7]
    return float(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))))


def main():
    m = g1_sit_env.build_plain_chair_model(sim_dt=0.002)
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.qpos[0:2] = g1_sit_env.SEAT_XY  # start with the pelvis above the seat center
    mujoco.mj_forward(m, d)

    seat_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "seat")
    pelv_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pelvis_collision")
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]

    default = d.qpos[7:].copy()
    target = default.copy()
    for hip_i, knee_i, ank_i in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        target[hip_i] = HIP_PITCH
        target[knee_i] = KNEE
        target[ank_i] = ANKLE_PITCH

    # offscreen renderer (CGL, plain python — never mujoco.viewer/mjpython on macOS)
    renderer = mujoco.Renderer(m, height=360, width=480)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance, cam.azimuth, cam.elevation = 2.6, 120, -10

    frames = []
    hist = []  # (t, pelvis_z, pitch_deg, seat_force_N, n_seat_contacts)
    f6 = np.zeros(6)
    n_steps = int(TOTAL_T / CTRL_DT)
    frame_every = max(1, int(round(1.0 / (FPS * CTRL_DT))))

    for k in range(n_steps):
        t = k * CTRL_DT
        alpha = min(1.0, t / T_DESCENT)
        d.ctrl[:] = np.clip(default + alpha * (target - default), lo, hi)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)

        force, ncon = 0.0, 0
        for i in range(d.ncon):
            c = d.contact[i]
            if {c.geom1, c.geom2} == {seat_gid, pelv_gid}:
                mujoco.mj_contactForce(m, d, i, f6)
                force += f6[0]  # normal component
                ncon += 1
        hist.append((t + CTRL_DT, d.qpos[2], pitch_deg(d), force, ncon))

        if k % frame_every == 0:
            renderer.update_scene(d, camera=cam)
            frames.append(Image.fromarray(renderer.render()))
    renderer.close()

    h = np.array(hist)
    sit_z = g1_sit_env.SIT_TARGET_Z
    good = (np.abs(h[:, 2]) < 30) & (np.abs(h[:, 1] - sit_z) < 0.08)
    hold = 0.0
    for g in good[::-1]:
        if not g:
            break
        hold += CTRL_DT
    last2 = h[h[:, 0] > TOTAL_T - 2.0]
    stable = bool(np.all(np.abs(last2[:, 2]) < 30)
                  and np.all(np.abs(last2[:, 1] - sit_z) < 0.08))
    pelvis_z = float(h[-1, 1])
    pitch = float(h[-1, 2])
    seat_force = float(h[-1, 3])
    on_seat = bool(h[-1, 4] > 0)
    success = stable and on_seat and seat_force > 80.0

    os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
    frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0)

    weight = float(np.sum(m.body_mass)) * 9.81
    print("=== G1 scripted sit-on-chair (no RL) ===")
    print(f"pose: hip_pitch={HIP_PITCH:+.3f} knee={KNEE:+.3f} ankle_pitch={ANKLE_PITCH:+.3f} "
          f"descent={T_DESCENT:.1f}s rollout={TOTAL_T:.1f}s")
    print(f"pelvis_z_final   = {pelvis_z:.4f}  (target {sit_z:.4f})")
    print(f"pitch_deg_final  = {pitch:+.2f}")
    print(f"on_seat          = {on_seat}  ({int(h[-1, 4])} seat-pelvis contacts)")
    print(f"seat_force_N     = {seat_force:.1f}  (robot weight {weight:.0f} N; "
          f"mean last 2 s {last2[:, 3].mean():.1f} N)")
    print(f"stable_last_2s   = {stable}   stable_hold = {hold:.2f} s")
    print(f"SUCCESS          = {success}")
    print(f"gif: {GIF_PATH}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
