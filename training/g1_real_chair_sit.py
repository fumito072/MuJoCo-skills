"""G1 sits on the REAL chair (３階講堂遠隔操作席, imported from FBX via the g1-isu
pipeline) starting ON its footrest platform. CPU-only macOS, no GPU, no cloud.

WHY PLATFORM-START: the seat top is 0.635 m (raycast-measured; chair_info says
0.648) while the G1's standing pelvis BOTTOM is 0.600 m — sitting from the floor
is geometrically impossible for this robot. The integrated footrest platform
(top 0.22 m, 0.6 x 0.3 m) is how the chair is meant to be used. Climbing onto it
is a separate, unsolved skill (Phase 3 in docs/AUTONOMOUS_SIT_PLAN.md).

WHY NO WALKING POLICY HERE (measured, 2026-06-11):
  - the 0.6 x 0.3 m platform cannot host the flat-terrain ONNX march: its
    cold-start scramble and correction steps walk off the edge (sweep: most
    spawns ended marching on the floor beside the chair);
  - every open-loop descent FROM a march instant failed: the legs never rest
    (min |leg qvel| ~0.6 rad/s even at "calm" footfalls) and the injected
    momentum rolls the robot off the cushion's curved front edge;
  - and none of that is needed: with SIT-mode stiff gains the static descent is
    SELF-FUNNELING — from any spawn in y 0.27..0.37, |dx| <= 5 cm, |dyaw| <= 8
    deg it converges to the SAME seated pose (12/12 probe sweep), because the
    stiff reference trajectory slides the pelvis ~20 cm deep onto the cushion
    regardless of where it started. Arrival uncertainty < descent basin =>
    no correction needed on the platform.

MODE-DEPENDENT GAINS (honest note): walking uses the playground gains the ONNX
policy was trained with (kp 75 legs / 20 ankles); this SIT mode stiffens the leg
actuators to kp=300, kd=8 — the same mechanism a real G1's mode controller (and
the g1-isu pretrained_walker gain swap, kp=500 for RL sitting) uses. Without it
the weak-ankle stand is a ~2 s forward-topple time bomb and the descent rolls
off sideways; with it the servos squash the momentum and the motion is
near-kinematic. No mass/inertia/gravity/contact change of any kind.

SEQUENCE (50 Hz ctrl, 500 Hz physics):
  STAND 0.4 s (stiff) -> DESCENT 0.6 s to the verified seated pose (hip -1.254,
  knee +1.611, ankle +0.137) -> SEAT-SETTLE 0.8 s -> POSE 1.5 s (ankle -> 0.0
  releases the toe press, waist_pitch -> -0.10 sets the torso upright) ->
  HOLD 3 s -> automatic verdict.

MEASURED RESULT (nominal): seated base z 0.800, pelvis pitch +0.1 deg, roll 0.0,
torso lean ~6 deg, chair support 120-260 N through pelvis box + both thighs
(feet excluded), armrests untouched. Same numbers across the whole spawn basin.

Modes:
  default            one rollout (spawn offset +3 cm, -2 cm, +6 deg), GIF
  G1RC_SWEEP=N       N randomized spawns (dx +-5 cm, y dock+[0.01,0.11],
                     dyaw +-8 deg), prints per-run lines + success rate
  G1RC_DEMOS=path    with G1RC_SWEEP: save successful rollouts (qpos, qvel,
                     ctrl @ 50 Hz) to <path>/demo_XXX.npz for the g1-isu RL
                     pipeline (state-level demos are gain-agnostic; ctrl here
                     is kp=300 targets — prefer state/DeepMimic-style tracking)
  G1RC_NOGIF=1, G1RC_DEBUG=1

Run:
    .venv-rl/bin/python training/g1_real_chair_sit.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

import g1_sit_env  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIF_PATH = os.path.join(REPO, "assets", "g1_real_chair_sit.gif")

CTRL_DT, SUBSTEPS = 0.02, 10
PLATFORM_TOP = 0.22
DOCK_Y = 0.26                 # descent reference; spawns land in y 0.27..0.37
YAW_TARGET = np.pi / 2        # facing chair +y (back to the seat)
SPAWN_NOISE_XY, SPAWN_NOISE_YAW = 0.05, np.deg2rad(8)

# SIT-mode gains
SIT_KP, SIT_KD = 300.0, 8.0

# the verified recipe (probe-swept on THIS chair, 2026-06-11)
HIP, KNEE, ANKLE = -1.254, +1.611, +0.137
ANKLE2, WAIST2 = 0.0, -0.10
T_STAND, T_DESC, T_SEAT, T_POSE, T_HOLD = 0.4, 0.6, 0.8, 1.5, 3.0
LEG_IDX = ((0, 3, 4), (6, 9, 10))
WAIST_PITCH_IDX = 14

# verdict (the honest measured seated state)
Z_RANGE = (0.70, 0.84)
PITCH_MAX, TORSO_LEAN_MAX, ROLL_MAX = 25.0, 12.0, 8.0
SEAT_FORCE_MIN = 60.0

DEBUG = bool(os.environ.get("G1RC_DEBUG"))
NO_GIF = bool(os.environ.get("G1RC_NOGIF")) or bool(os.environ.get("G1RC_SWEEP"))
SWEEP = int(os.environ.get("G1RC_SWEEP", "0"))
DEMOS = os.environ.get("G1RC_DEMOS")
FPS = 20


def rpy(d):
    w, x, y, z = d.qpos[3:7]
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    return float(roll), float(pitch)


def torso_lean_deg(m, d):
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    zax = d.xmat[tid].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arctan2(np.hypot(zax[0], zax[1]), zax[2])))


def rollout(m, spawn_dxy=(0.0, 0.0), spawn_dyaw=0.0, render=False, record=None):
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    mujoco.mj_resetDataKeyframe(m, d, key)
    yaw0 = YAW_TARGET + spawn_dyaw
    d.qpos[0:3] = (spawn_dxy[0], DOCK_Y + 0.04 + spawn_dxy[1], 0.755 + PLATFORM_TOP)
    d.qpos[3:7] = (np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2))
    mujoco.mj_forward(m, d)

    # SIT-mode stiff gains (restored at the end; m is shared across rollouts)
    leg_aids = list(range(12))
    orig_gain = m.actuator_gainprm[leg_aids].copy()
    orig_bias = m.actuator_biasprm[leg_aids].copy()
    for a in leg_aids:
        m.actuator_gainprm[a, 0] = SIT_KP
        m.actuator_biasprm[a, 1] = -SIT_KP
        m.actuator_biasprm[a, 2] = -SIT_KD

    default_pose = np.array(m.key_qpos[key][7:])
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    sit_t = default_pose.copy()
    pose_t = default_pose.copy()
    for hi_i, kn_i, an_i in LEG_IDX:
        sit_t[hi_i], sit_t[kn_i], sit_t[an_i] = HIP, KNEE, ANKLE
        pose_t[hi_i], pose_t[kn_i], pose_t[an_i] = HIP, KNEE, ANKLE2
    pose_t[WAIST_PITCH_IDX] = WAIST2
    sit_t, pose_t = np.clip(sit_t, lo, hi), np.clip(pose_t, lo, hi)

    renderer = cam = None
    frames = []
    if render:
        renderer = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        cam.distance, cam.azimuth, cam.elevation = 2.4, 155, -12
    frame_every = max(1, int(round(1.0 / (FPS * CTRL_DT))))

    rc_gids = {i for i in range(m.ngeom)
               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("rc_part")}
    foot_gids = {m.geom("left_foot").id, m.geom("right_foot").id}
    f6 = np.zeros(6)
    traj = {"qpos": [], "qvel": [], "ctrl": []} if record is not None else None

    T_TOTAL = T_STAND + T_DESC + T_SEAT + T_POSE + T_HOLD
    d.ctrl[:] = default_pose
    t, k, fail = 0.0, 0, None
    while t < T_TOTAL:
        if t < T_STAND:
            pass
        elif t < T_STAND + T_DESC:
            a = (t - T_STAND) / T_DESC
            d.ctrl[:] = default_pose + a * (sit_t - default_pose)
        elif t < T_STAND + T_DESC + T_SEAT:
            d.ctrl[:] = sit_t
        elif t < T_STAND + T_DESC + T_SEAT + T_POSE:
            a = (t - T_STAND - T_DESC - T_SEAT) / T_POSE
            d.ctrl[:] = sit_t + a * (pose_t - sit_t)
        else:
            d.ctrl[:] = pose_t
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        t += CTRL_DT
        k += 1
        if traj is not None:
            traj["qpos"].append(d.qpos.copy())
            traj["qvel"].append(d.qvel.copy())
            traj["ctrl"].append(d.ctrl.copy())
        roll, pitch = rpy(d)
        if DEBUG and k % 10 == 0:
            print(f"    DBG t={t:5.2f} base=({d.qpos[0]:+.3f},{d.qpos[1]:+.3f},"
                  f"{d.qpos[2]:.3f}) pitch={pitch:+6.1f} roll={roll:+6.1f}")
        if render and k % frame_every == 1:
            renderer.update_scene(d, camera=cam)
            frames.append(Image.fromarray(renderer.render())
                          .convert("P", palette=Image.ADAPTIVE, colors=128))
        if abs(roll) > 60 or d.qpos[2] < 0.35:
            fail = f"fell at t={t:.2f}s (z={d.qpos[2]:.3f}, roll={roll:+.1f})"
            break

    force_seat, ncon = 0.0, 0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in rc_gids or c.geom2 in rc_gids:
            other = c.geom2 if c.geom1 in rc_gids else c.geom1
            if other not in foot_gids:
                mujoco.mj_contactForce(m, d, i, f6)
                force_seat += f6[0]
                ncon += 1
    roll, pitch = rpy(d)
    tlean = torso_lean_deg(m, d)
    z = float(d.qpos[2])
    success = (fail is None and Z_RANGE[0] < z < Z_RANGE[1] and abs(pitch) < PITCH_MAX
               and tlean < TORSO_LEAN_MAX and abs(roll) < ROLL_MAX
               and force_seat > SEAT_FORCE_MIN)
    info = dict(z=z, pitch=pitch, roll=roll, torso_lean=tlean,
                seat_force=force_seat, ncon=ncon, fail=fail)
    if render and frames:
        os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
        frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:],
                       duration=int(1000 / FPS), loop=0, optimize=True)
        info["gif"], info["frames"] = GIF_PATH, len(frames)
    if renderer is not None:
        renderer.close()
    if traj is not None and success:
        np.savez_compressed(record, qpos=np.array(traj["qpos"]),
                            qvel=np.array(traj["qvel"]), ctrl=np.array(traj["ctrl"]),
                            dt=CTRL_DT, spawn_dxy=np.array(spawn_dxy),
                            spawn_dyaw=spawn_dyaw, sit_kp=SIT_KP, sit_kd=SIT_KD)
    # restore walk-mode gains (m shared across rollouts)
    m.actuator_gainprm[leg_aids] = orig_gain
    m.actuator_biasprm[leg_aids] = orig_bias
    return success, info


def main():
    m = g1_sit_env.build_fbx_chair_model(0.002)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720

    if SWEEP:
        rng = np.random.default_rng(0)
        n_ok = 0
        if DEMOS:
            os.makedirs(DEMOS, exist_ok=True)
        for i in range(SWEEP):
            dxy = np.array([rng.uniform(-SPAWN_NOISE_XY, SPAWN_NOISE_XY),
                            rng.uniform(-0.03, 0.07)])
            dyaw = rng.uniform(-SPAWN_NOISE_YAW, SPAWN_NOISE_YAW)
            rec = os.path.join(DEMOS, f"demo_{i:03d}.npz") if DEMOS else None
            ok, info = rollout(m, tuple(dxy), float(dyaw), record=rec)
            n_ok += ok
            print(f"run {i:03d}: spawn=({dxy[0]:+.3f},{dxy[1]:+.3f},"
                  f"{np.degrees(dyaw):+5.1f}deg) -> {'OK  ' if ok else 'FAIL'} "
                  f"z={info['z']:.3f} pitch={info['pitch']:+5.1f} "
                  f"torso={info['torso_lean']:+5.1f} seatF={info['seat_force']:5.0f}N"
                  + (f"  [{info['fail']}]" if info["fail"] else ""))
        print(f"\nSUCCESS RATE: {n_ok}/{SWEEP} = {100 * n_ok / SWEEP:.0f}%")
        return 0 if n_ok >= 0.9 * SWEEP else 1

    ok, info = rollout(m, (0.03, -0.02), np.deg2rad(6.0), render=not NO_GIF)
    print()
    print("=== G1 REAL-CHAIR sit from the footrest platform (stiff-mode descent) ===")
    print(f"seated base z        = {info['z']:.4f}  (expect {Z_RANGE[0]:.2f}..{Z_RANGE[1]:.2f})")
    print(f"pelvis pitch / roll  = {info['pitch']:+.1f} / {info['roll']:+.1f} deg")
    print(f"torso lean           = {info['torso_lean']:+.1f} deg")
    print(f"chair support force  = {info['seat_force']:.1f} N over {info['ncon']} contacts "
          f"(feet excluded)")
    if "gif" in info:
        print(f"gif: {info['gif']} ({info['frames']} frames)")
    print("RESULT:", "REAL-CHAIR SIT (platform start) ✓" if ok
          else f"FAILED — {info['fail'] or 'verdict thresholds not met'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
