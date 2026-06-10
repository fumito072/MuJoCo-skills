"""END-TO-END autonomous G1 demo: WALK -> AVOID (raycast VFH) -> ALIGN -> SIT on a seat.
One model, one continuous rollout, one video. CPU-only on macOS (plain python, offscreen CGL
rendering; NEVER mujoco.viewer / mjpython).

Pipeline (FSM at 50 Hz):
  NAVIGATE  official pretrained 29-DOF joystick ONNX walk (models/policies/g1_joystick_29dof.onnx,
            obs recipe + calibrated creep trim verified in training/g1_walk_onnx.py), steered by
            the stateless raycast-VFH planner from the mujoco-obstacle-navigation skill
            (21 mj_ray casts, 200 deg FOV, rays masked to obstacle geom group 4 only),
            goal = seat center. Exits when dist(base, seat) < 0.6 m.
  ALIGN     same ONNX walk; command = body-frame P-servo on (seat_xy - base_xy) PLUS a yaw
            servo wz = clip(-1.2*yaw, +-0.5) toward the seat-facing heading (the sit basin
            is +-45 deg about yaw=0 — attempt #1 docked at 2.5 cm but yaw +64 deg and
            toppled), on top of the calibrated stationary trim (vx -0.20, vy +0.07).
            Docking tolerance 0.055 m / 25 deg =~ 70%/55% of the measured basin.
  SETTLE    trimmed zero command for 1.0 s; verify drift < 0.15 m/s and error still in tolerance
            (else back to ALIGN, up to 3 retries), then wait (<= 2 s) for a BOTH-FEET-DOWN
            sway-calm instant of the march (low lateral velocity alone is NOT enough —
            attempt #2 cut mid-march and rolled over sideways within 0.4 s of the descent).
  FREEZE    STOP querying the ONNX; ramp d.ctrl to the keyframe stand (0.4 s), settle (0.6 s),
            then verify a statically stable stand (both feet down, |v|<0.10 m/s, |roll|<3 deg,
            |pitch|<10 deg, still inside the basin). This reproduces EXACTLY the verified
            initial condition of the scripted sit. On failure: re-engage the ONNX from its
            verified cold-start state (phase=[0,pi], last_action=0) and retry ALIGN.
  SIT       linear d.ctrl ramp (1.0 s) from the keyframe stand to the verified scripted seated
            pose (hip_pitch=-1.254, knee=+1.611, ankle_pitch=+0.137 — the deliberate
            plantarflex bias; see training/g1_sit_scripted.py), then hold.
  HOLD      3 s; final seated metrics (pelvis z vs SIT_TARGET_Z, pitch, seat contact + normal
            force via mj_contactForce).

Scene: playground G1 feetonly flat terrain + the g1_sit_env chair geoms (seat moved to world
(3.5, 0), pelvis_collision box identical) + 3 obstacles (group 4, contype/conaffinity 1) that
block the straight start->seat line so VFH must steer.

HONEST collision caveats (verified model facts): the G1 feet have contype=0 and collide ONLY via
explicit floor<->feet <pair> elements, and shins/thighs have no collision geoms — so the FEET
pass through the seat and the obstacles; only the pelvis box is physical to them. Obstacle
avoidance is therefore enforced by the raycast VFH (and checked with a min-clearance metric),
not by full-body contact. The seat catches the pelvis box during the SIT descent exactly as in
the verified scripted sit.

Hard rules respected: no qpos teleports after t=0, no welds/mocap/xfrc/gravity edits.

Run:
    /Users/hoshinafumito/development/Colapis_project/MuJoCo-skills/.venv-rl/bin/python \
        training/g1_walk_to_sit.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402
import onnxruntime as rt  # noqa: E402
from PIL import Image  # noqa: E402

import g1_sit_env  # noqa: E402  (chair constants — single source of truth)
from mujoco_playground._src.locomotion.g1 import base as g1_base  # noqa: E402
from mujoco_playground._src.locomotion.g1 import g1_constants as consts  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(REPO, "models", "policies", "g1_joystick_29dof.onnx")
GIF_PATH = os.path.join(REPO, "assets", "g1_walk_to_sit.gif")

# --- timing ---
CTRL_DT, SUBSTEPS = 0.02, 10          # 50 Hz policy, 10 x 0.002 s physics substeps
ACTION_SCALE, GAIT_FREQ = 0.5, 1.5
MAX_T = 45.0                          # hard rollout cap (timeout -> honest failure)
NAV_TIMEOUT = 30.0
WALK_GUARD_T = 40.0                   # must have started SIT by here

# --- calibrated command trim (measured; see training/g1_walk_onnx.py docstring) ---
TRIM_VX, TRIM_VY = -0.20, +0.07

# --- scene ---
SEAT_WORLD = np.array([3.5, 0.0])
OBST_GROUP = 4
# (name, x, y, kind, sx, sy, half_h, rgba). Heights >= 1.2 m so the z=0.5 rays see them.
OBSTACLES = [
    ("obs0", 1.5, 0.40, "cyl", 0.30, 0.0, 0.75, (0.85, 0.30, 0.15, 1.0)),
    ("obs1", 2.7, -0.65, "cyl", 0.30, 0.0, 0.75, (0.20, 0.45, 0.85, 1.0)),
    ("obs2", 2.0, 1.20, "box", 0.30, 0.30, 0.60, (0.90, 0.72, 0.15, 1.0)),
]

# --- VFH planner (params copied from the verified g1_nav_demo.py) ---
N_RAYS, FOV, RMAX, SAFE = 21, np.deg2rad(200), 4.0, 1.1
ANGLES = np.array([-FOV / 2 + FOV * i / (N_RAYS - 1) for i in range(N_RAYS)])
WIDEN = 3   # blocked-sector widening (+-WIDEN); was +-2 — the pelvis box (circumradius
            # ~0.14 m) scraped an obstacle at 0.07 m base clearance with +-2

# Docking NOMINAL: the 0.08 m basin was measured around base = seat_center - SEAT_XY
# (g1_sit_env puts the seat at (-0.08, 0) with the robot at the origin). Attempt #2 docked
# at the seat CENTER = -8 cm in x from nominal — exactly the one fragile basin direction.
DOCK_WORLD = SEAT_WORLD - np.array(g1_sit_env.SEAT_XY)   # = SEAT_WORLD + (0.08, 0)

# --- FSM tuning ---
NAV_EXIT_DIST = 0.6
ALIGN_TOL = 0.055        # ~70% of the measured 0.08 m all-direction docking basin
ALIGN_DEADBAND = 0.030   # inside this, stop chasing the march-in-place wobble
ALIGN_HOLD_S = 0.5
# Yaw servo: the 0.08 m / +-45 deg basin was measured with the robot facing the seat's +x.
# First E2E attempt docked at 2.5 cm but yaw +64 deg (VFH detour heading, never corrected,
# and it kept drifting during ALIGN/SETTLE) -> toppled. ALIGN must servo yaw -> 0 too.
YAW_TOL = np.deg2rad(25)        # ALIGN/SETTLE exit gate (~55% of the +-45 deg basin)
YAW_DEADBAND = np.deg2rad(8)    # inside this, stop chasing the march wobble
YAW_HARD = np.deg2rad(45)       # never start the descent outside the measured basin
SETTLE_S = 1.0
SETTLE_SPEED_MAX = 0.15
MAX_SETTLE_RETRIES = 3
# Sit-entry gate: cut to the scripted descent only at a SWAY-CALM instant of the march.
# (Measured: double-stance instants are the weight-transfer instants — body-lateral velocity
# peaks there at ~0.15-0.2 m/s, and entering the descent with that sway rolled the robot off
# the seat sideways. Calm instants recur every ~0.66 s gait cycle.)
CALM_WAIT_MAX = 2.0      # max wait for a calm instant; on timeout cut anyway (noted)
CALM_VX, CALM_VY = 0.12, 0.06   # m/s body-frame pelvis velocity bounds
CALM_GYRO = 0.5          # rad/s |gyro x|, |gyro y| bound
CALM_ROLL = 2.0          # deg |roll| bound

# FREEZE: the verified scripted descent starts from the STATIC symmetric keyframe stand.
# Attempt #2 cut from the march straight into the descent and rolled over sideways within
# 0.4 s (mid-march legs = asymmetric support; roll -1 -> +24 -> +86 deg, 0.8 m sideways
# slide past the seat). So: at a calm BOTH-FEET-DOWN instant, ramp d.ctrl to the keyframe
# stand (FREEZE_RAMP_S), let it settle (rest of FREEZE_S), verify static stability, and
# only then descend — the descent initial condition is then EXACTLY the verified one.
# If the stand check fails, re-engage the ONNX from its verified cold-start condition
# (phase=[0,pi], last_action=0 — how every walk rollout begins) and retry ALIGN.
FREEZE_RAMP_S = 0.4
FREEZE_WAIT_MAX = 3.0            # after the ramp, POLL every tick for the first stable
                                 # instant (the cut leaves a decaying lateral sway — a
                                 # fixed 1.0 s check landed mid-oscillation and failed)
FREEZE_SPEED_MAX = 0.10          # m/s |pelvis local vx|,|vy|
FREEZE_ROLL_MAX, FREEZE_PITCH_MAX = 3.0, 10.0   # deg

# --- scripted sit (verified in g1_sit_scripted.py; basin swept to 0.08 m / +-45 deg) ---
HIP_PITCH, KNEE, ANKLE_PITCH = -1.254, +1.611, +0.137
T_DESCENT, T_HOLD = 1.0, 3.0
LEG_IDX = ((0, 3, 4), (6, 9, 10))    # (hip_pitch, knee, ankle_pitch) ctrl indices, L / R

FPS = 20
DEBUG = bool(os.environ.get("G1W2S_DEBUG"))     # per-0.1s SIT/HOLD telemetry
NO_GIF = bool(os.environ.get("G1W2S_NOGIF"))    # skip rendering (fast iteration)


def yaw_of(d):
    w, x, y, z = d.qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def pitch_deg(d):
    w, x, y, z = d.qpos[3:7]
    return float(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))))


def roll_deg(d):
    w, x, y, z = d.qpos[3:7]
    return float(np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))))


def build_scene(sim_dt=0.002):
    """Playground G1 feetonly flat terrain + chair geoms (seat at SEAT_WORLD) + obstacles."""
    assets = g1_base.get_assets()
    spec = mujoco.MjSpec.from_string(consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets)

    # seat — same dims/rgba as g1_sit_env, relocated to SEAT_WORLD
    seat = spec.worldbody.add_geom()
    seat.name = "seat"
    seat.type = mujoco.mjtGeom.mjGEOM_BOX
    seat.size = list(g1_sit_env.SEAT_HALF)
    seat.pos = [SEAT_WORLD[0], SEAT_WORLD[1], g1_sit_env.SEAT_TOP - g1_sit_env.SEAT_HALF[2]]
    seat.rgba = [0.55, 0.38, 0.22, 1.0]
    seat.contype, seat.conaffinity = 1, 1

    # pelvis collision box — exactly as g1_sit_env._add_chair_geoms
    pelvis = [b for b in spec.bodies if b.name == "pelvis"][0]
    pc = pelvis.add_geom()
    pc.name = "pelvis_collision"
    pc.type = mujoco.mjtGeom.mjGEOM_BOX
    pc.size = list(g1_sit_env.PELVIS_COL_SIZE)
    pc.pos = list(g1_sit_env.PELVIS_COL_POS)
    pc.rgba = [0.8, 0.2, 0.2, 0.4]
    pc.contype, pc.conaffinity = 1, 1

    # obstacles — geom group 4 (the VFH rays are masked to this group), physical to the pelvis
    for name, ox, oy, kind, sx, sy, hh, rgba in OBSTACLES:
        g = spec.worldbody.add_geom()
        g.name = name
        g.group = OBST_GROUP
        g.rgba = list(rgba)
        g.contype, g.conaffinity = 1, 1
        if kind == "cyl":
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = [sx, hh, 0]
        else:
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [sx, sy, hh]
        g.pos = [ox, oy, hh]

    spec.assets = assets   # REQUIRED for mesh resolution
    m = spec.compile()
    m.opt.timestep = sim_dt
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
    return m


def make_planner(m, d, goal):
    """Raycast VFH (copied from the mujoco-obstacle-navigation skill), goal-parameterized."""
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[OBST_GROUP] = 1            # rays hit ONLY obstacle geoms (never the robot/seat)
    gid = np.zeros(1, dtype=np.int32)

    def plan():
        origin = np.array([d.qpos[0], d.qpos[1], 0.5])
        yaw = yaw_of(d)
        rng = np.full(N_RAYS, RMAX)
        for i, a in enumerate(ANGLES):
            vec = np.array([np.cos(yaw + a), np.sin(yaw + a), 0.0])
            dist = mujoco.mj_ray(m, d, origin, vec, geomgroup, 1, -1, gid)
            if dist >= 0:
                rng[i] = min(dist, RMAX)
        dx, dy = goal[0] - d.qpos[0], goal[1] - d.qpos[1]
        goal_dir = (np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(goal_dir) > np.pi / 2:
            # goal behind the fan: turn toward it in place instead of orbiting at speed
            return 0.0, float(np.clip(1.5 * goal_dir, -0.6, 0.6))
        blocked = rng < SAFE
        blk = blocked.copy()
        for i in range(N_RAYS):
            if blocked[i]:
                for j in range(i - WIDEN, i + WIDEN + 1):
                    if 0 <= j < N_RAYS:
                        blk[j] = True
        free = np.where(~blk)[0]
        if len(free) == 0:
            return 0.0, 0.6              # boxed in: turn in place to search
        best = free[np.argmin(np.abs(ANGLES[free] - goal_dir))]
        steer = ANGLES[best]
        vx = 0.5 * float(np.clip(rng[best] / RMAX, 0.3, 1.0))
        wz = float(np.clip(1.5 * steer, -0.6, 0.6))
        return vx, wz
    return plan


def main():
    m = build_scene()
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.qpos[0:2] = (0.0, 0.0)             # start at origin, facing +x (setup only, t=0)
    mujoco.mj_forward(m, d)

    default_pose = np.array(m.key_qpos[key][7:])
    imu_site = m.site("imu_in_pelvis").id
    seat_gid = m.geom("seat").id
    pelv_gid = m.geom("pelvis_collision").id
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]

    sit_target = default_pose.copy()
    for hip_i, knee_i, ank_i in LEG_IDX:
        sit_target[hip_i], sit_target[knee_i], sit_target[ank_i] = HIP_PITCH, KNEE, ANKLE_PITCH
    sit_target = np.clip(sit_target, lo, hi)

    policy = rt.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    plan = make_planner(m, d, DOCK_WORLD)

    renderer = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance, cam.azimuth, cam.elevation = 3.0, 135, -15
    vopt = mujoco.MjvOption()
    vopt.geomgroup[OBST_GROUP] = 1       # render the obstacles (group 4)
    frame_every = max(1, int(round(1.0 / (FPS * CTRL_DT))))
    frames = []

    last_action = np.zeros(m.nu, dtype=np.float32)
    phase = np.array([0.0, np.pi])       # init [0, pi]; NEVER freeze (topples the policy)
    phase_dt = 2 * np.pi * GAIT_FREQ * CTRL_DT
    f6 = np.zeros(6)

    state = "NAVIGATE"
    timeline = [f"[t=  0.00s] NAVIGATE  start (0.00,0.00) -> dock point "
                f"({DOCK_WORLD[0]:.2f},{DOCK_WORLD[1]:.2f}) (seat center "
                f"{SEAT_WORLD[0]:.2f},{SEAT_WORLD[1]:.2f})"]
    print(timeline[0])
    durations = {}
    navigated = False
    fell, fail = False, None
    min_clear = float("inf")
    clear_per_obs = {name: float("inf") for name, *_ in OBSTACLES}
    align_hold = 0.0
    settle_t0, settle_xy0 = None, None
    waiting_calm, calm_t0 = False, 0.0
    freeze_t0, freeze_ctrl0 = None, None
    retries = 0
    sit_t0, sit_ctrl0, hold_t0 = None, None, None
    sit_yaw_deg, sit_err_cm = None, None
    seat_hist = []                       # (t, z, pitch_deg, seat_force_N, ncon) in SIT/HOLD
    t, k = 0.0, 0

    def goto(new, why):
        nonlocal state
        msg = f"[t={t:6.2f}s] {state} -> {new}  ({why})"
        print(msg)
        timeline.append(msg)
        state = new

    while t < MAX_T:
        # --- pre-step pose / errors ---
        x, y = float(d.qpos[0]), float(d.qpos[1])
        yaw = yaw_of(d)
        exw, eyw = DOCK_WORLD[0] - x, DOCK_WORLD[1] - y
        dist = float(np.hypot(exw, eyw))
        ex = np.cos(yaw) * exw + np.sin(yaw) * eyw      # error in the base yaw frame
        ey = -np.sin(yaw) * exw + np.cos(yaw) * eyw

        # --- control ---
        if state in ("NAVIGATE", "ALIGN", "SETTLE"):
            if state == "NAVIGATE":
                vx, wz = plan()
                vx *= float(np.clip(dist / 1.5, 0.30, 1.0))   # approach slowdown (curvature)
                cmd = np.array([vx, 0.0, wz])
                if DEBUG and k % 25 == 0:
                    print(f"    DBG t={t:6.2f} NAV pos=({x:+.2f},{y:+.2f}) "
                          f"yaw={np.degrees(yaw):+5.0f} dist={dist:.2f} "
                          f"cmd=({vx:+.2f},{wz:+.2f})")
            elif state == "ALIGN":
                if dist < ALIGN_DEADBAND:
                    svx = svy = 0.0
                else:
                    svx = float(np.clip(1.2 * ex, -0.35, 0.35))
                    svy = float(np.clip(1.2 * ey, -0.25, 0.25))
                swz = (0.0 if abs(yaw) < YAW_DEADBAND
                       else float(np.clip(-1.2 * yaw, -0.5, 0.5)))
                cmd = np.array([TRIM_VX + svx, TRIM_VY + svy, swz])
            else:  # SETTLE — calibrated stationary march-in-place
                cmd = np.array([TRIM_VX, TRIM_VY, 0.0])

            # 103-dim obs, EXACTLY the verified recipe (g1_walk_onnx.py)
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
        elif state == "FREEZE":
            alpha = min(1.0, (t - freeze_t0) / FREEZE_RAMP_S)
            d.ctrl[:] = np.clip(freeze_ctrl0 + alpha * (default_pose - freeze_ctrl0), lo, hi)
        elif state == "SIT":
            alpha = min(1.0, (t - sit_t0) / T_DESCENT)
            d.ctrl[:] = np.clip(sit_ctrl0 + alpha * (sit_target - sit_ctrl0), lo, hi)
        else:  # HOLD
            d.ctrl[:] = sit_target

        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        t += CTRL_DT
        k += 1
        durations[state] = durations.get(state, 0.0) + CTRL_DT

        # --- post-step metrics ---
        x, y, z = float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2])
        pitch = pitch_deg(d)
        yaw = yaw_of(d)
        dist = float(np.hypot(DOCK_WORLD[0] - x, DOCK_WORLD[1] - y))

        if state in ("NAVIGATE", "ALIGN", "SETTLE", "FREEZE"):
            for name, ox, oy, kind, sx, sy, hh, rgba in OBSTACLES:
                r = float(np.hypot(sx, sy)) if kind == "box" else sx   # circumscribed radius
                c = float(np.hypot(x - ox, y - oy)) - r
                clear_per_obs[name] = min(clear_per_obs[name], c)
                if state == "NAVIGATE":
                    min_clear = min(min_clear, c)
            if z < 0.4 or abs(pitch) > 60:
                fell = True
                fail = f"fell during {state} (pelvis_z={z:.3f}, pitch={pitch:+.1f} deg)"
            if DEBUG and state == "SETTLE":
                lv = d.sensor("local_linvel_pelvis").data
                gy = d.sensor("gyro_pelvis").data
                lf = d.sensor("left_foot_floor_found").data[0] > 0
                rf = d.sensor("right_foot_floor_found").data[0] > 0
                print(f"    DBG t={t:6.2f} SETTLE lvel=({lv[0]:+.2f},{lv[1]:+.2f}) "
                      f"gyro=({gy[0]:+.2f},{gy[1]:+.2f}) roll={roll_deg(d):+5.1f} "
                      f"L={int(lf)} R={int(rf)}")
        else:
            force, ncon = 0.0, 0
            for i in range(d.ncon):
                c = d.contact[i]
                if {c.geom1, c.geom2} == {seat_gid, pelv_gid}:
                    mujoco.mj_contactForce(m, d, i, f6)
                    force += f6[0]       # normal component
                    ncon += 1
            seat_hist.append((t, z, pitch, force, ncon))
            if DEBUG and k % 5 == 0:
                px, py, pz = d.geom_xpos[pelv_gid]
                print(f"    DBG t={t:6.2f} {state:4s} relseat=({px - SEAT_WORLD[0]:+.3f},"
                      f"{py - SEAT_WORLD[1]:+.3f}) base_z={z:.3f} pbox_z={pz:.3f} "
                      f"pitch={pitch:+6.1f} roll={roll_deg(d):+6.1f} "
                      f"F={force:6.1f} ncon={ncon}")

        if not NO_GIF and k % frame_every == 1:
            renderer.update_scene(d, camera=cam, scene_option=vopt)
            frames.append(Image.fromarray(renderer.render())
                          .convert("P", palette=Image.ADAPTIVE, colors=128))

        # --- transitions ---
        if fail:
            msg = f"[t={t:6.2f}s] ABORT: {fail}"
            print(msg)
            timeline.append(msg)
            break

        if state in ("NAVIGATE", "ALIGN", "SETTLE", "FREEZE") and t > WALK_GUARD_T:
            fail = (f"did not reach SIT by t={WALK_GUARD_T:.0f}s "
                    f"(state={state}, dist={dist:.2f} m)")
            timeline.append(f"[t={t:6.2f}s] ABORT: {fail}")
            print(timeline[-1])
            break

        if state == "NAVIGATE":
            if dist < NAV_EXIT_DIST:
                navigated = True
                goto("ALIGN", f"dist={dist:.2f} m < {NAV_EXIT_DIST:.1f} m")
            elif t > NAV_TIMEOUT:
                fail = f"NAVIGATE timeout (dist still {dist:.2f} m at t={t:.1f}s)"
                timeline.append(f"[t={t:6.2f}s] ABORT: {fail}")
                print(timeline[-1])
                break
        elif state == "ALIGN":
            align_hold = (align_hold + CTRL_DT
                          if dist < ALIGN_TOL and abs(yaw) < YAW_TOL else 0.0)
            if align_hold >= ALIGN_HOLD_S:
                align_hold = 0.0
                settle_t0, settle_xy0 = t, np.array([x, y])
                waiting_calm = False
                goto("SETTLE", f"|err|={dist * 100:.1f} cm < {ALIGN_TOL * 100:.1f} cm "
                               f"held {ALIGN_HOLD_S:.1f}s; yaw={np.degrees(yaw):+.0f} deg")
        elif state == "SETTLE":
            if not waiting_calm and t - settle_t0 >= SETTLE_S:
                speed = float(np.linalg.norm(np.array([x, y]) - settle_xy0)
                              / (t - settle_t0))
                if (speed < SETTLE_SPEED_MAX and dist < ALIGN_TOL
                        and abs(yaw) < YAW_TOL):
                    waiting_calm, calm_t0 = True, t
                elif retries < MAX_SETTLE_RETRIES:
                    retries += 1
                    goto("ALIGN", f"settle check failed (drift={speed:.2f} m/s, "
                                  f"err={dist * 100:.1f} cm, yaw={np.degrees(yaw):+.0f} deg) "
                                  f"— retry {retries}/{MAX_SETTLE_RETRIES}")
                elif dist < 0.08 and abs(yaw) < YAW_HARD:   # hard basin budget — warn & go
                    waiting_calm, calm_t0 = True, t
                    msg = (f"[t={t:6.2f}s] WARN: settle retries exhausted, err="
                           f"{dist * 100:.1f} cm / yaw {np.degrees(yaw):+.0f} deg within "
                           f"the 8 cm / 45 deg hard budget — proceeding")
                    print(msg)
                    timeline.append(msg)
                else:
                    fail = (f"settle retries exhausted, err={dist * 100:.1f} cm / "
                            f"yaw {np.degrees(yaw):+.0f} deg outside the "
                            f"8 cm / +-45 deg docking basin")
                    timeline.append(f"[t={t:6.2f}s] ABORT: {fail}")
                    print(timeline[-1])
                    break
            if waiting_calm:
                lv = d.sensor("local_linvel_pelvis").data
                gy = d.sensor("gyro_pelvis").data
                lf = d.sensor("left_foot_floor_found").data[0] > 0
                rf = d.sensor("right_foot_floor_found").data[0] > 0
                calm = (lf and rf
                        and abs(lv[0]) < CALM_VX and abs(lv[1]) < CALM_VY
                        and abs(gy[0]) < CALM_GYRO and abs(gy[1]) < CALM_GYRO
                        and abs(roll_deg(d)) < CALM_ROLL)
                timed_out = t - calm_t0 >= CALM_WAIT_MAX
                if ((calm or timed_out) and abs(yaw) > YAW_HARD
                        and retries < MAX_SETTLE_RETRIES):
                    retries += 1     # yaw drifted out of the basin during the calm wait
                    goto("ALIGN", f"yaw {np.degrees(yaw):+.0f} deg drifted outside the "
                                  f"+-45 deg basin during calm wait — retry "
                                  f"{retries}/{MAX_SETTLE_RETRIES}")
                elif calm or timed_out:
                    freeze_t0, freeze_ctrl0 = t, d.ctrl.copy()
                    goto("FREEZE", f"{'both-feet sway-calm' if calm else 'calm-wait TIMEOUT'} "
                                   f"(lvel=({lv[0]:+.2f},{lv[1]:+.2f}), roll={roll_deg(d):+.1f} deg, "
                                   f"L={int(lf)} R={int(rf)}), err={dist * 100:.1f} cm, "
                                   f"yaw={np.degrees(yaw):+.0f} deg — ONNX off, "
                                   f"ramp to keyframe stand")
        elif state == "FREEZE":
            if t - freeze_t0 >= FREEZE_RAMP_S:
                lv = d.sensor("local_linvel_pelvis").data
                lf = d.sensor("left_foot_floor_found").data[0] > 0
                rf = d.sensor("right_foot_floor_found").data[0] > 0
                stable = (lf and rf
                          and abs(lv[0]) < FREEZE_SPEED_MAX and abs(lv[1]) < FREEZE_SPEED_MAX
                          and abs(roll_deg(d)) < FREEZE_ROLL_MAX
                          and abs(pitch) < FREEZE_PITCH_MAX
                          and dist < 0.08 and abs(yaw) < YAW_HARD)
                if stable:
                    sit_t0, sit_ctrl0 = t, d.ctrl.copy()
                    sit_yaw_deg, sit_err_cm = float(np.degrees(yaw)), dist * 100
                    goto("SIT", f"static stand verified after "
                                f"{t - freeze_t0:.2f}s (lvel=({lv[0]:+.2f},{lv[1]:+.2f}), "
                                f"roll={roll_deg(d):+.1f}, pitch={pitch:+.1f} deg), "
                                f"err={dist * 100:.1f} cm, yaw={np.degrees(yaw):+.0f} deg "
                                f"— verified scripted descent")
                elif t - freeze_t0 >= FREEZE_WAIT_MAX:
                    if retries < MAX_SETTLE_RETRIES:
                        retries += 1
                        # verified ONNX cold-start condition, then trim-march to recover
                        phase = np.array([0.0, np.pi])
                        last_action = np.zeros(m.nu, dtype=np.float32)
                        settle_t0, settle_xy0 = t, np.array([x, y])
                        waiting_calm = False
                        goto("SETTLE", f"stand never stabilized in {FREEZE_WAIT_MAX:.0f}s "
                                       f"(lvel=({lv[0]:+.2f},{lv[1]:+.2f}), "
                                       f"roll={roll_deg(d):+.1f}, pitch={pitch:+.1f}, "
                                       f"L={int(lf)} R={int(rf)}, err={dist * 100:.1f} cm, "
                                       f"yaw={np.degrees(yaw):+.0f} deg) — ONNX re-engaged "
                                       f"trim march, retry {retries}/{MAX_SETTLE_RETRIES}")
                    else:
                        fail = "freeze stand never verified and retries exhausted"
                        timeline.append(f"[t={t:6.2f}s] ABORT: {fail}")
                        print(timeline[-1])
                        break
        elif state == "SIT":
            if t - sit_t0 >= T_DESCENT:
                hold_t0 = t
                goto("HOLD", f"descent done (z={z:.3f}, pitch={pitch:+.1f} deg)")
        elif state == "HOLD":
            if t - hold_t0 >= T_HOLD:
                msg = f"[t={t:6.2f}s] HOLD complete (z={z:.3f}, pitch={pitch:+.1f} deg)"
                print(msg)
                timeline.append(msg)
                break

    if t >= MAX_T and fail is None and not (hold_t0 and t - hold_t0 >= T_HOLD - 1e-9):
        fail = f"45 s rollout cap hit in state {state}"
        timeline.append(f"[t={t:6.2f}s] ABORT: {fail}")
        print(timeline[-1])
    renderer.close()

    # --- final metrics ---
    sit_z = g1_sit_env.SIT_TARGET_Z
    if seat_hist:
        sh = np.array(seat_hist)
        last1 = sh[sh[:, 0] > sh[-1, 0] - 1.0]
        pelvis_z, pitch_f = float(sh[-1, 1]), float(sh[-1, 2])
        force_f, ncon_f = float(sh[-1, 3]), int(sh[-1, 4])
        force_mean1 = float(last1[:, 3].mean())
    else:
        pelvis_z, pitch_f = float(d.qpos[2]), pitch_deg(d)
        force_f, ncon_f, force_mean1 = 0.0, 0, 0.0
    on_seat = ncon_f > 0
    held = hold_t0 is not None and t - hold_t0 >= T_HOLD - 1e-9
    seated = (abs(pelvis_z - sit_z) < 0.08 and abs(pitch_f) < 30
              and on_seat and force_f > 80.0)
    success = (navigated and not fell and fail is None and min_clear > 0
               and held and seated)

    if frames:
        os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
        frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:],
                       duration=int(1000 / FPS), loop=0, optimize=True)

    print()
    print("=== G1 autonomous WALK -> AVOID -> ALIGN -> SIT (one continuous rollout) ===")
    print("state durations: " + "  ".join(f"{s}={v:.2f}s" for s, v in durations.items()))
    print(f"navigated            = {navigated}   fell = {fell}")
    print("min obstacle clearance (NAVIGATE, base-to-axis minus radius): "
          f"{min_clear:.3f} m  " +
          " ".join(f"[{n}:{v:.2f}]" for n, v in clear_per_obs.items()))
    if sit_err_cm is not None:
        print(f"docking at SIT entry = {sit_err_cm:.1f} cm from the nominal dock point, "
              f"yaw {sit_yaw_deg:+.1f} deg (basin: 8 cm / +-45 deg)")
    print(f"pelvis_z_final       = {pelvis_z:.4f}  (target {sit_z:.4f})")
    print(f"pitch_deg_final      = {pitch_f:+.2f}")
    print(f"on_seat              = {on_seat}  ({ncon_f} seat-pelvis contacts)")
    print(f"seat_force_N         = {force_f:.1f}  (mean last 1 s {force_mean1:.1f} N)")
    print(f"held_3s              = {held}")
    if frames:
        print(f"gif: {GIF_PATH} ({os.path.getsize(GIF_PATH) / 1e6:.2f} MB, "
              f"{len(frames)} frames)")
    if success:
        print("RESULT: AUTONOMOUS WALK→AVOID→SIT ✓")
    else:
        stage = fail if fail else (
            "SIT/HOLD: not seated "
            f"(z={pelvis_z:.3f} vs {sit_z:.3f}, pitch={pitch_f:+.1f}, on_seat={on_seat}, "
            f"F={force_f:.0f} N)" if navigated else "NAVIGATE: never reached the seat")
        print(f"RESULT: FAILED — {stage}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
