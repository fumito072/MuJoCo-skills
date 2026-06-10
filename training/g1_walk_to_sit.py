"""END-TO-END autonomous G1 demo: WALK -> AVOID -> TURN -> BACK UP -> SIT on a REAL chair.
One model, one continuous rollout, one video. CPU-only on macOS (plain python, offscreen CGL
rendering; NEVER mujoco.viewer / mjpython).

The chair is REALISTIC and COLLIDABLE: seat board + backrest + 4 legs (g1_sit_env
.add_real_chair_geoms), with explicit contact pairs to the robot's OWN thigh/shin/foot
collision geoms (contype=0 capsules/boxes that ship with the playground model) plus a
torso proxy box — so legs can NOT pass through the chair, and the approach must be
human-like: walk to the chair FRONT, turn 180 deg, back up until the calves are at the
front edge, then sit. Seat height 0.39 m = sized to the 1.32 m robot (a 0.45 m human
chair is a tall stool for the G1: the mesh-calibrated pelvis bottom cannot reach it and
the robot perches leaning ~34 deg forward — measured; human-chair sitting = RL track).

Pipeline (FSM at 50 Hz):
  NAVIGATE  official pretrained 29-DOF joystick ONNX walk (models/policies/g1_joystick_29dof
            .onnx, obs recipe + calibrated creep trim verified in training/g1_walk_onnx.py),
            steered by the raycast-VFH planner (21 mj_ray casts, 200 deg FOV, rays masked to
            obstacle geom group 4), goal = dock point. Exits at dist < 0.7 m (before the
            physical chair).
  TURN      turn in place (trim + wz servo) until the heading is within 20 deg of the
            seat-facing yaw (= pi: back to the chair).
  ALIGN     body-frame P-servo on (dock - base) — mostly WALKING BACKWARD — plus the yaw
            servo, on top of the calibrated stationary trim (vx -0.20, vy +0.07). The
            dock point = chair front edge + 1.5 cm, the center of the measured REAL-chair
            basin (fwd +8 / back -5 / lateral +-8 cm, yaw +-20 deg verified; yaw 35 deg
            has a topple hole, so the yaw budget is tight).
  SETTLE    station-keeping march (trim + small closed-loop position servo — the open-loop
            trim is heading/history dependent and drifted 3 cm/s into the chair), then wait
            (<= 2 s) for the cut gate: both feet down + feet abreast + ASYMMETRIC velocity
            bounds (away-from-chair vx is the killer; toward-chair drift gets caught by the
            seat) + low gyro/roll. Out-of-budget or never-calm -> retry via ALIGN (5x).
  SIT       at the gate instant, STOP the ONNX and ramp d.ctrl (0.7 s) STRAIGHT to the
            verified seated pose (hip_pitch=-1.254, knee=+1.611, ankle_pitch=+0.137
            plantarflex bias; training/g1_sit_scripted.py). There is deliberately NO
            stand phase in between: the open-loop keyframe stand is an unsteerable
            ~1 s glide window (measured: it topples forward in ~2 s under its weak
            ankles, never kills the cut momentum, and 0.1 m/s of residual glide = a
            10 cm miss — 'sitting into empty air'). The direct cut shrinks the
            uncontrolled exposure to ~0.55 s and the SEAT itself kills the residual
            motion. On the real chair the weight lands on BOTH the pelvis box and the
            thighs (measured ~212 N total) — human-like load sharing.
  HOLD      3 s; final seated metrics (pelvis z vs SIT_TARGET_Z=0.545, pitch, chair contact
            + normal force via mj_contactForce).

Scene: playground G1 feetonly flat terrain + REAL chair at world (3.5, 0) facing -x (the
robot approaches from the front) + 3 obstacles (group 4) blocking the straight line.

HONEST caveats: obstacle avoidance is still enforced by the raycast VFH + pelvis-box
contact only (leg<->obstacle pairs are not yet wired); the pelvis box and torso box are
calibrated proxies (pelvis bottom = mesh bbox bottom, base-0.155). The chair itself is
fully physical to pelvis, torso, thighs, shins and feet via explicit pairs.

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
MAX_T = 60.0                          # hard rollout cap (timeout -> honest failure)
NAV_TIMEOUT = 30.0
WALK_GUARD_T = 55.0                   # must have started SIT by here

# --- calibrated command trim (measured; see training/g1_walk_onnx.py docstring) ---
TRIM_VX, TRIM_VY = -0.20, +0.07

# --- scene ---
SEAT_WORLD = np.array([3.5, 0.0])     # seat center
CHAIR_YAW = np.pi                     # sitter faces -x: the robot approaches the chair FRONT
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

# Docking NOMINAL: the REAL-chair basin was measured around base = front edge + 1.5 cm,
# robot facing away from the backrest (probe sweep 2026-06-10: fwd +8 / back -5 /
# lateral +-8 cm OK, yaw +-20 deg OK with a topple hole at 35 deg).
DOCK_AHEAD = 0.04   # was 0.015: a -5 cm dock error left only ~6 mm between
                    # the standing shin tops (z 0.405) and the board (top 0.39)
_dock_local = g1_sit_env.REAL_DOCK_OFFSET[0] + DOCK_AHEAD
DOCK_WORLD = SEAT_WORLD + _dock_local * np.array([np.cos(CHAIR_YAW), np.sin(CHAIR_YAW)])
YAW_TARGET = CHAIR_YAW           # seated heading = the chair's facing direction

# --- FSM tuning ---
NAV_EXIT_DIST = 0.7      # stop short of the (now physical) chair, then turn around
TURN_EXIT = np.deg2rad(20)
ALIGN_TOL = 0.045        # inside the verified -5 cm back / +-8 cm basin with margin
ALIGN_DEADBAND = 0.030   # inside this, stop chasing the march-in-place wobble
ALIGN_HOLD_S = 0.5
DIST_HARD = 0.065        # never descend further out than this (basin back edge -5 cm)
# Yaw budget is TIGHT on the real chair: 20 deg verified OK, 35 deg topples (rotated
# legs/board-edge interaction), so gate at 15 deg and hard-cap at 25 deg.
YAW_TOL = np.deg2rad(15)        # ALIGN/SETTLE exit gate
YAW_DEADBAND = np.deg2rad(8)    # inside this, stop chasing the march wobble
YAW_HARD = np.deg2rad(25)       # never start the descent outside the verified basin
SETTLE_S = 1.0
SETTLE_SPEED_MAX = 0.15
MAX_SETTLE_RETRIES = 5
# Sit-entry gate: cut to the scripted descent only at a SWAY-CALM instant of the march.
# (Measured: double-stance instants are the weight-transfer instants — body-lateral velocity
# peaks there at ~0.15-0.2 m/s, and entering the descent with that sway rolled the robot off
# the seat sideways. Calm instants recur every ~0.66 s gait cycle.)
CALM_WAIT_MAX = 2.0      # max wait for a calm instant (backward-entry march sways more)
# Body +x faces AWAY from the chair when docked. Away-drift at the cut GLIDES the whole
# freeze+descent (0.12 m/s cut velocity x ~1 s = the 13 cm miss measured in attempt #10 —
# the robot sat into empty air in front of the seat). Toward-chair drift is the SAFE
# direction (the seat catches it), so the bounds are asymmetric.
CALM_VX_AWAY, CALM_VX_TOWARD = 0.03, 0.08   # m/s body vx in (-TOWARD, +AWAY)
CALM_VY = 0.08                  # m/s body-frame lateral bound
CALM_GYRO = 0.9          # rad/s |gyro x|, |gyro y| bound
CALM_ROLL = 4.0          # deg |roll| bound (the backward-entry march sways ~+-6 deg)
FEET_ABREAST = 0.10      # feet side-by-side (|x_L - x_R| body frame): no mid-stride cuts
# MEASURED: double stance in this march lasts only 1-3 ticks (0.02-0.06 s) at footfalls —
# there IS no quiet double-support window to wait for. So: cut at the FIRST decent
# footfall instant (gate above) and descend immediately; a rejected instant just
# waits for the next footfall (or retries via ALIGN on timeout).


# --- scripted sit (verified in g1_sit_scripted.py; basin swept to 0.08 m / +-45 deg) ---
HIP_PITCH, KNEE, ANKLE_PITCH = -1.254, +1.611, +0.137
T_DESCENT, T_HOLD = 0.7, 3.0   # 0.7 s: the march-cut entry advances the topple
                               # countdown, so the seat catch must come sooner
                               # (1.0 s entry from a march toppled at pitch +62;
                               # 0.6/0.7 s re-verified SEATED-STABLE across the
                               # basin spot-checks)
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


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


CHAIR_GEOMS = ("seat", "backrest", "chair_leg0", "chair_leg1", "chair_leg2", "chair_leg3")
ROBOT_SUPPORT_GEOMS = ("pelvis_collision", "torso_collision") + g1_sit_env.ROBOT_LEG_GEOMS


def build_scene(sim_dt=0.002):
    """Playground G1 feetonly flat terrain + REAL collidable chair + obstacles."""
    assets = g1_base.get_assets()
    spec = mujoco.MjSpec.from_string(consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets)

    # realistic chair (board + backrest + 4 legs) + pelvis/torso proxies + leg<->chair pairs
    g1_sit_env.add_real_chair_geoms(spec, center_xy=tuple(SEAT_WORLD), yaw=CHAIR_YAW)

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
    chair_gids = {m.geom(n).id for n in CHAIR_GEOMS}
    lfoot_gid, rfoot_gid = m.geom("left_foot").id, m.geom("right_foot").id
    support_gids = {m.geom(n).id for n in ROBOT_SUPPORT_GEOMS}
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
    navigated, turned = False, False
    fell, fail = False, None
    min_clear = float("inf")
    clear_per_obs = {name: float("inf") for name, *_ in OBSTACLES}
    align_hold = 0.0
    settle_t0, settle_xy0 = None, None
    waiting_calm, calm_t0 = False, 0.0
    calm_streak = 0.0
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
        yerr = wrap(yaw - YAW_TARGET)                   # heading error vs the seated heading
        exw, eyw = DOCK_WORLD[0] - x, DOCK_WORLD[1] - y
        dist = float(np.hypot(exw, eyw))
        ex = np.cos(yaw) * exw + np.sin(yaw) * eyw      # error in the base yaw frame
        ey = -np.sin(yaw) * exw + np.cos(yaw) * eyw

        # --- control ---
        if state in ("NAVIGATE", "TURN", "ALIGN", "SETTLE"):
            if state == "NAVIGATE":
                vx, wz = plan()
                vx *= float(np.clip(dist / 1.5, 0.30, 1.0))   # approach slowdown (curvature)
                cmd = np.array([vx, 0.0, wz])
                if DEBUG and k % 25 == 0:
                    print(f"    DBG t={t:6.2f} NAV pos=({x:+.2f},{y:+.2f}) "
                          f"yaw={np.degrees(yaw):+5.0f} dist={dist:.2f} "
                          f"cmd=({vx:+.2f},{wz:+.2f})")
            elif state == "TURN":     # turn in place: back to the chair
                cmd = np.array([TRIM_VX, TRIM_VY,
                                float(np.clip(-1.2 * yerr, -0.6, 0.6))])
            elif state == "ALIGN":
                if dist < ALIGN_DEADBAND:
                    svx = svy = 0.0
                else:
                    svx = float(np.clip(1.2 * ex, -0.30, 0.35))   # mostly backward here
                    svy = float(np.clip(1.2 * ey, -0.25, 0.25))
                swz = (0.0 if abs(yerr) < YAW_DEADBAND
                       else float(np.clip(-1.2 * yerr, -0.5, 0.5)))
                cmd = np.array([TRIM_VX + svx, TRIM_VY + svy, swz])
            else:  # SETTLE — station-keeping march: trim + SMALL closed-loop position
                # servo. Pure open-loop trim drifted ~3 cm/s backward INTO the chair here:
                # the calibrated creep bias is heading/history dependent (it was measured
                # facing +x after forward walking; this march follows BACKWARD walking).
                svx = float(np.clip(1.0 * ex, -0.12, 0.12))
                svy = float(np.clip(1.0 * ey, -0.08, 0.08))
                swz = float(np.clip(-0.8 * yerr, -0.2, 0.2))
                cmd = np.array([TRIM_VX + svx, TRIM_VY + svy, swz])

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
        yerr = wrap(yaw - YAW_TARGET)
        dist = float(np.hypot(DOCK_WORLD[0] - x, DOCK_WORLD[1] - y))

        if state in ("NAVIGATE", "TURN", "ALIGN", "SETTLE"):
            for name, ox, oy, kind, sx, sy, hh, rgba in OBSTACLES:
                r = float(np.hypot(sx, sy)) if kind == "box" else sx   # circumscribed radius
                c = float(np.hypot(x - ox, y - oy)) - r
                clear_per_obs[name] = min(clear_per_obs[name], c)
                if state == "NAVIGATE":
                    min_clear = min(min_clear, c)
            if z < 0.4 or abs(pitch) > 60:
                fell = True
                fail = f"fell during {state} (pelvis_z={z:.3f}, pitch={pitch:+.1f} deg)"
            elif z < 0.62 and abs(pitch) > 25:
                fail = (f"propped/stuck on the chair during {state} "
                        f"(pelvis_z={z:.3f}, pitch={pitch:+.1f} deg) — unrecoverable "
                        f"for the walk policy, aborting honestly")
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
                if ((c.geom1 in chair_gids and c.geom2 in support_gids)
                        or (c.geom2 in chair_gids and c.geom1 in support_gids)):
                    mujoco.mj_contactForce(m, d, i, f6)
                    force += f6[0]       # normal component
                    ncon += 1
            seat_hist.append((t, z, pitch, force, ncon))
            if DEBUG and k % 5 == 0:
                print(f"    DBG t={t:6.2f} {state:4s} reldock=({x - DOCK_WORLD[0]:+.3f},"
                      f"{y - DOCK_WORLD[1]:+.3f}) base_z={z:.3f} "
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

        if state in ("NAVIGATE", "TURN", "ALIGN", "SETTLE") and t > WALK_GUARD_T:
            fail = (f"did not reach SIT by t={WALK_GUARD_T:.0f}s "
                    f"(state={state}, dist={dist:.2f} m)")
            timeline.append(f"[t={t:6.2f}s] ABORT: {fail}")
            print(timeline[-1])
            break

        if state == "NAVIGATE":
            if dist < NAV_EXIT_DIST:
                navigated = True
                goto("TURN", f"dist={dist:.2f} m < {NAV_EXIT_DIST:.1f} m — "
                             f"turn 180 deg, back to the chair")
            elif t > NAV_TIMEOUT:
                fail = f"NAVIGATE timeout (dist still {dist:.2f} m at t={t:.1f}s)"
                timeline.append(f"[t={t:6.2f}s] ABORT: {fail}")
                print(timeline[-1])
                break
        elif state == "TURN":
            if abs(yerr) < TURN_EXIT:
                turned = True
                goto("ALIGN", f"heading {np.degrees(yerr):+.0f} deg from seated yaw — "
                              f"back up to the dock point")
        elif state == "ALIGN":
            align_hold = (align_hold + CTRL_DT
                          if dist < ALIGN_TOL and abs(yerr) < YAW_TOL else 0.0)
            if align_hold >= ALIGN_HOLD_S:
                align_hold = 0.0
                settle_t0, settle_xy0 = t, np.array([x, y])
                waiting_calm = False
                goto("SETTLE", f"|err|={dist * 100:.1f} cm < {ALIGN_TOL * 100:.1f} cm "
                               f"held {ALIGN_HOLD_S:.1f}s; yaw_err={np.degrees(yerr):+.0f} deg")
        elif state == "SETTLE":
            if dist > 0.08 and retries < MAX_SETTLE_RETRIES:
                # drifted way off mid-settle (e.g. backward INTO the chair) — bail out
                # early, BEFORE the legs reach the board edge
                retries += 1
                waiting_calm = False
                goto("ALIGN", f"drifted to {dist * 100:.1f} cm mid-settle — "
                              f"retry {retries}/{MAX_SETTLE_RETRIES}")
            elif not waiting_calm and t - settle_t0 >= SETTLE_S:
                speed = float(np.linalg.norm(np.array([x, y]) - settle_xy0)
                              / (t - settle_t0))
                if (speed < SETTLE_SPEED_MAX and dist < ALIGN_TOL
                        and abs(yerr) < YAW_TOL):
                    waiting_calm, calm_t0, calm_streak = True, t, 0.0
                elif retries < MAX_SETTLE_RETRIES:
                    retries += 1
                    goto("ALIGN", f"settle check failed (drift={speed:.2f} m/s, "
                                  f"err={dist * 100:.1f} cm, yaw_err={np.degrees(yerr):+.0f} deg) "
                                  f"— retry {retries}/{MAX_SETTLE_RETRIES}")
                elif dist < DIST_HARD and abs(yerr) < YAW_HARD:  # hard basin budget — warn & go
                    waiting_calm, calm_t0, calm_streak = True, t, 0.0
                    msg = (f"[t={t:6.2f}s] WARN: settle retries exhausted, err="
                           f"{dist * 100:.1f} cm / yaw_err {np.degrees(yerr):+.0f} deg within "
                           f"the hard budget — proceeding")
                    print(msg)
                    timeline.append(msg)
                else:
                    fail = (f"settle retries exhausted, err={dist * 100:.1f} cm / "
                            f"yaw_err {np.degrees(yerr):+.0f} deg outside the "
                            f"verified docking basin")
                    timeline.append(f"[t={t:6.2f}s] ABORT: {fail}")
                    print(timeline[-1])
                    break
            if waiting_calm:
                lv = d.sensor("local_linvel_pelvis").data
                gy = d.sensor("gyro_pelvis").data
                lf = d.sensor("left_foot_floor_found").data[0] > 0
                rf = d.sensor("right_foot_floor_found").data[0] > 0
                pL, pR = d.geom_xpos[lfoot_gid], d.geom_xpos[rfoot_gid]
                dxf = float(np.cos(yaw) * (pL[0] - pR[0])
                            + np.sin(yaw) * (pL[1] - pR[1]))
                calm = (lf and rf and abs(dxf) < FEET_ABREAST
                        and -CALM_VX_TOWARD < lv[0] < CALM_VX_AWAY
                        and abs(lv[1]) < CALM_VY
                        and abs(gy[0]) < CALM_GYRO and abs(gy[1]) < CALM_GYRO
                        and abs(roll_deg(d)) < CALM_ROLL)
                if DEBUG and k % 3 == 0:
                    print(f"    DBG t={t:6.2f} CALMGATE dxf={dxf:+.3f} "
                          f"lv=({lv[0]:+.2f},{lv[1]:+.2f}) gy=({gy[0]:+.2f},{gy[1]:+.2f}) "
                          f"roll={roll_deg(d):+4.1f} L={int(lf)} R={int(rf)}")
                timed_out = t - calm_t0 >= CALM_WAIT_MAX
                in_budget = dist < 0.05 and abs(yerr) < YAW_HARD   # direct-commit: tight
                # NEVER cut while out of budget or still moving: attempt #4 cut
                # on a calm-wait timeout at err 14 cm / -0.11 m/s, backed into the chair
                # mid-freeze and got propped on the seat edge — unrecoverable for the
                # walk policy. Budget+calm are mandatory; timeout means retry, and an
                # uncalm last-resort cut is allowed only IN budget with retries spent.
                if calm and in_budget:
                    # CUT STRAIGHT INTO THE DESCENT — no stand in between. The open-loop
                    # keyframe stand is an unsteerable ~1 s glide window (measured: it
                    # topples in ~2 s, never kills the cut momentum, and 0.1 m/s of glide
                    # = a 10 cm miss). Descending immediately shrinks the uncontrolled
                    # exposure to ~0.55 s and the seat itself kills the residual motion.
                    sit_t0, sit_ctrl0 = t, d.ctrl.copy()
                    sit_yaw_deg, sit_err_cm = float(np.degrees(yerr)), dist * 100
                    goto("SIT", f"both-feet sway-calm "
                                f"(lvel=({lv[0]:+.2f},{lv[1]:+.2f}), roll={roll_deg(d):+.1f} deg, "
                                f"L={int(lf)} R={int(rf)}), err={dist * 100:.1f} cm, "
                                f"yaw_err={np.degrees(yerr):+.0f} deg — ONNX off, "
                                f"direct scripted descent")
                elif timed_out:
                    if retries < MAX_SETTLE_RETRIES:
                        retries += 1
                        goto("ALIGN", f"calm-wait timeout ({'out of budget: ' if not in_budget else ''}"
                                      f"err={dist * 100:.1f} cm, yaw_err={np.degrees(yerr):+.0f} deg, "
                                      f"lvel=({lv[0]:+.2f},{lv[1]:+.2f})) — retry "
                                      f"{retries}/{MAX_SETTLE_RETRIES}")
                    elif in_budget:
                        msg = (f"[t={t:6.2f}s] WARN: never calm, retries spent, but in budget "
                               f"(err={dist * 100:.1f} cm) — last-resort cut")
                        print(msg)
                        timeline.append(msg)
                        sit_t0, sit_ctrl0 = t, d.ctrl.copy()
                        sit_yaw_deg, sit_err_cm = float(np.degrees(yerr)), dist * 100
                        goto("SIT", "last-resort cut — ONNX off, direct scripted descent")
                    else:
                        fail = (f"never calm and out of budget (err={dist * 100:.1f} cm, "
                                f"yaw_err={np.degrees(yerr):+.0f} deg), retries exhausted")
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
    seated = (abs(pelvis_z - sit_z) < 0.06 and abs(pitch_f) < 30
              and on_seat and force_f > 80.0)
    success = (navigated and turned and not fell and fail is None and min_clear > 0
               and held and seated)

    if frames:
        os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
        frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:],
                       duration=int(1000 / FPS), loop=0, optimize=True)

    print()
    print("=== G1 autonomous WALK -> AVOID -> TURN -> BACK-UP -> SIT (real chair, one rollout) ===")
    print("state durations: " + "  ".join(f"{s}={v:.2f}s" for s, v in durations.items()))
    print(f"navigated            = {navigated}   turned = {turned}   fell = {fell}")
    print("min obstacle clearance (NAVIGATE, base-to-axis minus radius): "
          f"{min_clear:.3f} m  " +
          " ".join(f"[{n}:{v:.2f}]" for n, v in clear_per_obs.items()))
    if sit_err_cm is not None:
        print(f"docking at SIT entry = {sit_err_cm:.1f} cm from the nominal dock point, "
              f"yaw_err {sit_yaw_deg:+.1f} deg (verified basin: +8/-5 cm, +-20 deg)")
    print(f"pelvis_z_final       = {pelvis_z:.4f}  (target {sit_z:.4f})")
    print(f"pitch_deg_final      = {pitch_f:+.2f}")
    print(f"on_chair             = {on_seat}  ({ncon_f} chair-robot contacts)")
    print(f"chair_force_N        = {force_f:.1f}  (mean last 1 s {force_mean1:.1f} N)")
    print(f"held_3s              = {held}")
    if frames:
        print(f"gif: {GIF_PATH} ({os.path.getsize(GIF_PATH) / 1e6:.2f} MB, "
              f"{len(frames)} frames)")
    if success:
        print("RESULT: AUTONOMOUS WALK→AVOID→TURN→SIT (REAL CHAIR) ✓")
    else:
        stage = fail if fail else (
            "SIT/HOLD: not seated "
            f"(z={pelvis_z:.3f} vs {sit_z:.3f}, pitch={pitch_f:+.1f}, on_seat={on_seat}, "
            f"F={force_f:.0f} N)" if navigated else "NAVIGATE: never reached the seat")
        print(f"RESULT: FAILED — {stage}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
