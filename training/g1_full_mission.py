"""FULL MISSION (sim harness): G1 walks across the room, avoids obstacles, turns,
backs up to the real chair, climbs the 0.22 m footrest platform with the learned
policy, and sits. One continuous rollout, CPU-only macOS.

THE CONTROLLER UNDER TEST IS THE DEPLOYMENT ARTIFACT: deploy/g1_mission_controller
.MissionController (numpy + onnxruntime only). This file is just the MuJoCo
adapter — it builds the scene, feeds the controller the same state dict the
robot's Linux adapter will feed it, and applies the returned targets/gains.
Passing here = the exact code that ships has completed the mission in physics.

State adapter (sim -> contract):           on hardware:
  base_xy / base_yaw / base_z  qpos        LiDAR/mocap localization + estimator
  gravity, gyro, linvel        sensors     IMU + state estimator
  joint_pos/vel                qpos/qvel   motor encoders
  foot_contact                 contacts    foot force sensors
  rays (21, 200 deg, z 0.5)    mj_ray      2D LiDAR fan (group-4 mask -> "not floor")

Gains: in WALK phases the sim keeps the XML gains the ONNX was trained with; in
CLIMB/SIT the controller's kp=300/kd=8 is applied (mode switch, verified here).
HW maps both through the motor-command kp/kd fields (README_DEPLOY sim2real #1).

Run:  .venv-rl/bin/python training/g1_full_mission.py     (G1FM_NOGIF=1 for fast)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "deploy"))

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

import g1_sit_env  # noqa: E402
from g1_mission_controller import MissionController  # noqa: E402
from mujoco_playground._src.locomotion.g1 import base as g1_base  # noqa: E402
from mujoco_playground._src.locomotion.g1 import g1_constants as consts  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIF_PATH = os.path.join(REPO, "assets", "g1_full_mission.gif")

CTRL_DT, SUBSTEPS = 0.02, 10
MAX_T = 75.0
START_XY = (0.0, 4.2)
OBST_GROUP = 4
OBSTACLES = [
    ("obs0", -0.45, 3.0, 0.30, 0.75, (0.85, 0.30, 0.15, 1.0)),
    ("obs1", 0.55, 2.2, 0.30, 0.75, (0.20, 0.45, 0.85, 1.0)),
]
N_RAYS, FOV = 21, np.deg2rad(200)
RAY_ANGLES = np.array([-FOV / 2 + FOV * i / (N_RAYS - 1) for i in range(N_RAYS)])

DEBUG = bool(os.environ.get("G1FM_DEBUG"))
NO_GIF = bool(os.environ.get("G1FM_NOGIF"))
FPS = 20


def yaw_of(d):
    w, x, y, z = d.qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rpy(d):
    w, x, y, z = d.qpos[3:7]
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    return float(roll), float(pitch)


def torso_lean_deg(m, d):
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    zax = d.xmat[tid].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arctan2(np.hypot(zax[0], zax[1]), zax[2])))


def build_scene():
    assets = g1_base.get_assets()
    spec = mujoco.MjSpec.from_string(consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets)
    g1_sit_env.add_fbx_chair_geoms(spec, center_xy=(0.0, 0.0), yaw=0.0)
    for name, ox, oy, r, hh, rgba in OBSTACLES:
        g = spec.worldbody.add_geom()
        g.name = name
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [r, hh, 0]
        g.pos = [ox, oy, hh]
        g.group = OBST_GROUP
        g.contype, g.conaffinity = 1, 1
        g.rgba = list(rgba)
    spec.assets = assets
    m = spec.compile()
    m.opt.timestep = 0.002
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
    return m


def main():
    m = build_scene()
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    mujoco.mj_resetDataKeyframe(m, d, key)
    yaw0 = -np.pi / 2
    d.qpos[0:3] = (START_XY[0], START_XY[1], 0.755)
    d.qpos[3:7] = (np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2))
    mujoco.mj_forward(m, d)

    imu_site = m.site("imu_in_pelvis").id
    lf, rf = m.geom("left_foot").id, m.geom("right_foot").id
    rc_gids = {i for i in range(m.ngeom)
               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("rc_part")}
    geomid = np.zeros(1, dtype=np.int32)
    f6 = np.zeros(6)

    ctrl = MissionController(
        config_path=os.path.join(REPO, "deploy", "config.json"),
        walk_onnx=os.path.join(REPO, "models", "policies", "g1_joystick_29dof.onnx"),
        climb_onnx=os.path.join(REPO, "deploy", "g1_climb_backstep.onnx"))

    leg_aids = list(range(12))
    orig_gain = m.actuator_gainprm[leg_aids].copy()
    orig_bias = m.actuator_biasprm[leg_aids].copy()

    def apply_gains(phase):
        if phase in ("CLIMB", "SIT"):
            for a in leg_aids:
                m.actuator_gainprm[a, 0] = ctrl.cfg["sit_mode_kp"]
                m.actuator_biasprm[a, 1] = -ctrl.cfg["sit_mode_kp"]
                m.actuator_biasprm[a, 2] = -ctrl.cfg["sit_mode_kd"]
        else:
            m.actuator_gainprm[leg_aids] = orig_gain
            m.actuator_biasprm[leg_aids] = orig_bias

    renderer = cam = None
    frames = []
    if not NO_GIF:
        renderer = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        cam.distance, cam.azimuth, cam.elevation = 3.2, 135, -14
    frame_every = max(1, int(round(1.0 / (FPS * CTRL_DT))))

    t, k = 0.0, 0
    fail = None
    min_clear = float("inf")
    seen = {"NAVIGATE"}
    prev_phase = "NAVIGATE"
    print(f"[t=  0.00s] NAVIGATE  start {START_XY}")

    while t < MAX_T:
        x, y = float(d.qpos[0]), float(d.qpos[1])
        yaw = yaw_of(d)
        # --- build the state dict (the HW adapter does exactly this from sensors) ---
        on_l = on_r = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if lf in (c.geom1, c.geom2):
                on_l = 1.0
            if rf in (c.geom1, c.geom2):
                on_r = 1.0
        rays = np.empty(N_RAYS)
        for i, a in enumerate(RAY_ANGLES):
            vec = np.array([np.cos(yaw + a), np.sin(yaw + a), 0.0])
            dist = mujoco.mj_ray(m, d, np.array([x, y, 0.5]), vec, None, 1, -1, geomid)
            grp = m.geom_group[geomid[0]] if geomid[0] >= 0 else -1
            rays[i] = dist if (dist >= 0 and grp == OBST_GROUP) else 4.0
        state = {
            "joint_pos": d.qpos[7:].copy(), "joint_vel": d.qvel[6:].copy(),
            "gravity": d.site_xmat[imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0]),
            "gyro": d.sensor("gyro_pelvis").data.copy(),
            "linvel": d.sensor("local_linvel_pelvis").data.copy(),
            "base_xy": np.array([x, y]), "base_yaw": yaw, "base_z": float(d.qpos[2]),
            "foot_contact": np.array([on_l, on_r]), "rays": rays,
        }
        out = ctrl.step(state)
        if out["phase"] != prev_phase:
            print(f"[t={t:6.2f}s] {prev_phase} -> {out['phase']}")
            seen.add(out["phase"])
            prev_phase = out["phase"]
            apply_gains(out["phase"])
        d.ctrl[:] = out["target"]
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
        t += CTRL_DT
        k += 1

        z = float(d.qpos[2])
        roll, pitch = rpy(d)
        if out["phase"] == "NAVIGATE":
            for name, ox, oy, r, *_ in OBSTACLES:
                min_clear = min(min_clear, float(np.hypot(x - ox, y - oy)) - r)
        if not NO_GIF and k % frame_every == 1:
            renderer.update_scene(d, camera=cam)
            frames.append(Image.fromarray(renderer.render())
                          .convert("P", palette=Image.ADAPTIVE, colors=128))
        if DEBUG and k % 25 == 0:
            print(f"    DBG t={t:6.2f} {out['phase']:8s} ({x:+.2f},{y:+.2f},{z:.3f}) "
                  f"p={pitch:+5.1f} r={roll:+5.1f}")
        if out["failed"]:
            fail = out["failed"]
            print(f"[t={t:6.2f}s] ABORT: {fail}")
            break
        if out["done"]:
            print(f"[t={t:6.2f}s] mission done")
            break
        if out["phase"] in ("NAVIGATE", "TURN", "BACKUP") and (z < 0.55 or abs(pitch) > 50):
            fail = f"fell during {out['phase']} (z={z:.3f}, pitch={pitch:+.1f})"
            print(f"[t={t:6.2f}s] ABORT: {fail}")
            break
        if out["phase"] in ("CLIMB", "SIT") and (abs(roll) > 60 or z < 0.30):
            fail = f"fell during {out['phase']} (z={z:.3f}, roll={roll:+.1f})"
            print(f"[t={t:6.2f}s] ABORT: {fail}")
            break

    # verdict
    force_seat, ncon = 0.0, 0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in rc_gids or c.geom2 in rc_gids:
            other = c.geom2 if c.geom1 in rc_gids else c.geom1
            if other not in (lf, rf):
                mujoco.mj_contactForce(m, d, i, f6)
                force_seat += f6[0]
                ncon += 1
    roll, pitch = rpy(d)
    tlean = torso_lean_deg(m, d)
    z = float(d.qpos[2])
    v = ctrl.cfg["verdict"]
    seated = (v["z_range"][0] < z < v["z_range"][1] and abs(pitch) < v["pitch_max_deg"]
              and tlean < v["torso_lean_max_deg"] and abs(roll) < v["roll_max_deg"]
              and force_seat > v["seat_force_min_N"])
    success = (fail is None and seated
               and {"NAVIGATE", "TURN", "BACKUP", "CLIMB", "SIT"} <= seen)

    if frames:
        os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
        frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:],
                       duration=int(1000 / FPS), loop=0, optimize=True)
    print()
    print("=== G1 FULL MISSION (deploy controller under sim test) ===")
    print(f"phases: {' -> '.join(p for p in ['NAVIGATE','TURN','BACKUP','CLIMB','SIT'] if p in seen)}")
    print(f"min obstacle clearance = {min_clear:.3f} m")
    print(f"seated: z={z:.4f} pitch={pitch:+.1f} roll={roll:+.1f} torso={tlean:+.1f} "
          f"chairF={force_seat:.1f} N ({ncon} contacts)")
    if frames:
        print(f"gif: {GIF_PATH} ({len(frames)} frames)")
    print("RESULT:", "FULL MISSION ✓" if success
          else f"FAILED — {fail or 'verdict thresholds not met'}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
