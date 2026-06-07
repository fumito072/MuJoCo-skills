"""Stage C: "go to the chair (avoiding obstacles) and sit down" — a two-phase executor.

This runs the plan the NLU emits for "椅子まで行って座って" / "go to the chair and sit":
  Phase 1 (goto, avoid):  the 12-DOF pretrained WALK model + a VFH obstacle-avoidance planner
                          navigates around obstacles to a spot in front of the chair.
  -- hand-off --
  Phase 2 (sit):          the 29-DOF position model replays the CEM-optimized floor SIT-DOWN.

HONEST two-model split: walking is the pretrained 12-DOF torque policy, sitting is the 29-DOF
position model + CEM trajectory — different MuJoCo models, so the two phases are separate sims
stitched into one video (the base pose is the hand-off). And the sit is a *floor* sit-down (the
solved, stable transition): lowering onto the ELEVATED seat is a balance-critical support transfer
that open-loop control topples (documented in g1-sit-recipe) — so "sit" = sit down at the chair,
not perch on the seat. NVIDIA-free, CPU + MPS.

Usage:  .venv-llm/bin/python chair_goto_sit.py --video assets/chair_goto_sit.gif
"""
import os
import argparse
import importlib.util
import numpy as np
import mujoco
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CB = os.path.normpath(os.path.join(HERE, "..", "..", "mujoco-controller-baselines", "scripts"))
PD = os.path.normpath(os.path.join(HERE, "..", "..", "mujoco-pretrained-deploy", "scripts"))
SIT_MODEL = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "models", "unitree_g1", "scene.xml"))


def _imp(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


gw = _imp("g1_walk", os.path.join(PD, "g1_walk.py"))
opt = _imp("g1_sitdown_optimize", os.path.join(CB, "g1_sitdown_optimize.py"))

# scene layout
CHAIR = np.array([3.0, 0.0])
WALK_GOAL = np.array([2.35, 0.0])          # stop in front of the chair
OBSTACLES = [(1.3, 0.45, "cyl"), (2.0, -0.5, "cyl"), (1.6, 0.95, "box")]
OBST_GROUP, CHAIR_GROUP = 4, 5
N, FOV, RMAX, SAFE = 21, np.deg2rad(200), 4.0, 1.1
ANGLES = np.array([-FOV/2 + FOV*i/(N-1) for i in range(N)])


def add_chair(sp, x, y):
    seat_h = 0.43
    parts = [("seat", [x, y, seat_h], [0.18, 0.18, 0.03]),
             ("back", [x + 0.15, y, seat_h + 0.22], [0.03, 0.18, 0.22]),
             ("leg0", [x - 0.14, y - 0.14, seat_h / 2], [0.02, 0.02, seat_h / 2]),
             ("leg1", [x - 0.14, y + 0.14, seat_h / 2], [0.02, 0.02, seat_h / 2]),
             ("leg2", [x + 0.14, y - 0.14, seat_h / 2], [0.02, 0.02, seat_h / 2]),
             ("leg3", [x + 0.14, y + 0.14, seat_h / 2], [0.02, 0.02, seat_h / 2])]
    for nm, pos, sz in parts:
        g = sp.worldbody.add_geom()
        g.name = "chair_" + nm; g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = sz; g.pos = pos; g.rgba = [0.45, 0.32, 0.22, 1]
        g.group = CHAIR_GROUP; g.contype = 0; g.conaffinity = 0   # visual landmark, not an obstacle


def yaw_of(d):
    w, x, y, z = d.qpos[3:7]
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def phase1_walk(frames, fps):
    sp = mujoco.MjSpec.from_file(os.path.join(gw.VENDOR, "model", "scene.xml"))
    for i, (ox, oy, kind) in enumerate(OBSTACLES):
        g = sp.worldbody.add_geom(); g.name = f"obs{i}"; g.group = OBST_GROUP
        g.rgba = [0.8, 0.25, 0.2, 1]
        if kind == "cyl":
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER; g.size = [0.28, 0.6, 0]; g.pos = [ox, oy, 0.6]
        else:
            g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.3, 0.3, 0.6]; g.pos = [ox, oy, 0.6]
    add_chair(sp, *CHAIR)
    m = sp.compile(); m.opt.timestep = gw.SIM_DT
    d = mujoco.MjData(m)
    policy = torch.jit.load(os.path.join(gw.VENDOR, "motion.pt"))
    gg = np.zeros(6, dtype=np.uint8); gg[OBST_GROUP] = 1; gid = np.zeros(1, dtype=np.int32)

    def planner(_t=0.0):
        origin = np.array([d.qpos[0], d.qpos[1], 0.5]); yaw = yaw_of(d)
        rng = np.full(N, RMAX)
        for i, a in enumerate(ANGLES):
            dist = mujoco.mj_ray(m, d, origin, np.array([np.cos(yaw+a), np.sin(yaw+a), 0.0]), gg, 1, -1, gid)
            if dist >= 0:
                rng[i] = min(dist, RMAX)
        dx, dy = WALK_GOAL - d.qpos[:2]
        goal_dir = (np.arctan2(dy, dx) - yaw + np.pi) % (2*np.pi) - np.pi
        blk = (rng < SAFE).copy()
        for i in range(N):
            if rng[i] < SAFE:
                for j in (i-2, i-1, i+1, i+2):
                    if 0 <= j < N:
                        blk[j] = True
        free = np.where(~blk)[0]
        if len(free) == 0:
            return [0.0, 0.0, 0.6]
        best = free[np.argmin(np.abs(ANGLES[free] - goal_dir))]
        dist_goal = np.hypot(dx, dy)
        vx = 0.5 * float(np.clip(min(rng[best] / RMAX, dist_goal / 1.0), 0.15, 1.0))
        return [vx, 0.0, float(np.clip(1.5 * ANGLES[best], -0.6, 0.6))]

    cam = mujoco.MjvCamera(); cam.lookat = [1.5, 0.1, 0.3]; cam.distance, cam.azimuth, cam.elevation = 5.2, 90, -52
    vopt = mujoco.MjvOption(); vopt.geomgroup[OBST_GROUP] = 1; vopt.geomgroup[CHAIR_GROUP] = 1
    rend = mujoco.Renderer(m, 360, 480)
    every = int(round(1.0 / (fps * gw.SIM_DT)))
    reached = [False]; st = {"k": 0}

    def log(dd):
        if np.hypot(*(WALK_GOAL - dd.qpos[:2])) < 0.3:
            reached[0] = True
        if st["k"] % every == 0:
            rend.update_scene(dd, camera=cam, scene_option=vopt)
            from PIL import Image
            frames.append(Image.fromarray(rend.render()).convert("P", palette=Image.ADAPTIVE, colors=64))
        st["k"] += 1

    gw.walk(m, d, policy, planner, int(16.0 / gw.SIM_DT), log=log)
    print(f"  Phase 1: arrived ({d.qpos[0]:.2f},{d.qpos[1]:.2f}) reached_chair_front={reached[0]} upright={d.qpos[2]>0.5}")
    return reached[0]


def phase2_sit(frames, fps):
    sp = mujoco.MjSpec.from_file(SIT_MODEL)
    add_chair(sp, -0.42, 0.0)                       # chair behind the robot's sit spot
    m = sp.compile(); d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(m, d, sid if sid >= 0 else 0)
    home = d.qpos[7:].copy()
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    params = np.load(os.path.join(CB, "..", "assets", "g1_sitdown_traj.npz"))["params"]
    k1 = opt.pose_from_channels(home, np.clip(params[:6], opt.CH_LO, opt.CH_HI))
    k2 = opt.pose_from_channels(home, np.clip(params[6:], opt.CH_LO, opt.CH_HI))
    seat = opt.pose_from_channels(home, opt.SEATED_CH)
    wps = [home, k1, k2, seat]; T = [0.0, opt.T_DESC*0.4, opt.T_DESC*0.75, opt.T_DESC]

    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = pid; cam.distance, cam.azimuth, cam.elevation = 2.6, 120, -10
    vopt = mujoco.MjvOption(); vopt.geomgroup[CHAIR_GROUP] = 1
    rend = mujoco.Renderer(m, 360, 480)
    n = int((opt.T_DESC + opt.T_SETTLE) / m.opt.timestep)
    every = int(round(1.0 / (fps * m.opt.timestep)))
    for i in range(n):
        t = min(i * m.opt.timestep, opt.T_DESC)
        tgt = wps[-1]
        for k in range(len(T) - 1):
            if t <= T[k+1]:
                tgt = opt.smooth(wps[k], wps[k+1], (t - T[k]) / (T[k+1] - T[k])); break
        d.ctrl[:] = tgt
        mujoco.mj_step(m, d)
        if i % every == 0:
            rend.update_scene(d, camera=cam, scene_option=vopt)
            from PIL import Image
            frames.append(Image.fromarray(rend.render()).convert("P", palette=Image.ADAPTIVE, colors=64))
    pz = d.xpos[pid][2]
    w, x, y, zq = d.qpos[3:7]
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y-zq*x), -1, 1)))
    ok = pz < 0.25 and abs(pitch) < 35
    print(f"  Phase 2: sat down, pelvis z={pz:.3f} pitch={pitch:+.0f} -> {'SEATED ✓' if ok else 'check'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    print('executing plan: [goto chair (avoid obstacles), sit]')
    frames = []
    arrived = phase1_walk(frames, args.fps)
    n1 = len(frames)
    seated = phase2_sit(frames, args.fps)
    print(f"RESULT: walked to chair={arrived}, sat down={seated}  ({n1} walk + {len(frames)-n1} sit frames)")

    if args.video and frames:
        out = os.path.abspath(args.video); os.makedirs(os.path.dirname(out), exist_ok=True)
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000/args.fps), loop=0, optimize=True)
        print(f"video -> {out} ({os.path.getsize(out)/1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
