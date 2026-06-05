"""GO2 obstacle-avoidance navigation: the model-based trot steered by the SAME VFH planner.

Demonstrates the reuse promise: the identical sense->plan->act stack from g1_nav_demo rides a
different locomotion layer (the steerable GO2 trot, gt.trot) instead of the G1 walk. The only
coupling is the (vx, vy, wz) velocity command. NVIDIA-free, CPU-only, geometric sensing.

Usage: python go2_nav_demo.py [--video out.gif]
"""
import os
import argparse
import importlib.util
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.normpath(os.path.join(HERE, "..", "..", "mujoco-controller-baselines", "scripts", "go2_trot.py"))
spec = importlib.util.spec_from_file_location("go2_trot", GT)
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)   # go2_trot.main is __main__-guarded, so this does not parse args

SCENE = "/tmp/mjm/unitree_go2/scene.xml"
N, FOV, RMAX, SAFE = 21, np.deg2rad(200), 3.5, 0.9
ANGLES = np.array([-FOV/2 + FOV*i/(N-1) for i in range(N)])
GOAL = np.array([3.2, 0.0])
OBST_GROUP = 4
OBSTACLES = [(1.3, 0.3, 0.22), (2.1, -0.45, 0.22), (1.8, 0.85, 0.22)]


def build_nav_scene():
    sp = mujoco.MjSpec.from_file(SCENE)
    for i, (ox, oy, r) in enumerate(OBSTACLES):
        g = sp.worldbody.add_geom()
        g.name = f"obs{i}"
        g.group = OBST_GROUP
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [r, 0.35, 0]
        g.pos = [ox, oy, 0.35]
        g.rgba = [0.8, 0.25, 0.2, 1]
    goal = sp.worldbody.add_geom()
    goal.name = "goal"
    goal.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    goal.size = [0.15, 0.004, 0]
    goal.pos = [GOAL[0], GOAL[1], 0.004]
    goal.rgba = [0.1, 0.9, 0.2, 0.6]
    goal.contype = goal.conaffinity = 0
    goal.group = 5
    return sp.compile()


def yaw_of(d):
    w, x, y, z = d.qpos[3:7]
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def make_planner(m, d):
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[OBST_GROUP] = 1
    gid = np.zeros(1, dtype=np.int32)

    def planner(_t=0.0):
        origin = np.array([d.qpos[0], d.qpos[1], 0.25])
        yaw = yaw_of(d)
        rng = np.full(N, RMAX)
        for i, a in enumerate(ANGLES):
            vec = np.array([np.cos(yaw + a), np.sin(yaw + a), 0.0])
            dist = mujoco.mj_ray(m, d, origin, vec, geomgroup, 1, -1, gid)
            if dist >= 0:
                rng[i] = min(dist, RMAX)
        dx, dy = GOAL[0] - d.qpos[0], GOAL[1] - d.qpos[1]
        goal_dir = (np.arctan2(dy, dx) - yaw + np.pi) % (2*np.pi) - np.pi
        blocked = rng < SAFE
        blk = blocked.copy()
        for i in range(N):
            if blocked[i]:
                for j in (i-2, i-1, i+1, i+2):
                    if 0 <= j < N:
                        blk[j] = True
        free = np.where(~blk)[0]
        if len(free) == 0:
            return [0.05, 0.0, 0.6]
        best = free[np.argmin(np.abs(ANGLES[free] - goal_dir))]
        steer = ANGLES[best]
        vx = 0.26 * float(np.clip(rng[best] / RMAX, 0.3, 1.0))
        wz = float(np.clip(1.5 * steer, -0.7, 0.7))
        return [vx, 0.0, wz]
    return planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--secs", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    m = build_nav_scene()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    planner = make_planner(m, d)

    renderer = cam = frames = vopt = None
    every = int(round(1.0 / (args.fps * m.opt.timestep)))
    if args.video:
        cam = mujoco.MjvCamera()
        cam.lookat = [1.7, 0.1, 0.2]
        cam.distance, cam.azimuth, cam.elevation = 4.6, 90, -55
        vopt = mujoco.MjvOption()
        vopt.geomgroup[OBST_GROUP] = 1
        vopt.geomgroup[5] = 1
        renderer = mujoco.Renderer(m, height=360, width=480)
        frames = []

    reached = [False]
    path = []
    st = {"k": 0}

    def log(dd):
        if st["k"] % 50 == 0:
            path.append((round(dd.qpos[0], 2), round(dd.qpos[1], 2)))
        if np.hypot(GOAL[0]-dd.qpos[0], GOAL[1]-dd.qpos[1]) < 0.3:
            reached[0] = True
        if renderer is not None and st["k"] % every == 0:
            renderer.update_scene(dd, camera=cam, scene_option=vopt)
            from PIL import Image
            frames.append(Image.fromarray(renderer.render()).convert("P", palette=Image.ADAPTIVE, colors=64))
        st["k"] += 1

    gt.trot(m, d, planner, int(args.secs / m.opt.timestep), log=log)
    print(f"path (x,y): {path[::3]}")
    print(f"final=({d.qpos[0]:.2f},{d.qpos[1]:.2f}) goal=({GOAL[0]},{GOAL[1]})  z={d.qpos[2]:.3f}  reached={reached[0]}")
    up = d.qpos[2] > 0.2
    print(f"RESULT: {'NAVIGATED TO GOAL ✓' if reached[0] and up else ('reached?' if reached[0] else 'did not reach')}")

    if args.video and frames:
        out = os.path.abspath(args.video)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000/args.fps), loop=0, optimize=True)
        print(f"video -> {out} ({os.path.getsize(out)/1e6:.2f} MB, {len(frames)} frames)")
    return 0 if reached[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
