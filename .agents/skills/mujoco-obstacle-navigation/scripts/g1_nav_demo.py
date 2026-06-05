"""G1 obstacle-avoidance navigation: the pretrained walk steered by a reactive VFH planner.

Hierarchical, NVIDIA-free, CPU-only:
  SENSE   - cast a fan of rays (mujoco.mj_ray) from the pelvis against OBSTACLE geoms only
            (a dedicated geom group, so rays never hit the robot itself).
  PLAN    - a stateless VFH gap-finder turns the range fan into a body velocity command
            (vx, wz) toward the goal, around obstacles.
  ACT     - that (vx, vy, wz) command drives the pretrained G1 walk (g1_walk.walk), which
            already tracks velocity commands. The planner is passed as the walk's callable cmd.

The ONLY coupling between perception and locomotion is the (vx, vy, wz) command — so the same
planner can later ride any velocity-tracking gait (e.g. a steerable GO2 trot).

Usage: python g1_nav_demo.py [--video out.gif]
"""
import os
import argparse
import importlib.util
import numpy as np
import mujoco
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GW = os.path.normpath(os.path.join(HERE, "..", "..", "mujoco-pretrained-deploy", "scripts", "g1_walk.py"))
spec = importlib.util.spec_from_file_location("g1_walk", GW)
gw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gw)

# fan + planner params
N, FOV, RMAX, SAFE = 21, np.deg2rad(200), 4.0, 1.1
ANGLES = np.array([-FOV/2 + FOV*i/(N-1) for i in range(N)])
GOAL = np.array([4.0, 0.0])
OBST_GROUP = 4
# (x, y, kind, sx, sy) obstacles ahead of the robot
OBSTACLES = [(1.8, 0.35, "cyl", 0.28, 0.0),
             (2.7, -0.55, "cyl", 0.28, 0.0),
             (2.4, 1.0, "box", 0.3, 0.3)]


def build_nav_scene():
    sp = mujoco.MjSpec.from_file(os.path.join(gw.VENDOR, "model", "scene.xml"))
    for i, (ox, oy, kind, sx, sy) in enumerate(OBSTACLES):
        g = sp.worldbody.add_geom()
        g.name = f"obs{i}"
        g.group = OBST_GROUP
        g.rgba = [0.8, 0.25, 0.2, 1]
        if kind == "cyl":
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = [sx, 0.6, 0]
            g.pos = [ox, oy, 0.6]
        else:
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [sx, sy, 0.6]
            g.pos = [ox, oy, 0.6]
    goal = sp.worldbody.add_geom()
    goal.name = "goal"
    goal.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    goal.size = [0.18, 0.005, 0]
    goal.pos = [GOAL[0], GOAL[1], 0.005]
    goal.rgba = [0.1, 0.9, 0.2, 0.6]
    goal.contype = 0
    goal.conaffinity = 0
    goal.group = 5
    return sp.compile()


def yaw_of(d):
    w, x, y, z = d.qpos[3:7]
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def make_planner(m, d, pid):
    # walk() calls cmd(t); we ignore t and read the SAME live MjData `d` it is stepping.
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[OBST_GROUP] = 1                      # rays hit ONLY obstacle geoms
    gid = np.zeros(1, dtype=np.int32)

    def planner(_t=0.0):
        origin = np.array([d.qpos[0], d.qpos[1], 0.5])
        yaw = yaw_of(d)
        rng = np.full(N, RMAX)
        for i, a in enumerate(ANGLES):
            vec = np.array([np.cos(yaw + a), np.sin(yaw + a), 0.0])
            dist = mujoco.mj_ray(m, d, origin, vec, geomgroup, 1, -1, gid)
            if dist >= 0:
                rng[i] = min(dist, RMAX)
        # goal heading in body frame
        dx, dy = GOAL[0] - d.qpos[0], GOAL[1] - d.qpos[1]
        goal_dir = (np.arctan2(dy, dx) - yaw + np.pi) % (2*np.pi) - np.pi
        # VFH: blocked sectors widened by robot radius, pick free sector nearest goal
        blocked = rng < SAFE
        blk = blocked.copy()
        for i in range(N):
            if blocked[i]:
                for j in (i-2, i-1, i+1, i+2):
                    if 0 <= j < N:
                        blk[j] = True
        free = np.where(~blk)[0]
        if len(free) == 0:
            return [0.0, 0.0, 0.6]                 # boxed in: turn in place to search
        best = free[np.argmin(np.abs(ANGLES[free] - goal_dir))]
        steer = ANGLES[best]
        vx = 0.5 * float(np.clip(rng[best] / RMAX, 0.3, 1.0))
        wz = float(np.clip(1.5 * steer, -0.6, 0.6))
        return [vx, 0.0, wz]
    return planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--secs", type=float, default=14.0)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    m = build_nav_scene()
    m.opt.timestep = gw.SIM_DT
    d = mujoco.MjData(m)
    policy = torch.jit.load(os.path.join(gw.VENDOR, "motion.pt"))
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    planner = make_planner(m, d, pid)

    renderer = cam = frames = vopt = None
    if args.video:
        cam = mujoco.MjvCamera()
        cam.lookat = [2.2, 0.2, 0.3]
        cam.distance, cam.azimuth, cam.elevation = 5.5, 90, -55
        vopt = mujoco.MjvOption()
        vopt.geomgroup[OBST_GROUP] = 1                 # show obstacles (group 4)
        vopt.geomgroup[5] = 1                          # and the goal marker (group 5)
        renderer = mujoco.Renderer(m, height=360, width=480)
        frames = []
        every = int(round(1.0 / (args.fps * gw.SIM_DT)))

    reached = [False]
    path = []
    st = {"k": 0}

    def log(dd):
        if st["k"] % 50 == 0:
            path.append((round(dd.qpos[0], 2), round(dd.qpos[1], 2)))
        if np.hypot(GOAL[0]-dd.qpos[0], GOAL[1]-dd.qpos[1]) < 0.35:
            reached[0] = True
        if renderer is not None and st["k"] % every == 0:
            renderer.update_scene(dd, camera=cam, scene_option=vopt)
            from PIL import Image
            frames.append(Image.fromarray(renderer.render()).convert("P", palette=Image.ADAPTIVE, colors=64))
        st["k"] += 1

    gw.walk(m, d, policy, planner, int(args.secs / gw.SIM_DT), log=log)
    fx, fy = d.qpos[0], d.qpos[1]
    upright = d.qpos[2] > 0.5
    print(f"path (x,y) sampled: {path[::3]}")
    print(f"final pos=({fx:.2f},{fy:.2f}) goal=({GOAL[0]},{GOAL[1]})  upright={upright}  reached={reached[0]}")
    print(f"RESULT: {'NAVIGATED TO GOAL ✓' if reached[0] and upright else ('reached but check' if reached[0] else 'did not reach')}")

    if args.video and frames:
        out = os.path.abspath(args.video)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000/args.fps), loop=0, optimize=True)
        print(f"video -> {out} ({os.path.getsize(out)/1e6:.2f} MB, {len(frames)} frames)")
    return 0 if reached[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
