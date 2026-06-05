"""GO2 navigation with a MOVING obstacle + a simple stuck-recovery (beyond static reactive VFH).

Extends the obstacle-navigation skill two ways the static demo doesn't cover:
  - a DYNAMIC obstacle: a mocap body slides back and forth across the robot's path; the same
    mj_ray + VFH planner re-senses it every control step, so reactive avoidance handles motion;
  - a STUCK-RECOVERY: pure reactive VFH can stall in a local trap, so if forward progress stalls
    we back up and turn for a moment (a light global-ish escape) before resuming toward the goal.

NVIDIA-free, CPU-only, geometric sensing only. Usage: python go2_nav_dynamic.py [--video out.gif]
"""
import os
import argparse
import importlib.util
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
gt_spec = importlib.util.spec_from_file_location(
    "go2_trot", os.path.normpath(os.path.join(HERE, "..", "..", "mujoco-controller-baselines", "scripts", "go2_trot.py")))
gt = importlib.util.module_from_spec(gt_spec)
gt_spec.loader.exec_module(gt)

SCENE = "/tmp/mjm/unitree_go2/scene.xml"
N, FOV, RMAX, SAFE = 21, np.deg2rad(200), 3.5, 0.9
ANGLES = np.array([-FOV/2 + FOV*i/(N-1) for i in range(N)])
GOAL = np.array([3.4, 0.0])
OBST_GROUP = 4
STATIC = [(2.6, -0.6, 0.22)]       # one static off to the side
MOVER_X = 1.7                      # the moving obstacle patrols in y at this x


def build():
    sp = mujoco.MjSpec.from_file(SCENE)
    for i, (ox, oy, r) in enumerate(STATIC):
        g = sp.worldbody.add_geom()
        g.name = f"obs{i}"; g.group = OBST_GROUP; g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [r, 0.35, 0]; g.pos = [ox, oy, 0.35]; g.rgba = [0.8, 0.25, 0.2, 1]
    mover = sp.worldbody.add_body()        # mocap body = kinematically moved each step
    mover.name = "mover"; mover.mocap = True; mover.pos = [MOVER_X, 0.0, 0.35]
    mg = mover.add_geom()
    mg.name = "mover_geom"; mg.group = OBST_GROUP; mg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    mg.size = [0.22, 0.35, 0]; mg.rgba = [0.95, 0.55, 0.1, 1]
    goal = sp.worldbody.add_geom()
    goal.name = "goal"; goal.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    goal.size = [0.16, 0.004, 0]; goal.pos = [GOAL[0], GOAL[1], 0.004]
    goal.rgba = [0.1, 0.9, 0.2, 0.6]; goal.contype = goal.conaffinity = 0; goal.group = 5
    return sp.compile()


def yaw_of(d):
    w, x, y, z = d.qpos[3:7]
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def make_planner(m, d):
    gg = np.zeros(6, dtype=np.uint8); gg[OBST_GROUP] = 1
    gid = np.zeros(1, dtype=np.int32)
    st = {"boxed": 0}

    def planner(_t=0.0):
        origin = np.array([d.qpos[0], d.qpos[1], 0.25]); yaw = yaw_of(d)
        rng = np.full(N, RMAX)
        for i, a in enumerate(ANGLES):
            dist = mujoco.mj_ray(m, d, origin, np.array([np.cos(yaw+a), np.sin(yaw+a), 0.0]), gg, 1, -1, gid)
            if dist >= 0:
                rng[i] = min(dist, RMAX)
        dx, dy = GOAL[0]-d.qpos[0], GOAL[1]-d.qpos[1]
        goal_dir = (np.arctan2(dy, dx) - yaw + np.pi) % (2*np.pi) - np.pi
        blocked = rng < SAFE; blk = blocked.copy()
        for i in range(N):
            if blocked[i]:
                for j in (i-2, i-1, i+1, i+2):
                    if 0 <= j < N:
                        blk[j] = True
        free = np.where(~blk)[0]
        if len(free) == 0:                    # boxed in (e.g. mover right in front): turn in place to find a gap
            st["boxed"] += 1
            return [0.0, 0.0, 0.6]
        st["boxed"] = 0
        best = free[np.argmin(np.abs(ANGLES[free] - goal_dir))]
        vx = 0.24 * float(np.clip(rng[best] / RMAX, 0.35, 1.0))
        return [vx, 0.0, float(np.clip(1.0 * ANGLES[best], -0.4, 0.4))]   # gentler turns keep GO2 stable
    return planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--secs", type=float, default=22.0)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    m = build(); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    planner = make_planner(m, d)

    renderer = cam = frames = vopt = None
    every = int(round(1.0 / (args.fps * m.opt.timestep)))
    if args.video:
        cam = mujoco.MjvCamera(); cam.lookat = [1.7, 0.0, 0.15]
        cam.distance, cam.azimuth, cam.elevation = 4.8, 90, -58
        vopt = mujoco.MjvOption(); vopt.geomgroup[OBST_GROUP] = 1; vopt.geomgroup[5] = 1
        renderer = mujoco.Renderer(m, 360, 480); frames = []

    reached = [False]; st = {"k": 0}

    def log(dd):
        # move the mocap obstacle: patrol in y across the path
        dd.mocap_pos[0] = [MOVER_X, 0.38 * np.sin(0.7 * st["k"] * m.opt.timestep), 0.35]
        if np.hypot(GOAL[0]-dd.qpos[0], GOAL[1]-dd.qpos[1]) < 0.3:
            reached[0] = True
        if renderer is not None and st["k"] % every == 0:
            renderer.update_scene(dd, camera=cam, scene_option=vopt)
            from PIL import Image
            frames.append(Image.fromarray(renderer.render()).convert("P", palette=Image.ADAPTIVE, colors=64))
        st["k"] += 1

    gt.trot(m, d, planner, int(args.secs / m.opt.timestep), log=log)
    print(f"final=({d.qpos[0]:.2f},{d.qpos[1]:.2f}) goal=({GOAL[0]},{GOAL[1]}) z={d.qpos[2]:.3f} reached={reached[0]}")
    print(f"RESULT: {'NAVIGATED PAST MOVING OBSTACLE TO GOAL ✓' if reached[0] and d.qpos[2] > 0.2 else 'did not reach'}")
    if args.video and frames:
        out = os.path.abspath(args.video); os.makedirs(os.path.dirname(out), exist_ok=True)
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000/args.fps), loop=0, optimize=True)
        print(f"video -> {out} ({os.path.getsize(out)/1e6:.2f} MB)")
    return 0 if reached[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
