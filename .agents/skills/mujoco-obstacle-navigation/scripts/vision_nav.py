"""Camera-driven obstacle avoidance: the G1 avoids using its own forward DEPTH CAMERA, not rays.

Unlike g1_nav_demo (abstract mj_ray fan from the pelvis), here the robot perceives obstacles the
way a real robot does — a forward-facing depth camera mounted on the body renders what it sees
each control tick; per-bearing minimum depth gives the obstacle distances; a VFH gap-finder with a
generous safety bubble + speed-scaling turns that into a (vx, wz) command for the pretrained walk.
This both honours "use the camera to decide how to move" and fixes the bumping (bigger margins,
slow down when something is close). NVIDIA-free, CPU + CGL depth on macOS.

Usage: .venv-llm/bin/python vision_nav.py [--video out.gif]
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
gw = importlib.util.module_from_spec(spec); spec.loader.exec_module(gw)

GOAL = np.array([4.0, 0.0])
OBST_GROUP = 4
OBSTACLES = [(1.5, 0.3, "cyl"), (2.5, -0.5, "cyl"), (2.1, 1.0, "box")]
CAM_W, CAM_H, FOVY = 160, 120, 100.0
FOVX = np.degrees(2 * np.arctan(np.tan(np.radians(FOVY) / 2) * CAM_W / CAM_H))
NB = 16                                   # bearings extracted from the depth image (divides CAM_W=160)
SAFE, SLOW = 0.95, 1.8                    # block within SAFE m; slow down within SLOW m


def build():
    sp = mujoco.MjSpec.from_file(os.path.join(gw.VENDOR, "model", "scene.xml"))
    for i, (ox, oy, k) in enumerate(OBSTACLES):
        g = sp.worldbody.add_geom(); g.name = f"obs{i}"; g.group = OBST_GROUP; g.rgba = [0.8, 0.25, 0.2, 1]
        if k == "cyl":
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER; g.size = [0.28, 0.6, 0]; g.pos = [ox, oy, 0.6]
        else:
            g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.3, 0.3, 0.6]; g.pos = [ox, oy, 0.6]
    goal = sp.worldbody.add_geom(); goal.name = "goal"; goal.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    goal.size = [0.18, 0.005, 0]; goal.pos = [GOAL[0], GOAL[1], 0.005]; goal.rgba = [0.1, 0.9, 0.2, 0.6]
    goal.contype = goal.conaffinity = 0; goal.group = 5
    pelvis = [b for b in sp.bodies if b.name == "pelvis"][0]
    cam = pelvis.add_camera(); cam.name = "head"; cam.pos = [0.12, 0, 0.42]; cam.fovy = FOVY
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float).flatten())
    cam.quat = q
    return sp.compile()


def yaw_of(d):
    w, x, y, z = d.qpos[3:7]
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def make_planner(m, d, rend, cid, vopt):
    bears = np.linspace(-FOVX/2, FOVX/2, NB)        # +left ... -right (image cols, flipped to body)
    body_bears = -np.radians(bears)                 # image column -> body bearing (left = +yaw)

    def depth_ranges():
        rend.enable_depth_rendering()
        rend.update_scene(d, camera=cid, scene_option=vopt)
        dep = rend.render()
        rend.disable_depth_rendering()
        band = dep[int(CAM_H*0.42):int(CAM_H*0.62), :]      # horizon band
        cols = band.min(axis=0)
        per = cols.reshape(NB, CAM_W // NB).min(axis=1)      # min depth per bearing
        return np.clip(per, 0.1, 6.0)

    def planner(_t=0.0):
        rng = depth_ranges()
        yaw = yaw_of(d)
        dx, dy = GOAL - d.qpos[:2]
        goal_dir = (np.arctan2(dy, dx) - yaw + np.pi) % (2*np.pi) - np.pi
        blk = (rng < SAFE).copy()
        for i in range(NB):                                  # widen blocked sectors (robot width)
            if rng[i] < SAFE:
                for j in (i-1, i+1):
                    if 0 <= j < NB:
                        blk[j] = True
        free = np.where(~blk)[0]
        if len(free) == 0:
            return [0.0, 0.0, 0.6]                           # all blocked ahead -> turn to scan
        # prefer the free bearing closest to the goal direction (clamped to camera FOV)
        gd = np.clip(goal_dir, body_bears.min(), body_bears.max())
        best = free[np.argmin(np.abs(body_bears[free] - gd))]
        near = rng.min()
        vx = 0.5 * float(np.clip((near - SAFE) / (SLOW - SAFE), 0.22, 1.0))   # slow as obstacles near
        wz = float(np.clip(1.4 * body_bears[best], -0.6, 0.6))
        return [vx, 0.0, wz]
    return planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--secs", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    m = build(); m.opt.timestep = gw.SIM_DT
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    policy = torch.jit.load(os.path.join(gw.VENDOR, "motion.pt"))
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head")
    dvopt = mujoco.MjvOption(); dvopt.geomgroup[OBST_GROUP] = 1          # depth cam sees obstacles
    depth_rend = mujoco.Renderer(m, CAM_H, CAM_W)
    planner = make_planner(m, d, depth_rend, cid, dvopt)

    # obstacle clearance check + optional side-by-side video
    obs_xy = [(o[0], o[1]) for o in OBSTACLES]
    state = {"k": 0, "minclear": 9.9, "reached": False}
    view_rend = cam = frames = vvopt = None
    if args.video:
        cam = mujoco.MjvCamera(); cam.lookat = [2.2, 0.2, 0.3]; cam.distance, cam.azimuth, cam.elevation = 5.5, 90, -55
        vvopt = mujoco.MjvOption(); vvopt.geomgroup[OBST_GROUP] = 1; vvopt.geomgroup[5] = 1
        view_rend = mujoco.Renderer(m, 360, 360); frames = []
    every = int(round(1.0 / (args.fps * gw.SIM_DT)))

    def log(dd):
        for ox, oy in obs_xy:
            state["minclear"] = min(state["minclear"], np.hypot(ox-dd.qpos[0], oy-dd.qpos[1]) - 0.28)
        if np.hypot(*(GOAL - dd.qpos[:2])) < 0.35:
            state["reached"] = True
        if view_rend is not None and state["k"] % every == 0:
            from PIL import Image
            depth_rend.enable_depth_rendering(); depth_rend.update_scene(dd, camera=cid, scene_option=dvopt)
            dep = depth_rend.render(); depth_rend.disable_depth_rendering()
            dn = (255 * (1 - np.clip(dep/5.0, 0, 1))).astype(np.uint8)         # near=bright
            dimg = Image.fromarray(dn).convert("RGB").resize((360, 360))
            view_rend.update_scene(dd, camera=cam, scene_option=vvopt)
            over = Image.fromarray(view_rend.render())
            canvas = Image.new("RGB", (720, 360)); canvas.paste(dimg, (0, 0)); canvas.paste(over, (360, 0))
            frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=64))
        state["k"] += 1

    gw.walk(m, d, policy, planner, int(args.secs / gw.SIM_DT), log=log)
    print(f"final=({d.qpos[0]:.2f},{d.qpos[1]:.2f}) reached={state['reached']} upright={d.qpos[2]>0.5}")
    print(f"min clearance to any obstacle = {state['minclear']:.2f} m  -> {'NO BUMP ✓' if state['minclear']>0.0 else 'BUMPED ✗'}")
    if args.video and frames:
        out = os.path.abspath(args.video); os.makedirs(os.path.dirname(out), exist_ok=True)
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000/args.fps), loop=0, optimize=True)
        print(f"video -> {out} ({os.path.getsize(out)/1e6:.2f} MB)  [left=depth cam view, right=overview]")


if __name__ == "__main__":
    main()
