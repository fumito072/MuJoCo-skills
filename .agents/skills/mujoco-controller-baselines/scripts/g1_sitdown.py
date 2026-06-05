"""G1 stand -> floor-sit: replay the CEM-optimized open-loop descent trajectory.

The trajectory in assets/g1_sitdown_traj.npz was found offline by g1_sitdown_optimize.py
(hand-scripting this balance-critical transition fails; search finds a non-toppling path).
This is the sit-down SKILL: load the params, replay, the G1 sits down on the floor.

Sim-valid open-loop (deterministic). Real-robot transfer needs added feedback (sim-to-real
loop) — by design we iterate: optimize in sim -> test on hardware -> feed back -> re-optimize.

Usage: python g1_sitdown.py [scene.xml] [--video out.gif]
"""
import os
import argparse
import importlib.util
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("opt", os.path.join(HERE, "g1_sitdown_optimize.py"))
opt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="/tmp/mjm/unitree_g1/scene.xml")
    ap.add_argument("--traj", default=os.path.join(HERE, "..", "assets", "g1_sitdown_traj.npz"))
    ap.add_argument("--video", default="")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    z = np.load(args.traj)
    params = z["params"]
    sim = opt.Sim(args.scene)

    renderer = cam = frames = None
    if args.video:
        pid = sim.pid
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = pid
        cam.distance, cam.azimuth, cam.elevation = 2.4, 135, -12
        renderer = mujoco.Renderer(sim.m, height=240, width=320)
        frames = []
        every = int(round(1.0 / (args.fps * sim.dt)))
        st = {"k": 0}

        # patch rollout to capture frames: re-run the trajectory here with rendering
    # rebuild + run the trajectory (mirror of opt.Sim.rollout, with optional render)
    m, d = sim.m, sim.d
    k1 = opt.pose_from_channels(sim.home, np.clip(params[:6], opt.CH_LO, opt.CH_HI))
    k2 = opt.pose_from_channels(sim.home, np.clip(params[6:], opt.CH_LO, opt.CH_HI))
    seat = opt.pose_from_channels(sim.home, opt.SEATED_CH)
    wps = [sim.home, k1, k2, seat]
    T = [0.0, opt.T_DESC * 0.4, opt.T_DESC * 0.75, opt.T_DESC]
    mujoco.mj_resetDataKeyframe(m, d, 0)
    n = int((opt.T_DESC + opt.T_SETTLE) / sim.dt)
    max_tilt = 0.0
    for i in range(n):
        t = min(i * sim.dt, opt.T_DESC)
        for k in range(len(T) - 1):
            if t <= T[k + 1]:
                tgt = opt.smooth(wps[k], wps[k + 1], (t - T[k]) / (T[k + 1] - T[k]))
                break
        else:
            tgt = wps[-1]
        d.ctrl[:] = tgt
        mujoco.mj_step(m, d)
        w, x, y, zq = d.qpos[3:7]
        roll = np.degrees(np.arctan2(2*(w*x+y*zq), 1-2*(x*x+y*y)))
        pitch = np.degrees(np.arcsin(np.clip(2*(w*y-zq*x), -1, 1)))
        max_tilt = max(max_tilt, abs(roll), abs(pitch))
        if renderer is not None and i % every == 0:
            renderer.update_scene(d, camera=cam)
            from PIL import Image
            frames.append(Image.fromarray(renderer.render()).convert("P", palette=Image.ADAPTIVE, colors=48))

    pz = d.xpos[sim.pid][2]
    w, x, y, zq = d.qpos[3:7]
    f_roll = np.degrees(np.arctan2(2*(w*x+y*zq), 1-2*(x*x+y*y)))
    f_pitch = np.degrees(np.arcsin(np.clip(2*(w*y-zq*x), -1, 1)))
    ok = pz < 0.22 and abs(f_roll) < 25 and abs(f_pitch) < 30 and max_tilt < 80
    print(f"sit-down: final pelvis z={pz:.3f}  final tilt(roll {f_roll:+.1f}, pitch {f_pitch:+.1f})  max_tilt={max_tilt:.0f}")
    print(f"RESULT: {'SAT DOWN ON FLOOR ✓' if ok else 'FAILED'}")

    if args.video and frames:
        out = os.path.abspath(args.video)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000/args.fps), loop=0, optimize=True)
        print(f"video -> {out} ({os.path.getsize(out)/1e6:.2f} MB, {len(frames)} frames)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
