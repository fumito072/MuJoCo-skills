"""Replay / evaluate a trained one-leg-stand policy on the Mac and render a GIF
of the best episode. NVIDIA-free (CPU + onnx-free; uses the SB3 model directly).

  .venv-rl/bin/python training/g1_oneleg_play.py \
      --model runs_oneleg/ppo_oneleg_latest --episodes 20 \
      --video assets/g1_oneleg.gif
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1_oneleg_env import G1OneLegEnv, CTRL_DT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path stem (without .zip)")
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--video", default=None, help="GIF of the longest-hold episode")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    cls = PPO if args.algo == "ppo" else SAC
    model = cls.load(args.model, device="cpu")
    vecnorm = None
    vn_path = args.model + "_vecnorm.pkl"
    if os.path.exists(vn_path):
        vecnorm = VecNormalize.load(vn_path, DummyVecEnv([lambda: G1OneLegEnv()]))
        vecnorm.training = False
        vecnorm.norm_reward = False

    env = G1OneLegEnv(seed=12345)
    m, d = env.m, env.d

    rend = cam = None
    if args.video:
        m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
        rend = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        cam.distance, cam.azimuth, cam.elevation = 2.4, 130, -10

    holds, succ, angs = [], 0, []
    best_frames, best_hold = None, -1.0
    for ep in range(args.episodes):
        obs, _ = env.reset()
        frames = []
        max_held = 0.0
        ever = False
        ang_while_clean, z_while, waist_while = [], [], []
        swing_g0 = env.rf if env._stance_left else env.lf
        for k in range(int(6.0 / CTRL_DT)):
            o = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            act, _ = model.predict(o, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            max_held = max(max_held, info["held"])
            ever = ever or info["success"]
            if info["held"] > 0:                              # currently in a clean hold
                ang_while_clean.append(
                    float(np.linalg.norm(d.sensor("gyro_pelvis").data)))
                z_while.append(float(d.geom_xpos[swing_g0][2]))
                waist_while.append(abs(float(d.qpos[7 + 14] - env.default_pose[14])))
            if rend is not None:
                rend.update_scene(d, camera=cam)
                from PIL import Image
                frames.append(Image.fromarray(rend.render())
                              .convert("P", palette=Image.ADAPTIVE, colors=128))
            if term or trunc:
                break
        holds.append(max_held)
        succ += int(ever)
        mean_ang = float(np.mean(ang_while_clean)) if ang_while_clean else float("nan")
        mean_z = float(np.mean(z_while)) if z_while else float("nan")
        mean_waist = float(np.degrees(np.mean(waist_while))) if waist_while else float("nan")
        if ang_while_clean:
            angs.append(mean_ang)
        side = "L" if env._stance_left else "R"
        print(f"ep{ep:2d} stance={side}  hold={max_held:4.2f}s  "
              f"lift={mean_z:4.2f}m  torso_bend={mean_waist:4.1f}deg  "
              f"spin={mean_ang:4.2f}rad/s  {'SUCCESS' if ever else ''}")
        if args.video and max_held > best_hold:
            best_hold, best_frames = max_held, frames

    spin = float(np.mean(angs)) if angs else float("nan")
    print(f"\none-leg success (>=1.5 s STILL hold): {succ}/{args.episodes}  "
          f"| median max-hold {np.median(holds):.2f}s  best {max(holds):.2f}s"
          f"  | mean spin while held {spin:.2f} rad/s (lower = more static)")
    if args.video and best_frames:
        os.makedirs(os.path.dirname(args.video), exist_ok=True)
        step = max(1, int(1 / (args.fps * CTRL_DT)))
        fr = best_frames[::step]
        fr[0].save(args.video, save_all=True, append_images=fr[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
        print(f"gif ({best_hold:.2f}s hold): {args.video}")
    return 0 if succ > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
