"""Evaluate a DR step-up policy: RSI clean-stand success, FLOOR-start success, and
mean pitch, per height. Prints a JSON summary line for easy parsing.

  .venv-rl/bin/python training/g1_climb_dr_eval.py --model runs_climb/climb_dr_latest
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1_climb_dr_env import G1ClimbDREnv


def load_model(stem):
    try:
        return PPO.load(stem, device="cpu")
    except Exception:
        return SAC.load(stem, device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heights", default="0.02,0.08,0.15,0.22")
    ap.add_argument("--episodes", type=int, default=30)
    args = ap.parse_args()

    venv = VecNormalize.load(args.model + "_vecnorm.pkl",
                             DummyVecEnv([lambda: G1ClimbDREnv(fixed_h=0.1)]))
    venv.training = False
    venv.norm_reward = False
    m = load_model(args.model)
    heights = [float(x) for x in args.heights.split(",")]
    out = {}
    for H in heights:
        e = G1ClimbDREnv(seed=int(H * 1000), fixed_h=H)
        # RSI clean-stand hold
        e.rsi_p = 1.0
        rsi_ok = 0
        pitches = []
        for ep in range(args.episodes):
            o, _ = e.reset()
            for k in range(300):
                a, _ = m.predict(venv.normalize_obs(o), deterministic=True)
                o, r, term, trunc, info = e.step(a)
                d = e.d
                w, x_, y_, z_ = d.qpos[3:7]
                pitches.append(abs(np.degrees(np.arcsin(np.clip(2 * (w * y_ - z_ * x_), -1, 1)))))
                if term or trunc:
                    break
            rsi_ok += int(info["success"])
        # FLOOR-start climb+stand
        e.rsi_p = 0.0
        floor_ok = 0
        both = 0
        for ep in range(args.episodes):
            o, _ = e.reset()
            mf = 0
            for k in range(300):
                a, _ = m.predict(venv.normalize_obs(o), deterministic=True)
                o, r, term, trunc, info = e.step(a)
                mf = max(mf, info["feet_on_step"])
                if term or trunc:
                    break
            floor_ok += int(info["success"])
            both += int(mf >= 2)
        out[f"{H:.2f}"] = {
            "rsi_succ": round(rsi_ok / args.episodes, 2),
            "floor_succ": round(floor_ok / args.episodes, 2),
            "floor_both": round(both / args.episodes, 2),
            "mean_pitch": round(float(np.mean(pitches)), 1),
        }
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    raise SystemExit(main())
