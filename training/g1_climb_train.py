"""PPO training for the backward step-up skill (g1_climb_env) on Mac CPU.

Usage:
    .venv-rl/bin/python training/g1_climb_train.py [--steps 6000000] [--envs 10]
    .venv-rl/bin/python training/g1_climb_train.py --resume runs_climb/ppo_climb_latest

Checkpoints + VecNormalize land in runs_climb/ (gitignored artifacts except the
final policy we ship). Success rate is logged every rollout from episode infos.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "runs_climb")


def make_env(rank):
    def _f():
        from g1_climb_env import G1ClimbEnv
        from stable_baselines3.common.monitor import Monitor
        return Monitor(G1ClimbEnv(seed=1000 + rank), info_keywords=("success",))
    return _f


class SuccessLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.hist = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.hist.append(1.0 if info["episode"].get("success") else 0.0)
        return True

    def _on_rollout_end(self):
        if self.hist:
            recent = self.hist[-200:]
            self.logger.record("custom/success_rate", float(np.mean(recent)))
            print(f"[success_rate last {len(recent)} eps] {np.mean(recent) * 100:.1f}%",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6_000_000)
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    venv = SubprocVecEnv([make_env(i) for i in range(args.envs)])
    if args.resume:
        venv = VecNormalize.load(args.resume + "_vecnorm.pkl", venv)
        model = PPO.load(args.resume, env=venv)
        print(f"resumed from {args.resume}")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy", venv, verbose=1, device="cpu",
            n_steps=512, batch_size=4096, learning_rate=3e-4,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.005, clip_range=0.2,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=None,
        )

    ckpt = CheckpointCallback(save_freq=max(500_000 // args.envs, 1),
                              save_path=RUNS, name_prefix="ppo_climb",
                              save_vecnormalize=True)
    try:
        model.learn(total_timesteps=args.steps, callback=[ckpt, SuccessLogger()],
                    reset_num_timesteps=not bool(args.resume))
    finally:
        model.save(os.path.join(RUNS, "ppo_climb_latest"))
        venv.save(os.path.join(RUNS, "ppo_climb_latest_vecnorm.pkl"))
        print("saved", os.path.join(RUNS, "ppo_climb_latest"))


if __name__ == "__main__":
    main()
