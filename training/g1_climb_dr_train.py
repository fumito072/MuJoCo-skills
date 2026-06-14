"""Train the domain-randomized step-up (g1_climb_dr_env). One policy for ALL
heights 0.02-0.22 m (height randomized per episode + observed), with a
potential-based reward. SB3 PPO, Mac CPU + PyTorch.

  .venv-rl/bin/python training/g1_climb_dr_train.py --steps 30000000 --envs 24

Floor-start success is logged bucketed by height (low 0.02-0.09 / mid 0.09-0.16 /
high 0.16-0.22) so you can see whether the TALL steps are learning, not just the
easy ones. RSI starts are excluded from the metric.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "runs_climb")


def make_env(rank):
    def _f():
        from g1_climb_dr_env import G1ClimbDREnv
        return Monitor(G1ClimbDREnv(seed=3000 + rank),
                       info_keywords=("success", "rsi", "step_h"))
    return _f


class SuccessLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.hist = []   # (height, success) for floor-start episodes

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None and not ep.get("rsi"):
                self.hist.append((float(ep.get("step_h", 0.0)),
                                  1.0 if ep.get("success") else 0.0))
        return True

    def _on_rollout_end(self):
        if not self.hist:
            return
        recent = self.hist[-400:]
        def rate(lo, hi):
            v = [s for h, s in recent if lo <= h < hi]
            return (100 * np.mean(v), len(v)) if v else (float("nan"), 0)
        allr = 100 * np.mean([s for _, s in recent])
        lo = rate(0.02, 0.09); mid = rate(0.09, 0.16); hi = rate(0.16, 0.221)
        self.logger.record("custom/success_rate", float(allr) / 100)
        print(f"[floor-start success] all {allr:.0f}% | "
              f"low {lo[0]:.0f}%(n{lo[1]}) mid {mid[0]:.0f}%(n{mid[1]}) "
              f"high {hi[0]:.0f}%(n{hi[1]})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30_000_000)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--ent_coef", type=float, default=0.005)
    args = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    name = "climb_dr"
    venv = SubprocVecEnv([make_env(i) for i in range(args.envs)])
    if args.resume:
        venv = VecNormalize.load(args.resume + "_vecnorm.pkl", venv)
        venv.training = True
        venv.norm_reward = True
        model = PPO.load(args.resume, env=venv, device=args.device)
        model.ent_coef = args.ent_coef
        print(f"resumed from {args.resume}", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", venv, verbose=1, device=args.device,
                    n_steps=512, batch_size=4096, learning_rate=3e-4,
                    gamma=0.99, gae_lambda=0.95, ent_coef=args.ent_coef, clip_range=0.2,
                    policy_kwargs=dict(net_arch=[256, 256]))

    ckpt = CheckpointCallback(save_freq=max(1_000_000 // args.envs, 1),
                              save_path=RUNS, name_prefix=name, save_vecnormalize=True)
    try:
        model.learn(total_timesteps=args.steps, callback=[ckpt, SuccessLogger()],
                    reset_num_timesteps=not bool(args.resume))
    finally:
        model.save(os.path.join(RUNS, name + "_latest"))
        venv.save(os.path.join(RUNS, name + "_latest_vecnorm.pkl"))
        print("saved", os.path.join(RUNS, name + "_latest"), flush=True)


if __name__ == "__main__":
    main()
