"""Train G1 one-leg standing on Mac CPU with PyTorch (SB3). PPO by default; SAC
is a near-drop-in for the planned A/B (off-policy may fit the Mac's low-env
regime — see docs/FULL_MISSION_DEPLOY.md).

  .venv-rl/bin/python training/g1_oneleg_train.py --algo ppo --steps 8000000 --envs 24
  .venv-rl/bin/python training/g1_oneleg_train.py --algo sac --steps 3000000 --envs 8

Checkpoints + VecNormalize land in runs_oneleg/ (gitignored). One-leg success
rate (episode held a >=1.5 s one-leg stand) is printed every rollout.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "runs_oneleg")


def make_env(rank):
    def _f():
        from g1_oneleg_env import G1OneLegEnv
        return Monitor(G1OneLegEnv(seed=1000 + rank), info_keywords=("success",))
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
            print(f"[one-leg success last {len(recent)} eps] "
                  f"{np.mean(recent) * 100:.1f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--steps", type=int, default=8_000_000)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    name = f"{args.algo}_oneleg"
    venv = SubprocVecEnv([make_env(i) for i in range(args.envs)])
    # off-policy + reward-normalization interact badly (buffer holds stale norms);
    # only PPO normalizes the reward
    venv = VecNormalize(venv, norm_obs=True, norm_reward=(args.algo == "ppo"),
                        clip_obs=10.0)

    if args.resume:
        cls = PPO if args.algo == "ppo" else SAC
        venv = VecNormalize.load(args.resume + "_vecnorm.pkl",
                                 SubprocVecEnv([make_env(i) for i in range(args.envs)]))
        model = cls.load(args.resume, env=venv)
        print(f"resumed {args.algo} from {args.resume}")
    elif args.algo == "ppo":
        model = PPO("MlpPolicy", venv, verbose=1, device=args.device,
                    n_steps=512, batch_size=4096, learning_rate=3e-4,
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.005, clip_range=0.2,
                    policy_kwargs=dict(net_arch=[256, 256]))
    else:
        model = SAC("MlpPolicy", venv, verbose=1, device=args.device,
                    buffer_size=300_000, batch_size=256, learning_rate=3e-4,
                    train_freq=1, gradient_steps=1, learning_starts=10_000,
                    policy_kwargs=dict(net_arch=[256, 256]))

    ckpt = CheckpointCallback(save_freq=max(500_000 // args.envs, 1),
                              save_path=RUNS, name_prefix=name,
                              save_vecnormalize=True)
    try:
        model.learn(total_timesteps=args.steps, callback=[ckpt, SuccessLogger()],
                    reset_num_timesteps=not bool(args.resume))
    finally:
        model.save(os.path.join(RUNS, name + "_latest"))
        venv.save(os.path.join(RUNS, name + "_latest_vecnorm.pkl"))
        print("saved", os.path.join(RUNS, name + "_latest"))


if __name__ == "__main__":
    main()
