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
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from g1_climb_dr_env import H_MIN, H_MAX

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "runs_climb")


def make_env(rank):
    def _f():
        from g1_climb_dr_env import G1ClimbDREnv
        return Monitor(G1ClimbDREnv(seed=3000 + rank),
                       info_keywords=("success", "rsi", "step_h"))
    return _f


class AutoCurriculum(BaseCallback):
    """Grow the randomization cap as the TOP of the current range is mastered
    (user's expanding-range idea). Always sampling [H_MIN, cap] keeps low heights
    practiced (no forgetting); cap only rises when the top band succeeds (gives
    difficulty ordering). Floor-start episodes only."""
    def __init__(self, start_cap, step, band, thresh, min_n):
        super().__init__()
        self.cap = start_cap
        self.step, self.band, self.thresh, self.min_n = step, band, thresh, min_n
        self.recent = []   # rolling (height, success), floor-start only

    def _on_training_start(self):
        self.training_env.env_method("set_h_cap", self.cap)
        print(f"[curriculum] start cap={self.cap:.2f}", flush=True)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None and not ep.get("rsi"):
                self.recent.append((float(ep.get("step_h", 0.0)),
                                    1.0 if ep.get("success") else 0.0))
        if len(self.recent) > 800:
            self.recent = self.recent[-800:]
        return True

    def _on_rollout_end(self):
        if not self.recent:
            return
        band_lo = max(H_MIN, self.cap - self.band)
        top = [s for h, s in self.recent if h >= band_lo]
        allr = 100 * np.mean([s for _, s in self.recent])
        topr = (100 * np.mean(top), len(top)) if top else (float("nan"), 0)
        self.logger.record("custom/h_cap", self.cap)
        self.logger.record("custom/success_rate",
                           float(np.mean([s for _, s in self.recent])))
        print(f"[curriculum] cap={self.cap:.2f} | floor success all {allr:.0f}% | "
              f"top[{band_lo:.2f}-{self.cap:.2f}] {topr[0]:.0f}%(n{topr[1]})", flush=True)
        if self.cap < H_MAX and topr[1] >= self.min_n and topr[0] >= self.thresh * 100:
            self.cap = min(H_MAX, round(self.cap + self.step, 3))
            self.training_env.env_method("set_h_cap", self.cap)
            self.recent = []
            print(f"[curriculum] >>> MASTERED top band, EXPAND cap -> {self.cap:.2f}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30_000_000)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--ent_coef", type=float, default=0.005)
    ap.add_argument("--start_cap", type=float, default=0.04)
    ap.add_argument("--cap_step", type=float, default=0.01)
    ap.add_argument("--cap_band", type=float, default=0.03)
    ap.add_argument("--cap_thresh", type=float, default=0.4)
    ap.add_argument("--cap_min_n", type=int, default=80)
    ap.add_argument("--name", type=str, default="climb_dr")
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target_kl", type=float, default=None,
                    help="cap PPO update KL to prevent the divergence (peak-then-collapse)")
    args = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    name = args.name
    venv = SubprocVecEnv([make_env(i) for i in range(args.envs)])
    cls = PPO if args.algo == "ppo" else SAC
    # off-policy (SAC) + reward-normalization interact badly (buffer holds stale norms)
    norm_rew = (args.algo == "ppo")
    if args.resume:
        venv = VecNormalize.load(args.resume + "_vecnorm.pkl", venv)
        venv.training = True
        venv.norm_reward = norm_rew
        model = cls.load(args.resume, env=venv, device=args.device)
        if args.algo == "ppo":
            model.ent_coef = args.ent_coef
            if args.target_kl is not None:
                model.target_kl = args.target_kl
        model.learning_rate = args.lr
        print(f"resumed {args.algo} from {args.resume}", flush=True)
    elif args.algo == "ppo":
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", venv, verbose=1, device=args.device,
                    n_steps=512, batch_size=4096, learning_rate=args.lr,
                    gamma=0.99, gae_lambda=0.95, ent_coef=args.ent_coef, clip_range=0.2,
                    target_kl=args.target_kl,
                    policy_kwargs=dict(net_arch=[256, 256]))
    else:  # SAC — off-policy, replay buffer in (unified) RAM, far more stable for
           # continuous "reach and hold" than PPO (which diverged on the trivial stand)
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        model = SAC("MlpPolicy", venv, verbose=1, device=args.device,
                    buffer_size=400_000, batch_size=512, learning_rate=args.lr,
                    train_freq=1, gradient_steps=1, learning_starts=5_000,
                    ent_coef="auto", gamma=0.99,
                    policy_kwargs=dict(net_arch=[256, 256]))

    ckpt = CheckpointCallback(save_freq=max(1_000_000 // args.envs, 1),
                              save_path=RUNS, name_prefix=name, save_vecnormalize=True)
    curric = AutoCurriculum(args.start_cap, args.cap_step, args.cap_band,
                            args.cap_thresh, args.cap_min_n)
    try:
        model.learn(total_timesteps=args.steps, callback=[ckpt, curric],
                    reset_num_timesteps=not bool(args.resume))
    finally:
        model.save(os.path.join(RUNS, name + "_latest"))
        venv.save(os.path.join(RUNS, name + "_latest_vecnorm.pkl"))
        print("saved", os.path.join(RUNS, name + "_latest"), flush=True)


if __name__ == "__main__":
    main()
