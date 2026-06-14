"""Train the G1 STEP-UP by curriculum, RESUMING from the v4 one-leg policy
(2026-06-14). Same Pattern B as one-leg: SB3 PPO, Mac CPU + PyTorch, no GPU.

The whole point: --resume runs_oneleg/ppo_oneleg_latest loads the v4 weights
(obs/action are byte-identical) so the step-up starts from a policy that already
lifts the swing foot high with a steady torso. Then a HEIGHT curriculum:

  # Stage 1 (0.08 m) -- resume from the v4 one-leg policy
  .venv-rl/bin/python training/g1_climb_curriculum_train.py --step_h 0.08 \
      --resume runs_oneleg/ppo_oneleg_latest --steps 12000000 --envs 24
  # Stage 2 (0.14 m) -- resume from stage 1
  .venv-rl/bin/python training/g1_climb_curriculum_train.py --step_h 0.14 \
      --resume runs_climb/climb_curric_h0.08_latest --steps 12000000 --envs 24
  # Stage 3 (0.22 m) -- resume from stage 2
  .venv-rl/bin/python training/g1_climb_curriculum_train.py --step_h 0.22 \
      --resume runs_climb/climb_curric_h0.14_latest --steps 16000000 --envs 24

Checkpoints land in runs_climb/ (gitignored). Step-up success rate (episode
held a >=1.0 s two-foot platform stand) prints every rollout.
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


def make_env(rank, step_h):
    def _f():
        from g1_climb_curriculum_env import G1ClimbCurriculumEnv
        return Monitor(G1ClimbCurriculumEnv(seed=2000 + rank, step_h=step_h),
                       info_keywords=("success", "rsi"))
    return _f


class SuccessLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.hist = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info and not info["episode"].get("rsi"):
                # only FLOOR-start episodes count — RSI starts are near-success and
                # would inflate the metric; this number is the real climb-from-floor rate
                self.hist.append(1.0 if info["episode"].get("success") else 0.0)
        return True

    def _on_rollout_end(self):
        if self.hist:
            recent = self.hist[-200:]
            self.logger.record("custom/success_rate", float(np.mean(recent)))
            print(f"[step-up success (floor-start) last {len(recent)} eps] "
                  f"{np.mean(recent) * 100:.1f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step_h", type=float, required=True)
    ap.add_argument("--steps", type=int, default=12_000_000)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", type=str, default=None,
                    help="stem to warm-start from (v4 one-leg or a prior stage)")
    ap.add_argument("--reset_std", type=float, default=None,
                    help="on resume, reset the action std (v4 ended at ~22 = bang-bang; "
                         "0.5 restores fine control while keeping the learned features)")
    ap.add_argument("--ent_coef", type=float, default=0.005)
    args = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    name = f"climb_curric_h{args.step_h:.2f}"
    venv = SubprocVecEnv([make_env(i, args.step_h) for i in range(args.envs)])

    if args.resume:
        # warm-start: carry the policy AND the obs running-stats (they keep adapting
        # to the new step-env distribution since training stays True)
        venv = VecNormalize.load(args.resume + "_vecnorm.pkl", venv)
        venv.training = True
        venv.norm_reward = True
        model = PPO.load(args.resume, env=venv, device=args.device)
        model.ent_coef = args.ent_coef
        if args.reset_std is not None:
            import torch
            with torch.no_grad():
                model.policy.log_std.data.fill_(float(np.log(args.reset_std)))
            print(f"reset action std -> {args.reset_std}", flush=True)
        print(f"resumed from {args.resume} (step_h={args.step_h}, "
              f"ent_coef={args.ent_coef})", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", venv, verbose=1, device=args.device,
                    n_steps=512, batch_size=4096, learning_rate=3e-4,
                    gamma=0.99, gae_lambda=0.95, ent_coef=args.ent_coef, clip_range=0.2,
                    policy_kwargs=dict(net_arch=[256, 256]))

    ckpt = CheckpointCallback(save_freq=max(1_000_000 // args.envs, 1),
                              save_path=RUNS, name_prefix=name,
                              save_vecnormalize=True)
    try:
        model.learn(total_timesteps=args.steps, callback=[ckpt, SuccessLogger()],
                    reset_num_timesteps=True)
    finally:
        model.save(os.path.join(RUNS, name + "_latest"))
        venv.save(os.path.join(RUNS, name + "_latest_vecnorm.pkl"))
        print("saved", os.path.join(RUNS, name + "_latest"), flush=True)


if __name__ == "__main__":
    main()
