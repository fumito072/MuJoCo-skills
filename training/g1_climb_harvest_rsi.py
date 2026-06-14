"""Harvest MID-TRANSFER states from a policy that climbs (2026-06-14). The
curriculum env's RSI only seeded the two ENDS (floor start / standing on the
platform); the floor->platform transition in between was never practiced, so the
floor policy got the lead foot up but never completed the second-foot transfer.

This is the fix: the h0.02 policy reaches BOTH feet on the step ~93% of the time,
so its rollouts ARE physically-valid floor->platform trajectories. Save every
frame where a foot is ON the step (the one-foot-on-step transition + the on-step
stand) as RSI seeds. Spawning training episodes from these gives the policy DENSE
practice at exactly the missing phase (DeepMimic / reverse-curriculum, with the
climbing policy as its own demonstrator).

  .venv-rl/bin/python training/g1_climb_harvest_rsi.py \
      --model runs_climb/climb_curric_h0.02_latest --step_h 0.02 \
      --episodes 400 --out runs_climb/climb_rsi_harvest_h0.02.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1_climb_curriculum_env import G1ClimbCurriculumEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--step_h", type=float, required=True)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_states", type=int, default=6000)
    args = ap.parse_args()

    venv = VecNormalize.load(args.model + "_vecnorm.pkl",
                             DummyVecEnv([lambda: G1ClimbCurriculumEnv(step_h=args.step_h)]))
    venv.training = False
    venv.norm_reward = False
    m = PPO.load(args.model, device="cpu")

    env = G1ClimbCurriculumEnv(seed=4242, step_h=args.step_h)
    env.rsi_p = 0.0  # always floor-start so the trajectories are real climbs

    qpos_bank, qvel_bank = [], []
    n_climbed = 0
    for ep in range(args.episodes):
        obs, _ = env.reset()
        traj_q, traj_v = [], []
        reached_both = False
        # half deterministic (clean), half stochastic (variety)
        deterministic = (ep % 2 == 0)
        for k in range(300):
            o = venv.normalize_obs(obs)
            a, _ = m.predict(o, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(a)
            if info["feet_on_step"] >= 1:                  # transition + on-step frames
                traj_q.append(env.d.qpos.copy())
                traj_v.append(env.d.qvel.copy())
            if info["feet_on_step"] >= 2:
                reached_both = True
            if term or trunc:
                break
        if reached_both and traj_q:                        # only harvest real climbs
            n_climbed += 1
            qpos_bank.extend(traj_q)
            qvel_bank.extend(traj_v)

    qpos_bank = np.asarray(qpos_bank)
    qvel_bank = np.asarray(qvel_bank)
    if len(qpos_bank) > args.max_states:                   # subsample to cap size
        idx = np.random.default_rng(0).choice(len(qpos_bank), args.max_states, replace=False)
        qpos_bank, qvel_bank = qpos_bank[idx], qvel_bank[idx]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, qpos=qpos_bank, qvel=qvel_bank)
    print(f"climbed (reached both feet) {n_climbed}/{args.episodes} episodes")
    print(f"harvested {len(qpos_bank)} mid-transfer/on-step states -> {args.out}")


if __name__ == "__main__":
    main()
