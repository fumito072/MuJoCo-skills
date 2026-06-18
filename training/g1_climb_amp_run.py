"""Run AMP climb training on a LOCAL NVIDIA GPU (e.g. RTX 3060) — no Colab.

Setup on the GPU machine (Linux + CUDA):
    git clone https://github.com/fumito072/MuJoCo-skills && cd MuJoCo-skills
    python -m venv .venv-gpu
    .venv-gpu/bin/pip install -U pip
    .venv-gpu/bin/pip install "jax[cuda12]" brax mujoco mujoco-mjx playground flax optax
    .venv-gpu/bin/python training/g1_climb_amp_run.py        # full run (2048 envs, 300M)

Disconnect/crash-safe: pickle checkpoints land in CKPT (default runs_amp_v2); re-run to
auto-resume. Env-var overrides for a quick smoke test:
    NUM_ENVS=32 NUM_STEPS=1280 NUM_EVALS=2 CKPT=/tmp/amp_smoke python training/g1_climb_amp_run.py
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco_playground import registry

import amp
import amp_train as AT
import g1_climb_amp_env as E

env = registry.load("G1ClimbAMP")
AMP_EXPERT, _ = amp.build_expert_pairs(
    env._amp_features, env.mj_model, env.mjx_model, E.REF_LEGS, E.REF_BASE, E.REF_YAW, E.REF_DT)
print("env + expert ready:", AMP_EXPERT.shape, "| devices:", jax.devices(), flush=True)


def progress(step, m):
    print(f"[{datetime.now():%H:%M:%S}] step {step:>11,}  style {m.get('train/style_mean', 0):.3f}"
          f"  D(exp){m.get('disc/d_expert', 0):+.2f} D(pol){m.get('disc/d_policy', 0):+.2f}"
          f"  target {m.get('train/task_mean', 0):.3f}  loss {m.get('total_loss', 0):7.2f}", flush=True)


def true_floor_eval(step, make_policy, state, n=20):
    """HONEST eval: FORCE qpos to reference frame 0 (floor, z0.755) — never a phase
    relabel. start_z must print ~0.755 or the eval is lying."""
    infer = jax.jit(make_policy((state.obs_norm, state.ppo_params.policy), deterministic=True))
    rfn, sfn = jax.jit(env.reset), jax.jit(env.step)
    rl0, rb0 = E.REF_LEGS[0], E.REF_BASE[0]
    quat = jp.array([jp.cos(E.REF_YAW / 2), 0., 0., jp.sin(E.REF_YAW / 2)])
    succ, sz = 0, 0.0
    for i in range(n):
        st = rfn(jax.random.PRNGKey(7000 + i))
        base = jp.array([0., rb0[0], rb0[1]])
        joints = st.data.qpos[7:].at[0:12].set(rl0)
        data = mjx.forward(env.mjx_model, st.data.replace(
            qpos=jp.concatenate([base, quat, joints]), qvel=jp.zeros_like(st.data.qvel)))
        info = dict(st.info)
        for k in ("amp_obs", "amp_obs_prev"):
            info[k] = env._amp_features(data)
        info["ref_frame0"] = jp.int32(0); info["step"] = jp.int32(0)
        contact = jp.array([data.sensordata[env._mj_model.sensor_adr[s]] > 0
                            for s in env._feet_floor_found_sensor])
        st = st.replace(data=data, obs=env._get_obs(data, info, contact), info=info)
        sz = float(st.data.qpos[2]); ever = False
        for _ in range(400):
            st = sfn(st, infer(st.obs, jax.random.PRNGKey(0))[0])
            ever = ever or bool(st.metrics.get("reward/climb_stand", 0.0) > 0.3)
            if bool(st.done):
                break
        succ += int(ever)
    print(f"  >> TRUE FLOOR-START {succ}/{n}  (start_z {sz:.3f} must be ~0.755)", flush=True)


NUM_ENVS = int(os.environ.get("NUM_ENVS", 2048))    # 3060 12GB; raise on a bigger GPU
NUM_STEPS = int(os.environ.get("NUM_STEPS", 300_000_000))
NUM_EVALS = int(os.environ.get("NUM_EVALS", 60))
CKPT = os.environ.get("CKPT", os.path.join(os.path.dirname(__file__), "..", "runs_amp_v2"))

make_policy, state = AT.amp_train(
    env, AMP_EXPERT, E.AMP_FEAT_DIM, num_timesteps=NUM_STEPS, episode_length=400,
    num_envs=NUM_ENVS, unroll_length=20, batch_size=256, num_minibatches=32,
    num_updates_per_batch=4, num_evals=NUM_EVALS, learning_rate=3e-4,
    w_task=0.5, w_style=0.5, grad_pen=5.0, disc_lr=1e-4,
    ckpt_dir=CKPT, progress_fn=progress, eval_fn=true_floor_eval, seed=0)
print("DONE.", flush=True)
