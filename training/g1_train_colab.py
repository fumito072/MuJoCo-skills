"""Train a Unitree G1 policy on (free) Colab GPU with MuJoCo Playground, save it for Mac inference.

Run on a Colab GPU runtime (see training/COLAB_GUIDE.md). Two tasks:
  --task walk : the stock G1JoystickFlatTerrain (known to converge) — validates your Colab pipeline.
  --task sit  : the custom G1Sit env (training/g1_sit_env.py) — the goal (lower into a stable seat).

Saves <out>_params.pkl (policy params) + <out>_config.json (obs/act sizes, network) so the policy
can later be replayed in MuJoCo on the Mac (CPU/MPS), NVIDIA-free — same split as the walk replay.

Setup (Colab cell):  !pip install -q playground "jax[cuda12]"
Example:             !python g1_train_colab.py --task sit --steps 40_000_000 --out /content/drive/MyDrive/g1_sit
"""
import argparse
import functools
import json
import os
import pickle
import shutil

import jax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["walk", "sit"], default="walk")
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--out", default="g1_policy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import locomotion_params
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks

    if args.task == "sit":
        import g1_sit_env  # noqa: F401  (importing registers "G1Sit")
        env_name = "G1Sit"
    else:
        env_name = "G1JoystickFlatTerrain"

    print(f"jax devices: {jax.devices()}")
    print(f"task={args.task} env={env_name} steps={args.steps:,}")
    # Force the pure-JAX MJX backend. Newer Playground defaults config.impl to "warp",
    # which needs mujoco-warp (absent on stock Colab) -> AttributeError GraphMode.WARP.
    cfg = registry.get_default_config(env_name)
    cfg.impl = "jax"
    print(f"mjx impl = {cfg.impl}")
    env = registry.load(env_name, config=cfg)
    eval_env = registry.load(env_name, config=cfg)

    # reuse the tuned G1 PPO config (network + hyperparams) for both tasks
    ppo_params = locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")
    ppo_params.num_timesteps = args.steps

    net_kwargs = dict(ppo_params.network_factory)
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_kwargs)
    train_kwargs = {k: v for k, v in ppo_params.items() if k != "network_factory"}

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    latest_path, best_path = out + "_latest_params.pkl", out + "_best_params.pkl"
    best = {"r": float("-inf")}

    # brax calls these per eval in this order: policy_params_fn -> run_evaluation -> progress_fn.
    # So at progress() time, latest_params.pkl already holds THIS eval's params -> promote if best.
    def save_ckpt(step, make_policy, params):     # brax passes (current_step, make_policy, params)
        del step, make_policy
        with open(latest_path, "wb") as f:
            pickle.dump(params, f)

    def progress(step, metrics):
        r = metrics.get("eval/episode_reward", float("nan"))
        tag = ""
        if r == r and r > best["r"]:               # r == r skips NaN
            best["r"] = r
            if os.path.exists(latest_path):
                shutil.copyfile(latest_path, best_path)
                tag = "  <- new best (saved)"
        print(f"  step {step:>12,}  eval_reward={r:8.2f}{tag}", flush=True)

    print("training... (checkpoints: *_latest / *_best every eval; survives early stop & disconnect)")
    _, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        policy_params_fn=save_ckpt,
        seed=args.seed,
        **train_kwargs,
    )

    # Save params FIRST (the thing we can't regenerate) so a later hiccup can't lose the policy.
    with open(out + "_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print(f"\nSAVED final policy -> {out}_params.pkl  (best eval={best['r']:.2f} -> {best_path})")

    # config.json is best-effort metadata; observation_size is a dict for asymmetric obs, so don't int() it.
    try:
        obs_size = env.observation_size
        obs_size = ({k: list(v) for k, v in obs_size.items()}
                    if hasattr(obs_size, "items") else int(obs_size))
        json.dump(
            {"env": env_name, "task": args.task,
             "obs_size": obs_size, "act_size": int(env.action_size),
             "ctrl_dt": 0.02, "sim_dt": 0.002, "action_scale": 0.5,
             "network": net_kwargs},
            open(out + "_config.json", "w"), default=str, indent=2)
        print(f"SAVED config -> {out}_config.json")
    except Exception as e:  # never let metadata sink the run
        print(f"(config.json skipped: {e})")
    print("Download g1_sit_params.pkl to your Mac (models/policies/) and replay with training/g1_sit_play.py.")


if __name__ == "__main__":
    main()
