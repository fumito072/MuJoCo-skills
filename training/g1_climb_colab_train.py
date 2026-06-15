# ============================================================================
#  G1 CLIMB — Colab / GPU (MJX) training. Paste each "# === CELL N ===" block
#  into its own Colab cell (Runtime -> Change runtime type -> GPU) and run in order.
#
#  PREREQ (do once, on your Mac, in this Claude Code session):  ! git push
#  so Colab's `git clone` pulls the height-aware env (commit 5fdfab8+).
#
#  Bakes in the 2026-06-15 Mac-CPU campaign lessons (docs/COLAB_CLIMB_PLAN.md):
#   - height DOMAIN RANDOMIZATION [0.04..0.22] + height in obs (fixed-0.22 was 0/20)
#   - DISCONNECT-SAFE checkpointing to Google Drive every eval + AUTO-RESUME
#     (if Colab stops, re-run CELL 3 — it restores from Drive and continues)
#   - GPU-scale brax PPO (8192 envs) for the hard single-support exploration
#  Reviewed by 4 expert agents; all blocker fixes applied (wrap_env_fn, padded
#  checkpoint resume path, brax pin, sum_reward log key, per-height eval rebind).
# ============================================================================

# === CELL 1 — install + GPU/version asserts (run once; ~4 min) ===
import subprocess
print(subprocess.run(["nvidia-smi","-L"], capture_output=True, text=True).stdout or "NO GPU!")
# mujoco_playground pulls a compatible mujoco/mjx; pin a brax new enough to have
# save_checkpoint_path + wrap_env_fn (0.10.x lacks both). jax[cuda12] for the GPU.
%pip -q install "mujoco_playground>=0.0.4" "brax>=0.12.1" orbax-checkpoint
%pip -q install -U "jax[cuda12]"
import inspect, jax, brax
from brax.training.agents.ppo import train as _ppo
assert "save_checkpoint_path" in inspect.signature(_ppo.train).parameters, "brax too old: %pip install -U brax"
assert "wrap_env_fn" in inspect.signature(_ppo.train).parameters, "brax too old: %pip install -U brax"
assert any(d.platform == "gpu" for d in jax.devices()), "No GPU — Runtime->Change runtime type->GPU, then rerun CELL 1"
print("OK: brax", brax.__version__, "| jax", jax.__version__, "|", jax.devices())
# NOTE: if the jax[cuda12] line REINSTALLED jax, do Runtime->Restart, then rerun CELL 1 once.


# === CELL 2 — Drive + clone repo + env + HEIGHT DOMAIN RANDOMIZATION ===
import os, sys, functools
import jax, jax.numpy as jp
from google.colab import drive
drive.mount("/content/drive")
CKPT_DIR = "/content/drive/MyDrive/g1_climb_ckpts"     # checkpoints persist here
os.makedirs(CKPT_DIR, exist_ok=True)

REPO = "/content/MuJoCo-skills"
if not os.path.isdir(REPO):
    subprocess.run(["git","clone","--depth","1",
                    "https://github.com/fumito072/MuJoCo-skills", REPO], check=True)
else:
    subprocess.run(["git","-C",REPO,"pull"], check=False)
sys.path.insert(0, os.path.join(REPO, "training"))

import g1_climb_mjx_env                  # registers "G1ClimbBox" (height-aware)
from mujoco_playground import registry
env = registry.load("G1ClimbBox")
assert hasattr(env, "_plat_h_model"), "old env on GitHub — run `git push` on the Mac first, then re-run CELL 2"
PLAT_GID = env._plat_gid
H_MIN, H_MAX = 0.04, 0.22

def height_domain_randomization(model, rng):
    """Per-env platform height in [H_MIN,H_MAX]. brax/Playground pass `rng` already
    batched (num_envs,); vmap over it and return (batched_model, in_axes). The env
    reads h back from the geom, so ONE policy learns ALL heights (no forgetting)."""
    @jax.vmap
    def per_env(key):
        h = jax.random.uniform(key, (), minval=H_MIN, maxval=H_MAX)
        return (model.geom_pos.at[PLAT_GID, 2].set(h / 2.0),
                model.geom_size.at[PLAT_GID, 2].set(h / 2.0))
    gpos, gsize = per_env(rng)
    model = model.tree_replace({"geom_pos": gpos, "geom_size": gsize})
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({"geom_pos": 0, "geom_size": 0})
    return model, in_axes
print("env ready. platform height randomized in", (H_MIN, H_MAX))


# === CELL 3 — TRAIN (disconnect-safe: Drive checkpoint every eval + AUTO-RESUME) ===
from datetime import datetime
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import wrapper            # REQUIRED for Playground env + DR
from etils import epath

NUM_ENVS = 8192
NUM_TIMESTEPS = 300_000_000
NUM_EVALS = 60                                   # ~ a checkpoint every 5M steps

def latest_ckpt(d):                              # brax saves zero-padded dirs: 000005000000
    p = epath.Path(d)
    dirs = [c for c in p.iterdir() if c.is_dir() and c.name.isdigit()] if p.exists() else []
    return max(dirs, key=lambda c: int(c.name)).as_posix() if dirs else None

restore = latest_ckpt(CKPT_DIR)
print("RESUMING from" if restore else "starting fresh:", restore or "(none)")

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_obs_key="state", value_obs_key="privileged_state",   # asymmetric actor/critic
    policy_hidden_layer_sizes=(512, 256, 128),
    value_hidden_layer_sizes=(512, 256, 128))

def progress(step, metrics):
    r = metrics.get("eval/episode_sum_reward", float("nan"))      # this env has no bare 'reward'
    stand = metrics.get("eval/episode_reward/climb_stand", float("nan"))
    print(f"[{datetime.now():%H:%M:%S}] step {int(step):>11,}  sum_reward {r:8.2f}  "
          f"climb_stand {stand:6.3f}", flush=True)

make_inference_fn, params, _ = ppo.train(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,   # <-- Playground wrapper (DR + .mjx_model)
    randomization_fn=height_domain_randomization, # <-- height DR
    num_timesteps=NUM_TIMESTEPS, num_evals=NUM_EVALS, episode_length=400,
    num_envs=NUM_ENVS, batch_size=256, num_minibatches=32, unroll_length=20,
    num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2,
    discounting=0.97, gae_lambda=0.95, clipping_epsilon=0.2,
    normalize_observations=True, network_factory=network_factory,
    save_checkpoint_path=CKPT_DIR,                # <-- saves to Drive every eval
    restore_checkpoint_path=restore,             # <-- auto-resume (None = fresh)
    progress_fn=progress, seed=0)
print("DONE. latest checkpoint:", latest_ckpt(CKPT_DIR))
# If Colab disconnects mid-run: just RE-RUN CELL 3 — it restores the latest Drive
# checkpoint and continues (weights+normalizer carry over).


# === CELL 4 — eval per height + render a GIF ===
import mediapy as media
def eval_height(h, n=20):
    # rebind the env model to height h so the eval truly uses it (then re-jit)
    env._mjx_model = env.mjx_model.tree_replace({
        "geom_pos":  env.mjx_model.geom_pos.at[PLAT_GID, 2].set(h / 2),
        "geom_size": env.mjx_model.geom_size.at[PLAT_GID, 2].set(h / 2)})
    infer = jax.jit(make_inference_fn(params, deterministic=True))
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    succ = 0
    for i in range(n):
        st = reset(jax.random.PRNGKey(1000 + i)); ever = False
        for _ in range(400):
            act = infer(st.obs, jax.random.PRNGKey(0))[0]
            st = step(st, act)
            ever = ever or bool(st.metrics.get("reward/climb_stand", 0.0) > 0)
            if st.done: break
        succ += int(ever)
    return succ, n

for h in (0.06, 0.12, 0.18, 0.22):
    s, n = eval_height(h); print(f"h={h:.2f}: reached platform-stand on {s}/{n} floor starts")

env._mjx_model = env.mjx_model.tree_replace({
    "geom_pos":  env.mjx_model.geom_pos.at[PLAT_GID, 2].set(0.22 / 2),
    "geom_size": env.mjx_model.geom_size.at[PLAT_GID, 2].set(0.22 / 2)})
infer = jax.jit(make_inference_fn(params, deterministic=True))
reset, step = jax.jit(env.reset), jax.jit(env.step)
st = reset(jax.random.PRNGKey(0)); frames = []
for _ in range(400):
    act = infer(st.obs, jax.random.PRNGKey(0))[0]
    st = step(st, act); frames.append(env.render([st], height=240, width=320)[0])
    if st.done: break
media.show_video(frames, fps=1.0 / env.dt)


# === CELL 5 — save params for Mac inference (train-on-GPU / run-on-Mac) ===
import pickle
with open(f"{CKPT_DIR}/g1_climb_params.pkl", "wb") as f:
    pickle.dump(params, f)
print("saved ->", f"{CKPT_DIR}/g1_climb_params.pkl  (also in the checkpoint dirs)")
