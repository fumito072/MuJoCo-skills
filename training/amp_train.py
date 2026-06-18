"""Single-device AMP training loop for the G1 climb (brax/MJX). Reuses brax PPO
components (networks, compute_ppo_loss, generate_unroll, running_statistics) and OWNS
the outer loop so it can, every iteration: (1) roll out, (2) compute the discriminator
style reward from the rollout's amp_obs pairs and rewrite the reward to
w_task*task + w_style*style BEFORE GAE, (3) update the discriminator (LSGAN+R1) on
expert-vs-policy pairs, (4) PPO-update on the combined reward.

Single device (Colab = 1 GPU): no pmap (compute_ppo_loss normalizes advantage locally,
no pmean). Checkpoint = pickle of the full AMP state to CKPT_DIR (disconnect-safe).
"""
import functools
import os
import pickle
import time

import flax
import jax
import jax.numpy as jp
import numpy as np
import optax

from brax.training import acting, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import wrapper

import amp


@flax.struct.dataclass
class AMPState:
    ppo_params: ppo_losses.PPONetworkParams
    ppo_opt_state: optax.OptState
    obs_norm: running_statistics.RunningStatisticsState
    disc_params: any
    disc_opt_state: optax.OptState
    amp_norm: running_statistics.RunningStatisticsState
    env_steps: jp.ndarray


def _save(ckpt_dir, step, state):
    os.makedirs(ckpt_dir, exist_ok=True)
    blob = jax.tree_util.tree_map(np.asarray, state)
    with open(os.path.join(ckpt_dir, f"amp_{step:012d}.pkl"), "wb") as f:
        pickle.dump(blob, f)


def _latest(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    cks = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("amp_") and f.endswith(".pkl"))
    return os.path.join(ckpt_dir, cks[-1]) if cks else None


def amp_train(environment, expert_pairs, amp_feat_dim, *,
              num_timesteps, episode_length, num_envs=2048, unroll_length=20,
              batch_size=256, num_minibatches=32, num_updates_per_batch=4,
              num_evals=10, learning_rate=3e-4, entropy_cost=1e-2, discounting=0.97,
              gae_lambda=0.95, clipping_epsilon=0.2,
              w_task=0.5, w_style=0.5, grad_pen=5.0, disc_lr=1e-4, disc_hidden=(256, 256),
              hidden=(512, 256, 128), seed=0, ckpt_dir=None, progress_fn=None,
              eval_fn=None):
    key = jax.random.PRNGKey(seed)
    env = wrapper.wrap_for_brax_training(
        environment, episode_length=episode_length, action_repeat=1)

    env_steps_per_iter = batch_size * num_minibatches * unroll_length
    num_iters = max(1, num_timesteps // env_steps_per_iter)
    iters_per_eval = max(1, num_iters // max(1, num_evals))
    n_rollouts = batch_size * num_minibatches // num_envs

    # --- networks / normalizers / optimizers ---
    normalize = running_statistics.normalize
    nets = ppo_networks.make_ppo_networks(
        environment.observation_size, environment.action_size,
        preprocess_observations_fn=normalize,
        policy_hidden_layer_sizes=hidden, value_hidden_layer_sizes=hidden,
        policy_obs_key="state", value_obs_key="privileged_state")
    make_policy = ppo_networks.make_inference_fn(nets)
    ppo_opt = optax.adam(learning_rate)

    disc = amp.Discriminator(hidden=disc_hidden)
    disc_apply = lambda p, x: disc.apply(p, x)
    disc_opt = optax.adam(disc_lr)
    amp_spec = specs.Array((2 * amp_feat_dim,), jp.float32)

    loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss, ppo_network=nets, entropy_cost=entropy_cost,
        discounting=discounting, gae_lambda=gae_lambda, clipping_epsilon=clipping_epsilon)

    def pairs_from_data(data):
        e = data.extras["state_extras"]
        return jp.concatenate([e["amp_obs_prev"], e["amp_obs"]], axis=-1)  # (B,T,2F)

    def disc_update(disc_params, disc_opt_state, amp_norm, policy_pairs_flat, key):
        kx, ke = jax.random.split(key)
        n = policy_pairs_flat.shape[0]
        idx_e = jax.random.randint(ke, (n,), 0, expert_pairs.shape[0])
        exp = normalize(expert_pairs[idx_e], amp_norm)
        pol = normalize(policy_pairs_flat, amp_norm)
        (loss, metrics), g = jax.value_and_grad(
            lambda p: amp.disc_loss(disc_apply, p, exp, pol, grad_pen), has_aux=True)(disc_params)
        upd, disc_opt_state = disc_opt.update(g, disc_opt_state)
        return optax.apply_updates(disc_params, upd), disc_opt_state, metrics

    def sgd_step(carry, _, data, obs_norm):
        opt_state, params, key = carry
        key, kperm, kgrad = jax.random.split(key, 3)
        # shuffle the batch and split into num_minibatches (proper PPO minibatching)
        def convert(x):
            x = jax.random.permutation(kperm, x)
            return x.reshape((num_minibatches, -1) + x.shape[1:])
        shuffled = jax.tree_util.tree_map(convert, data)

        def mb_step(c, mb):
            ostate, ps, k = c
            k, kl = jax.random.split(k)
            (loss, metrics), g = jax.value_and_grad(loss_fn, has_aux=True)(ps, obs_norm, mb, kl)
            upd, ostate = ppo_opt.update(g, ostate)
            return (ostate, optax.apply_updates(ps, upd), k), metrics

        (opt_state, params, _), metrics = jax.lax.scan(
            mb_step, (opt_state, params, kgrad), shuffled)
        return (opt_state, params, key), jax.tree_util.tree_map(jp.mean, metrics)

    @jax.jit
    def training_step(state: AMPState, env_state, key):
        key, k_unroll, k_disc, k_sgd = jax.random.split(key, 4)
        policy = make_policy((state.obs_norm, state.ppo_params.policy))

        def roll(carry, _):
            es, k = carry
            k, k2 = jax.random.split(k)
            es, data = acting.generate_unroll(
                env, es, policy, k, unroll_length,
                extra_fields=("truncation", "amp_obs", "amp_obs_prev"))
            return (es, k2), data

        (env_state, _), data = jax.lax.scan(roll, (env_state, k_unroll), (), length=n_rollouts)
        data = jax.tree_util.tree_map(lambda x: jp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jp.reshape(x, (-1,) + x.shape[2:]), data)

        # --- AMP: style reward, rewrite reward BEFORE GAE ---
        pairs = pairs_from_data(data)                                   # (B,T,2F)
        flat = pairs.reshape(-1, pairs.shape[-1])                       # (B*T,2F)
        style = amp.style_reward(disc_apply, state.disc_params, normalize(flat, state.amp_norm))
        style = style.reshape(pairs.shape[:2])                          # (B,T)
        data = data._replace(reward=w_task * data.reward + w_style * style)

        obs_norm = running_statistics.update(state.obs_norm, data.observation)
        amp_norm = running_statistics.update(state.amp_norm, flat)
        disc_params, disc_opt_state, dmetrics = disc_update(
            state.disc_params, state.disc_opt_state, amp_norm, flat, k_disc)

        (ppo_opt_state, ppo_params, _), pmetrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, obs_norm=obs_norm),
            (state.ppo_opt_state, state.ppo_params, k_sgd), (), length=num_updates_per_batch)

        new_state = AMPState(
            ppo_params=ppo_params, ppo_opt_state=ppo_opt_state, obs_norm=obs_norm,
            disc_params=disc_params, disc_opt_state=disc_opt_state, amp_norm=amp_norm,
            env_steps=state.env_steps + env_steps_per_iter)
        metrics = {**jax.tree_util.tree_map(jp.mean, pmetrics), **dmetrics,
                   "train/style_mean": jp.mean(style), "train/task_mean": jp.mean(data.reward)}
        return new_state, env_state, metrics

    # --- init ---
    key, k_pol, k_disc = jax.random.split(key, 3)
    init_ppo = ppo_losses.PPONetworkParams(
        policy=nets.policy_network.init(k_pol),
        value=nets.value_network.init(jax.random.fold_in(k_pol, 1)))
    obs_norm0 = running_statistics.init_state(
        specs.Array(environment.observation_size["state"], jp.float32)
        if not isinstance(environment.observation_size, dict) else
        {k: specs.Array(v, jp.float32) for k, v in environment.observation_size.items()})
    state = AMPState(
        ppo_params=init_ppo, ppo_opt_state=ppo_opt.init(init_ppo), obs_norm=obs_norm0,
        disc_params=disc.init(k_disc, expert_pairs[:2]), disc_opt_state=None,
        amp_norm=running_statistics.init_state(amp_spec), env_steps=jp.int32(0))
    state = state.replace(disc_opt_state=disc_opt.init(state.disc_params))

    if ckpt_dir and _latest(ckpt_dir):
        with open(_latest(ckpt_dir), "rb") as f:
            blob = pickle.load(f)
        state = jax.tree_util.tree_map(jp.asarray, blob)
        print("RESUMED from", _latest(ckpt_dir), "step", int(state.env_steps), flush=True)

    key, k_reset = jax.random.split(key)
    env_state = jax.jit(env.reset)(jax.random.split(k_reset, num_envs))

    for it in range(num_iters):
        key, k = jax.random.split(key)
        t = time.time()
        state, env_state, metrics = training_step(state, env_state, k)
        if it % iters_per_eval == 0 or it == num_iters - 1:
            jax.block_until_ready(state.env_steps)
            if ckpt_dir:
                _save(ckpt_dir, int(state.env_steps), state)
            if progress_fn:
                progress_fn(int(state.env_steps),
                            {k: float(v) for k, v in metrics.items()})
            if eval_fn:
                eval_fn(int(state.env_steps), make_policy, state)
    return make_policy, state
