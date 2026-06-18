"""Adversarial Motion Priors (AMP) building blocks for the G1 climb.

Pure, env-agnostic pieces: the discriminator network, its LSGAN+R1 loss, the style
reward, and an expert-transition builder. The climb's KINEMATIC reference can't
supply the active single-leg balance that DeepMimic TRACKING needs (over-constrains
-> true floor-start 0/20). AMP instead rewards the policy for producing motion whose
*style* matches the reference distribution (a learned discriminator), paired with a
task reward, leaving the policy free to find its own dynamic balance.

The discriminator sees a TRANSITION feature pair concat([phi(s_t), phi(s_{t+1})]).
phi is computed by the env (`_amp_features`); the expert pairs are built here by
FK-ing each reference frame with a CORRECT finite-difference qvel (via
mujoco.mj_differentiatePos, so the free-joint convention matches the robot exactly —
this is the #1 silent-failure trap: mismatched expert/robot features make the
discriminator separate on an artifact and the style reward degenerate).
"""
import numpy as np

import jax
import jax.numpy as jp
import flax.linen as nn
import mujoco
from mujoco import mjx


class Discriminator(nn.Module):
    """AMP discriminator: raw logit, trained LSGAN-style (+1 expert, -1 policy)."""
    hidden: tuple = (256, 256)

    @nn.compact
    def __call__(self, x):                       # x: (B, 2*feat_dim), pre-normalized
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(1)(x)[..., 0]            # (B,)


def disc_loss(disc_apply, params, expert_pairs, policy_pairs, grad_pen=5.0):
    """LSGAN discriminator loss + R1 gradient penalty on the (real) expert inputs.
    expert_pairs / policy_pairs are ALREADY normalized. Returns (loss, metrics)."""
    d_exp = disc_apply(params, expert_pairs)
    d_pol = disc_apply(params, policy_pairs)
    lsgan = 0.5 * (jp.mean((d_exp - 1.0) ** 2) + jp.mean((d_pol + 1.0) ** 2))

    # R1: penalize the gradient of D w.r.t. the expert inputs (stabilizes the GAN).
    def d_sum(x):
        return jp.sum(disc_apply(params, x))
    g = jax.grad(d_sum)(expert_pairs)            # (B, 2*feat_dim)
    r1 = jp.mean(jp.sum(g ** 2, axis=-1))
    loss = lsgan + grad_pen * r1
    return loss, {
        "disc/loss": loss, "disc/lsgan": lsgan, "disc/r1": r1,
        "disc/d_expert": jp.mean(d_exp), "disc/d_policy": jp.mean(d_pol),
    }


def style_reward(disc_apply, params, policy_pairs):
    """AMP style reward in [0,1]: 1 when D=+1 (expert-like), 0 when D<=-1.
    policy_pairs ALREADY normalized."""
    d = disc_apply(params, policy_pairs)
    return jp.clip(1.0 - 0.25 * (d - 1.0) ** 2, 0.0, 1.0)


def build_expert_pairs(amp_features_fn, mj_model, mjx_model,
                       ref_legs, ref_base, ref_yaw, dt):
    """Build expert transition pairs (N-1, 2*feat) from the KINEMATIC reference.

    For each frame: qpos = (base from ref, yaw quat, legs from ref); qvel via central
    finite-diff with mj_differentiatePos (correct free-joint tangent); mjx.forward;
    phi = amp_features_fn(data). Pairs = concat([phi[k], phi[k+1]]). CPU/host build.
    """
    N = len(ref_legs)
    base0 = np.array(mj_model.qpos0)                            # full default qpos
    yaw = float(ref_yaw)
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

    def frame_qpos(k):
        q = base0.copy()
        q[0:3] = (0.0, float(ref_base[k][0]), float(ref_base[k][1]))
        q[3:7] = quat
        q[7:19] = np.asarray(ref_legs[k])
        return q

    qpos = np.stack([frame_qpos(k) for k in range(N)])          # (N, nq)
    # central-diff velocities via mj_differentiatePos (handles the free-joint quat)
    qvel = np.zeros((N, mj_model.nv))
    dq = np.zeros(mj_model.nv)
    for k in range(N):
        kp, kn = max(k - 1, 0), min(k + 1, N - 1)
        span = (kn - kp) * dt
        mujoco.mj_differentiatePos(mj_model, dq, span if span > 0 else dt,
                                   qpos[kp], qpos[kn])
        qvel[k] = dq

    # VMAP the FK + feature extraction over all frames (MJX batches physics) — one
    # compile + parallel, vs a slow 561-call Python loop.
    d0 = mjx.make_data(mjx_model)

    def one(qp, qv):
        dk = d0.replace(qpos=qp, qvel=qv)
        dk = mjx.forward(mjx_model, dk)
        return amp_features_fn(dk)

    feats = jax.vmap(one)(jp.asarray(qpos), jp.asarray(qvel))   # (N, feat)
    pairs = jp.concatenate([feats[:-1], feats[1:]], axis=-1)    # (N-1, 2*feat)
    return pairs.astype(jp.float32), feats.astype(jp.float32)
