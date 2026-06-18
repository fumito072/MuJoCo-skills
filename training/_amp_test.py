import sys; sys.path.insert(0, "training")
import numpy as np, jax, jax.numpy as jp, optax, mujoco
from mujoco import mjx
from mujoco_playground import registry
import g1_climb_amp_env as E
import amp

env = registry.load("G1ClimbAMP")
reset = jax.jit(env.reset); step = jax.jit(env.step)

# ---- V0: env constructs; obs size unchanged; climb_mimic gone ----
st = reset(jax.random.PRNGKey(0))
assert tuple(env.observation_size["state"]) == (119,), env.observation_size
assert tuple(env.observation_size["privileged_state"]) == (232,), env.observation_size
mk = [k for k in st.metrics if k.startswith("reward/")]
assert "reward/climb_mimic" not in mk, "climb_mimic must be gone"
assert "reward/climb_feet" in mk and "reward/climb_stand" in mk
print("V0 OK: obs", dict(env.observation_size), "| no climb_mimic; feet+stand present")

# ---- V1: amp_obs shape/finite/updates; leg block == qpos[7:19] ----
f = st.info["amp_obs"]
assert f.shape == (E.AMP_FEAT_DIM,) and bool(jp.all(jp.isfinite(f))), f.shape
assert bool(jp.allclose(f[0:12], st.data.qpos[7:19])), "leg block mismatch"
st2 = step(st, jp.zeros(env.action_size))
f2 = st2.info["amp_obs"]
assert bool(jp.all(jp.isfinite(f2))) and not bool(jp.allclose(f, f2)), "amp_obs not updating"
print("V1 OK: amp_obs", f.shape, "finite, leg block == qpos[7:19], updates after step")

exp_pairs, exp_feats = amp.build_expert_pairs(
    env._amp_features, env.mj_model, env.mjx_model, E.REF_LEGS, E.REF_BASE, E.REF_YAW, E.REF_DT)
assert exp_pairs.shape == (E.REF_N - 1, 2 * E.AMP_FEAT_DIM), exp_pairs.shape
print("V1b OK: expert pairs", exp_pairs.shape)

# ---- V2 (the trap): robot features at frame k == expert features; vel plausible ----
base0 = np.array(env.mj_model.qpos0)
quat = np.array([np.cos(E.REF_YAW / 2), 0, 0, np.sin(E.REF_YAW / 2)])
def fq(k):
    q = base0.copy(); q[0:3] = (0, float(E.REF_BASE[k][0]), float(E.REF_BASE[k][1]))
    q[3:7] = quat; q[7:19] = np.asarray(E.REF_LEGS[k]); return q
k = 100
dq = np.zeros(env.mj_model.nv)
mujoco.mj_differentiatePos(env.mj_model, dq, 2 * E.REF_DT, fq(k - 1), fq(k + 1))
d = mjx.make_data(env.mjx_model).replace(qpos=jp.asarray(fq(k)), qvel=jp.asarray(dq))
d = mjx.forward(env.mjx_model, d)
fr = np.asarray(env._amp_features(d))
diff = float(np.abs(fr - np.asarray(exp_feats[k])).max())
assert diff < 1e-4, ("V2 feature mismatch", diff)
legvel_max = float(np.abs(np.asarray(exp_feats[:, 12:24])).max())
assert legvel_max < 20, ("vel too large -> dt wrong", legvel_max)
print(f"V2 OK: robot==expert at frame {k} (max diff {diff:.1e}); expert legvel max {legvel_max:.2f} rad/s")

# ---- V3: discriminator + style reward finite; sign correct after a few disc steps ----
disc = amp.Discriminator()
dparams = disc.init(jax.random.PRNGKey(1), exp_pairs[:2])
da = lambda p, x: disc.apply(p, x)
pol = exp_pairs + jax.random.normal(jax.random.PRNGKey(2), exp_pairs.shape) * 0.5
sr = amp.style_reward(da, dparams, pol)
assert bool(jp.all(jp.isfinite(sr)) & jp.all(sr >= 0) & jp.all(sr <= 1)), "style reward range"
loss0, _ = amp.disc_loss(da, dparams, exp_pairs, pol)
assert bool(jp.isfinite(loss0)), "disc loss finite"
opt = optax.adam(1e-3); ostate = opt.init(dparams)
for _ in range(50):
    (l, _), g = jax.value_and_grad(lambda p: amp.disc_loss(da, p, exp_pairs, pol), has_aux=True)(dparams)
    upd, ostate = opt.update(g, ostate); dparams = optax.apply_updates(dparams, upd)
de, dp = float(jp.mean(da(dparams, exp_pairs))), float(jp.mean(da(dparams, pol)))
assert de > dp, ("V3 sign wrong", de, dp)
print(f"V3 OK: style_r in [0,1]; loss finite; after 50 steps D(expert)={de:.2f} > D(policy)={dp:.2f}")
print("\nALL V0-V3 PASSED")
