import sys; sys.path.insert(0, "training")
import jax, jax.numpy as jp, numpy as np, glob, pickle, tempfile
from mujoco_playground import registry
import g1_climb_amp_env as E, amp, amp_train as AT

print("setup: env + expert...", flush=True)
env = registry.load("G1ClimbAMP")
ep, ef = amp.build_expert_pairs(env._amp_features, env.mj_model, env.mjx_model,
                                E.REF_LEGS, E.REF_BASE, E.REF_YAW, E.REF_DT)
ck = tempfile.mkdtemp()

def prog(step, m):
    print(f"  step {step:5d}  style {m.get('train/style_mean', 0):.3f}  "
          f"dE {m.get('disc/d_expert', 0):+.2f} dP {m.get('disc/d_policy', 0):+.2f}  "
          f"r1 {m.get('disc/r1', 0):.3f}  total_loss {m.get('total_loss', 0):.2f}", flush=True)

print("V4: amp_train tiny run (compiles the full AMP training_step on CPU, slow)...", flush=True)
mp, st = AT.amp_train(env, ep, E.AMP_FEAT_DIM, num_timesteps=1280, episode_length=100,
                      num_envs=32, unroll_length=10, batch_size=32, num_minibatches=2,
                      num_updates_per_batch=2, num_evals=2, ckpt_dir=ck, progress_fn=prog, seed=0)
print("V4 OK: amp_train completed, env_steps", int(st.env_steps), flush=True)

# V5: checkpoint round-trips bit-identical (ppo + disc + amp_norm + optimizers)
latest = sorted(glob.glob(ck + "/amp_*.pkl"))[-1]
blob = pickle.load(open(latest, "rb"))
eq = all(bool(jp.all(jp.asarray(a) == b))
         for a, b in zip(jax.tree_util.tree_leaves(blob), jax.tree_util.tree_leaves(st)))
assert eq, "checkpoint round-trip NOT identical"
print("V5 OK: checkpoint (ppo+disc+amp_norm+opt states) round-trips bit-identical", flush=True)

# V5b: resume continues from the saved step (disconnect-safety)
print("V5b: resume from checkpoint...", flush=True)
mp2, st2 = AT.amp_train(env, ep, E.AMP_FEAT_DIM, num_timesteps=640, episode_length=100,
                        num_envs=32, unroll_length=10, batch_size=32, num_minibatches=2,
                        num_updates_per_batch=2, num_evals=1, ckpt_dir=ck, seed=1)
assert int(st2.env_steps) > int(st.env_steps), ("resume did not continue", int(st2.env_steps))
print("V5b OK: resumed + continued, env_steps", int(st2.env_steps), flush=True)
print("ALL V4-V5 PASSED")
