# ============================================================================
#  G1 CLIMB — Colab/GPU (MJX) + DeepMimic REFERENCE, SELF-CONTAINED.
#  Paste each "# === CELL N ===" block into its own Colab cell (Runtime->GPU).
#
#  WHY THIS over g1_climb_colab_train.py: that one uses TASK rewards (blind
#  exploration of the first-foot-up) — the approach the prior ~51 GPU runs used,
#  0/20. This session (Mac CPU) we found the missing ingredient: a DeepMimic
#  REFERENCE with the CoM placed OVER the support foot (CoM-IK), which made the
#  climb trackable (CPU frontier 0.55, brace 0.42). Here we hand GPU that same
#  reference — spawn the policy ALONG the climb path (reference RSI, frontier-
#  weighted toward the floor) and reward TRACKING it — so GPU's massive
#  parallelism only has to crack the single-leg first-foot-up, with the whole
#  motion scaffolded. This is a genuinely NEW combination, not a repeat.
#
#  Stage 1 = 12-DOF leg climb (this file). If GPU still can't crack the first
#  foot-up, escalate to the 29-DOF hand-brace (swap the reference + arm gains).
# ============================================================================

# === CELL 1 — install + asserts. RUN FIRST in every (re)started runtime. ===
import subprocess
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout or "NO GPU!")
%pip -q install mujoco mujoco-mjx playground brax "jax[cuda12]" orbax-checkpoint mediapy
import importlib
for mod in ("mujoco", "mujoco.mjx", "jax", "brax", "mujoco_playground", "flax", "orbax.checkpoint"):
    importlib.import_module(mod)
import inspect, jax, brax
from brax.training.agents.ppo import train as _ppo
assert "save_checkpoint_path" in inspect.signature(_ppo.train).parameters, "brax too old: %pip install -U brax"
assert any(d.platform == "gpu" for d in jax.devices()), \
    "No GPU — Runtime->Restart runtime, then RE-RUN CELL 1"
print("OK: brax", brax.__version__, "| jax", jax.__version__, "|", jax.devices())


# === CELL 2 — Drive + reference + INLINE DeepMimic env ===
import os, functools
import numpy as np
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from ml_collections import config_dict
from google.colab import drive

from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.g1 import base as g1_base

drive.mount("/content/drive")
CKPT_DIR = "/content/drive/MyDrive/g1_climb_dm_ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)

# --- CoM-IK leg-climb reference EMBEDDED as base64 (561 frames: 12 leg joints +
# base y,z). Fully self-contained — NO download, push, or upload needed. -------
import base64, io
REF_B64 = "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"
_ref = np.load(io.BytesIO(base64.b64decode(REF_B64)))
REF_LEGS = jp.asarray(_ref["legs"], jp.float32)          # (N,12)
REF_BASE = jp.asarray(_ref["base"], jp.float32)          # (N,2) = y,z
REF_N = int(REF_LEGS.shape[0])
REF_YAW = float(_ref["yaw"])                              # -pi/2, facing the step
print(f"reference: {REF_N} frames, base {np.array(_ref['base'])[0].round(3)} -> "
      f"{np.array(_ref['base'])[-1].round(3)}")

# real footrest platform (box; top at 0.22, ~matches the FBX footrest the ref used)
PLAT_HALF = (0.30, 0.15, 0.11)
PLAT_CENTER = (0.0, 0.35, 0.11)
SIT_KP, SIT_KD = 300.0, 8.0
LEG_GEOMS = ("left_thigh", "right_thigh", "left_shin", "right_shin",
             "left_foot", "right_foot")
ACTION_SCALE_REF = 0.45                  # residual control gain around the reference


def climb_config():
    cfg = g1_joystick.default_config()
    cfg.impl = "jax"
    cfg.episode_length = 400
    try:
        cfg.njmax = 29 * 2 + 16 * 4
    except (AttributeError, KeyError):
        pass
    s = cfg.reward_config.scales
    for k in ("tracking_lin_vel", "tracking_ang_vel", "feet_phase", "feet_air_time",
              "feet_height", "feet_clearance", "feet_slip", "lin_vel_z", "ang_vel_xy",
              "stand_still", "pose", "joint_deviation_knee", "base_height"):
        if k in s:
            s[k] = 0.0
    s.joint_deviation_hip = -0.05
    s.orientation = -0.5
    s.alive = 0.2
    s.climb_mimic = 1.0       # DeepMimic tracking (the scaffold)
    s.climb_feet = 0.5        # both feet on the platform top
    s.climb_stand = 2.0       # brief upright stand on top = the goal
    return cfg


class G1ClimbDeepMimic(g1_joystick.Joystick):
    """G1 climbs the 0.22 m platform by TRACKING the CoM-IK reference (residual
    control + reference RSI), with GPU exploration for the first-foot-up."""

    def __init__(self, config=climb_config(), config_overrides=None):
        super().__init__(task="flat_terrain", config=config,
                         config_overrides=config_overrides)
        self._add_platform()

    def _add_platform(self):
        assets = g1_base.get_assets()
        spec = mujoco.MjSpec.from_string(
            consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets)
        g = spec.worldbody.add_geom()
        g.name, g.type = "platform", mujoco.mjtGeom.mjGEOM_BOX
        g.size, g.pos = list(PLAT_HALF), list(PLAT_CENTER)
        g.rgba = [0.5, 0.45, 0.4, 1.0]
        g.contype, g.conaffinity = 1, 1
        for rg in LEG_GEOMS:                  # feet are contype=0: pairs are mandatory
            spec.add_pair(geomname1=rg, geomname2="platform")
        spec.assets = assets
        m = spec.compile()
        m.opt.timestep = self.sim_dt
        for a in range(12):                   # SIT-mode stiff legs
            m.actuator_gainprm[a, 0] = SIT_KP
            m.actuator_biasprm[a, 1] = -SIT_KP
            m.actuator_biasprm[a, 2] = -SIT_KD
        self._mj_model = m
        self._mjx_model = mjx.put_model(m, impl=self._config.impl)
        self._lf_gid = m.geom("left_foot").id
        self._rf_gid = m.geom("right_foot").id
        self._plat_gid = m.geom("platform").id
        self._default_pose = jp.asarray(m.qpos0[7:])
        self._post_init()

    def sample_command(self, rng):
        del rng
        return jp.zeros(3)

    # ----------------------------------------------------------------- reset --
    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        rng, kf, kx, ky, kj, kv = jax.random.split(rng, 6)

        # REFERENCE RSI: spawn at a random frame, frontier-weighted toward the floor
        # (frame 0) so the hard first-foot-up gets the most coverage (square weight).
        frac = jax.random.uniform(kf) ** 2                  # concentrate near 0
        frame0 = (frac * (REF_N - 1)).astype(jp.int32)
        rl = REF_LEGS[frame0]
        rb = REF_BASE[frame0]
        yaw = REF_YAW + jax.random.uniform(kx, (), minval=-0.06, maxval=0.06)
        pos = jp.array([jax.random.uniform(ky, (), minval=-0.03, maxval=0.03),
                        rb[0], rb[1]])
        quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
        joints = qpos[7:].at[0:12].set(rl) + \
            jax.random.uniform(kj, (29,), minval=-0.04, maxval=0.04)
        qpos = jp.concatenate([pos, quat, joints])
        qvel = qvel.at[0:3].set(jax.random.uniform(kv, (3,), minval=-0.08, maxval=0.08))

        mk = dict(qpos=qpos, qvel=qvel, ctrl=qpos[7:], impl=self.mjx_model.impl.value)
        try:
            data = mjx_env.make_data(self.mj_model, **mk)
        except TypeError:
            data = mjx.make_data(self.mjx_model).replace(
                qpos=qpos, qvel=qvel, ctrl=qpos[7:])
        data = mjx.forward(self.mjx_model, data)

        rng, cmd_rng, push_rng = jax.random.split(rng, 3)
        push_interval = jax.random.uniform(
            push_rng, minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1])
        info = {
            "rng": rng, "step": 0, "command": self.sample_command(cmd_rng),
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2), "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2), "phase_dt": 2 * jp.pi * self.dt * 1.375,
            "phase": jp.array([0.0, jp.pi]), "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": jp.round(push_interval / self.dt).astype(jp.int32),
            "ref_frame0": frame0,                          # <-- DeepMimic phase anchor
        }
        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())
        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._feet_floor_found_sensor])
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _cur_frame(self, info):
        # base Joystick.step() increments info["step"] and PRESERVES custom keys, so
        # the phase advances without overriding step(). ref_frame0 is set in reset().
        return jp.clip(info["ref_frame0"] + info["step"], 0, REF_N - 1)

    # NOTE: we deliberately do NOT override step() — the base Joystick maps action to
    # motor targets (default_pose + scale*action) and we let it. The DeepMimic scaffold
    # comes from reference RSI (reset) + the tracking reward (_get_reward) + ref obs.
    # OPTIONAL UPGRADE once this runs: residual control (motor = ref_legs + scale*action)
    # is more sample-efficient but needs the exact base action mapping — add it then.

    # ------------------------------------------------------------------- obs --
    def _get_obs(self, data, info, contact):
        obs = super()._get_obs(data, info, contact)
        frame = self._cur_frame(info)
        rl, rb = REF_LEGS[frame], REF_BASE[frame]
        base = data.qpos[0:3]
        phase = frame.astype(jp.float32) / REF_N
        leg_err = rl - data.qpos[7:19]
        extra = jp.concatenate([
            jp.array([phase, base[2], rb[0] - base[1], rb[1] - base[2]]),
            leg_err])                              # 4 + 12 = 16 extra dims
        if isinstance(obs, dict):
            return {k: jp.concatenate([v, extra]) for k, v in obs.items()}
        return jp.concatenate([obs, extra])

    # ---------------------------------------------------------------- reward --
    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact)
        frame = self._cur_frame(info)
        rl, rb = REF_LEGS[frame], REF_BASE[frame]
        base = data.qpos[0:3]
        leg_err = jp.mean(jp.square(data.qpos[7:19] - rl))
        base_err = (base[0] ** 2 + jp.square(base[1] - rb[0])
                    + 1.5 * jp.square(base[2] - rb[1]))
        # PRODUCT tracking: leg mimicry pays nothing off the reference body path
        rewards["climb_mimic"] = 2.2 * jp.exp(-8.0 * leg_err) * jp.exp(-15.0 * base_err)

        plat_h = 2.0 * data.geom_xpos[self._plat_gid, 2]

        def on_plat(gid):
            p = data.geom_xpos[gid]
            return ((p[2] > plat_h - 0.03) & (p[2] < plat_h + 0.08)
                    & (jp.abs(p[0]) < PLAT_HALF[0])
                    & (jp.abs(p[1] - PLAT_CENTER[1]) < PLAT_HALF[1]))
        feet = on_plat(self._lf_gid).astype(jp.float32) + on_plat(self._rf_gid).astype(jp.float32)
        rewards["climb_feet"] = 0.15 * feet + 0.6 * (feet == 2)
        up = self.get_gravity(data, "torso")
        upright = jp.exp(-jp.sum(jp.square(up - jp.array([0.0, 0.0, 1.0]))) / 0.1)
        standing = (feet == 2) & (base[2] > 0.755 + plat_h - 0.06)
        rewards["climb_stand"] = jp.where(standing, upright, 0.0)
        return rewards


registry.locomotion.register_environment("G1ClimbDeepMimic", G1ClimbDeepMimic, climb_config)
env = registry.load("G1ClimbDeepMimic")
print("env ready. DeepMimic reference RSI (frontier-weighted to floor) + tracking reward.")


# === CELL 3 — TRAIN (disconnect-safe: Drive checkpoint every eval + AUTO-RESUME) ===
from datetime import datetime
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import wrapper
from etils import epath
NUM_ENVS, NUM_TIMESTEPS, NUM_EVALS = 8192, 300_000_000, 60
def latest_ckpt(d):
    p = epath.Path(d)
    dirs = [c for c in p.iterdir() if c.is_dir() and c.name.isdigit()] if p.exists() else []
    return max(dirs, key=lambda c: int(c.name)).as_posix() if dirs else None
restore = latest_ckpt(CKPT_DIR)
print("RESUMING from" if restore else "starting fresh:", restore or "(none)")
network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_obs_key="state", value_obs_key="privileged_state",
    policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128))
def progress(step, m):
    print(f"[{datetime.now():%H:%M:%S}] step {int(step):>11,}  "
          f"reward {m.get('eval/episode_sum_reward', float('nan')):8.2f}  "
          f"mimic {m.get('eval/episode_reward/climb_mimic', float('nan')):6.2f}  "
          f"stand {m.get('eval/episode_reward/climb_stand', float('nan')):6.3f}", flush=True)
make_inference_fn, params, _ = ppo.train(
    environment=env, wrap_env_fn=wrapper.wrap_for_brax_training,
    num_timesteps=NUM_TIMESTEPS, num_evals=NUM_EVALS, episode_length=400,
    num_envs=NUM_ENVS, batch_size=256, num_minibatches=32, unroll_length=20,
    num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2,
    discounting=0.97, gae_lambda=0.95, clipping_epsilon=0.2,
    normalize_observations=True, network_factory=network_factory,
    save_checkpoint_path=CKPT_DIR, restore_checkpoint_path=restore,
    progress_fn=progress, seed=0)
print("DONE. latest checkpoint:", latest_ckpt(CKPT_DIR))


# === CELL 4 — eval the full floor-start climb (frame 0, deterministic) ===
def eval_floor(n=20):
    infer = jax.jit(make_inference_fn(params, deterministic=True))
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    succ = 0
    for i in range(n):
        st = reset(jax.random.PRNGKey(7000 + i))
        # force floor start = reference frame 0
        st.info["ref_frame0"] = jp.int32(0)
        ever = False
        for _ in range(400):
            act = infer(st.obs, jax.random.PRNGKey(0))[0]
            st = step(st, act)
            ever = ever or bool(st.metrics.get("reward/climb_stand", 0.0) > 0.3)
            if st.done:
                break
        succ += int(ever)
    return succ, n
s, n = eval_floor()
print(f"FLOOR-START full climb: reached platform-stand on {s}/{n}")


# === CELL 5 — save params for Mac inference ===
import pickle
with open(f"{CKPT_DIR}/g1_climb_dm_params.pkl", "wb") as f:
    pickle.dump(params, f)
print("saved ->", f"{CKPT_DIR}/g1_climb_dm_params.pkl")


# === CELL 6 — PEEK: eval the latest checkpoint WITHOUT finishing CELL 3 ===
# See the REAL floor-start N/20 mid-training, so you can stop as soon as it's good
# (no need to sit through all 300M). Verified on local CPU MJX.
#   1) Runtime -> Interrupt execution  (stops CELL 3; the Drive checkpoint is safe)
#   2) Run THIS cell                   (needs CELL 1+2 already run this session)
#   3) Re-run CELL 3 to RESUME from the same checkpoint (or CELL 5 to save & stop)
# Uses the low-level load() + manual inference rebuild (load_policy has a brax config
# quirk on some versions; this path is version-independent).
from brax.training.agents.ppo import checkpoint as ppo_ckpt, networks as ppo_networks
from brax.training.acme import running_statistics
from etils import epath
import functools, jax, jax.numpy as jp
def _latest(d):
    p = epath.Path(d)
    dd = [c for c in p.iterdir() if c.is_dir() and c.name.isdigit()] if p.exists() else []
    return max(dd, key=lambda c: int(c.name)).as_posix() if dd else None
_CK = _latest(CKPT_DIR); print("eval checkpoint:", _CK)
_nf = functools.partial(ppo_networks.make_ppo_networks,
    policy_obs_key="state", value_obs_key="privileged_state",
    policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128))
_params = ppo_ckpt.load(_CK)
_net = _nf(env.observation_size, env.action_size,
           preprocess_observations_fn=running_statistics.normalize)
_infer = jax.jit(ppo_networks.make_inference_fn(_net)(_params, deterministic=True))
_reset, _step = jax.jit(env.reset), jax.jit(env.step)
_succ = 0
for i in range(20):
    st = _reset(jax.random.PRNGKey(7000 + i)); st.info["ref_frame0"] = jp.int32(0)
    ever = False
    for _ in range(400):
        act = _infer(st.obs, jax.random.PRNGKey(0))[0]
        st = _step(st, act)
        ever = ever or bool(st.metrics.get("reward/climb_stand", 0.0) > 0.3)
        if st.done:
            break
    _succ += int(ever)
print(f"FLOOR-START full climb @ latest checkpoint: {_succ}/20")
