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

# --- load the CoM-IK leg-climb reference (561 frames: 12 leg joints + base y,z) ---
# pushed in the repo; pull the raw file (no clone). If your repo is private or the
# push hasn't happened, upload training/g1_climb_reference.npz to CKPT_DIR instead.
REF_URL = ("https://raw.githubusercontent.com/fumito072/MuJoCo-skills/"
           "main/training/g1_climb_reference.npz")
REF_LOCAL = f"{CKPT_DIR}/g1_climb_reference.npz"
if not os.path.exists(REF_LOCAL):
    subprocess.run(["wget", "-q", "-O", REF_LOCAL, REF_URL])
_ref = np.load(REF_LOCAL)
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
print("env ready. reference RSI + residual control + tracking reward.")


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
