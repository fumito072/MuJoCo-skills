# ============================================================================
#  G1 CLIMB — Colab/GPU (MJX) training, SELF-CONTAINED (no git clone, no push).
#  Paste each "# === CELL N ===" block into its own Colab cell (Runtime->GPU) in order.
#  CELL 2 has the full env inlined, so you can copy-paste straight from this file.
# ============================================================================

# === CELL 1 — install + asserts. RUN THIS FIRST in every (re)started runtime. ===
# IMPORTANT: a Colab restart/reset wipes pip installs. If you restart the runtime,
# RE-RUN CELL 1 before CELL 2/3, or you'll get "No module named mujoco".
import subprocess
print(subprocess.run(["nvidia-smi","-L"], capture_output=True, text=True).stdout or "NO GPU!")
# install mujoco/mjx explicitly (don't rely on playground pulling them) + a CUDA jax
%pip -q install mujoco mujoco-mjx playground brax "jax[cuda12]" orbax-checkpoint mediapy
import importlib
for mod in ("mujoco", "mujoco.mjx", "jax", "brax", "mujoco_playground", "flax", "orbax.checkpoint"):
    importlib.import_module(mod)               # fail loudly here if any didn't install
import inspect, jax, brax
from brax.training.agents.ppo import train as _ppo
assert "save_checkpoint_path" in inspect.signature(_ppo.train).parameters, "brax too old: %pip install -U brax"
assert "wrap_env_fn" in inspect.signature(_ppo.train).parameters, "brax too old: %pip install -U brax"
assert any(d.platform == "gpu" for d in jax.devices()), \
    "No GPU visible — Runtime->Restart runtime, then RE-RUN CELL 1 (don't skip it)"
print("OK: brax", brax.__version__, "| jax", jax.__version__, "|", jax.devices())


# === CELL 2 — Drive + (INLINE env, no clone/push) + height domain randomization ===
import os, functools
import jax
from google.colab import drive
drive.mount("/content/drive")
CKPT_DIR = "/content/drive/MyDrive/g1_climb_ckpts"     # checkpoints persist here
os.makedirs(CKPT_DIR, exist_ok=True)

# ----- inlined training/g1_climb_mjx_env.py (height-aware; registers G1ClimbBox) -----
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from ml_collections import config_dict

from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.g1 import base as g1_base

# real footrest platform (measured: docs/FULL_MISSION_DEPLOY.md)
PLAT_HALF = (0.30, 0.15, 0.11)        # top at 0.22
PLAT_CENTER = (0.0, 0.35, 0.11)
TARGET = jp.array([0.0, 0.33])        # stand point = the verified sit-descent basin
YAW_GOAL = jp.pi / 2                  # final heading: +y (back to the seat)
SPAWN_FLOOR_Y = 0.68
SIT_KP, SIT_KD = 300.0, 8.0
LEG_GEOMS = ("left_thigh", "right_thigh", "left_shin", "right_shin",
             "left_foot", "right_foot")
HAND_GEOMS = ("left_hand_collision", "right_hand_collision")

# simplified brace surfaces (full-hull measurements; hands-only pairs).
# armrest catch zone = where the probe's hands actually rest (surface ~0.98
# at y ~0.13); the seat caught deeper falls at 0.635.
ARMREST_HALF = (0.035, 0.18, 0.035)
ARMREST_CENTERS = ((0.27, -0.03, 0.935), (-0.27, -0.03, 0.935))   # tops at 0.97
SEAT_HALF = (0.20, 0.20, 0.03)
SEAT_CENTER = (0.0, -0.05, 0.605)                                  # top at 0.635

# measured 4-point bridge state (training/g1_handbrace_probe.py BRIDGE_T=7.0):
# pitch ~34 deg, hands ~50 N each on the rests, qvel norm 0.08
BRIDGE_QPOS = jp.array([
    0.0052, 0.5000, 0.7053,                                        # base pos
    0.6662, 0.2041, 0.2051, -0.6873,                               # quat (yaw -90 + pitch 34)
    -0.4501, 0.0022, 0.0013, 0.8238, -0.4241, -0.0026,             # L leg
    -0.4723, 0.0016, 0.0015, 0.8304, -0.4350, -0.0000,             # R leg
    -0.0438, 0.0506, -0.5323,                                      # waist y/r/p
    -0.4928, 0.9851, 0.0436, 0.0518, 0.0059, -0.0358, 0.0113,      # L arm
    -0.4929, -0.9712, -0.0521, 0.0307, 0.0143, -0.1314, -0.0344,   # R arm
])


def climb_config():
    cfg = g1_joystick.default_config()
    cfg.impl = "jax"
    cfg.episode_length = 400                  # 8 s
    try:
        cfg.njmax = 29 * 2 + 16 * 4
    except (AttributeError, KeyError):        # field renamed in newer playground
        pass
    rc = cfg.reward_config
    s = rc.scales
    for k in ("tracking_lin_vel", "tracking_ang_vel", "feet_phase", "feet_air_time",
              "feet_height", "feet_clearance", "feet_slip", "lin_vel_z", "ang_vel_xy",
              "stand_still", "pose", "joint_deviation_knee", "base_height"):
        if k in s:
            s[k] = 0.0
    s.joint_deviation_hip = -0.05
    s.orientation = -1.0
    s.alive = 0.2
    # climb terms (computed in _get_reward; scales must exist here)
    s.climb_target = 1.0
    s.climb_feet = 1.0
    s.climb_stand = 1.0
    s.climb_lean = -1.0
    s.climb_yaw = -0.3
    return cfg


class G1ClimbBox(g1_joystick.Joystick):
    """G1 learns to climb the 0.22 m platform and stand on it facing +y."""

    def __init__(self, config: config_dict.ConfigDict = climb_config(),
                 config_overrides=None):
        super().__init__(task="flat_terrain", config=config,
                         config_overrides=config_overrides)
        self._add_platform()

    def _add_platform(self):
        assets = g1_base.get_assets()
        spec = mujoco.MjSpec.from_string(
            consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets)
        g = spec.worldbody.add_geom()
        g.name = "platform"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(PLAT_HALF)
        g.pos = list(PLAT_CENTER)
        g.rgba = [0.5, 0.45, 0.4, 1.0]
        g.contype, g.conaffinity = 1, 1
        for rg in LEG_GEOMS:                  # feet are contype=0: pairs are mandatory
            spec.add_pair(geomname1=rg, geomname2="platform")
        # brace surfaces: pair-only (contype=0) and ONLY with the hands, so leg
        # collision behavior is identical to the previous runs
        braces = [("seat_brace", SEAT_HALF, SEAT_CENTER)]
        for i, c in enumerate(ARMREST_CENTERS):
            braces.append((f"armrest_{i}", ARMREST_HALF, c))
        for name, half, center in braces:
            b = spec.worldbody.add_geom()
            b.name = name
            b.type = mujoco.mjtGeom.mjGEOM_BOX
            b.size = list(half)
            b.pos = list(center)
            b.rgba = [0.6, 0.55, 0.5, 1.0]
            b.contype, b.conaffinity = 0, 0
            for hg in HAND_GEOMS:
                spec.add_pair(geomname1=hg, geomname2=name)
        spec.assets = assets
        m = spec.compile()
        m.opt.timestep = self.sim_dt
        for a in range(12):                   # SIT-mode stiff legs (mode switch on HW)
            m.actuator_gainprm[a, 0] = SIT_KP
            m.actuator_biasprm[a, 1] = -SIT_KP
            m.actuator_biasprm[a, 2] = -SIT_KD
        # probe-matched arm mode gains: a braced press needs stiff shoulders/
        # elbows, and stock kp=2 wrists flop and roll off the rests
        for a in (15, 16, 17, 18, 22, 23, 24, 25):
            m.actuator_gainprm[a, 0] = 150.0
            m.actuator_biasprm[a, 1] = -150.0
            m.actuator_biasprm[a, 2] = -4.0
        for a in (19, 20, 21, 26, 27, 28):
            m.actuator_gainprm[a, 0] = 80.0
            m.actuator_biasprm[a, 1] = -80.0
            m.actuator_biasprm[a, 2] = -2.0
        m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
        self._mj_model = m
        self._mjx_model = mjx.put_model(m, impl=self._config.impl)
        self._lf_gid = m.geom("left_foot").id
        self._rf_gid = m.geom("right_foot").id
        self._plat_gid = m.geom("platform").id
        self._post_init()

    # HEIGHT-AWARE helpers (2026-06-15): read the platform height PER ENV so a
    # domain-randomization fn can vary it. With no DR these return the fixed 0.22.
    # A geom at pos_z=h/2, size_z=h/2 sits on the floor with its TOP at h.
    def _plat_h_model(self):                 # in reset (self.mjx_model is per-env under DR)
        return 2.0 * self.mjx_model.geom_pos[self._plat_gid, 2]

    def _plat_h_data(self, data):            # in reward/obs (geom_xpos is per-env)
        return 2.0 * data.geom_xpos[self._plat_gid, 2]

    def sample_command(self, rng):
        del rng
        return jp.zeros(3)

    # ----------------------------------------------------------------- reset --
    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        plat_h = self._plat_h_model()         # per-env platform height (DR-aware)

        rng, k_mode, k1, k2, k3, k4, k5, kd1, kd2 = jax.random.split(rng, 9)
        u = jax.random.uniform(k_mode)
        bridge = u < 0.25                     # mode C: measured 4-point bridge (RSI)
        oneleg = (u >= 0.25) & (u < 0.40)     # mode D: single-support curriculum
        on_platform = (u >= 0.40) & (u < 0.65)

        # mode A: floor, facing the step (-y), forward approach
        yaw_a = -jp.pi / 2 + jax.random.uniform(k1, (), minval=-0.10, maxval=0.10)
        pos_a = jp.array([0.0, SPAWN_FLOOR_Y, qpos[2]])
        pos_a = pos_a.at[0].add(jax.random.uniform(k2, (), minval=-0.04, maxval=0.04))
        pos_a = pos_a.at[1].add(jax.random.uniform(k3, (), minval=-0.03, maxval=0.03))

        # mode B: turn-ladder platform stand (procedural RSI): random heading on top,
        # slight crouch — the value of the goal region propagates back from here
        yaw_b = jax.random.uniform(k1, (), minval=-jp.pi / 2 - 0.1,
                                   maxval=jp.pi / 2 + 0.1)
        crouch = jax.random.uniform(k4, (), minval=0.0, maxval=0.30)
        pos_b = jp.array([0.0, 0.33, qpos[2] + plat_h - crouch * 0.3])
        pos_b = pos_b.at[0].add(jax.random.uniform(k2, (), minval=-0.05, maxval=0.05))
        pos_b = pos_b.at[1].add(jax.random.uniform(k3, (), minval=-0.04, maxval=0.06))

        yaw = jp.where(on_platform, yaw_b, yaw_a)
        pos = jp.where(on_platform, pos_b, pos_a)
        quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])

        joints = qpos[7:]
        leg_crouch = jp.zeros(29)
        for hp, kn, ap in ((0, 3, 4), (6, 9, 10)):
            leg_crouch = leg_crouch.at[hp].add(-crouch * 1.2)
            leg_crouch = leg_crouch.at[kn].add(crouch * 2.0)
            leg_crouch = leg_crouch.at[ap].add(-crouch * 0.6)
        joints = joints + jp.where(on_platform, leg_crouch, jp.zeros(29))

        # mode D: one-leg stance at the step edge — the human curriculum
        # ("shift weight onto one leg, raise the other to ~90 deg, balance").
        # 10 scripted probes (2026-06-13) showed this state is REACHABLE but
        # knife-edge open-loop: the lateral CoM transfer is an avalanche
        # (closed-chain leg push, ~8 cm/s at center crossing) that hand-tuned
        # PD can't time. Spawning IN the state lets PPO learn single-support
        # balance exactly where the climb needs it — the raised foot is one
        # small motion from the platform. lift_amt grades the curriculum
        # (50-100% of the 90-deg tuck). Yaw/quat from mode A already apply.
        stance_left = jax.random.bernoulli(kd1)
        lift_amt = jax.random.uniform(kd2, (), minval=0.5, maxval=1.0)
        pos_d = jp.array([0.0, 0.65, qpos[2] - 0.005])
        pos_d = pos_d.at[0].add(jp.where(stance_left, 0.07, -0.07))   # CoM over stance foot
        pos_d = pos_d.at[0].add(jax.random.uniform(k2, (), minval=-0.02, maxval=0.02))
        pos_d = pos_d.at[1].add(jax.random.uniform(k3, (), minval=-0.03, maxval=0.03))
        swing_pose = jp.array([-1.6, 1.6, -0.2]) * lift_amt   # hip pitch, knee, ankle
        sw_r = (jp.zeros(29).at[6].set(swing_pose[0])
                .at[9].set(swing_pose[1]).at[10].set(swing_pose[2]))
        sw_l = (jp.zeros(29).at[0].set(swing_pose[0])
                .at[3].set(swing_pose[1]).at[4].set(swing_pose[2]))
        mask_r = jp.zeros(29).at[6].set(1.0).at[9].set(1.0).at[10].set(1.0)
        mask_l = jp.zeros(29).at[0].set(1.0).at[3].set(1.0).at[4].set(1.0)
        swing = jp.where(stance_left, sw_r, sw_l)
        mask = jp.where(stance_left, mask_r, mask_l)
        # stance hip roll leans the pelvis over the stance foot (sign measured:
        # negative = lean left; mirrored positive for the right stance)
        hiproll = jp.where(stance_left,
                           jp.zeros(29).at[1].set(-0.12),
                           jp.zeros(29).at[7].set(0.12))
        joints_d = joints * (1 - mask) + swing + hiproll
        pos = jp.where(oneleg, pos_d, pos)
        joints = jp.where(oneleg, joints_d, joints)

        # mode C overrides: the probe's quasi-static brace, tighter noise so
        # the pressed hands stay in contact
        pos = jp.where(bridge, BRIDGE_QPOS[0:3], pos)
        quat = jp.where(bridge, BRIDGE_QPOS[3:7], quat)
        joints = jp.where(bridge, BRIDGE_QPOS[7:], joints)
        rng, k6 = jax.random.split(rng)
        noise = jax.random.uniform(k6, (29,), minval=-0.08, maxval=0.08)
        joints = joints + noise * jp.where(bridge, 0.3, 1.0)

        qpos = jp.concatenate([pos, quat, joints])
        rng, k7 = jax.random.split(rng)
        qvel = qvel.at[0:2].set(
            jax.random.uniform(k7, (2,), minval=-0.12, maxval=0.12)
            * jp.where(bridge | oneleg, 0.0, 1.0))

        # playground/mjx API drift: the config fields were RENAMED (nconmax ->
        # naconmax, Colab 2026-06) and the kwargs differ across versions. Read
        # whichever exists; fall back to letting make_data size the arenas.
        mk = dict(qpos=qpos, qvel=qvel, ctrl=qpos[7:], impl=self.mjx_model.impl.value)
        for kw, names in (("nconmax", ("nconmax", "naconmax")),
                          ("njmax", ("njmax", "najmax"))):
            for n in names:
                try:
                    mk[kw] = self._config[n]
                    break
                except KeyError:
                    continue
        try:
            data = mjx_env.make_data(self.mj_model, **mk)
        except TypeError:
            mk.pop("nconmax", None)
            mk.pop("njmax", None)
            try:
                data = mjx_env.make_data(self.mj_model, **mk)
            except TypeError:
                # playground/mjx skew where the WRAPPER passes renamed kwargs
                # internally (older playground + newer mjx, e.g. local Mac):
                # build the data object from mjx directly.
                data = mjx.make_data(self.mjx_model).replace(
                    qpos=qpos, qvel=qvel, ctrl=qpos[7:])
        data = mjx.forward(self.mjx_model, data)

        phase = jp.array([0.0, jp.pi])
        rng, cmd_rng = jax.random.split(rng)
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        info = {
            "rng": rng,
            "step": 0,
            "command": self.sample_command(cmd_rng),
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            "phase_dt": 2 * jp.pi * self.dt * 1.375,
            "phase": phase,
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": jp.round(push_interval / self.dt).astype(jp.int32),
        }
        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())
        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._feet_floor_found_sensor
        ])
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ------------------------------------------------------------------- obs --
    def _get_obs(self, data, info, contact):
        obs = super()._get_obs(data, info, contact)
        base = data.qpos[0:3]
        w, x_, y_, z_ = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        yaw = jp.arctan2(2 * (w * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
        ye = jp.mod(yaw - YAW_GOAL + jp.pi, 2 * jp.pi) - jp.pi
        exw, eyw = TARGET[0] - base[0], TARGET[1] - base[1]
        ex = jp.cos(yaw) * exw + jp.sin(yaw) * eyw
        ey = -jp.sin(yaw) * exw + jp.cos(yaw) * eyw
        # gravity/IMU are yaw-invariant: without these 5 dims the turn-to-goal
        # is unobservable (on HW they come from chair-frame localization).
        # +plat_h so the policy knows the step height it faces (domain randomized).
        extra = jp.array([jp.sin(ye), jp.cos(ye), ex, ey, base[2], self._plat_h_data(data)])
        if isinstance(obs, dict):
            return {k: jp.concatenate([v, extra]) for k, v in obs.items()}
        return jp.concatenate([obs, extra])

    # ---------------------------------------------------------------- reward --
    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact)

        base = data.qpos[0:3]
        w, x_, y_, z_ = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        yaw = jp.arctan2(2 * (w * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
        ye = jp.mod(yaw - YAW_GOAL + jp.pi, 2 * jp.pi) - jp.pi
        plat_h = self._plat_h_data(data)              # per-env platform height (DR-aware)
        stand_z = 0.755 + plat_h                       # pelvis height standing on top
        dist = jp.linalg.norm(base[0:2] - TARGET)
        dz = jp.abs(base[2] - stand_z)

        # absolute potential (no per-step delta plumbing needed in this API)
        rewards["climb_target"] = jp.exp(-2.5 * dist) + 1.2 * jp.exp(-6.0 * dz)

        # feet geometrically on the platform TOP (per-env height; contact proxy).
        # gate on the platform-top band so a foot grazing the front face/edge of a
        # low step is NOT a false positive (a trap we hit on CPU).
        def on_plat(gid):
            p = data.geom_xpos[gid]
            return ((p[2] > plat_h - 0.03) & (p[2] < plat_h + 0.08)
                    & (jp.abs(p[0]) < PLAT_HALF[0])
                    & (jp.abs(p[1] - PLAT_CENTER[1]) < PLAT_HALF[1]))
        lf = on_plat(self._lf_gid)
        rf = on_plat(self._rf_gid)
        feet = lf.astype(jp.float32) + rf.astype(jp.float32)
        rewards["climb_feet"] = 0.15 * feet + 0.6 * (feet == 2)

        # the goal region pays PER STEP: stand on target, upright, facing +y.
        # BRIEF arrival is enough — the stiff gains / mission FSM hold the stand
        # (forcing RL to hold a long still stand was the CPU dead-end).
        up = self.get_gravity(data, "torso")
        upright = jp.exp(-jp.sum(jp.square(up - jp.array([0.073, 0.0, 1.0]))) / 0.1)
        standing = ((feet == 2) & (base[2] > stand_z - 0.055) & (dist < 0.15)
                    & (jp.abs(ye) < 0.45))
        rewards["climb_stand"] = jp.where(standing, 5.0 * upright, 0.0)

        # sitting/leaning on the platform edge instead of climbing = exploit
        leaning = (base[2] < 0.62) & (base[1] < 0.58) & (feet < 2)
        rewards["climb_lean"] = leaning.astype(jp.float32)

        rewards["climb_yaw"] = jp.abs(ye) * (feet == 2)
        return rewards


registry.locomotion.register_environment("G1ClimbBox", G1ClimbBox, climb_config)

# ----- end inlined env -----
from mujoco_playground import registry
env = registry.load("G1ClimbBox")
assert hasattr(env, "_plat_h_model"), "env inline mismatch"
PLAT_GID = env._plat_gid
H_MIN, H_MAX = 0.04, 0.22

def height_domain_randomization(model, rng):
    """Per-env platform height in [H_MIN,H_MAX]. brax/Playground pass rng already
    batched (num_envs,); vmap over it, return (batched_model, in_axes). The env reads
    h back from the geom so ONE policy learns ALL heights (no catastrophic forgetting)."""
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
print("env ready (inline). platform height randomized in", (H_MIN, H_MAX))

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
          f"sum_reward {m.get('eval/episode_sum_reward', float('nan')):8.2f}  "
          f"climb_stand {m.get('eval/episode_reward/climb_stand', float('nan')):6.3f}", flush=True)
make_inference_fn, params, _ = ppo.train(
    environment=env, wrap_env_fn=wrapper.wrap_for_brax_training,
    randomization_fn=height_domain_randomization,
    num_timesteps=NUM_TIMESTEPS, num_evals=NUM_EVALS, episode_length=400,
    num_envs=NUM_ENVS, batch_size=256, num_minibatches=32, unroll_length=20,
    num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2,
    discounting=0.97, gae_lambda=0.95, clipping_epsilon=0.2,
    normalize_observations=True, network_factory=network_factory,
    save_checkpoint_path=CKPT_DIR, restore_checkpoint_path=restore,
    progress_fn=progress, seed=0)
print("DONE. latest checkpoint:", latest_ckpt(CKPT_DIR))


# === CELL 4 — eval per height + render ===
import mediapy as media
def eval_height(h, n=20):
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


# === CELL 5 — save params for Mac inference ===
import pickle
with open(f"{CKPT_DIR}/g1_climb_params.pkl", "wb") as f:
    pickle.dump(params, f)
print("saved ->", f"{CKPT_DIR}/g1_climb_params.pkl")
