"""G1 climb env for AMP (Adversarial Motion Priors), MJX. Adapted from the DeepMimic
env (g1_climb_deepmimic_colab.py): SAME platform, reference RSI, residual control, and
TRUE-floor eval — but the per-frame TRACKING reward (climb_mimic) is REMOVED. The
"style" now comes from a discriminator (trained in amp_train) that rewards motion
resembling the reference distribution, leaving the policy free to find its own dynamic
single-leg balance for the first foot-lift. The env emits per-step motion features
`info["amp_obs"]`; the training loop turns them into the style reward.

Local module (loads the reference from g1_climb_reference.npz). The Colab notebook
inlines this with the reference embedded as base64.
"""
import os

import numpy as np
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

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ref = np.load(os.path.join(_REPO, "training", "g1_climb_reference.npz"))
REF_LEGS = jp.asarray(_ref["legs"], jp.float32)          # (N,12)
REF_BASE = jp.asarray(_ref["base"], jp.float32)          # (N,2) = y,z
REF_N = int(REF_LEGS.shape[0])
REF_YAW = float(_ref["yaw"])
REF_DT = float(_ref["dt"])

PLAT_HALF = (0.30, 0.15, 0.11)
PLAT_CENTER = (0.0, 0.35, 0.11)
SIT_KP, SIT_KD = 300.0, 8.0
LEG_GEOMS = ("left_thigh", "right_thigh", "left_shin", "right_shin",
             "left_foot", "right_foot")
AMP_FEAT_DIM = 30                                        # see _amp_features


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
    # NO climb_mimic — AMP replaces per-frame tracking with the discriminator style.
    # TASK rewards BOOSTED (v2): v1 style(~0.4/step) drowned a tiny task(~0.05) => 0/20.
    # climb_target is a DENSE goal-approach term so the climb is actually PURSUED.
    s.climb_target = 10.0     # dense: approach the platform-stand position (TASK)
    s.climb_feet = 2.0        # feet on the platform top (TASK)
    s.climb_stand = 10.0      # upright stand on top = the goal (TASK)
    return cfg


class G1ClimbAMP(g1_joystick.Joystick):
    """G1 climbs the 0.22 m platform. Reward = TASK only (feet+stand); the AMP style
    reward is added in the training loop from info['amp_obs']. Keeps reference RSI +
    residual control (a ~33% open-loop head start independent of the reward)."""

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
        for rg in LEG_GEOMS:
            spec.add_pair(geomname1=rg, geomname2="platform")
        spec.assets = assets
        m = spec.compile()
        m.opt.timestep = self.sim_dt
        for a in range(12):
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

    # ---- AMP motion features (30-D), reference-derivable, consistent expert<->robot.
    # 12 leg pos + 12 leg vel + [base_z, pelvis world z-vel, pelvis world y-vel(fwd),
    # torso tilt x, torso tilt y, foot height-diff]. NO xy/yaw/arms (reference-blind).
    def _amp_features(self, data):
        legs = data.qpos[7:19]
        legvel = data.qvel[6:18]
        gv = self.get_global_linvel(data, "pelvis")
        up = self.get_gravity(data, "torso")
        feet_dz = data.geom_xpos[self._lf_gid, 2] - data.geom_xpos[self._rf_gid, 2]
        return jp.concatenate([
            legs, legvel,
            jp.array([data.qpos[2], gv[2], gv[1], up[0], up[1], feet_dz])])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        rng, kf, kx, ky, kj, kv = jax.random.split(rng, 6)
        frac = jax.random.uniform(kf) ** 2
        frame0 = (frac * (REF_N - 1)).astype(jp.int32)
        rl, rb = REF_LEGS[frame0], REF_BASE[frame0]
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
            "ref_frame0": frame0,
            # AMP motion features. Carry BOTH phi(s_t) and phi(s_{t+1}) per transition
            # so the loop builds the discriminator pair with NO off-by-one.
            "amp_obs": self._amp_features(data),           # phi(s_{t+1}) after a step
            "amp_obs_prev": self._amp_features(data),      # phi(s_t)
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
        return jp.clip(info["ref_frame0"] + info["step"], 0, REF_N - 1)

    # RESIDUAL control (motor_legs = REF_LEGS[frame] + action*scale; default cancels),
    # then refresh the AMP features on the NEW data.
    def step(self, state, action):
        frame = self._cur_frame(state.info)
        off = jp.zeros(self.mjx_model.nu).at[0:12].set(
            (REF_LEGS[frame] - self._default_pose[0:12]) / self._config.action_scale)
        state.info["amp_obs_prev"] = state.info["amp_obs"]      # phi(s_t)
        state = super().step(state, action + off)
        state.info["amp_obs"] = self._amp_features(state.data)  # phi(s_{t+1})
        return state

    def _get_obs(self, data, info, contact):
        obs = super()._get_obs(data, info, contact)
        frame = self._cur_frame(info)
        rl, rb = REF_LEGS[frame], REF_BASE[frame]
        base = data.qpos[0:3]
        phase = frame.astype(jp.float32) / REF_N
        leg_err = rl - data.qpos[7:19]
        extra = jp.concatenate([
            jp.array([phase, base[2], rb[0] - base[1], rb[1] - base[2]]), leg_err])
        if isinstance(obs, dict):
            return {k: jp.concatenate([v, extra]) for k, v in obs.items()}
        return jp.concatenate([obs, extra])

    # TASK reward only (NO climb_mimic). climb_feet + climb_stand + base regularizers.
    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact)
        base = data.qpos[0:3]
        # DENSE goal-approach (TASK): reward being near the platform-stand position so
        # the climb is pursued from the floor (v1 had only sparse on-platform reward).
        tgt = REF_BASE[-1]                                   # (y, z) at the stand
        dist = jp.abs(base[1] - tgt[0]) + jp.abs(base[0])    # forward + lateral drift
        rewards["climb_target"] = jp.exp(-3.0 * dist) + jp.exp(-6.0 * jp.abs(base[2] - tgt[1]))
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


registry.locomotion.register_environment("G1ClimbAMP", G1ClimbAMP, climb_config)
