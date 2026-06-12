"""MJX (GPU) env for the 0.22 m step-up — "G1ClimbBox", a MuJoCo Playground env.

WHY THIS EXISTS: ~250M CPU PPO steps (task + DeepMimic variants) plateaued at
single-digit success on the climb — the one skill blocking the full mission
(docs/FULL_MISSION_DEPLOY.md §3). This env ports the task to Playground/MJX so
it can train at GPU scale (thousands of parallel envs) on a free Colab T4 —
see training/g1_climb_colab.ipynb. Train once on GPU, replay on the Mac
(training/g1_climb_play.py), same split as the G1Sit pipeline.

KEY SIMPLIFICATION (what makes MJX fast here): during the climb the robot only
touches the floor and the footrest PLATFORM — never the chair's 153 collision
hulls. So the training scene is the playground G1 + ONE BOX with the real
platform's exact dimensions (top 0.22 m, x in [-0.3, 0.3], y in [0.2, 0.5]).
The trained policy is then verified on the Mac in the full 153-hull scene.

Matches the CPU task (training/g1_climb_env.py): SIT-mode stiff gains
(kp=300/kd=8) baked into the model, legs-only actions would need obs surgery —
here we keep the FULL joystick action space (29) but the reward only cares
about the result; command is zeroed. Spawns: floor facing the step (forward
approach) OR a turn-ladder platform stand (the RSI bank, procedural in jax).
Success region (standing on the platform near (0, 0.33), facing +y = back to
the seat) pays a large PER-STEP bonus instead of a terminal bonus (simplest
within the Joystick step plumbing).

The playground feet are contype=0 (floor contact via explicit pairs only), so
the platform gets explicit pairs to feet+shins+thighs — without them the robot
falls THROUGH the box (the same trap the CPU chair import hit).
"""
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
        spec.assets = assets
        m = spec.compile()
        m.opt.timestep = self.sim_dt
        for a in range(12):                   # SIT-mode stiff legs (mode switch on HW)
            m.actuator_gainprm[a, 0] = SIT_KP
            m.actuator_biasprm[a, 1] = -SIT_KP
            m.actuator_biasprm[a, 2] = -SIT_KD
        m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
        self._mj_model = m
        self._mjx_model = mjx.put_model(m, impl=self._config.impl)
        self._lf_gid = m.geom("left_foot").id
        self._rf_gid = m.geom("right_foot").id
        self._post_init()

    def sample_command(self, rng):
        del rng
        return jp.zeros(3)

    # ----------------------------------------------------------------- reset --
    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        rng, k_mode, k1, k2, k3, k4, k5 = jax.random.split(rng, 7)
        on_platform = jax.random.bernoulli(k_mode, 0.4)

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
        pos_b = jp.array([0.0, 0.33, qpos[2] + 0.22 - crouch * 0.3])
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
        rng, k6 = jax.random.split(rng)
        joints = joints + jax.random.uniform(k6, (29,), minval=-0.08, maxval=0.08)

        qpos = jp.concatenate([pos, quat, joints])
        rng, k7 = jax.random.split(rng)
        qvel = qvel.at[0:2].set(jax.random.uniform(k7, (2,), minval=-0.12, maxval=0.12))

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
            data = mjx_env.make_data(self.mj_model, **mk)
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
        # is unobservable (on HW they come from chair-frame localization)
        extra = jp.array([jp.sin(ye), jp.cos(ye), ex, ey, base[2]])
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
        dist = jp.linalg.norm(base[0:2] - TARGET)
        dz = jp.abs(base[2] - 0.975)

        # absolute potential (no per-step delta plumbing needed in this API)
        rewards["climb_target"] = jp.exp(-2.5 * dist) + 1.2 * jp.exp(-6.0 * dz)

        # feet geometrically on the platform (contact proxy; MJX-cheap)
        def on_plat(gid):
            p = data.geom_xpos[gid]
            return ((p[2] > 0.20) & (p[2] < 0.30)
                    & (jp.abs(p[0]) < PLAT_HALF[0])
                    & (jp.abs(p[1] - PLAT_CENTER[1]) < PLAT_HALF[1]))
        lf = on_plat(self._lf_gid)
        rf = on_plat(self._rf_gid)
        feet = lf.astype(jp.float32) + rf.astype(jp.float32)
        rewards["climb_feet"] = 0.15 * feet + 0.6 * (feet == 2)

        # the goal region pays PER STEP: stand on target, upright, facing +y
        up = self.get_gravity(data, "torso")
        upright = jp.exp(-jp.sum(jp.square(up - jp.array([0.073, 0.0, 1.0]))) / 0.1)
        standing = ((feet == 2) & (base[2] > 0.92) & (dist < 0.15)
                    & (jp.abs(ye) < 0.45))
        rewards["climb_stand"] = jp.where(standing, 5.0 * upright, 0.0)

        # sitting/leaning on the platform edge instead of climbing = exploit
        leaning = (base[2] < 0.62) & (base[1] < 0.58) & (feet < 2)
        rewards["climb_lean"] = leaning.astype(jp.float32)

        rewards["climb_yaw"] = jp.abs(ye) * (feet == 2)
        return rewards


registry.locomotion.register_environment("G1ClimbBox", G1ClimbBox, climb_config)
