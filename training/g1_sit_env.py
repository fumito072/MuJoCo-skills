"""Custom G1 SIT-ON-CHAIR task for MuJoCo Playground — learn (via RL) to lower onto a seat and hold a
stable, upright seated posture. This is the task the open-loop / CEM controllers toppled on.

It subclasses Playground's G1 Joystick env and changes three things:
  1. MODEL: add a static SEAT box (collidable) the robot lowers onto, plus a collision geom on the
     pelvis so its weight is physically SUPPORTED by the seat — that support is what makes a stable
     upright sit reachable (a deep *unsupported* squat is balance-critical and ill-posed).
  2. COMMAND: always zero — the robot holds position and sits, it does not walk.
  3. REWARD: pull the pelvis DOWN onto the seat while staying upright and over the seat, and DROP the
     walking-pose incentives (stand_still / pose / knee-deviation) that otherwise fight sitting.

Why a chair (and not free space): the seat gives support, so "pelvis at seat height + upright" is a
stable physical attractor. Termination is orientation-based (topple), NOT height-based, so a low
seated pelvis is not mistaken for a fall.

Curriculum knob: raise SEAT_TOP for an easy/shallow sit, lower it for a deep/hard one.

Import this module before loading "G1Sit" — importing it registers the env. The plain-MuJoCo model
builder (build_plain_chair_model) is reused by the Mac-side replay so train and inference share the
exact same chaired model.
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

# --- chair geometry (curriculum: raise SEAT_TOP = easier/shallower, lower = harder/deeper) ---
# 0.39 m is the seat height at which the G1's FULL seated pose (hip -1.254, knee +1.611)
# puts the mesh-calibrated pelvis bottom exactly on the board: base 0.546 - 0.155 = 0.391.
# A 0.45 m human chair is a tall stool for a 1.32 m robot — the descent gets caught ~6 cm
# early and the robot perches leaning ~34 deg forward (measured 2026-06-10). Sitting on
# human-height chairs is a separate skill (scoot-back / perch balance -> RL track).
SEAT_TOP = 0.39                       # seat surface height (m), sized to the robot
SEAT_XY = (-0.08, 0.0)                # seat center, under where the pelvis descends
SEAT_HALF = (0.18, 0.20, 0.02)        # seat box half-sizes
# Pelvis collision box, calibrated to the model's own meshes (measured 2026-06-10):
# the pelvis mesh bbox bottom sits at base-0.155 (hip housings hang BELOW the hip axes),
# so the box bottom is -0.155 — the original -0.105 bottom was 5 cm too shallow, which
# made the "butt" unable to reach a seat once the thigh capsules rest on it.
PELVIS_COL_SIZE = (0.09, 0.11, 0.08)   # pelvis collision box half-sizes
PELVIS_COL_POS = (-0.02, 0.0, -0.075)  # offset from pelvis body origin (bottom -0.155)
# pelvis base-z when the pelvis collision rests on the seat top
SIT_TARGET_Z = SEAT_TOP + (-PELVIS_COL_POS[2] + PELVIS_COL_SIZE[2])  # ≈ 0.555


def _add_pelvis_collision(spec):
    pelvis = [b for b in spec.bodies if b.name == "pelvis"][0]
    pc = pelvis.add_geom()
    pc.name = "pelvis_collision"
    pc.type = mujoco.mjtGeom.mjGEOM_BOX
    pc.size = list(PELVIS_COL_SIZE)
    pc.pos = list(PELVIS_COL_POS)
    pc.rgba = [0.8, 0.2, 0.2, 0.4]
    pc.contype, pc.conaffinity = 1, 1


def _add_chair_geoms(spec):
    """Add the static seat + a pelvis collision geom to a loaded MjSpec (in place)."""
    seat = spec.worldbody.add_geom()
    seat.name = "seat"
    seat.type = mujoco.mjtGeom.mjGEOM_BOX
    seat.size = list(SEAT_HALF)
    seat.pos = [SEAT_XY[0], SEAT_XY[1], SEAT_TOP - SEAT_HALF[2]]
    seat.rgba = [0.55, 0.38, 0.22, 1.0]
    seat.contype, seat.conaffinity = 1, 1
    _add_pelvis_collision(spec)


# --- REALISTIC chair (seat board + backrest + 4 legs) + leg<->chair collision ------------
# Chair frame: origin = seat center on the floor, +x = the direction the SITTER faces.
# All numbers below derive from MEASURED leg-capsule extents (probe, 2026-06-10), robot
# base at the dock point, coordinates relative to that base:
#   standing thigh capsule (r=0.06) comes no closer than x ~ +0.06 to the seat-top band
#     z in [0.41, 0.45]  -> the board FRONT EDGE must sit at/behind the base (x <= 0.0);
#   seated pelvis box spans x [-0.231, -0.051]  -> a 0.42 m-deep board whose front edge
#     is AT the base still fully supports it;
#   seated thigh capsules sink ~5 cm BELOW the seat-top plane -> with collision enabled
#     the thighs REST ON the board (human-like, weight through the thighs), so the final
#     seated pelvis height is HIGHER than the no-collision 0.555 m — measure, don't assume;
#   standing shin capsules top out at z=0.405 < board bottom 0.41 (clear), and the front
#     chair legs must stay >= ~0.10 behind the front edge to clear marching feet/shins.
REAL_SEAT_TOP = SEAT_TOP
REAL_SEAT_HALF = (0.21, 0.20, 0.02)      # board half-sizes (0.42 m deep, 0.40 m wide)
REAL_BACK_HALF = (0.015, 0.20, 0.20)     # backrest panel half-sizes (rises 0.45->0.85 m)
REAL_LEG_R, REAL_LEG_HALF = 0.025, (SEAT_TOP - 2 * REAL_SEAT_HALF[2]) / 2
REAL_LEG_XY = ((0.11, 0.17), (0.11, -0.17), (-0.18, 0.17), (-0.18, -0.17))
# dock pose in the chair frame: robot base at the seat FRONT EDGE, facing chair +x
REAL_DOCK_OFFSET = (REAL_SEAT_HALF[0], 0.0)
# These contype=0 collision geoms already exist in the playground G1 model (they are
# used by its own explicit pairs) — pairing them with the chair cannot disturb walking.
ROBOT_LEG_GEOMS = ("left_thigh", "right_thigh", "left_shin", "right_shin",
                   "left_foot", "right_foot")


def add_real_chair_geoms(spec, center_xy=(0.0, 0.0), yaw=0.0):
    """Add a collidable realistic chair at center_xy with the sitter facing `yaw`, the
    pelvis collision box, and explicit robot-leg<->chair contact pairs. Returns the
    list of chair geom names."""
    import math
    c, s = math.cos(yaw), math.sin(yaw)
    qz = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]

    names = []

    def add(name, gtype, size, px, py, z, rgba):
        g = spec.worldbody.add_geom()
        g.name = name
        g.type = gtype
        g.size = list(size)
        g.pos = [center_xy[0] + c * px - s * py, center_xy[1] + s * px + c * py, z]
        g.quat = qz
        g.rgba = list(rgba)
        g.contype, g.conaffinity = 1, 1
        names.append(name)

    wood, dark = (0.55, 0.38, 0.22, 1.0), (0.40, 0.26, 0.13, 1.0)
    add("seat", mujoco.mjtGeom.mjGEOM_BOX, REAL_SEAT_HALF,
        0.0, 0.0, REAL_SEAT_TOP - REAL_SEAT_HALF[2], wood)
    add("backrest", mujoco.mjtGeom.mjGEOM_BOX, REAL_BACK_HALF,
        -(REAL_SEAT_HALF[0] + REAL_BACK_HALF[0]), 0.0,
        REAL_SEAT_TOP + REAL_BACK_HALF[2], wood)
    for i, (lx, ly) in enumerate(REAL_LEG_XY):
        add(f"chair_leg{i}", mujoco.mjtGeom.mjGEOM_CYLINDER,
            (REAL_LEG_R, REAL_LEG_HALF, 0), lx, ly, REAL_LEG_HALF, dark)

    _add_pelvis_collision(spec)          # contype=1: collides with all chair geoms

    # torso collision proxy (sized from the torso mesh bbox): contype=0, paired with the
    # backrest/seat only — so leaning back is CAUGHT instead of ghosting through the panel
    torso = [b for b in spec.bodies if b.name == "torso_link"][0]
    tc = torso.add_geom()
    tc.name = "torso_collision"
    tc.type = mujoco.mjtGeom.mjGEOM_BOX
    tc.size = [0.07, 0.10, 0.15]
    tc.pos = [0.0, 0.0, 0.15]
    tc.rgba = [0.8, 0.2, 0.2, 0.3]
    tc.contype, tc.conaffinity = 0, 0

    for rg in ROBOT_LEG_GEOMS + ("torso_collision",):   # collide with the chair ONLY via pairs
        for cg in names:
            spec.add_pair(geomname1=rg, geomname2=cg)
    return names


def build_real_chair_model(sim_dt=0.002, center_xy=(0.0, 0.0), yaw=0.0):
    """Plain-MuJoCo G1 feetonly model WITH the realistic chair + leg collision pairs."""
    assets = g1_base.get_assets()
    spec = mujoco.MjSpec.from_string(consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets)
    add_real_chair_geoms(spec, center_xy=center_xy, yaw=yaw)
    spec.assets = assets
    m = spec.compile()
    m.opt.timestep = sim_dt
    return m


# --- REAL-WORLD chair: ３階講堂遠隔操作席, imported from FBX via the g1-isu pipeline ----
# assets/real_chair/ = 153 convex-decomposed collision hulls + 1 visual mesh +
# chair_info.json (auto-detected seat center/height/facing, in the CHAIR frame).
# Facts (chair_info.json): seat top 0.6483 m, seat center (0.0008, -0.1029, 0.6483),
# sitter faces ~+y, integrated footrest platform (the robot CANNOT sit from the floor:
# its standing pelvis bottom is 0.600 m < the 0.648 m seat — the platform is mandatory).
import json as _json
import os as _os

REAL_CHAIR_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                               "assets", "real_chair")


def load_fbx_chair_info():
    with open(_os.path.join(REAL_CHAIR_DIR, "chair_info.json")) as f:
        return _json.load(f)


def add_fbx_chair_geoms(spec, center_xy=(0.0, 0.0), yaw=0.0, pair_legs=True):
    """Add the FBX-imported real chair (static body, 153 convex hulls + visual mesh)
    at center_xy with the whole chair frame rotated by `yaw`, plus the pelvis/torso
    collision proxies and (optionally) explicit robot-leg<->hull contact pairs.
    Returns the list of hull geom names."""
    import math
    body = spec.worldbody.add_body()
    body.name = "real_chair"
    body.pos = [center_xy[0], center_xy[1], 0.0]
    body.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]

    vis = spec.add_mesh()
    vis.name = "rc_visual"
    vis.file = _os.path.join(REAL_CHAIR_DIR, "visual", "chair_visual.obj")
    g = body.add_geom()
    g.name = "rc_visual"
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = "rc_visual"
    g.contype, g.conaffinity = 0, 0
    g.group = 2
    g.rgba = [0.35, 0.40, 0.50, 1.0]

    names = []
    files = sorted(_os.listdir(_os.path.join(REAL_CHAIR_DIR, "collision")))
    for f in files:
        if not f.endswith(".obj"):
            continue
        mesh = spec.add_mesh()
        mesh.name = f"rc_{f[:-4]}"
        mesh.file = _os.path.join(REAL_CHAIR_DIR, "collision", f)
        g = body.add_geom()
        g.name = f"rc_{f[:-4]}"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = mesh.name
        g.contype, g.conaffinity = 1, 1     # collides with the pelvis box automatically
        g.condim = 3
        g.friction = [0.8, 0.02, 0.01]
        g.group = 3
        g.rgba = [0.6, 0.6, 0.65, 0.0]      # invisible: the visual mesh shows the chair
        names.append(g.name)

    _add_pelvis_collision(spec)
    torso = [b for b in spec.bodies if b.name == "torso_link"][0]
    tc = torso.add_geom()
    tc.name = "torso_collision"
    tc.type = mujoco.mjtGeom.mjGEOM_BOX
    tc.size = [0.07, 0.10, 0.15]
    tc.pos = [0.0, 0.0, 0.15]
    tc.rgba = [0.8, 0.2, 0.2, 0.3]
    tc.contype, tc.conaffinity = 0, 0

    if pair_legs:
        for rg in ROBOT_LEG_GEOMS + ("torso_collision",):
            for cg in names:
                spec.add_pair(geomname1=rg, geomname2=cg)
    return names


def build_fbx_chair_model(sim_dt=0.002, center_xy=(0.0, 0.0), yaw=0.0, pair_legs=True):
    """Plain-MuJoCo G1 feetonly model WITH the real FBX chair + leg collision pairs."""
    assets = g1_base.get_assets()
    spec = mujoco.MjSpec.from_string(consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets)
    add_fbx_chair_geoms(spec, center_xy=center_xy, yaw=yaw, pair_legs=pair_legs)
    spec.assets = assets
    m = spec.compile()
    m.opt.timestep = sim_dt
    return m


def build_plain_chair_model(sim_dt=0.002):
    """Plain-MuJoCo (C engine) G1 feetonly flat-terrain model WITH the chair — for Mac replay."""
    assets = g1_base.get_assets()
    spec = mujoco.MjSpec.from_string(consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets)
    _add_chair_geoms(spec)
    spec.assets = assets
    m = spec.compile()
    m.opt.timestep = sim_dt
    return m


def sit_config():
    cfg = g1_joystick.default_config()
    cfg.impl = "jax"                  # pure-JAX MJX backend (Colab has no mujoco-warp)
    cfg.njmax = 29 * 2 + 16 * 4       # a bit more constraint room for the extra seat contacts
    rc = cfg.reward_config
    rc.base_height_target = SIT_TARGET_Z
    s = rc.scales
    # OFF: all locomotion / gait incentives
    for k in ("tracking_lin_vel", "tracking_ang_vel", "feet_phase", "feet_air_time",
              "feet_height", "feet_clearance", "feet_slip", "lin_vel_z", "ang_vel_xy"):
        if k in s:
            s[k] = 0.0
    # OFF: terms that pull back to the STANDING pose and thus fight sitting
    s.stand_still = 0.0               # (this was the +1.0 sign-bug that rewarded any deviation)
    s.pose = 0.0                      # don't regularize toward the standing pose
    s.joint_deviation_knee = 0.0      # knees MUST bend to sit
    s.joint_deviation_hip = -0.1      # keep mild anti-splay only
    # ON: sit incentives
    s.base_height = -10.0             # pull pelvis down to the seat rest height
    s.orientation = -5.0              # stay upright (stronger than walking's -2)
    s.alive = 1.0                     # survive the episode (don't topple)
    # NEW chair terms (added in G1Sit._get_reward; scales must exist here)
    s.pelvis_over_seat = -2.0         # stay horizontally over the seat
    s.seated = 5.0                    # bonus for resting upright on the seat
    # (termination stays -100; dof_pos_limits/collision/contact_force keep their defaults)
    return cfg


class G1Sit(g1_joystick.Joystick):
    """G1 learns to sit down onto a seat: lower the pelvis onto it, stay upright, don't topple."""

    def __init__(self, config: config_dict.ConfigDict = sit_config(), config_overrides=None):
        super().__init__(task="flat_terrain", config=config, config_overrides=config_overrides)
        self._add_chair()

    def _add_chair(self):
        assets = g1_base.get_assets()
        spec = mujoco.MjSpec.from_string(consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets)
        _add_chair_geoms(spec)
        spec.assets = assets
        m = spec.compile()
        m.opt.timestep = self.sim_dt
        m.vis.global_.offwidth, m.vis.global_.offheight = 3840, 2160
        self._mj_model = m
        self._mjx_model = mjx.put_model(m, impl=self._config.impl)
        self._seat_geom_id = m.geom("seat").id
        self._pelvis_collision_id = m.geom("pelvis_collision").id
        self._seat_xy = jp.array(SEAT_XY)
        self._post_init()             # recompute indices/ids against the chaired model

    def sample_command(self, rng):
        del rng                        # no walking command — always "stay / sit"
        return jp.zeros(3)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # Start standing OVER the seat with only small perturbations (seat is at a fixed spot,
        # so no large xy/yaw randomization — otherwise the robot would start off the seat).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)
        qpos = qpos.at[0:2].set(self._seat_xy + dxy)
        rng, key = jax.random.split(rng)
        qpos = qpos.at[7:].set(qpos[7:] * jax.random.uniform(key, (29,), minval=0.9, maxval=1.1))
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.2, maxval=0.2))

        data = mjx_env.make_data(
            self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax, njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        phase = jp.array([0.0, jp.pi])
        phase_dt = 2 * jp.pi * self.dt * 1.375          # fixed gait freq (irrelevant at command=0)
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            "phase_dt": phase_dt,
            "phase": phase,
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
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

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact
        )
        over = jp.linalg.norm(data.qpos[0:2] - self._seat_xy)
        rewards["pelvis_over_seat"] = over                       # cost (weight < 0)
        height_term = jp.exp(-jp.square(data.qpos[2] - SIT_TARGET_Z) / 0.005)
        up = self.get_gravity(data, "torso")
        upright_term = jp.exp(-jp.sum(jp.square(up - jp.array([0.073, 0.0, 1.0]))) / 0.1)
        over_term = jp.exp(-jp.square(over) / 0.02)
        rewards["seated"] = height_term * upright_term * over_term  # bonus (weight > 0)
        return rewards


registry.locomotion.register_environment("G1Sit", G1Sit, sit_config)
