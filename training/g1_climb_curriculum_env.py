"""Gymnasium env: GROW the v4 one-leg stand into a STEP-UP onto a 0.22 m platform
(2026-06-14). This is curriculum step #1 the user chose: wire the learned one-leg
balance into the climb.

KEY DESIGN — RESUME COMPATIBILITY: the obs (100-D) and action (29-DOF full body)
are BYTE-IDENTICAL to g1_oneleg_env, so SB3 can `--resume` straight from the v4
checkpoint (ppo_oneleg_latest). The v4 policy already lifts the swing foot to
~0.27 m with a steady torso — exactly the FIRST HALF of a step-up. Here we add a
parametric box step in front and reward planting that lifted foot ON the step and
transferring weight up. Full body (arms help balance) is why one-leg worked; the
12-DOF legs-only climb env stalled at 0/20, so we do NOT revive it.

HEIGHT CURRICULUM: step_h is a constructor arg. Train 0.08 -> 0.14 -> 0.22 m,
each stage resuming the previous. base_z senses the climb; the step pose is fixed
so no target coords are needed in obs (the price of resume; fine for one chair).

Task: spawn on the floor facing +x toward the step, lead (swing) foot randomized.
Lift the lead foot to clear step_h, plant it on the step, transfer weight, bring
the trailing foot up. Success = BOTH feet on the step, base risen ~step_h,
upright, calm, near the step-center target. Then the mission FSM hands off to the
verified 30/30 stiff-descent sit.

Control: 50 Hz, action in [-1,1]^29, target = default + 0.5*action, SIT-mode
stiff LEG gains kp300/kd8 (mode-switch like the real controller).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import mujoco
from gymnasium import spaces

from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.g1 import base as g1_base

CTRL_DT, SUBSTEPS = 0.02, 10
ACTION_SCALE = 0.5            # match v4 so the resumed policy behaves consistently
EP_SECONDS = 6.0
HOLD_FOR = 0.1               # brief calm arrival = success (matches the mission FSM,
#                              which takes over and holds the stand itself; the real
#                              g1_climb_env success is instantaneous, not a long hold)

# parametric box step (top at z = step_h). Placed just ahead of the toes so the
# lead foot steps FORWARD onto it (toe-first loads naturally — user's insight).
STEP_FRONT_X = 0.10
STEP_DEPTH = 0.45
STEP_WIDTH = 0.70
STEP_CENTER_X = STEP_FRONT_X + STEP_DEPTH / 2   # target x = middle of the step top
DEFAULT_BASE_Z = 0.755       # knees_bent pelvis height on flat ground

SIT_KP, SIT_KD = 300.0, 8.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def build_step_model(step_h, sim_dt=0.002):
    """Playground G1 feetonly on flat ground + a box step of height step_h."""
    assets = g1_base.get_assets()
    spec = mujoco.MjSpec.from_string(consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets)
    spec.assets = assets
    h = max(float(step_h), 1e-3)
    step = spec.worldbody.add_geom()
    step.name = "step"
    step.type = mujoco.mjtGeom.mjGEOM_BOX
    step.size = [STEP_DEPTH / 2, STEP_WIDTH / 2, h / 2]
    step.pos = [STEP_CENTER_X, 0.0, h / 2]
    step.condim = 3
    step.friction = [0.9, 0.02, 0.01]
    step.rgba = [0.55, 0.38, 0.22, 1.0]
    # the G1 feet are contype/conaff 0 and collide with the floor ONLY via explicit
    # pairs (floor<->left_foot, floor<->right_foot). Without the same pairs the feet
    # pass THROUGH this box — so add them, else the step is a non-colliding phantom.
    for foot in ("left_foot", "right_foot"):
        pair = spec.add_pair(geomname1=foot, geomname2="step")
        pair.condim = 3
        pair.friction = [0.9, 0.9, 0.02, 0.01, 0.01]
    m = spec.compile()
    m.opt.timestep = sim_dt
    return m


class G1ClimbCurriculumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=None, step_h=0.08):
        self.step_h = float(step_h)
        self.m = build_step_model(self.step_h)
        for a in range(12):                                   # legs: SIT-mode stiff
            self.m.actuator_gainprm[a, 0] = SIT_KP
            self.m.actuator_biasprm[a, 1] = -SIT_KP
            self.m.actuator_biasprm[a, 2] = -SIT_KD
        self.d = mujoco.MjData(self.m)
        self.key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
        self.default_pose = np.array(self.m.key_qpos[self.key][7:])
        self.lo = self.m.actuator_ctrlrange[:, 0].copy()
        self.hi = self.m.actuator_ctrlrange[:, 1].copy()
        self.nu = self.m.nu                                   # 29
        self.imu_site = self.m.site("imu_in_pelvis").id
        self.lf = self.m.geom("left_foot").id
        self.rf = self.m.geom("right_foot").id
        self.floor = self.m.geom("floor").id
        self.step_g = self.m.geom("step").id
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), np.float32)
        obs_dim = 3 + 3 + 3 + 1 + self.nu + self.nu + 2 + 1 + self.nu  # == 100 (v4)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self._last_a = np.zeros(self.nu, np.float32)
        self._steps = 0
        self._held = 0.0
        self._stance_left = True
        self.target_z = DEFAULT_BASE_Z + self.step_h
        # RSI (reverse curriculum): the v4 one-leg balance basin is so stable that
        # the policy plants both feet but stays CROUCHED and never stands up on the
        # step. Seeding a fraction of episodes from already-on-the-step states lets
        # that high-value "climbed" state propagate back (the project's proven
        # escape hatch). Built once here by drop-and-settle at several base_x.
        self.rsi_p = float(os.environ.get("G1CLIMB_RSI_P", 0.4))
        self._rsi_q, self._rsi_v = self._build_rsi_bank()
        self._from_rsi = False

    def _build_rsi_bank(self):
        qs, vs = [], []
        for bx in (0.16, 0.20, 0.24, 0.28, 0.32):
            mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
            self.d.qpos[0] = bx
            self.d.qpos[2] = DEFAULT_BASE_Z + self.step_h + 0.03
            self.d.ctrl[:] = self.default_pose
            for _ in range(200):
                mujoco.mj_step(self.m, self.d)
            _, on_step = self._foot_contacts()
            if (int(on_step[self.lf]) + int(on_step[self.rf]) >= 1
                    and self.d.qpos[2] > 0.70 and abs(self.d.qpos[1]) < 0.2):
                qs.append(self.d.qpos.copy())
                vs.append(self.d.qvel.copy())
        # HARVESTED mid-transfer states (g1_climb_harvest_rsi.py): the constructed
        # states above are only the END (standing on the step); these are real
        # one-foot-on-step / weight-transfer frames from a climbing policy, so the
        # missing TRANSITION phase gets dense practice too (the whole point).
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hp = os.path.join(repo, "runs_climb", f"climb_rsi_harvest_h{self.step_h:.2f}.npz")
        if os.path.exists(hp) and not os.environ.get("G1CLIMB_NO_HARVEST"):
            z = np.load(hp)
            qs.extend(z["qpos"])
            vs.extend(z["qvel"])
        return qs, vs

    # --- helpers -------------------------------------------------------------
    def _foot_contacts(self):
        """on_support = foot touching floor OR step; on_step = foot on the step TOP
        only. A foot standing on the floor in front of the step grazes the step's
        vertical FRONT FACE — that must NOT count as a plant, so gate on the contact
        point height being near the step top (else the success metric is faked)."""
        on_support = {self.lf: False, self.rf: False}
        on_step = {self.lf: False, self.rf: False}
        top_z = self.step_h - 0.02
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in on_support and o in (self.floor, self.step_g):
                    on_support[g] = True
                    if o == self.step_g and c.pos[2] > top_z:
                        on_step[g] = True
        return on_support, on_step

    def _obs(self):
        d = self.d
        gravity = d.site_xmat[self.imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        gyro = d.sensor("gyro_pelvis").data
        linvel = d.sensor("local_linvel_pelvis").data
        on_support, _ = self._foot_contacts()
        stance = 1.0 if self._stance_left else -1.0           # which leg leads (stays down)
        return np.hstack([
            gravity, gyro, linvel, [d.qpos[2]],
            d.qpos[7:] - self.default_pose, d.qvel[6:],
            [float(on_support[self.lf]), float(on_support[self.rf])], [stance],
            self._last_a,
        ]).astype(np.float32)

    def _potential(self):
        d = self.d
        dist = float(np.hypot(STEP_CENTER_X - d.qpos[0], 0.0 - d.qpos[1]))
        dz = abs(d.qpos[2] - self.target_z)
        return -(1.0 * dist + 3.0 * dz)

    # --- gym API -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
        self._stance_left = bool(self.rng.integers(2))
        if self._rsi_q and self.rng.random() < self.rsi_p:
            # reverse-curriculum spawn: already on (or stepping onto) the platform
            i = self.rng.integers(len(self._rsi_q))
            self.d.qpos[:] = self._rsi_q[i]
            self.d.qvel[:] = self._rsi_v[i]
            self.d.qpos[7:] += self.rng.uniform(-0.04, 0.04, self.nu)
            self.d.qvel[6:] += self.rng.uniform(-0.10, 0.10, self.nu)
            self._from_rsi = True
        else:
            # spawn on the floor, facing +x toward the step, lead foot randomized
            self.d.qpos[0] += self.rng.uniform(-0.03, 0.03)   # small x jitter
            self.d.qpos[1] += self.rng.uniform(-0.03, 0.03)
            self.d.qpos[7:] += self.rng.uniform(-0.05, 0.05, self.nu)
            # handoff-from-a-march transient (the project lesson): initial motion
            self.d.qvel[0:2] = self.rng.uniform(-0.10, 0.10, 2)
            self.d.qvel[6:] += self.rng.uniform(-0.10, 0.10, self.nu)
            self._from_rsi = False
        self.d.ctrl[:] = self.default_pose
        mujoco.mj_forward(self.m, self.d)
        self._last_a[:] = 0
        self._steps = 0
        self._held = 0.0
        self._ever_success = False
        self._yaw0 = self._yaw()
        self._prev_pot = self._potential()
        return self._obs(), {}

    def _yaw(self):
        w, x_, y_, z_ = self.d.qpos[3:7]
        return np.arctan2(2 * (w * z_ + x_ * y_), 1 - 2 * (y_**2 + z_**2))

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        tgt = self.default_pose + ACTION_SCALE * a
        self.d.ctrl[:] = np.clip(tgt, self.lo, self.hi)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
        self._steps += 1
        d = self.d
        w, x_, y_, z_ = d.qpos[3:7]
        roll = np.degrees(np.arctan2(2 * (w * x_ + y_ * z_), 1 - 2 * (x_**2 + y_**2)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (w * y_ - z_ * x_), -1, 1)))
        yaw_drift = abs(wrap(self._yaw() - self._yaw0))
        gravity = d.site_xmat[self.imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        upright = float(np.exp(-5.0 * np.sum((gravity - np.array([0, 0, -1.0])) ** 2)))

        stance_g = self.lf if self._stance_left else self.rf  # trailing foot (stays down)
        swing_g = self.rf if self._stance_left else self.lf   # lead foot (steps up first)
        swing_z = float(d.geom_xpos[swing_g][2])
        swing_x = float(d.geom_xpos[swing_g][0])              # lead-foot x (toward step)
        on_support, on_step = self._foot_contacts()
        feet_on_step = int(on_step[self.lf]) + int(on_step[self.rf])

        ang = float(np.linalg.norm(d.sensor("gyro_pelvis").data))
        lin = float(np.linalg.norm(d.sensor("local_linvel_pelvis").data[:2]))
        waist_dev = d.qpos[7 + 12:7 + 15] - self.default_pose[12:15]
        arm_dev = d.qpos[7 + 15:7 + 29] - self.default_pose[15:29]

        r = 1.5 * upright                                    # keep pelvis upright
        # LEAD FOOT MAGNET: pull the swing foot toward a landing spot ON the step
        # top (forward AND down). This is the key fix vs the flat "lift high" basin
        # that trapped the policy in v4's in-place high hold — exp() gives a smooth
        # gradient out of (x~0, z~0.34) toward (over-step, z=step_h).
        tx = STEP_FRONT_X + 0.13
        dxz = float(np.hypot(swing_x - tx, swing_z - self.step_h))
        r += 2.5 * np.exp(-4.0 * dxz)
        if on_step[swing_g]:
            r += 2.0                                          # solid plant bonus (dominates)
            # WEIGHT TRANSFER once the lead foot is down: bring the trailing foot
            # up and pull the base forward onto the step (harder, vs leaning back).
            r += 1.5 if on_step[stance_g] else 0.3
            r += 1.2 * np.exp(-3.0 * abs(d.qpos[0] - (STEP_FRONT_X + 0.18)))
        else:
            r -= 0.2 if on_support[swing_g] else 0.0          # mild: don't park on the floor
        if feet_on_step == 2:
            r += 1.0                                          # BOTH feet on the platform
            # ONE BIG on-step reward that REQUIRES calm * tall * default-pose, so a
            # clean STILL stand strictly beats "both feet but fidgeting/crouched"
            # (the milking attractor that collapsed success as training converged:
            # the old rewards were collected regardless of calm). calm->0 when
            # moving so milking pays little; the convergent optimum IS the success.
            calm = float(np.exp(-1.0 * ang) * np.exp(-2.0 * lin))
            tall = float(np.exp(-8.0 * max(0.0, self.target_z - d.qpos[2])))
            leg_dev = float(np.sum((d.qpos[7:7 + 12] - self.default_pose[:12]) ** 2))
            r += 9.0 * calm * tall * np.exp(-2.0 * leg_dev)

        # keep the v4 steady-torso shaping (no forward bend, no arm flail)
        r -= 0.6 * float(np.sum(waist_dev ** 2))
        r -= 0.15 * float(np.mean(arm_dev ** 2))
        r += 0.05                                             # alive (small)
        r -= 0.15 * ang - 0.0                                 # discourage wobble
        r -= 0.20 * yaw_drift                                 # hold heading
        r -= 0.10 * lin
        r -= 0.01 * float(np.sum((a - self._last_a) ** 2))
        r -= 0.0003 * float(np.sum(d.qvel[6:] ** 2))
        # leaning a non-foot body part on the step is not climbing
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if o == self.step_g and g not in (self.lf, self.rf):
                    r -= 0.15
                    break
        self._last_a = a

        # CLEAN platform stand: both feet up, risen ~step_h, upright, CALM, and the
        # base genuinely ON the step (past the front edge, not hanging off the side)
        climbed = (feet_on_step == 2 and d.qpos[2] > self.target_z - 0.06
                   and abs(roll) < 20 and abs(pitch) < 20
                   and d.qpos[0] > STEP_FRONT_X + 0.05 and abs(d.qpos[1]) < 0.25
                   and ang < 3.0 and lin < 0.6)
        # small extra reinforcement in the success window (the calm*tall term above
        # is the main driver; the climbed gate also advances the success flag)
        r += 2.0 if climbed else 0.0
        self._held = self._held + CTRL_DT if climbed else 0.0
        if self._held >= HOLD_FOR:
            self._ever_success = True

        terminated, truncated = False, False
        info = {"success": self._ever_success, "held": self._held,
                "feet_on_step": feet_on_step, "rsi": int(self._from_rsi)}
        # do NOT terminate on success: terminating forfeited ~250 frames of dense
        # reward for a +200 bonus, so the policy learned to AVOID a clean 5-frame
        # hold (hover just off-clean to keep milking). Now success is only a flag;
        # the episode runs full length and the dominant clean-stand reward above is
        # what the policy milks -> it settles into the success state on its own.
        if d.qpos[2] < 0.45 or abs(roll) > 50 or abs(pitch) > 50:
            r -= 20.0
            terminated = True
        elif self._steps >= int(EP_SECONDS / CTRL_DT):
            truncated = True
        return self._obs(), float(r), terminated, truncated, info
