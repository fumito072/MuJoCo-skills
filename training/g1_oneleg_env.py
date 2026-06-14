"""Gymnasium env: G1 learns to STAND ON ONE LEG — the user's curriculum step
toward the 0.22 m climb (2026-06-14). Flat ground, no chair, Mac CPU + PyTorch.

WHY THIS EXISTS: 16 hand-tuned probes proved single-support balance is an
underactuated feedback problem (CoM over the stance foot + attitude can't both
be fixed by scripted PD — capture-point territory). That is exactly what RL
discovers. This env trains that reactive balance in isolation; the policy is
the missing core skill behind the climb (lift one foot toward the platform
without the lateral avalanche).

Task: from a two-foot stand, shift weight onto the (randomized) stance leg and
LIFT the swing foot to ~0.18 m, staying upright and calm. Success = hold the
one-leg pose for >= 1.5 s. Reward is dense (upright + swing-foot height + CoM
over the stance foot + a per-step one-leg bonus), so partial progress has a
gradient — important on the Mac's ~32-env scale.

Control: 50 Hz, action in [-1,1]^29 (FULL body — arms genuinely help balance),
target = default_pose + 0.5*action, model default position-actuator gains.
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
ACTION_SCALE = 0.5
EP_SECONDS = 6.0
LIFT_TARGET = 0.18          # swing-foot world-z that counts as a clean lift
HOLD_FOR = 1.5             # seconds of continuous one-leg stand = success
# STATIC stand: the v1 policy balanced as a never-settling limit cycle (CoM on
# the heel, perpetual yaw rotation). v3 keeps v1's strong one-leg incentive but
# (a) scales the one-leg bonus by stillness (3.0*exp(-0.8*spin)*exp(-1.5*yaw))
# so the held pose settles, (b) adds gentle spin/drift penalties, and (c) rewards
# a FORWARD knee raise so the CoM sits over the foot center, not the heel.


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
SIT_KP, SIT_KD = 300.0, 8.0  # SIT-mode stiff LEG gains (mode-switch like the real
#                              controller): makes the two-foot stand STATICALLY
#                              stable so RL spends its budget on the one-leg
#                              transfer, not on rediscovering two-foot balance
#                              (default ankle kp20/roll kp2 = ~2 s topple bomb).
#                              Matches g1_climb_env so the policy transfers.


def build_plain_model(sim_dt=0.002):
    """Playground G1 feetonly on flat ground (stock foot-floor pairs), no chair."""
    assets = g1_base.get_assets()
    spec = mujoco.MjSpec.from_string(consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets)
    spec.assets = assets
    m = spec.compile()
    m.opt.timestep = sim_dt
    return m


class G1OneLegEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=None):
        self.m = build_plain_model()
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
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), np.float32)
        obs_dim = 3 + 3 + 3 + 1 + self.nu + self.nu + 2 + 1 + self.nu
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self._last_a = np.zeros(self.nu, np.float32)
        self._steps = 0
        self._held = 0.0
        self._stance_left = True

    # --- helpers -------------------------------------------------------------
    def _foot_on_floor(self):
        on = {self.lf: False, self.rf: False}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in on and o == self.floor:
                    on[g] = True
        return on

    def _obs(self):
        d = self.d
        gravity = d.site_xmat[self.imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        gyro = d.sensor("gyro_pelvis").data
        linvel = d.sensor("local_linvel_pelvis").data
        on = self._foot_on_floor()
        stance = 1.0 if self._stance_left else -1.0           # which leg to balance on
        return np.hstack([
            gravity, gyro, linvel, [d.qpos[2]],
            d.qpos[7:] - self.default_pose, d.qvel[6:],
            [float(on[self.lf]), float(on[self.rf])], [stance],
            self._last_a,
        ]).astype(np.float32)

    # --- gym API -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
        self._stance_left = bool(self.rng.integers(2))        # randomize stance side
        self.d.qpos[7:] += self.rng.uniform(-0.05, 0.05, self.nu)
        self.d.qvel[6:] += self.rng.uniform(-0.10, 0.10, self.nu)
        self.d.ctrl[:] = self.default_pose
        mujoco.mj_forward(self.m, self.d)
        self._last_a[:] = 0
        self._steps = 0
        self._held = 0.0
        self._ever_success = False
        self._start_xy = self.d.qpos[:2].copy()               # anchor: don't drift
        w, x_, y_, z_ = self.d.qpos[3:7]
        self._yaw0 = np.arctan2(2 * (w * z_ + x_ * y_), 1 - 2 * (y_**2 + z_**2))
        return self._obs(), {}

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
        yaw = np.arctan2(2 * (w * z_ + x_ * y_), 1 - 2 * (y_**2 + z_**2))
        gravity = d.site_xmat[self.imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        upright = float(np.exp(-5.0 * np.sum((gravity - np.array([0, 0, -1.0])) ** 2)))

        stance_g = self.lf if self._stance_left else self.rf
        swing_g = self.rf if self._stance_left else self.lf
        swing_z = float(d.geom_xpos[swing_g][2])
        over = float(np.linalg.norm(d.qpos[:2] - d.geom_xpos[stance_g][:2]))
        on = self._foot_on_floor()

        # stillness signals (drive a STATIC stand, not a limit cycle)
        ang = float(np.linalg.norm(d.sensor("gyro_pelvis").data))          # spin/wobble
        lin = float(np.linalg.norm(d.sensor("local_linvel_pelvis").data[:2]))
        yaw_drift = abs(wrap(yaw - self._yaw0))                            # net rotation
        xy_drift = float(np.linalg.norm(d.qpos[:2] - self._start_xy))      # wandering
        swing_hip_pitch = float(d.qpos[7 + (6 if self._stance_left else 0)])
        swing_flex = max(0.0, -swing_hip_pitch)               # FORWARD knee raise (90deg look)

        # v1 incentives (these drove one-leg standing to ~87%) — KEPT INTACT so
        # the one-leg pose stays far more rewarding than a safe two-foot stand
        # (over-penalizing motion in v2 made two-foot the optimum -> 0% one-leg)
        r = 1.0 * upright
        r += 1.5 * min(swing_z, LIFT_TARGET) / LIFT_TARGET    # lift the swing foot
        r += 0.4 * min(swing_flex, 1.2) / 1.2                 # raise it FORWARD (CoM over foot, not heel)
        r += 0.5 if on[stance_g] else 0.0                     # keep the stance foot down
        r -= 1.5 * over                                       # CoM over the foot CENTER (heel fix)
        r -= 0.5 * max(0.0, 0.62 - d.qpos[2])                 # don't squat/collapse
        r += 0.1                                              # alive
        r -= 0.3 if on[swing_g] else 0.0                      # swing foot should be UP
        # GENTLE static shaping (small vs the one-leg bonus below)
        r -= 0.15 * ang                                       # discourage spin/wobble
        r -= 0.30 * yaw_drift                                 # hold the original heading
        r -= 0.20 * xy_drift                                  # stay on the spot
        r -= 0.15 * lin                                       # no horizontal drift
        r -= 0.01 * float(np.sum((a - self._last_a) ** 2))
        r -= 0.0003 * float(np.sum(d.qvel[6:] ** 2))

        # one-leg hold (achievable v1 gate) -> bonus MAXIMIZED when still & not
        # rotating, so the held pose settles to a static stand instead of spinning
        clean = (swing_z > LIFT_TARGET * 0.9 and not on[swing_g]
                 and abs(roll) < 18 and abs(pitch) < 18)
        r += (3.0 * np.exp(-0.8 * ang) * np.exp(-1.5 * yaw_drift)) if clean else 0.0
        self._held = self._held + CTRL_DT if clean else 0.0
        if self._held >= HOLD_FOR:
            self._ever_success = True
        self._last_a = a

        terminated, truncated = False, False
        # "success" = this episode achieved at least one >=1.5 s one-leg hold
        info = {"success": self._ever_success, "held": self._held}
        if d.qpos[2] < 0.45 or abs(roll) > 50 or abs(pitch) > 50:
            r -= 20.0                                          # fell
            terminated = True
        elif self._steps >= int(EP_SECONDS / CTRL_DT):
            truncated = True
        return self._obs(), float(r), terminated, truncated, info
