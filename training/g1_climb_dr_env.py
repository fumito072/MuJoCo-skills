"""Domain-randomized step-up env (2026-06-14) — the user's two structural fixes:

(B) LADDER COLLAPSE was catastrophic forgetting / per-height over-specialization.
    Fix: DON'T make height a curriculum stage; randomize step height per episode
    over [0.02, 0.22] AND put it in the observation. The policy learns ALL heights
    at once, so it never forgets 0.02 while learning 0.22.

(A) MILKING (both feet up but fidgeting/crouched to farm per-step reward) was
    reward hacking, not a compute problem. Fix: POTENTIAL-BASED shaping. The step
    reward is r = Phi(s') - Phi(s) (telescoping), so staying in any state earns ~0
    -> nothing to milk. Phi rises as the robot climbs and stands cleanly; a small
    time penalty + a sparse success bonus with EARLY TERMINATION make the policy
    rush to a clean stand and stop (no future reward to forfeit, since staying = 0).

Same body/control as the curriculum env (29-DOF, SIT-mode stiff legs kp300). Obs
adds [height_norm, dx-to-step, dy] so the policy knows the step it faces.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import mujoco
from gymnasium import spaces

from g1_climb_curriculum_env import (
    build_step_model, SIT_KP, SIT_KD, wrap, CTRL_DT, SUBSTEPS, ACTION_SCALE,
    STEP_FRONT_X, STEP_CENTER_X, STEP_DEPTH, STEP_WIDTH, DEFAULT_BASE_Z)

H_MIN, H_MAX = 0.02, 0.22
EP_SECONDS = 6.0
HOLD_FOR = 0.1               # brief calm arrival = success (FSM then holds the stand)


class G1ClimbDREnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=None, fixed_h=None):
        self.fixed_h = fixed_h                                 # set for eval at one height
        self.m = build_step_model(H_MAX)                       # build once; mutate per reset
        for a in range(12):
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
        obs_dim = 3 + 3 + 3 + 1 + self.nu + self.nu + 2 + 1 + self.nu + 3  # 103
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.rsi_p = float(os.environ.get("G1CLIMB_RSI_P", 0.4))
        # standalone UPRIGHT penalty with a NON-vanishing gradient (linear in the
        # squared gravity error), active when standing on the step. The exp() upright
        # term in the stand-quality goes flat (~0) at a big lean -> no gradient to
        # escape; this keeps pushing pitch/roll down from any lean. Configurable for
        # the reward sweep (G1CLIMB_UPRIGHT_PEN).
        self.upright_pen = float(os.environ.get("G1CLIMB_UPRIGHT_PEN", 2.0))
        # AUTOMATIC CURRICULUM over the randomization RANGE (user's idea): sample
        # height from [H_MIN, h_cap]; h_cap starts low and the trainer grows it as
        # the top of the range is mastered. Always randomizing the WHOLE [H_MIN,
        # h_cap] keeps every lower height practiced -> no forgetting (the ladder's
        # flaw); growing the cap only when the top works gives difficulty ordering
        # (full-range DR lacked it, so tall steps were never discovered).
        self.h_cap = float(os.environ.get("G1CLIMB_HCAP_START", H_MAX))
        self._last_a = np.zeros(self.nu, np.float32)
        self._steps = 0
        self._held = 0.0
        self._stance_left = True
        self._from_rsi = False
        self.step_h = H_MAX
        self._set_height(H_MAX)

    def set_h_cap(self, cap):
        self.h_cap = float(np.clip(cap, H_MIN, H_MAX))
        return self.h_cap

    def get_h_cap(self):
        return self.h_cap

    def _set_height(self, h):
        self.step_h = float(h)
        self.m.geom_size[self.step_g] = [STEP_DEPTH / 2, STEP_WIDTH / 2, h / 2]
        self.m.geom_pos[self.step_g] = [STEP_CENTER_X, 0.0, h / 2]
        self.target_z = DEFAULT_BASE_Z + h

    # --- helpers -------------------------------------------------------------
    def _foot_contacts(self):
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

    def _state(self):
        """Common state quantities used by both the obs and the potential."""
        d = self.d
        grav = d.site_xmat[self.imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        upright = float(np.exp(-5.0 * np.sum((grav - np.array([0, 0, -1.0])) ** 2)))
        ang = float(np.linalg.norm(d.sensor("gyro_pelvis").data))
        lin = float(np.linalg.norm(d.sensor("local_linvel_pelvis").data[:2]))
        on_support, on_step = self._foot_contacts()
        feet = int(on_step[self.lf]) + int(on_step[self.rf])
        return grav, upright, ang, lin, on_support, on_step, feet

    def _phi_climb(self):
        """Climb-only potential (feet up / base height / forward onto step). Telescoping
        r = Phi(s')-Phi(s) drives getting UP without being milkable. The STAND quality
        (upright/calm/pose) is a separate dense term in step()."""
        d = self.d
        _, _, _, _, _, _, feet = self._state()
        fs = feet / 2.0
        hz = float(np.clip((d.qpos[2] - 0.6) / (self.target_z - 0.6), 0.0, 1.0))
        ox = float(np.clip(d.qpos[0] / STEP_CENTER_X, 0.0, 1.0))
        return 3.0 * fs + 2.0 * hz + 1.0 * ox

    def _obs(self):
        d = self.d
        grav, _, _, _, on_support, _, _ = self._state()
        gyro = d.sensor("gyro_pelvis").data
        linvel = d.sensor("local_linvel_pelvis").data
        stance = 1.0 if self._stance_left else -1.0
        return np.hstack([
            grav, gyro, linvel, [d.qpos[2]],
            d.qpos[7:] - self.default_pose, d.qvel[6:],
            [float(on_support[self.lf]), float(on_support[self.rf])], [stance],
            self._last_a,
            [self.step_h / H_MAX, STEP_CENTER_X - d.qpos[0], -d.qpos[1]],  # task obs
        ]).astype(np.float32)

    # --- gym API -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        h = self.fixed_h if self.fixed_h is not None else float(self.rng.uniform(H_MIN, self.h_cap))
        self._set_height(h)
        self._stance_left = bool(self.rng.integers(2))
        if self.rng.random() < self.rsi_p:
            # RSI: construct a standing-on-the-step state at THIS height, then settle
            mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
            self.d.qpos[0] = STEP_FRONT_X + 0.20 + self.rng.uniform(-0.03, 0.03)
            self.d.qpos[1] += self.rng.uniform(-0.03, 0.03)
            self.d.qpos[2] = self.target_z + 0.01
            self.d.qpos[7:] += self.rng.uniform(-0.04, 0.04, self.nu)
            self.d.ctrl[:] = self.default_pose
            for _ in range(60):
                mujoco.mj_step(self.m, self.d)
            self._from_rsi = True
        else:
            mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
            self.d.qpos[0] += self.rng.uniform(-0.03, 0.03)
            self.d.qpos[1] += self.rng.uniform(-0.03, 0.03)
            self.d.qpos[7:] += self.rng.uniform(-0.05, 0.05, self.nu)
            self.d.qvel[0:2] = self.rng.uniform(-0.10, 0.10, 2)
            self.d.qvel[6:] += self.rng.uniform(-0.10, 0.10, self.nu)
            self.d.ctrl[:] = self.default_pose
            mujoco.mj_forward(self.m, self.d)
            self._from_rsi = False
        self._last_a[:] = 0
        self._steps = 0
        self._held = 0.0
        self._ever_success = False
        self._prev_phi = self._phi_climb()
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
        grav, upright, ang, lin, _, on_step, feet = self._state()

        # CLIMB drive: potential-based shaping (telescoping) gets the feet UP without
        # being milkable. Phi here is climb-only (feet/height/forward); the STAND
        # quality is a separate dense term below.
        phi = self._phi_climb()
        r = (phi - self._prev_phi)
        self._prev_phi = phi

        # STAND QUALITY (the fix): a single dense reward whose UNIQUE maximum is the
        # clean stand — upright AND tall AND calm AND default-pose AND both feet.
        # The policy was leaning forward (pitch>15) because uprightness was never in
        # the reward; folding it into the product makes leaning strictly worse, so
        # the milkable optimum the policy converges to IS the success pose.
        if feet == 2:
            grav_err = float(np.sum((grav - np.array([0, 0, -1.0])) ** 2))
            up = float(np.exp(-12.0 * grav_err))
            tall = float(np.exp(-8.0 * max(0.0, self.target_z - d.qpos[2])))
            calm = float(np.exp(-1.0 * ang) * np.exp(-2.0 * lin))
            leg_dev = float(np.sum((d.qpos[7:7 + 12] - self.default_pose[:12]) ** 2))
            pose = float(np.exp(-2.0 * leg_dev))
            r += 8.0 * up * tall * calm * pose
            r -= self.upright_pen * grav_err                  # non-vanishing upright gradient
        r -= 0.02                                             # small time cost (finish/settle)
        r -= 0.01 * float(np.sum((a - self._last_a) ** 2))
        self._last_a = a

        # clean platform stand: both feet up, tall, UPRIGHT, on-step, calm
        climbed = (feet == 2 and d.qpos[2] > self.target_z - 0.06
                   and abs(roll) < 15 and abs(pitch) < 15
                   and d.qpos[0] > STEP_FRONT_X + 0.05 and abs(d.qpos[1]) < 0.25
                   and ang < 2.5 and lin < 0.5)
        self._held = self._held + CTRL_DT if climbed else 0.0
        if self._held >= HOLD_FOR:
            self._ever_success = True

        # NO success-terminate: the dense stand-quality max IS the clean stand, so
        # the policy holds it (milking the max = success). Terminating would make
        # the policy avoid a clean hold to keep collecting; instead let it stand.
        terminated, truncated = False, False
        info = {"success": self._ever_success, "feet_on_step": feet,
                "rsi": int(self._from_rsi), "step_h": self.step_h}
        if d.qpos[2] < 0.45 or abs(roll) > 50 or abs(pitch) > 50:
            r -= 10.0                                          # fell
            terminated = True
        elif self._steps >= int(EP_SECONDS / CTRL_DT):
            truncated = True
        return self._obs(), float(r), terminated, truncated, info
