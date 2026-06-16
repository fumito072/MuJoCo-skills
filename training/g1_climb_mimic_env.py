"""DeepMimic-style climb env: track the kinematic reference (g1_climb_reference.npz)
instead of discovering the climb. Reward = pose/base tracking + the original task
terms; reset = Reference State Initialization (spawn at a random reference frame,
joints set to the reference, base placed on its reference path) — the curriculum
is the reference itself.

Obs (57-D) = the 52-D task obs + [phase, ref-legs-error preview (next frame minus
current joints) summarized as 4 PCA-free scalars]:  kept simple — phase (1) +
mean/max leg ref error (2) + base y/z ref error (2).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import mujoco
from gymnasium import spaces

import g1_sit_env

CTRL_DT, SUBSTEPS = 0.02, 10
ACTION_SCALE = 0.45
SIT_KP, SIT_KD = 300.0, 8.0
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = np.load(os.path.join(REPO, "training", "g1_climb_reference.npz"))
REF_LEGS, REF_BASE = REF["legs"], REF["base"]
REF_N = len(REF_LEGS)
YAW_REF = float(REF["yaw"])           # facing -y during the climb
EXTRA_S = 1.0                          # grace after the reference ends to settle


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class G1ClimbMimicEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=None):
        self.m = g1_sit_env.build_fbx_chair_model(0.002)
        for a in range(12):
            self.m.actuator_gainprm[a, 0] = SIT_KP
            self.m.actuator_biasprm[a, 1] = -SIT_KP
            self.m.actuator_biasprm[a, 2] = -SIT_KD
        self.d = mujoco.MjData(self.m)
        self.key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
        self.default_pose = np.array(self.m.key_qpos[self.key][7:])
        self.lo = self.m.actuator_ctrlrange[:, 0].copy()
        self.hi = self.m.actuator_ctrlrange[:, 1].copy()
        self.imu_site = self.m.site("imu_in_pelvis").id
        self.lf = self.m.geom("left_foot").id
        self.rf = self.m.geom("right_foot").id
        self.rc = {i for i in range(self.m.ngeom)
                   if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                   .startswith("rc_part")}
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (57,), np.float32)
        self._last_a = np.zeros(12, np.float32)
        self._k = 0
        # reverse curriculum (EvoCoT) + Goldilocks adaptive difficulty: episodes
        # start in [rsi_min_phase, 1.0] of the reference. Start near the lunge so the
        # policy masters the LAST push (single-leg rise + trail-foot transfer) where
        # it was stuck; the trainer anneals rsi_min_phase toward 0 (full floor climb)
        # to keep success ~50% (the "edge of ability"). lunge is at phase ~0.5.
        self.rsi_min_phase = float(os.environ.get("RSI_MIN_PHASE", 0.45))

    def set_rsi_min_phase(self, p):
        self.rsi_min_phase = float(np.clip(p, 0.0, 0.95))
        return self.rsi_min_phase

    def _foot_contacts(self):
        on = {self.lf: 0.0, self.rf: 0.0}
        plat = {self.lf: False, self.rf: False}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in on:
                    on[g] = 1.0
                    if o in self.rc:
                        plat[g] = True
        return on, plat

    def _ref(self, k):
        kk = min(k, REF_N - 1)
        return REF_LEGS[kk], REF_BASE[kk]

    def _obs(self):
        d = self.d
        gravity = d.site_xmat[self.imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        gyro = d.sensor("gyro_pelvis").data
        linvel = d.sensor("local_linvel_pelvis").data
        yaw = np.arctan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]),
                         1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
        ye = wrap(yaw - YAW_REF)
        rl, rb = self._ref(self._k + 1)
        leg_err = rl - d.qpos[7:19]
        on, _ = self._foot_contacts()
        phase = min(self._k / REF_N, 1.5)
        return np.hstack([
            gravity, gyro, linvel, [d.qpos[2]], [np.sin(ye), np.cos(ye)],
            [0.0 - d.qpos[0], rb[0] - d.qpos[1]],
            d.qpos[7:19] - self.default_pose[:12], d.qvel[6:18],
            [on[self.lf], on[self.rf]], self._last_a,
            [phase, float(np.mean(np.abs(leg_err))), float(np.max(np.abs(leg_err))),
             rb[0] - d.qpos[1], rb[1] - d.qpos[2]],
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
        # reverse-curriculum RSI: spawn uniformly in [rsi_min_phase, 1.0] of the
        # reference. Early training rsi_min_phase ~0.45 (start at the lunge -> learn
        # the last push); the Goldilocks trainer anneals it toward 0 as success rises,
        # so the policy masters the hard ending first then extends back to the floor.
        lo = self.rsi_min_phase + (1.0 - self.rsi_min_phase) * self.rng.random()
        self._k = int(lo * (REF_N - 1))
        rl, rb = self._ref(self._k)
        yaw0 = YAW_REF + self.rng.uniform(-0.08, 0.08)
        self.d.qpos[0:3] = (self.rng.uniform(-0.04, 0.04),
                            rb[0] + self.rng.uniform(-0.02, 0.02),
                            rb[1])
        self.d.qpos[3:7] = (np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2))
        self.d.qpos[7:19] = rl + self.rng.uniform(-0.05, 0.05, 12)
        self.d.qvel[:] = 0
        self.d.qvel[0:2] = self.rng.uniform(-0.10, 0.10, 2)
        self.d.qvel[6:18] = self.rng.uniform(-0.5, 0.5, 12)
        self.d.ctrl[:] = self.default_pose
        self.d.ctrl[:12] = rl
        mujoco.mj_forward(self.m, self.d)
        self._last_a[:] = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        rl, rb = self._ref(self._k + 1)
        # action = correction AROUND the reference (residual control)
        tgt = self.default_pose.copy()
        tgt[:12] = rl + ACTION_SCALE * a
        self.d.ctrl[:] = np.clip(tgt, self.lo, self.hi)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
        self._k += 1
        d = self.d
        w, x_, y_, z_ = d.qpos[3:7]
        roll = np.degrees(np.arctan2(2 * (w * x_ + y_ * z_), 1 - 2 * (x_**2 + y_**2)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (w * y_ - z_ * x_), -1, 1)))
        yaw = np.arctan2(2 * (w * z_ + x_ * y_), 1 - 2 * (y_**2 + z_**2))
        ye = wrap(yaw - YAW_REF)
        on, plat = self._foot_contacts()
        feet_plat = plat[self.lf] + plat[self.rf]

        # DeepMimic-style tracking reward
        leg_err = float(np.mean((d.qpos[7:19] - rl) ** 2))
        base_err = float(1.0 * d.qpos[0] ** 2 + (d.qpos[1] - rb[0]) ** 2
                 + 1.5 * (d.qpos[2] - rb[1]) ** 2)   # x_ref = 0 (lateral
                 # flee was the first exploit found in this reward)
        # PRODUCT form: leg mimicry pays NOTHING unless the body is on the
        # reference path (additive form bred wall-hugging leg-pose farming)
        r = 2.2 * np.exp(-8.0 * leg_err) * np.exp(-15.0 * base_err)
        r -= 0.15 * abs(ye)
        r -= 0.10 * (abs(np.radians(roll)) + abs(np.radians(pitch)))
        r -= 0.01 * float(np.sum((a - self._last_a) ** 2))
        self._last_a = a

        terminated, truncated = False, False
        info = {"success": False}
        speed = float(np.linalg.norm(d.sensor("local_linvel_pelvis").data[:2]))
        done_window = self._k >= REF_N
        if (done_window and feet_plat == 2 and d.qpos[2] > 0.90 and abs(ye) < 0.35
                and abs(roll) < 25 and abs(pitch) < 25 and speed < 0.6
                and abs(d.qpos[1] - REF_BASE[-1][0]) < 0.15 and abs(d.qpos[0]) < 0.12):
            r += 100.0
            terminated = True
            info["success"] = True
        elif d.qpos[2] < 0.40 or abs(roll) > 55 or abs(pitch) > 65 or abs(d.qpos[0]) > 0.30:
            r -= 20.0
            terminated = True
        elif self._k >= REF_N + int(EXTRA_S / CTRL_DT):
            truncated = True
        return self._obs(), float(r), terminated, truncated, info
