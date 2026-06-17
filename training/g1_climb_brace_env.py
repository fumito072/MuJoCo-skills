"""29-DOF HAND-BRACE DeepMimic climb env (Mac-CPU). Tracks the braced-climb reference
(g1_climb_brace_reference.npz): start in the verified 4-point bridge (hands on the
chair armrests, leaned), step the lead foot onto the 0.22m footrest WHILE the hands
take load, then rise and release. The stiff arm gains + hand contact pairs provide the
support the open-loop bridge lacks; RL learns only the feedback.

Why this over the 12-DOF leg climb: the floor->first-foot-up is single-leg-balance-
hard; the hands turn it into a 4-point problem. The reference carries the base
ORIENTATION (the lean lives there, not just the joints) so RSI into the leaned brace
frames is correct.
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
ACTION_SCALE = 0.40
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = np.load(os.path.join(REPO, "training", "g1_climb_brace_reference.npz"))
REF_J, REF_BASE, REF_QUAT = REF["joints"], REF["base"], REF["quat"]
REF_N = len(REF_J)
YAW_REF = float(REF["yaw"])
EXTRA_S = 1.0
BRACE_PHASE = 0.55                 # hands expected on the armrests below this phase


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def quat_pitch(q):
    w, x, y, z = q
    return np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))


class G1ClimbBraceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=None):
        self.m = g1_sit_env.build_fbx_chair_model(0.002, pair_hands=True)
        # arm-mode gains (verified in the handbrace probe): legs stiff, shoulders/elbows
        # firm, wrists rigid struts (kp80) — floppy wrists roll off the narrow rests.
        for a in range(12):
            self.m.actuator_gainprm[a, 0] = 300.0
            self.m.actuator_biasprm[a, 1] = -300.0
            self.m.actuator_biasprm[a, 2] = -8.0
        for a in (15, 16, 17, 18, 22, 23, 24, 25):
            self.m.actuator_gainprm[a, 0] = 150.0
            self.m.actuator_biasprm[a, 1] = -150.0
            self.m.actuator_biasprm[a, 2] = -4.0
        for a in (19, 20, 21, 26, 27, 28):
            self.m.actuator_gainprm[a, 0] = 80.0
            self.m.actuator_biasprm[a, 1] = -80.0
            self.m.actuator_biasprm[a, 2] = -2.0
        self.d = mujoco.MjData(self.m)
        self.key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
        self.default_pose = np.array(self.m.key_qpos[self.key][7:])
        self.lo = self.m.actuator_ctrlrange[:, 0].copy()
        self.hi = self.m.actuator_ctrlrange[:, 1].copy()
        self.imu_site = self.m.site("imu_in_pelvis").id
        self.lf = self.m.geom("left_foot").id
        self.rf = self.m.geom("right_foot").id
        self.lh = self.m.geom("left_hand_collision").id
        self.rh = self.m.geom("right_hand_collision").id
        self.rc = {i for i in range(self.m.ngeom)
                   if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                   .startswith("rc_part")}
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, (29,), np.float32)
        n_obs = 3 + 3 + 3 + 1 + 2 + 2 + 29 + 29 + 4 + 29 + 4
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self._last_a = np.zeros(29, np.float32)
        self._k = 0
        self.rsi_min_phase = float(os.environ.get("RSI_MIN_PHASE", 0.0))
        self._f6 = np.zeros(6)

    def set_rsi_min_phase(self, p):
        self.rsi_min_phase = float(np.clip(p, 0.0, 0.95))
        return self.rsi_min_phase

    def _contacts(self):
        on = {self.lf: 0.0, self.rf: 0.0}
        plat = {self.lf: False, self.rf: False}
        handF = {self.lh: 0.0, self.rh: 0.0}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in on:
                    on[g] = 1.0
                    if o in self.rc:
                        plat[g] = True
                if g in handF and o in self.rc:
                    mujoco.mj_contactForce(self.m, self.d, i, self._f6)
                    handF[g] += abs(self._f6[0])
        return on, plat, handF

    def _ref(self, k):
        kk = min(k, REF_N - 1)
        return REF_J[kk], REF_BASE[kk], REF_QUAT[kk]

    def _obs(self):
        d = self.d
        gravity = d.site_xmat[self.imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        gyro = d.sensor("gyro_pelvis").data
        linvel = d.sensor("local_linvel_pelvis").data
        yaw = np.arctan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]),
                         1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
        ye = wrap(yaw - YAW_REF)
        rj, rb, rq = self._ref(self._k + 1)
        jerr = rj - d.qpos[7:]
        on, _, handF = self._contacts()
        phase = min(self._k / REF_N, 1.5)
        return np.hstack([
            gravity, gyro, linvel, [d.qpos[2]], [np.sin(ye), np.cos(ye)],
            [0.0 - d.qpos[0], rb[0] - d.qpos[1]],
            d.qpos[7:] - self.default_pose, d.qvel[6:],
            [on[self.lf], on[self.rf],
             1.0 if handF[self.lh] > 1 else 0.0, 1.0 if handF[self.rh] > 1 else 0.0],
            self._last_a,
            [phase, float(np.mean(np.abs(jerr))), rb[0] - d.qpos[1],
             quat_pitch(rq) - quat_pitch(d.qpos[3:7])],
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
        # reverse curriculum, frontier-weighted toward min_phase. Frame 0 = the bridge
        # brace; the early leaned frames carry the correct base quat so RSI is valid.
        lo = self.rsi_min_phase + (1.0 - self.rsi_min_phase) * self.rng.random() ** 2
        self._k = int(lo * (REF_N - 1))
        rj, rb, rq = self._ref(self._k)
        self.d.qpos[0:3] = (self.rng.uniform(-0.03, 0.03),
                            rb[0] + self.rng.uniform(-0.02, 0.02), rb[1])
        # small orientation noise around the reference quat
        q = rq + self.rng.uniform(-0.02, 0.02, 4)
        self.d.qpos[3:7] = q / np.linalg.norm(q)
        self.d.qpos[7:] = rj + self.rng.uniform(-0.04, 0.04, 29)
        self.d.qvel[:] = 0
        self.d.qvel[0:3] = self.rng.uniform(-0.08, 0.08, 3)
        self.d.ctrl[:] = rj
        mujoco.mj_forward(self.m, self.d)
        self._last_a[:] = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        rj, rb, rq = self._ref(self._k + 1)
        tgt = rj + ACTION_SCALE * a               # residual around the reference
        self.d.ctrl[:] = np.clip(tgt, self.lo, self.hi)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
        self._k += 1
        d = self.d
        w, x_, y_, z_ = d.qpos[3:7]
        roll = np.degrees(np.arctan2(2 * (w * x_ + y_ * z_), 1 - 2 * (x_**2 + y_**2)))
        yaw = np.arctan2(2 * (w * z_ + x_ * y_), 1 - 2 * (y_**2 + z_**2))
        ye = wrap(yaw - YAW_REF)
        on, plat, handF = self._contacts()
        feet_plat = plat[self.lf] + plat[self.rf]
        phase = self._k / REF_N

        # DeepMimic tracking: joints (product) x base position x lean
        jerr = float(np.mean((d.qpos[7:] - rj) ** 2))
        base_err = float(d.qpos[0] ** 2 + (d.qpos[1] - rb[0]) ** 2
                         + 1.5 * (d.qpos[2] - rb[1]) ** 2)
        lean_err = float((quat_pitch(d.qpos[3:7]) - quat_pitch(rq)) ** 2)
        r = 2.4 * np.exp(-6.0 * jerr) * np.exp(-12.0 * base_err) * np.exp(-4.0 * lean_err)
        # brace bonus: while the reference expects the hands down, reward pressing both
        # armrests (the load transfer that makes the foot-lift feasible)
        if phase < BRACE_PHASE:
            both = min(handF[self.lh], handF[self.rh])
            r += 0.5 * np.clip(both / 40.0, 0.0, 1.0)
        r -= 0.12 * abs(ye)
        r -= 0.01 * float(np.sum((a - self._last_a) ** 2))
        self._last_a = a

        terminated, truncated = False, False
        info = {"success": False}
        speed = float(np.linalg.norm(d.sensor("local_linvel_pelvis").data[:2]))
        done_window = self._k >= REF_N
        if (done_window and feet_plat == 2 and d.qpos[2] > 0.90 and abs(ye) < 0.35
                and abs(roll) < 25 and speed < 0.6
                and abs(d.qpos[1] - REF_BASE[-1][0]) < 0.15 and abs(d.qpos[0]) < 0.12):
            r += 100.0
            terminated = True
            info["success"] = True
        elif d.qpos[2] < 0.45 or abs(roll) > 55 or abs(d.qpos[0]) > 0.35:
            r -= 20.0
            terminated = True
        elif self._k >= REF_N + int(EXTRA_S / CTRL_DT):
            truncated = True
        return self._obs(), float(r), terminated, truncated, info
