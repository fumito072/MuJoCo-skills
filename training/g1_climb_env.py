"""Gymnasium env: G1 learns the BACKWARD step-up (0.22 m) onto the real chair's
footrest platform — the one skill hand-scripting could not crack (14 measured
failures: the swing-mass reaction, base retreat and lateral-roll couplings of a
single-support 0.22 m climb defeat open-loop waypoints + ankle-PD).

Task: spawn standing on the floor, back to the step (facing chair +y, away from
the seat), heels ~7 cm from the step edge. Succeed by STANDING on the platform
at base y ~0.33 (the verified sit-descent basin), still facing +y, calm.
Backward climb is chosen so the platform pivot-turn is never needed: the chain
is then  navigate -> turn on the floor -> back up -> CLIMB (this policy) ->
stiff-descent sit (verified 30/30).

Control: 50 Hz, action in [-1,1]^12 (legs only), target = default + 0.45*action,
SIT-mode stiff gains kp=300/kd=8 (mode-switched, like the real controller).
Obs (52-D): gravity(3) gyro(3) local linvel(3) base_z(1) yaw_err sin/cos(2)
base->target xy in body frame(2) leg qpos-default(12) leg qvel(12) feet
contact(2) last action(12). The xy-to-target term is privileged in sim; on
hardware it comes from the known chair pose (chair_info.json) + odometry.
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
EP_SECONDS = 6.0
YAW_TARGET = np.pi / 2
TARGET = np.array([0.0, 0.33])        # platform stand point = sit-descent basin center
PLATFORM_Z = 0.22
SPAWN_Y, SPAWN_NOISE = 0.68, (0.04, 0.03, 0.10)   # (x, y, yaw) noise amplitudes
# FORWARD-approach variant (user insight: the body is fore-aft ASYMMETRIC — toe-first
# landings load naturally, heel-first backward landings fight the foot lever): spawn
# FACING the step; success still requires facing +y (back to the seat) on the platform,
# so the policy may climb-then-pivot or pivot-while-climbing — its choice.
SPAWN_YAW = -np.pi / 2
SIT_KP, SIT_KD = 300.0, 8.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class G1ClimbEnv(gym.Env):
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
        self.floor = self.m.geom("floor").id
        self.rng = np.random.default_rng(seed)
        # reverse-curriculum (RSI): optional bank of harvested half-climb states —
        # spawning there puts the policy one weight-transfer from the success bonus,
        # the escape hatch from the one-foot-up local optimum it converged to
        self.rsi = None
        rsi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "runs_climb", "half_climb_states.npz")
        if os.path.exists(rsi_path) and not os.environ.get("G1CLIMB_NO_RSI"):
            z = np.load(rsi_path)
            self.rsi = (z["qpos"], z["qvel"])
        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (52,), np.float32)
        self._last_a = np.zeros(12, np.float32)
        self._steps = 0
        self._prev_pot = 0.0

    # --- helpers -------------------------------------------------------------
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

    def _obs(self):
        d, m = self.d, self.m
        gravity = d.site_xmat[self.imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        gyro = d.sensor("gyro_pelvis").data
        linvel = d.sensor("local_linvel_pelvis").data
        yaw = np.arctan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]),
                         1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
        ye = wrap(yaw - YAW_TARGET)
        exw, eyw = TARGET[0] - d.qpos[0], TARGET[1] - d.qpos[1]
        ex = np.cos(yaw) * exw + np.sin(yaw) * eyw
        ey = -np.sin(yaw) * exw + np.cos(yaw) * eyw
        on, _ = self._foot_contacts()
        return np.hstack([
            gravity, gyro, linvel, [d.qpos[2]], [np.sin(ye), np.cos(ye)], [ex, ey],
            d.qpos[7:19] - self.default_pose[:12], d.qvel[6:18],
            [on[self.lf], on[self.rf]], self._last_a,
        ]).astype(np.float32)

    def _potential(self):
        d = self.d
        dist = float(np.hypot(TARGET[0] - d.qpos[0], TARGET[1] - d.qpos[1]))
        dz = abs(d.qpos[2] - (0.755 + PLATFORM_Z))
        return -(1.0 * dist + 3.0 * dz)

    # --- gym API -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # reverse-curriculum (RSI): optional bank of harvested half-climb states —
        # spawning there puts the policy one weight-transfer from the success bonus,
        # the escape hatch from the one-foot-up local optimum it converged to
        self.rsi = None
        rsi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "runs_climb", "half_climb_states.npz")
        if os.path.exists(rsi_path) and not os.environ.get("G1CLIMB_NO_RSI"):
            z = np.load(rsi_path)
            self.rsi = (z["qpos"], z["qvel"])
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
        if self.rsi is not None and self.rng.random() < 0.5:
            i = self.rng.integers(len(self.rsi[0]))
            self.d.qpos[:] = self.rsi[0][i]
            self.d.qvel[:] = self.rsi[1][i]
            self.d.qpos[7:19] += self.rng.uniform(-0.03, 0.03, 12)
            self.d.ctrl[:] = self.default_pose
            mujoco.mj_forward(self.m, self.d)
            self._last_a[:] = 0
            self._steps = 0
            self._prev_pot = self._potential()
            return self._obs(), {}
        nx, ny, nyaw = (self.rng.uniform(-a, a) for a in SPAWN_NOISE)
        yaw0 = SPAWN_YAW + nyaw
        self.d.qpos[0:3] = (nx, SPAWN_Y + ny, 0.755)
        self.d.qpos[3:7] = (np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2))
        # the chain hands over FROM A MARCH: expose initial base/joint motion so the
        # policy learns to absorb the handoff transient (the lesson of this project)
        self.d.qvel[0:2] = self.rng.uniform(-0.15, 0.15, 2)
        self.d.qvel[6:18] = self.rng.uniform(-1.0, 1.0, 12)
        self.d.qpos[7:19] = (self.default_pose[:12]
                             + self.rng.uniform(-0.12, 0.12, 12))
        self.d.ctrl[:] = self.default_pose
        mujoco.mj_forward(self.m, self.d)
        self._last_a[:] = 0
        self._steps = 0
        self._prev_pot = self._potential()
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        tgt = self.default_pose.copy()
        tgt[:12] = self.default_pose[:12] + ACTION_SCALE * a
        self.d.ctrl[:] = np.clip(tgt, self.lo, self.hi)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
        self._steps += 1
        d = self.d
        w, x_, y_, z_ = d.qpos[3:7]
        roll = np.degrees(np.arctan2(2 * (w * x_ + y_ * z_), 1 - 2 * (x_**2 + y_**2)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (w * y_ - z_ * x_), -1, 1)))
        yaw = np.arctan2(2 * (w * z_ + x_ * y_), 1 - 2 * (y_**2 + z_**2))
        ye = wrap(yaw - YAW_TARGET)
        on, plat = self._foot_contacts()
        feet_plat = plat[self.lf] + plat[self.rf]

        pot = self._potential()
        r = 6.0 * (pot - self._prev_pot)                      # progress (potential-based)
        self._prev_pot = pot
        # one foot is cheap, BOTH feet pay; and a platform foot pays MORE the DEEPER
        # it is planted (the policy parked at a shallow toe-hook at y~0.56 — a dead-end
        # posture; this gradient walks the plant toward the transfer-feasible depth)
        for fg, on_plat in ((self.lf, plat[self.lf]), (self.rf, plat[self.rf])):
            if on_plat:
                fy = float(self.d.geom_xpos[fg][1])
                r += 0.05 + 0.45 * max(0.0, 0.55 - fy)
        r += 0.60 if feet_plat == 2 else 0.0
        r += 0.03                                              # alive (small: farming it
                                                               # bred a sit-on-the-edge exploit)
        # leaning ANY non-foot body part on the chair is not climbing — kill the
        # crouch-against-the-step local optimum the first run converged to
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if o in self.rc and g not in (self.lf, self.rf):
                    r -= 0.15
                    break
        r -= 0.10 * (abs(np.radians(roll)) + abs(np.radians(pitch)))
        r -= 0.20 * abs(ye)
        r -= 0.012 * float(np.sum((a - self._last_a) ** 2))
        r -= 0.0004 * float(np.sum(d.qvel[6:18] ** 2))
        self._last_a = a

        terminated, truncated = False, False
        info = {"success": False}
        speed = float(np.linalg.norm(d.sensor("local_linvel_pelvis").data[:2]))
        # success = GOT UP (scrappy ok): the mission FSM holds a stiff stand after
        # the climb and gates the sit on its own calm/position check — the policy
        # only has to deliver both feet onto the platform near the target, upright
        if (feet_plat == 2 and d.qpos[2] > 0.90 and abs(ye) < 0.35
                and abs(roll) < 25 and abs(pitch) < 25 and speed < 0.60
                and float(np.hypot(TARGET[0] - d.qpos[0], TARGET[1] - d.qpos[1])) < 0.15):
            r += 200.0
            terminated = True
            info["success"] = True
        elif d.qpos[2] < 0.45 or abs(roll) > 55 or abs(pitch) > 65:
            r -= 20.0
            terminated = True
        elif float(np.hypot(d.qpos[0], d.qpos[1] - SPAWN_Y)) > 1.2:
            r -= 10.0
            terminated = True
        elif self._steps >= int(EP_SECONDS / CTRL_DT):
            truncated = True
        return self._obs(), float(r), terminated, truncated, info
