"""Hardware-agnostic mission controller: navigate -> avoid -> turn -> back up ->
climb the footrest platform -> sit on the real chair (３階講堂遠隔操作席).

THIS FILE IS THE DEPLOYMENT ARTIFACT. It depends only on numpy + onnxruntime —
no MuJoCo, no torch. The exact same class is executed:
  - in simulation by training/g1_full_mission.py (the E2E test harness), and
  - on the robot's Linux PC by a thin adapter that feeds it sensor data and
    forwards its output to the motor bus (see deploy/README_DEPLOY.md).

I/O contract (chair frame: origin = chair center, sitter faces +y, z up):
  step(state) -> command, called at 50 Hz.
  state = {
    "joint_pos":   (29,) rad, actuator order (see config actuator_names)
    "joint_vel":   (29,) rad/s
    "gravity":     (3,)  gravity direction in the pelvis IMU frame (unit-ish)
    "gyro":        (3,)  rad/s, pelvis IMU
    "linvel":      (3,)  m/s, pelvis frame (estimator)
    "base_xy":     (2,)  m, chair frame (odometry / mocap / LiDAR localization)
    "base_yaw":    float rad, chair frame
    "foot_contact":(2,)  L,R in {0,1}
    "rays":        (21,) m, horizontal LiDAR fan: 200 deg FOV centered on
                   heading, beams at z~0.5 m, max range 4.0 (None disables
                   avoidance — NAVIGATE then steers straight at the goal)
  }
  command = {
    "target": (29,) rad joint position targets
    "kp":     (29,)  position gains   (mode-dependent: walk vs sit)
    "kd":     (29,)  velocity gains
    "phase":  str    current FSM state
    "done":   bool   mission complete (seated, hold finished)
    "failed": str|None
  }
"""
import json
import os

import numpy as np

try:
    import onnxruntime as rt
except ImportError as e:           # pragma: no cover
    raise ImportError("deploy needs onnxruntime (pip install onnxruntime)") from e


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class MissionController:
    def __init__(self, config_path=None, walk_onnx=None, climb_onnx=None):
        here = os.path.dirname(os.path.abspath(__file__))
        config_path = config_path or os.path.join(here, "config.json")
        with open(config_path) as f:
            self.cfg = json.load(f)
        c = self.cfg
        self.default_pose = np.array(c["default_pose"])
        self.lo = np.array(c["ctrl_lo"])
        self.hi = np.array(c["ctrl_hi"])
        self.walk_kp = np.array(c["walk_kp"])
        self.walk_kd = np.array(c["walk_kd"])
        self.sit_kp = self.walk_kp.copy()
        self.sit_kd = self.walk_kd.copy()
        self.sit_kp[:12] = c["sit_mode_kp"]
        self.sit_kd[:12] = c["sit_mode_kd"]
        self.dt = c["ctrl_dt"]

        self.walk = rt.InferenceSession(
            walk_onnx or os.path.join(here, c["walk_onnx"]),
            providers=["CPUExecutionProvider"])
        self.climb = rt.InferenceSession(
            climb_onnx or os.path.join(here, c["climb_onnx"]),
            providers=["CPUExecutionProvider"])

        n = c["vfh"]["n_rays"]
        fov = np.deg2rad(c["vfh"]["fov_deg"])
        self.ray_angles = np.array([-fov / 2 + fov * i / (n - 1) for i in range(n)])
        self.reset()

    # ------------------------------------------------------------------ FSM --
    def reset(self):
        self.state = "NAVIGATE"
        self.t = 0.0
        self.last_walk_action = np.zeros(29, dtype=np.float32)
        self.last_climb_action = np.zeros(12, dtype=np.float32)
        self.phase = np.array([0.0, np.pi])
        self.ix = self.iy = self.iw = 0.0
        self.t_climb = self.t_sit = None
        self.failed = None
        self.min_clear = float("inf")
        # precompute sit targets
        c = self.cfg
        self.sit_target = self.default_pose.copy()
        self.pose_target = self.default_pose.copy()
        for hi_i, kn_i, an_i in ((0, 3, 4), (6, 9, 10)):
            self.sit_target[[hi_i, kn_i, an_i]] = [c["sit"]["hip"], c["sit"]["knee"],
                                                   c["sit"]["ankle"]]
            self.pose_target[[hi_i, kn_i, an_i]] = [c["sit"]["hip"], c["sit"]["knee"],
                                                    c["sit"]["ankle2"]]
        self.pose_target[14] = c["sit"]["waist2"]
        self.sit_target = np.clip(self.sit_target, self.lo, self.hi)
        self.pose_target = np.clip(self.pose_target, self.lo, self.hi)

    # --------------------------------------------------------------- helpers --
    def _vfh(self, s, goal):
        c = self.cfg["vfh"]
        x, y = s["base_xy"]
        yaw = s["base_yaw"]
        rays = s.get("rays")
        if rays is None:
            rays = np.full(len(self.ray_angles), c["rmax"])
        rays = np.minimum(np.asarray(rays, dtype=float), c["rmax"])
        blocked = rays < c["safe"]
        wide = blocked.copy()
        w = c["widen"]
        for i in np.where(blocked)[0]:
            wide[max(0, i - w):i + w + 1] = True
        goal_dir = _wrap(np.arctan2(goal[1] - y, goal[0] - x) - yaw)
        free = np.where(~wide)[0]
        if len(free) == 0:
            return 0.0, 0.6
        best = free[np.argmin(np.abs(self.ray_angles[free] - goal_dir))]
        steer = self.ray_angles[best]
        vx = 0.5 * float(np.clip(rays[best] / c["rmax"], 0.3, 1.0))
        return vx, float(np.clip(1.5 * steer, -0.6, 0.6))

    def _walk_step(self, s, cmd):
        obs = np.hstack([
            s["linvel"], s["gyro"], s["gravity"], cmd,
            s["joint_pos"] - self.default_pose, s["joint_vel"],
            self.last_walk_action,
            np.concatenate([np.cos(self.phase), np.sin(self.phase)]),
        ]).astype(np.float32)
        a = self.walk.run(["continuous_actions"], {"obs": obs.reshape(1, -1)})[0][0]
        self.last_walk_action = a.copy()
        self.phase = np.fmod(self.phase + 2 * np.pi * self.cfg["gait_freq"] * self.dt
                             + np.pi, 2 * np.pi) - np.pi
        return np.clip(a * self.cfg["action_scale"] + self.default_pose,
                       self.lo, self.hi)

    def _climb_step(self, s):
        c = self.cfg
        yaw = s["base_yaw"]
        ye = _wrap(yaw - np.pi / 2)
        tx, ty = c["climb"]["target_xy"]
        exw, eyw = tx - s["base_xy"][0], ty - s["base_xy"][1]
        ex = np.cos(yaw) * exw + np.sin(yaw) * eyw
        ey = -np.sin(yaw) * exw + np.cos(yaw) * eyw
        obs = np.hstack([
            s["gravity"], s["gyro"], s["linvel"], [s["base_z"]],
            [np.sin(ye), np.cos(ye)], [ex, ey],
            s["joint_pos"][:12] - self.default_pose[:12], s["joint_vel"][:12],
            s["foot_contact"], self.last_climb_action,
        ]).astype(np.float32)
        a = self.climb.run(None, {"obs": obs.reshape(1, -1)})[0][0]
        self.last_climb_action = np.asarray(a, dtype=np.float32)
        tgt = self.default_pose.copy()
        tgt[:12] = self.default_pose[:12] + c["climb"]["action_scale"] * self.last_climb_action
        return np.clip(tgt, self.lo, self.hi)

    # ------------------------------------------------------------------ step --
    def step(self, s):
        c = self.cfg
        x, y = s["base_xy"]
        yaw = s["base_yaw"]
        yerr = _wrap(yaw - np.pi / 2)
        kp, kd = self.walk_kp, self.walk_kd
        done = False
        target = self.default_pose

        if self.state == "NAVIGATE":
            goal = np.array(c["pre_step_xy"])
            vx, wz = self._vfh(s, goal)
            dist = float(np.hypot(goal[0] - x, goal[1] - y))
            vx *= float(np.clip(dist / 1.2, 0.3, 1.0))
            target = self._walk_step(s, np.array([vx, 0.0, wz]))
            if s.get("rays") is not None:
                self.min_clear = min(self.min_clear, float(np.min(s["rays"])))
            if dist < c["gates"]["nav_exit"]:
                self.state = "TURN"
        elif self.state == "TURN":
            target = self._walk_step(s, np.array(
                [c["trim_vx"], c["trim_vy"], float(np.clip(-1.2 * yerr, -0.6, 0.6))]))
            if abs(yerr) < np.deg2rad(c["gates"]["turn_exit_deg"]):
                self.state = "BACKUP"
        elif self.state == "BACKUP":
            dx_, dy_ = c["climb"]["spawn_xy"][0] - x, c["climb"]["spawn_xy"][1] - y
            ex = np.cos(yaw) * dx_ + np.sin(yaw) * dy_
            ey = -np.sin(yaw) * dx_ + np.cos(yaw) * dy_
            self.ix = float(np.clip(self.ix + 0.35 * ex * self.dt, -0.30, 0.30))
            self.iy = float(np.clip(self.iy + 0.35 * ey * self.dt, -0.20, 0.20))
            self.iw = float(np.clip(self.iw - 0.50 * yerr * self.dt, -0.20, 0.20))
            target = self._walk_step(s, np.array([
                c["trim_vx"] + np.clip(1.2 * ex + self.ix, -0.15, 0.40),
                c["trim_vy"] + np.clip(1.2 * ey + self.iy, -0.20, 0.20),
                np.clip(-1.0 * yerr + self.iw, -0.30, 0.30)]))
            dist = float(np.hypot(dx_, dy_))
            lv = s["linvel"]
            if (dist < c["gates"]["backup_tol"]
                    and abs(yerr) < np.deg2rad(c["gates"]["backup_yaw_deg"])
                    and abs(lv[0]) < 0.15 and abs(lv[1]) < 0.12):
                self.state = "CLIMB"
                self.t_climb = self.t
                self.last_climb_action[:] = 0
        elif self.state == "CLIMB":
            kp, kd = self.sit_kp, self.sit_kd
            target = self._climb_step(s)
            tx, ty = c["climb"]["target_xy"]
            distt = float(np.hypot(tx - x, ty - y))
            speed = float(np.hypot(s["linvel"][0], s["linvel"][1]))
            feet_up = (s["foot_contact"][0] > 0 and s["foot_contact"][1] > 0
                       and s["base_z"] > 0.92)
            if (feet_up and abs(yerr) < 0.30 and speed < 0.35 and distt < 0.12
                    and abs(s["gravity"][0]) < 0.25 and abs(s["gravity"][1]) < 0.25):
                self.state = "SIT"
                self.t_sit = self.t
            elif self.t - self.t_climb > c["gates"]["climb_timeout"]:
                self.failed = "climb timeout"
        elif self.state == "SIT":
            kp, kd = self.sit_kp, self.sit_kd
            st = self.cfg["sit"]
            dt_ = self.t - self.t_sit
            t1, t2 = st["t_stand"], st["t_stand"] + st["t_desc"]
            t3 = t2 + st["t_seat"]
            t4 = t3 + st["t_pose"]
            if dt_ < t1:
                target = self.default_pose
            elif dt_ < t2:
                a = (dt_ - t1) / st["t_desc"]
                target = self.default_pose + a * (self.sit_target - self.default_pose)
            elif dt_ < t3:
                target = self.sit_target
            elif dt_ < t4:
                a = (dt_ - t3) / st["t_pose"]
                target = self.sit_target + a * (self.pose_target - self.sit_target)
            else:
                target = self.pose_target
                if dt_ > t4 + st["t_hold"]:
                    done = True

        self.t += self.dt
        return {"target": np.asarray(target), "kp": kp, "kd": kd,
                "phase": self.state, "done": done, "failed": self.failed}
