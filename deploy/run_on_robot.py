"""TEMPLATE: Linux-side adapter that runs MissionController on a real Unitree G1.

STATUS: structurally complete, NEVER executed on hardware — every block marked
TODO(HW) must be verified on the robot with the safety checklist in
README_DEPLOY.md before the first untethered run. This file intentionally
contains the entire hardware surface area so the sim-tested controller
(g1_mission_controller.py) needs zero changes.

Pattern follows unitree_rl_gym's deploy_mujoco/deploy_real split and uses
unitree_sdk2py (pip install unitree_sdk2py) low-level joint command interface:
500 Hz lowcmd loop, our controller decides at 50 Hz, targets are linearly
interpolated in between (standard practice).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_mission_controller import MissionController  # noqa: E402

CTRL_DT = 0.02          # controller decision period (50 Hz)
LOW_DT = 0.002          # lowcmd bus period (500 Hz)


class HardwareIO:
    """All robot-specific I/O lives here. Every method is a TODO(HW)."""

    def __init__(self):
        # TODO(HW): unitree_sdk2py channel setup, e.g.
        #   from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        #   from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        #   ChannelFactoryInitialize(0, "eth0")
        #   ... subscribe lowstate, create lowcmd publisher, CRC ...
        # TODO(HW): joint index map — config.json actuator_names order MUST be
        #   mapped to the G1 lowcmd motor indices (they differ!).
        raise NotImplementedError("wire unitree_sdk2py here")

    def read_state(self):
        """Return the MissionController state dict (see its docstring).
        TODO(HW):
          joint_pos/vel : lowstate.motor_state[q, dq] via the index map
          gravity, gyro : lowstate.imu_state (quaternion -> gravity in body)
          linvel        : a state estimator (e.g. Unitree's, or leg odometry)
          base_xy/yaw   : LOCALIZATION IN THE CHAIR FRAME — survey the chair
                          pose once (e.g. LiDAR scan-matching / AprilTag on the
                          chair / mocap) and compose with odometry
          base_z        : estimator height
          foot_contact  : lowstate foot force > threshold (e.g. 20 N)
          rays          : 2D LiDAR fan resampled to 21 beams, 200 deg, max 4 m,
                          points below z~0.2 and above ~1.2 filtered out
        """
        raise NotImplementedError

    def send_targets(self, target, kp, kd):
        """TODO(HW): for each joint i: lowcmd.motor_cmd[map[i]] =
        (q=target[i], dq=0, tau=0, kp=kp[i], kd=kd[i]); CRC; publish."""
        raise NotImplementedError

    def estop_pressed(self):
        """TODO(HW): wireless E-stop / controller button."""
        return False


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ctrl = MissionController(config_path=os.path.join(here, "config.json"))
    hw = HardwareIO()

    # safety: start in damping mode, operator confirms, then stand (Unitree FSM),
    # then hand over to the mission controller.  TODO(HW): implement the standard
    # unitree_rl_gym zero-torque -> default-pose ramp before this loop.
    prev_target = None
    next_decision = time.monotonic()
    sub = 0
    out = None
    while True:
        now = time.monotonic()
        if hw.estop_pressed():
            # TODO(HW): switch to damping mode
            print("E-STOP")
            return 1
        if now >= next_decision:
            s = hw.read_state()
            new_out = ctrl.step(s)
            prev_target = out["target"] if out is not None else new_out["target"]
            out = new_out
            sub = 0
            next_decision += CTRL_DT
            if out["failed"]:
                print("controller failed:", out["failed"])
                # TODO(HW): damping mode
                return 1
            if out["done"]:
                print("mission complete (seated)")
                return 0
        # 500 Hz interpolation between 50 Hz decisions
        alpha = min(1.0, (sub + 1) * LOW_DT / CTRL_DT)
        tgt = prev_target + alpha * (out["target"] - prev_target)
        hw.send_targets(tgt, out["kp"], out["kd"])
        sub += 1
        time.sleep(max(0.0, LOW_DT - (time.monotonic() - now)))


if __name__ == "__main__":
    raise SystemExit(main())
