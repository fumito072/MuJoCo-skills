"""Unitree G1 walking by REPLAYING a pretrained policy locally on CPU (no NVIDIA, no RL training).

This is a headless, vendored, self-contained version of unitree_rl_gym's deploy_mujoco.py:
no GUI viewer, no `legged_gym` import (replaced by a local path constant). The pretrained
torch-JIT policy (motion.pt) outputs 12 leg target-angle deltas at 50 Hz; a software PD turns
them into torques on the 12-DOF torque model. The command (vx, vy, wz) steers the gait and is
the single hook the obstacle-avoidance planner will write to.

Embodiment note: this 12-DOF legs-only TORQUE model (g1_12dof) is SEPARATE from the 29-DOF
position Menagerie model used by g1_stand.py / g1_squat.py. Do not mix them.

License: the vendored policy + model are Unitree's, BSD-3-Clause (see vendor/g1/LICENSE-*).

Usage: python g1_walk.py [--secs 6] [--vx 0.5] [--vy 0] [--wz 0]
"""
import os
import argparse
import numpy as np
import mujoco
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.normpath(os.path.join(HERE, "..", "vendor", "g1"))

# config (from vendored g1.yaml)
SIM_DT = 0.002
DECIMATION = 10                      # 50 Hz policy over 500 Hz sim
KPS = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float32)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)
ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.25, 1.0, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
NUM_ACT, NUM_OBS, PERIOD = 12, 47, 0.8


def gravity_orientation(quat):
    qw, qx, qy, qz = quat
    return np.array([2*(-qz*qx + qw*qy), -2*(qz*qy + qw*qx), 1 - 2*(qw*qw + qz*qz)])


def make():
    m = mujoco.MjModel.from_xml_path(os.path.join(VENDOR, "model", "scene.xml"))
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)
    policy = torch.jit.load(os.path.join(VENDOR, "motion.pt"))
    return m, d, policy


def walk(m, d, policy, cmd, steps, log=None):
    """Run the replay loop for `steps` physics steps with a fixed (vx,vy,wz) command.
    `cmd` may be a 3-vector or a callable(t)->3-vector (for the avoidance planner)."""
    action = np.zeros(NUM_ACT, dtype=np.float32)
    target = DEFAULT_ANGLES.copy()
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    for c in range(steps):
        tau = (target - d.qpos[7:]) * KPS + (0.0 - d.qvel[6:]) * KDS
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        if c % DECIMATION == 0:
            cmd_t = np.asarray(cmd(c * SIM_DT) if callable(cmd) else cmd, dtype=np.float32)
            qj = (d.qpos[7:] - DEFAULT_ANGLES) * DOF_POS_SCALE
            dqj = d.qvel[6:] * DOF_VEL_SCALE
            grav = gravity_orientation(d.qpos[3:7])
            omega = d.qvel[3:6] * ANG_VEL_SCALE
            phase = (c * SIM_DT) % PERIOD / PERIOD
            obs[:3] = omega
            obs[3:6] = grav
            obs[6:9] = cmd_t * CMD_SCALE
            obs[9:9+NUM_ACT] = qj
            obs[9+NUM_ACT:9+2*NUM_ACT] = dqj
            obs[9+2*NUM_ACT:9+3*NUM_ACT] = action
            obs[9+3*NUM_ACT:9+3*NUM_ACT+2] = [np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target = action * ACTION_SCALE + DEFAULT_ANGLES
        if log is not None:
            log(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=6.0)
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    args = ap.parse_args()

    m, d, policy = make()
    print(f"model nq={m.nq} nv={m.nv} nu={m.nu}  start base z={d.qpos[2]:.3f}")
    x0 = d.qpos[0]
    zs, rolls, pitches = [], [], []

    def rec(d):
        zs.append(d.qpos[2])
        w, x, y, z = d.qpos[3:7]
        rolls.append(np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))))
        pitches.append(np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1))))

    walk(m, d, policy, [args.vx, args.vy, args.wz], int(args.secs / SIM_DT), log=rec)
    zs = np.array(zs)
    dx, dy = d.qpos[0] - x0, d.qpos[1]
    upright = zs[-1] > 0.5 and abs(rolls[-1]) < 30 and abs(pitches[-1]) < 30
    print(f"cmd=({args.vx},{args.vy},{args.wz})  {args.secs}s")
    print(f"travel: dx={dx:+.3f} dy={dy:+.3f} m -> {dx/args.secs:+.3f} m/s   base z end={zs[-1]:.3f} min={zs.min():.3f}")
    print(f"final roll={rolls[-1]:+.1f} pitch={pitches[-1]:+.1f}")
    print(f"RESULT: {'WALKS ✓' if (upright and dx > 0.2) else ('STAYS UP' if upright else 'FELL ✗')}")
    return 0 if upright else 1


if __name__ == "__main__":
    raise SystemExit(main())
