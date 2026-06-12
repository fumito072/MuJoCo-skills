"""Replay the Colab-trained G1ClimbBox policy on the Mac — in the FULL real-chair
scene (153 collision hulls), plain C engine, NVIDIA-free.

This is the verification half of the climb pipeline (train on GPU box-world,
verify here in the full world): it rebuilds the brax policy from
g1_climb_params.pkl, reproduces the env observation (joystick 103 + the 5 climb
dims) from C-engine sensors, applies SIT-mode stiff gains, spawns on the floor
facing the step (the forward approach) and reports the FULL-CLIMB success rate.

  .venv-rl/bin/python training/g1_climb_play.py \
      --params models/policies/g1_climb_params.pkl --episodes 20 \
      --video assets/g1_climb_rl.gif
"""
import argparse
import os
import pickle
import sys

import numpy as np
import mujoco
import jax
import jax.numpy as jp

from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_sit_env  # noqa: E402

CTRL_DT, SIM_DT, ACTION_SCALE = 0.02, 0.002, 0.5
DECIMATION = int(round(CTRL_DT / SIM_DT))
GAIT_FREQ = 1.375
STATE_SIZE, PRIV_SIZE = 103 + 5, 216 + 5      # joystick obs + the 5 climb dims
TARGET = np.array([0.0, 0.33])
YAW_GOAL = np.pi / 2
SIT_KP, SIT_KD = 300.0, 8.0
EP_SECONDS = 8.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_policy(params_path, act_size):
    nk = dict(locomotion_params.brax_ppo_config("G1JoystickFlatTerrain").network_factory)
    net = ppo_networks.make_ppo_networks(
        {"state": (STATE_SIZE,), "privileged_state": (PRIV_SIZE,)},
        act_size,
        preprocess_observations_fn=running_statistics.normalize,
        **nk,
    )
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    return ppo_networks.make_inference_fn(net)(params, deterministic=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--video", default=None, help="GIF of the first SUCCESSFUL episode")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    # FULL chair: 153 hulls + leg pairs + hand pairs (the policy may hand-brace)
    m = g1_sit_env.build_fbx_chair_model(SIM_DT, pair_hands=True)
    for a in range(12):
        m.actuator_gainprm[a, 0] = SIT_KP
        m.actuator_biasprm[a, 1] = -SIT_KP
        m.actuator_biasprm[a, 2] = -SIT_KD
    # arm mode gains — must match g1_climb_mjx_env (stock kp=2 wrists flop)
    for a in (15, 16, 17, 18, 22, 23, 24, 25):
        m.actuator_gainprm[a, 0] = 150.0
        m.actuator_biasprm[a, 1] = -150.0
        m.actuator_biasprm[a, 2] = -4.0
    for a in (19, 20, 21, 26, 27, 28):
        m.actuator_gainprm[a, 0] = 80.0
        m.actuator_biasprm[a, 1] = -80.0
        m.actuator_biasprm[a, 2] = -2.0
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    gyro = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro_pelvis")
    lvel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "local_linvel_pelvis")
    imu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis")
    lf, rf = m.geom("left_foot").id, m.geom("right_foot").id
    rc_gids = {i for i in range(m.ngeom)
               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("rc_part")}
    policy = load_policy(args.params, m.nu)

    def sread(s):
        a = m.sensor_adr[s]
        return d.sensordata[a:a + m.sensor_dim[s]].copy()

    rng = np.random.default_rng(7)
    n_ok = 0
    rend = cam = None
    if args.video:
        rend = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        cam.distance, cam.azimuth, cam.elevation = 2.6, 150, -12

    for ep in range(args.episodes):
        mujoco.mj_resetDataKeyframe(m, d, kid)
        yaw0 = -np.pi / 2 + rng.uniform(-0.10, 0.10)
        d.qpos[0:3] = (rng.uniform(-0.04, 0.04), 0.68 + rng.uniform(-0.03, 0.03), 0.755)
        d.qpos[3:7] = (np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2))
        mujoco.mj_forward(m, d)
        default_pose = np.array(m.key_qpos[kid][7:])
        d.ctrl[:] = default_pose
        last = np.zeros(m.nu)
        ph = np.array([0.0, np.pi])
        pdt = 2 * np.pi * CTRL_DT * GAIT_FREQ
        frames = []
        ok = False
        for k in range(int(EP_SECONDS / CTRL_DT)):
            gravity = d.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
            w, qx, qy, qz = d.qpos[3:7]
            yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            ye = wrap(yaw - YAW_GOAL)
            exw, eyw = TARGET[0] - d.qpos[0], TARGET[1] - d.qpos[1]
            ex = np.cos(yaw) * exw + np.sin(yaw) * eyw
            ey = -np.sin(yaw) * exw + np.cos(yaw) * eyw
            state = np.concatenate([
                sread(lvel), sread(gyro), gravity, np.zeros(3),
                d.qpos[7:] - default_pose, d.qvel[6:], last,
                [np.cos(ph[0]), np.cos(ph[1]), np.sin(ph[0]), np.sin(ph[1])],
                [np.sin(ye), np.cos(ye), ex, ey, d.qpos[2]],
            ]).astype(np.float32)
            act, _ = policy({"state": jp.asarray(state)}, jax.random.PRNGKey(k))
            act = np.asarray(act)
            d.ctrl[:] = np.clip(default_pose + act * ACTION_SCALE,
                                m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            for _ in range(DECIMATION):
                mujoco.mj_step(m, d)
            last = act
            ph = (ph + pdt + np.pi) % (2 * np.pi) - np.pi
            if rend is not None and n_ok == 0 and k % max(1, int(1 / (args.fps * CTRL_DT))) == 0:
                rend.update_scene(d, camera=cam)
                from PIL import Image
                frames.append(Image.fromarray(rend.render())
                              .convert("P", palette=Image.ADAPTIVE, colors=128))
            # success check (same as the CPU env): both feet platform-contact, standing
            feet_plat = 0
            for i in range(d.ncon):
                c = d.contact[i]
                for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if g == lf and o in rc_gids:
                        feet_plat |= 1
                    if g == rf and o in rc_gids:
                        feet_plat |= 2
            w, qx, qy, qz = d.qpos[3:7]
            yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            ye = wrap(yaw - YAW_GOAL)
            dist = float(np.hypot(TARGET[0] - d.qpos[0], TARGET[1] - d.qpos[1]))
            speed = float(np.linalg.norm(sread(lvel)[:2]))
            if (feet_plat == 3 and d.qpos[2] > 0.90 and abs(ye) < 0.35
                    and dist < 0.15 and speed < 0.6):
                ok = True
                break
            if d.qpos[2] < 0.40:
                break
        n_ok += ok
        print(f"ep{ep}: {'SUCCESS' if ok else 'fail'}  "
              f"base=({d.qpos[0]:+.2f},{d.qpos[1]:+.2f},{d.qpos[2]:.3f})")
        if ok and frames and args.video:
            os.makedirs(os.path.dirname(args.video), exist_ok=True)
            frames[0].save(args.video, save_all=True, append_images=frames[1:],
                           duration=int(1000 / args.fps), loop=0, optimize=True)
            print(f"gif: {args.video}")
            rend = None
    print(f"\nFULL-CLIMB-FROM-FLOOR (full-hull scene): {n_ok}/{args.episodes}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
