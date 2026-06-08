"""Replay a Colab-trained G1Sit policy locally in MuJoCo on the Mac — CPU, NVIDIA-free.

This is the INFERENCE half of "train once on GPU (Colab), run local on Mac". It loads the brax PPO
params (normalizer + policy) saved by ``g1_train_colab.py --task sit``, rebuilds the policy with the
SAME network config the training used (pulled from ``locomotion_params`` so it matches exactly), and
rolls it out in PLAIN MuJoCo (the C engine — no MJX, no JAX physics) on the Playground G1 'feetonly'
flat-terrain model.

The observation vector and action mapping reproduce ``mujoco_playground``'s G1 Joystick env EXACTLY
(same sensors, same 103-dim layout, same ``motor_targets = default_pose + action*0.5``) — only the
observation noise is turned OFF, for a clean deterministic replay. The command is held at zero (the
sit task), so the policy should lower the base toward SIT_HEIGHT and hold it upright without toppling.

It prints sit metrics (pelvis height over time, final uprightness) and writes a GIF.

Run with the RL venv (it has playground + brax + jax + mujoco):
  .venv-rl/bin/python training/g1_sit_play.py \
      --params models/policies/g1_sit_params.pkl --video assets/g1_sit.gif --seconds 6
"""
import argparse
import os
import pickle

import numpy as np
import mujoco
import jax
import jax.numpy as jp

from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

# Must match the G1 Joystick env (mujoco_playground .../g1/joystick.py default_config).
CTRL_DT, SIM_DT, ACTION_SCALE = 0.02, 0.002, 0.5
DECIMATION = int(round(CTRL_DT / SIM_DT))      # 10 sim substeps per control step
GAIT_FREQ = 1.375                              # midpoint of training U(1.25,1.5); command=0 for sit
STATE_SIZE, PRIV_SIZE = 103, 216               # policy obs ("state") / value obs ("privileged_state")


def build_model():
    """Playground G1 feetonly flat-terrain model, loaded into the plain MuJoCo C engine."""
    m = mujoco.MjModel.from_xml_string(
        consts.FEET_ONLY_FLAT_TERRAIN_XML.read_text(), assets=g1_base.get_assets()
    )
    m.opt.timestep = SIM_DT
    return m


def load_policy(params_path, act_size):
    """Rebuild the trained policy: same network as training (from locomotion_params) + saved params."""
    nk = dict(locomotion_params.brax_ppo_config("G1JoystickFlatTerrain").network_factory)
    net = ppo_networks.make_ppo_networks(
        {"state": (STATE_SIZE,), "privileged_state": (PRIV_SIZE,)},
        act_size,
        preprocess_observations_fn=running_statistics.normalize,
        **nk,
    )
    with open(params_path, "rb") as f:
        params = pickle.load(f)               # (normalizer_params, policy_params, [value_params])
    # make_inference_fn uses params[0] (normalizer) + params[1] (policy) only.
    return ppo_networks.make_inference_fn(net)(params, deterministic=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="g1_sit_params.pkl from Colab")
    ap.add_argument("--video", default="assets/g1_sit.gif")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    m = build_model()
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    mujoco.mj_resetDataKeyframe(m, d, kid)
    mujoco.mj_forward(m, d)
    default_pose = d.qpos[7:].copy()

    gyro = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro_pelvis")
    lvel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "local_linvel_pelvis")
    imu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis")

    def sread(s):
        a = m.sensor_adr[s]
        return d.sensordata[a:a + m.sensor_dim[s]].copy()

    policy = load_policy(args.params, m.nu)

    # offscreen render (CGL, NVIDIA-free), camera tracking the floating base
    track = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if track < 0:
        track = m.body(consts.ROOT_BODY).id
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = track
    cam.distance, cam.azimuth, cam.elevation = 2.6, 120, -10
    rend = mujoco.Renderer(m, 360, 480)

    from PIL import Image
    last = np.zeros(m.nu)
    ph = np.array([0.0, np.pi])
    pdt = 2 * np.pi * CTRL_DT * GAIT_FREQ
    nsteps = int(args.seconds / CTRL_DT)
    every = max(1, int(round(1.0 / (args.fps * CTRL_DT))))
    frames, zs = [], []

    for k in range(nsteps):
        gravity = d.site_xmat[imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
        state = np.concatenate([
            sread(lvel),                       # local linvel (3)
            sread(gyro),                       # gyro (3)
            gravity,                           # projected gravity (3)
            np.zeros(3),                       # command = [0,0,0] (sit)
            d.qpos[7:] - default_pose,         # joint angles rel. default (29)
            d.qvel[6:],                        # joint velocities (29)
            last,                              # last action (29)
            [np.cos(ph[0]), np.cos(ph[1]), np.sin(ph[0]), np.sin(ph[1])],  # gait phase (4)
        ]).astype(np.float32)
        act, _ = policy({"state": jp.asarray(state)}, jax.random.PRNGKey(k))
        act = np.asarray(act)
        d.ctrl[:] = default_pose + act * ACTION_SCALE
        for _ in range(DECIMATION):
            mujoco.mj_step(m, d)
        last = act
        ph = (ph + pdt + np.pi) % (2 * np.pi) - np.pi
        zs.append(float(d.qpos[2]))
        if k % every == 0:
            rend.update_scene(d, camera=cam)
            frames.append(Image.fromarray(rend.render()).convert("P", palette=Image.ADAPTIVE, colors=64))

    w, x, y, zq = d.qpos[3:7]
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - zq * x), -1, 1)))
    pz = float(d.qpos[2])
    print(f"pelvis_z: start={zs[0]:.3f}  min={min(zs):.3f}  final={pz:.3f}  (sit target≈0.42)")
    print(f"final pitch={pitch:+.0f}deg  upright={abs(pitch) < 35}")
    seated = pz < 0.55 and abs(pitch) < 35
    print(f"RESULT: {'LOWERED & UPRIGHT (sit-like) ✓' if seated else 'did NOT settle low+upright (topple/stand?)'}")

    if args.video and frames:
        out = os.path.abspath(args.video)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.fps), loop=0, optimize=True)
        print(f"video -> {out} ({os.path.getsize(out) / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
