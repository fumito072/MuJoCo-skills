"""Unitree G1 floor sitting (long-sit) on the 29-DOF position-actuator model.

STATUS (verified on M5 Max):
  * The floor-sit POSE is stable: legs extended forward, torso upright, pelvis on the
    ground (z~0.15). The robot settles into and holds it cleanly. (mode=hold)
  * The controlled STAND -> FLOOR-SIT transition is NOT solved by open-loop position
    control. ~10 trajectory strategies were tried; all either fold forward (stay on feet)
    or topple when forced down. Floor sit-down is a balance-critical, support-transfer
    maneuver (feet -> buttocks) in the same difficulty class as bipedal walking, and needs
    closed-loop CoM/ZMP control, trajectory optimization, or a pretrained sit/getup policy
    (cf. how g1 WALK uses a pretrained policy). See references/g1-sit-recipe.md. (mode=sitdown)

This is a stepping stone toward chair-sitting. The stable seated pose is reusable; the
descent controller is the open problem.

Usage:
  python g1_sit.py [scene.xml] --mode hold      # settle into & hold the floor-sit (works)
  python g1_sit.py [scene.xml] --mode sitdown   # best-effort descent (honest: usually fails)
"""
import sys
import argparse
import mujoco
import numpy as np

# Verified stable long-sit leg angles (per leg): hip_pitch, knee, ankle_pitch
SIT_HIP, SIT_KNEE, SIT_ANKLE = -1.57, 0.2, 0.0


def leg_pose(home, hip, knee, ankle, waistp=0.0):
    t = home.copy()
    for s in (0, 6):                       # left legs 0-5, right legs 6-11
        t[s + 0] = hip
        t[s + 3] = knee
        t[s + 4] = ankle
    t[14] = waistp                          # waist_pitch
    return t


def base_state(d):
    w, x, y, z = d.qpos[3:7]
    roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
    return roll, pitch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="/tmp/mjm/unitree_g1/scene.xml")
    ap.add_argument("--mode", choices=["hold", "sitdown"], default="hold")
    ap.add_argument("--secs", type=float, default=4.0)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    home = d.qpos[7:].copy()
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    sit = leg_pose(home, SIT_HIP, SIT_KNEE, SIT_ANKLE)

    if args.mode == "hold":
        # place the robot in the floor-sit and let it settle / hold (verified stable)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[0:3] = [0, 0, 0.2]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[7:] = sit
        mujoco.mj_forward(m, d)
        for _ in range(int(args.secs / m.opt.timestep)):
            d.ctrl[:] = sit
            mujoco.mj_step(m, d)
        roll, pitch = base_state(d)
        pz = d.xpos[pid][2]
        ok = pz < 0.25 and abs(roll) < 25 and abs(pitch) < 30
        print(f"[hold] floor-sit: pelvis z={pz:.3f}  roll={roll:+.1f} pitch={pitch:+.1f}")
        print(f"RESULT: {'SITS STABLY ✓' if ok else 'UNSTABLE ✗'}")
        return 0 if ok else 1

    # mode == sitdown: best-effort open-loop descent (documented to be unreliable)
    def smooth(a, b, u):
        u = np.clip(u, 0, 1)
        return a + (b - a) * (u*u*(3-2*u))
    wps = [home,
           leg_pose(home, -0.8, 1.6, -0.8, 0.3),   # deep squat + lean
           sit]
    T = [0, 3, 7]
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for i in range(int(T[-1] / m.opt.timestep)):
        t = i * m.opt.timestep
        for k in range(len(T)-1):
            if t <= T[k+1]:
                tgt = smooth(wps[k], wps[k+1], (t-T[k])/(T[k+1]-T[k]))
                break
        else:
            tgt = wps[-1]
        d.ctrl[:] = tgt
        mujoco.mj_step(m, d)
    roll, pitch = base_state(d)
    pz = d.xpos[pid][2]
    ok = pz < 0.25 and abs(roll) < 25 and abs(pitch) < 35
    print(f"[sitdown] final pelvis z={pz:.3f} roll={roll:+.1f} pitch={pitch:+.1f}")
    print(f"RESULT: {'SAT DOWN ✓' if ok else 'FAILED (expected — open-loop descent is unsolved; see recipe)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
