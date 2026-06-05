"""G1 walk robustness stress test: long horizon + changing commands + fall detection.

The base g1_walk demo runs a few seconds forward. This drives the SAME pretrained policy for
minutes through a changing command schedule (forward / turn / sidestep / stop / fast) and watches
for a fall (pelvis drop / large tilt), reporting how long it stayed upright. NVIDIA-free, CPU.

Usage: python g1_walk_stress.py [--secs 120]
"""
import os
import argparse
import importlib.util
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("g1_walk", os.path.join(HERE, "g1_walk.py"))
gw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gw)

# command schedule: (duration_s, [vx, vy, wz]); cycles
SCHEDULE = [
    (6, [0.5, 0.0, 0.0]),     # forward
    (4, [0.3, 0.0, 0.6]),     # turn left while walking
    (4, [0.3, 0.0, -0.6]),    # turn right
    (4, [0.0, -0.3, 0.0]),    # sidestep
    (3, [0.0, 0.0, 0.0]),     # stop / stand
    (5, [0.7, 0.0, 0.0]),     # fast forward
    (4, [0.4, 0.0, 0.8]),     # sharp turn
]
CYCLE = sum(s for s, _ in SCHEDULE)


def sched_cmd(t):
    p = t % CYCLE
    for dur, c in SCHEDULE:
        if p < dur:
            return c
        p -= dur
    return SCHEDULE[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=120.0)
    args = ap.parse_args()

    m, d, policy = gw.make()
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    state = {"min_z": 9.0, "max_tilt": 0.0, "fell_at": None, "k": 0, "dist0": None}

    def watch(dd):
        z = dd.qpos[2]
        state["min_z"] = min(state["min_z"], z)
        w, x, y, zq = dd.qpos[3:7]
        roll = np.degrees(np.arctan2(2*(w*x+y*zq), 1-2*(x*x+y*y)))
        pitch = np.degrees(np.arcsin(np.clip(2*(w*y-zq*x), -1, 1)))
        state["max_tilt"] = max(state["max_tilt"], abs(roll), abs(pitch))
        if state["fell_at"] is None and (z < 0.4 or abs(roll) > 50 or abs(pitch) > 50):
            state["fell_at"] = state["k"] * gw.SIM_DT
        state["k"] += 1

    steps = int(args.secs / gw.SIM_DT)
    gw.walk(m, d, policy, sched_cmd, steps, log=watch)

    survived = state["fell_at"] is None and d.qpos[2] > 0.5
    print(f"stress: {args.secs:.0f}s, changing cmd schedule (cycle {CYCLE}s)")
    print(f"  travelled: ({d.qpos[0]:+.1f}, {d.qpos[1]:+.1f}) m   min pelvis z={state['min_z']:.3f}   max tilt={state['max_tilt']:.0f} deg")
    if state["fell_at"] is not None:
        print(f"  FELL at t={state['fell_at']:.1f}s")
    print(f"RESULT: {'STAYED UPRIGHT for the full run ✓' if survived else 'FELL ✗'}")
    return 0 if survived else 1


if __name__ == "__main__":
    raise SystemExit(main())
