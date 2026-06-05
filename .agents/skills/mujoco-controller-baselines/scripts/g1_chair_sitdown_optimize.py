"""CEM trajectory optimization for G1 stand -> SIT ON A CHAIR (final-goal stepping stone).

STATUS (2026-06-05): does NOT yet work. We expected chair-sit to be EASIER than floor-sit
(shallower descent), but it is HARDER in open-loop control: the raised seat is a small target
the drifting base keeps missing, and — verified directly — the moment the buttocks contact the
seat, that contact force perturbs the (otherwise stable) squat balance and TOPPLES the robot.
A stable squat (g1_squat, never topples) tips over as soon as a seat touches the buttocks.
So chair sit-down is a balance-critical SUPPORT-TRANSFER problem, the same hard class as floor
get-up — not the quick win first assumed. CEM here drives the cost down but stays toppled.
Reliable path: a pretrained sit/getup policy (mirrors WALK), or closed-loop CoM control.
This harness is kept as a starting point (reuses g1_sitdown_optimize CEM + g1_chair model).

Usage: python g1_chair_sitdown_optimize.py [scene.xml] [--gen 20] [--pop 40]
"""
import os
import argparse
import importlib.util
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


opt = _load("g1_sitdown_optimize")
chair = _load("g1_chair")
T_DESC, T_SETTLE = 3.0, 2.0


class ChairSim:
    def __init__(self, scene):
        self.m = chair.build_chaired_model(scene)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.home = self.d.qpos[7:].copy()
        self.pid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.dt = self.m.opt.timestep

    def rollout(self, params, record=False):
        m, d = self.m, self.d
        k1 = opt.pose_from_channels(self.home, np.clip(params[:6], opt.CH_LO, opt.CH_HI))
        k2 = opt.pose_from_channels(self.home, np.clip(params[6:], opt.CH_LO, opt.CH_HI))
        sit = opt.pose_from_channels(self.home, chair.CHAIR_SIT_CH)
        wps = [self.home, k1, k2, sit]
        T = [0.0, T_DESC * 0.4, T_DESC * 0.75, T_DESC]
        mujoco.mj_resetDataKeyframe(m, d, 0)
        n = int((T_DESC + T_SETTLE) / self.dt)
        tilt_over = 0.0
        max_tilt = 0.0
        traj = [] if record else None
        for i in range(n):
            t = min(i * self.dt, T_DESC)
            for k in range(len(T) - 1):
                if t <= T[k + 1]:
                    tgt = opt.smooth(wps[k], wps[k + 1], (t - T[k]) / (T[k + 1] - T[k]))
                    break
            else:
                tgt = wps[-1]
            d.ctrl[:] = tgt
            mujoco.mj_step(m, d)
            roll, pitch = chair.base_tilt(d)
            tilt = max(abs(roll), abs(pitch))
            max_tilt = max(max_tilt, tilt)
            tilt_over += max(0.0, tilt - 50) * self.dt
            if record:
                traj.append(tgt.copy())
        pz = d.xpos[self.pid][2]
        f_roll, f_pitch = chair.base_tilt(d)
        cforce = chair.seat_contact_force(m, d)
        settle = np.linalg.norm(d.qvel[:6])
        toppled = max_tilt > 80
        # want: chair bearing weight (cforce high) + upright + not collapsed below the seat
        cost = (6.0 * max(0, 1 - cforce / 150.0)
                + 0.05 * (abs(f_roll) + abs(f_pitch))
                + 0.3 * tilt_over
                + 0.1 * settle
                + 8.0 * max(0, 0.40 - pz)
                + (15.0 if toppled else 0.0))
        info = dict(pz=pz, f_roll=f_roll, f_pitch=f_pitch, max_tilt=max_tilt,
                    cforce=cforce, settle=settle, toppled=toppled, traj=traj)
        return cost, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="/tmp/mjm/unitree_g1/scene.xml")
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "assets", "g1_chair_sitdown_traj.npz"))
    args = ap.parse_args()

    sim = ChairSim(args.scene)
    print(f"optimizing G1 sit-on-chair: dim=12, gen={args.gen}, pop={args.pop}")
    # patch opt.cem's printed info to also show cforce
    best = opt.cem(sim, 12, args.gen, args.pop)
    cost, params, info = best
    print(f"\nBEST cost={cost:.3f}: pelvisZ={info['pz']:.3f} chairF={info['cforce']:.0f}N "
          f"final_tilt={abs(info['f_roll'])+abs(info['f_pitch']):.1f} toppled={info['toppled']}")
    ok = info['cforce'] > 120 and not info['toppled'] and abs(info['f_pitch']) < 25 and abs(info['f_roll']) < 20
    print(f"RESULT: {'SITS ON CHAIR ✓' if ok else 'not yet — tune/more gens'}")
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, params=params)
    print(f"saved -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
