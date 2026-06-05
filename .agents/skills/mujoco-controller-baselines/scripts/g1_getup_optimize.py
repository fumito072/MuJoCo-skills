"""Offline trajectory optimization for G1 floor-sit -> STAND (get-up), via CEM.

Reverse of g1_sitdown: the robot starts settled in the long-sit on the floor and must rise to
standing without toppling. Rising is the harder support transfer (buttocks -> feet, against
gravity). Same harness: optimize 2 intermediate knots (6 symmetric channels each), endpoints
fixed (settled floor-sit -> standing home pose). Reuses helpers from g1_sitdown_optimize.

CPU-only, NVIDIA-free.

STATUS (2026-06-05): this CEM harness does NOT yet solve floor get-up. Three settings were
tried (slow quasi-static T=4 + binary cost; slow + continuous cost + reversed-sit seed; fast
T=2 for momentum) — all topple (the robot collapses forward, pelvis stays ~0.06). Unlike
sit-down (where gravity assists and CEM solved it in ~65 s), rising from a legs-extended floor
sit is a multi-phase, momentum/contact-dependent maneuver that 2-knot open-loop position
trajectories cannot express. Known-hard in humanoid RL (dedicated "getup" tasks exist).
Paths that should work: (1) a pretrained getup policy (mirrors the WALK solution); (2) a richer
optimizer — more knots + arms-planting-on-ground push + contact scheduling, or a dynamic
optimizer (mjpc). Note: basic standing-up is already covered by g1_squat.py (sit-to-stand from a
squat) — only deep-floor-sit get-up is unsolved here. This harness is kept as a starting point.

Usage: python g1_getup_optimize.py [scene.xml] [--gen 25] [--pop 48] [--out traj.npz]
"""
import os
import argparse
import importlib.util
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("opt", os.path.join(HERE, "g1_sitdown_optimize.py"))
opt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt)

HOME_CH = np.array([-0.1, 0.0, 0.3, -0.2, 0.0, 0.2])   # standing leg/waist/shoulder channels
# rising is dynamic (needs momentum) — a fast push beats a slow quasi-static rise
T_RISE, T_SETTLE = 2.0, 3.0


class GetupSim:
    def __init__(self, scene):
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.home = self.d.qpos[7:].copy()
        self.pid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.dt = self.m.opt.timestep
        # settle into the floor-sit to get a realistic seated START state
        seat = opt.pose_from_channels(self.home, opt.SEATED_CH)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.d.qpos[0:3] = [0, 0, 0.2]
        self.d.qpos[3:7] = [1, 0, 0, 0]
        self.d.qpos[7:] = seat
        mujoco.mj_forward(self.m, self.d)
        for _ in range(int(1.5 / self.dt)):
            self.d.ctrl[:] = seat
            mujoco.mj_step(self.m, self.d)
        self.q0 = self.d.qpos.copy()
        self.v0 = self.d.qvel.copy()

    def rollout(self, params, record=False):
        m, d = self.m, self.d
        k1 = opt.pose_from_channels(self.home, np.clip(params[:6], opt.CH_LO, opt.CH_HI))
        k2 = opt.pose_from_channels(self.home, np.clip(params[6:], opt.CH_LO, opt.CH_HI))
        seat = opt.pose_from_channels(self.home, opt.SEATED_CH)
        stand = opt.pose_from_channels(self.home, HOME_CH)
        wps = [seat, k1, k2, stand]
        T = [0.0, T_RISE * 0.35, T_RISE * 0.7, T_RISE]
        d.qpos[:] = self.q0
        d.qvel[:] = self.v0
        mujoco.mj_forward(m, d)
        n = int((T_RISE + T_SETTLE) / self.dt)
        max_tilt = 0.0
        max_pz_upright = 0.0          # highest pelvis reached while still ~upright
        tilt_over = 0.0               # continuous topple signal
        traj = [] if record else None
        for i in range(n):
            t = min(i * self.dt, T_RISE)
            for k in range(len(T) - 1):
                if t <= T[k + 1]:
                    tgt = opt.smooth(wps[k], wps[k + 1], (t - T[k]) / (T[k + 1] - T[k]))
                    break
            else:
                tgt = wps[-1]
            d.ctrl[:] = tgt
            mujoco.mj_step(m, d)
            w, x, y, z = d.qpos[3:7]
            roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
            pitch = np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
            tilt = max(abs(roll), abs(pitch))
            max_tilt = max(max_tilt, tilt)
            if tilt < 40:
                max_pz_upright = max(max_pz_upright, d.xpos[self.pid][2])
            tilt_over += max(0.0, tilt - 50) * self.dt
            if record:
                traj.append(tgt.copy())
        pz = d.xpos[self.pid][2]
        w, x, y, z = d.qpos[3:7]
        f_roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
        f_pitch = np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
        settle = np.linalg.norm(d.qvel[:6])
        toppled = max_tilt > 80
        # CONTINUOUS cost so CEM has a gradient even when everything topples:
        # reward (a) the highest pelvis reached while upright and (b) ending upright+high.
        cost = (10.0 * max(0, 0.74 - max_pz_upright)   # get as high as possible, upright
                + 4.0 * max(0, 0.74 - pz)              # end standing
                + 0.05 * (abs(f_roll) + abs(f_pitch))  # end upright
                + 0.5 * tilt_over                      # continuous topple penalty
                + 0.1 * settle)
        info = dict(pz=pz, f_roll=f_roll, f_pitch=f_pitch, max_tilt=max_tilt,
                    settle=settle, toppled=toppled, traj=traj)
        return cost, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="/tmp/mjm/unitree_g1/scene.xml")
    ap.add_argument("--gen", type=int, default=25)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "assets", "g1_getup_traj.npz"))
    args = ap.parse_args()

    sim = GetupSim(args.scene)
    print(f"seated start: pelvis z={sim.d.xpos[sim.pid][2]:.3f}")
    # seed from the REVERSE of the sit-down trajectory (a plausible get-up guess)
    seed = None
    sd = os.path.join(HERE, "..", "assets", "g1_sitdown_traj.npz")
    if os.path.exists(sd):
        sp = np.load(sd)["params"]
        seed = np.concatenate([sp[6:], sp[:6]])   # reverse knot order
        print("seeded CEM from reversed sit-down trajectory")
    print(f"optimizing G1 get-up: dim=12, gen={args.gen}, pop={args.pop} "
          f"(~{args.gen*args.pop*0.09:.0f}s)")
    cost, params, info = opt.cem(sim, 12, args.gen, args.pop, seed_mean=seed)
    print(f"\nBEST cost={cost:.3f}: pz={info['pz']:.3f} final_tilt="
          f"{abs(info['f_roll'])+abs(info['f_pitch']):.1f} max_tilt={info['max_tilt']:.1f} "
          f"toppled={info['toppled']}")
    success = (info['pz'] > 0.70 and not info['toppled']
               and abs(info['f_roll']) < 20 and abs(info['f_pitch']) < 25)
    print(f"RESULT: {'GET-UP FOUND ✓' if success else 'not yet — more gens/pop or tune cost'}")
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, params=params, t_rise=T_RISE, t_settle=T_SETTLE)
    print(f"saved trajectory params -> {out}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
