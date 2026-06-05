"""Offline trajectory optimization for the G1 stand -> floor-sit descent (approach B).

Hand-scripting this balance-critical transition fails (see g1-sit-recipe.md). Instead we
SEARCH for a working open-loop trajectory: the descent goes stand (fixed) -> knot1 -> knot2 ->
floor-sit (fixed, verified-stable pose). The two intermediate knots (6 symmetric channels each:
hip_pitch, hip_roll, knee, ankle_pitch, waist_pitch, shoulder_pitch) are optimized by CEM
(cross-entropy method) so the robot descends to the seated pose WITHOUT toppling.

Deterministic sim, CPU-only, NVIDIA-free. Each rollout ~90 ms on M5 Max. The result is a
sim-valid open-loop trajectory (to be hardened with feedback for real-robot transfer later).

Usage: python g1_sitdown_optimize.py [scene.xml] [--gen 20] [--pop 40] [--out traj.npz]
"""
import os
import argparse
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

# 6 symmetric channels and their plausible bounds (radians)
CH_LO = np.array([-1.9, 0.0, 0.0, -0.9, -0.2, -1.2])
CH_HI = np.array([0.0, 0.4, 2.6, 0.5, 0.7, 1.5])
SEATED_CH = np.array([-1.57, 0.0, 0.2, 0.0, 0.0, 0.2])   # verified stable long-sit
T_DESC, T_SETTLE = 4.0, 2.0


def pose_from_channels(home, ch):
    hp, hr, kn, an, wp, sp = ch
    t = home.copy()
    t[0], t[1], t[3], t[4] = hp, +hr, kn, an      # left leg
    t[6], t[7], t[9], t[10] = hp, -hr, kn, an     # right leg (mirror roll)
    t[14] = wp                                     # waist_pitch
    t[15] = t[22] = sp                             # shoulder_pitch L/R
    return t


def smooth(a, b, u):
    u = np.clip(u, 0, 1)
    return a + (b - a) * (u * u * (3 - 2 * u))


class Sim:
    def __init__(self, scene):
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.home = self.d.qpos[7:].copy()
        self.pid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.dt = self.m.opt.timestep

    def rollout(self, params, record=False):
        m, d = self.m, self.d
        k1 = pose_from_channels(self.home, np.clip(params[:6], CH_LO, CH_HI))
        k2 = pose_from_channels(self.home, np.clip(params[6:], CH_LO, CH_HI))
        seat = pose_from_channels(self.home, SEATED_CH)
        wps = [self.home, k1, k2, seat]
        T = [0.0, T_DESC * 0.4, T_DESC * 0.75, T_DESC]
        mujoco.mj_resetDataKeyframe(m, d, 0)
        n = int((T_DESC + T_SETTLE) / self.dt)
        max_tilt = 0.0
        traj = [] if record else None
        for i in range(n):
            t = min(i * self.dt, T_DESC)
            for k in range(len(T) - 1):
                if t <= T[k + 1]:
                    tgt = smooth(wps[k], wps[k + 1], (t - T[k]) / (T[k + 1] - T[k]))
                    break
            else:
                tgt = wps[-1]
            d.ctrl[:] = tgt
            mujoco.mj_step(m, d)
            w, x, y, z = d.qpos[3:7]
            roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
            pitch = np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
            max_tilt = max(max_tilt, abs(roll), abs(pitch))
            if record:
                traj.append(tgt.copy())
        pz = d.xpos[self.pid][2]
        w, x, y, z = d.qpos[3:7]
        f_roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
        f_pitch = np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
        settle = np.linalg.norm(d.qvel[:6])
        toppled = max_tilt > 80
        cost = (12.0 * max(0, pz - 0.18)
                + 0.05 * (abs(f_roll) + abs(f_pitch))
                + 0.02 * max(0, max_tilt - 45)
                + 0.15 * settle
                + (40.0 if toppled else 0.0))
        info = dict(pz=pz, f_roll=f_roll, f_pitch=f_pitch, max_tilt=max_tilt,
                    settle=settle, toppled=toppled, traj=traj)
        return cost, info


def cem(sim, dim, gen, pop, elite_frac=0.25, seed_mean=None):
    mean = np.zeros(dim) if seed_mean is None else seed_mean.copy()
    # init sigma per dim ~ a quarter of the channel range, tiled for 2 knots
    rng_scale = (CH_HI - CH_LO) * 0.25
    sigma = np.tile(rng_scale, 2)
    n_elite = max(2, int(pop * elite_frac))
    best = (1e9, None, None)
    # deterministic-ish sampling without Math.random: use a fixed-seed numpy Generator
    g = np.random.default_rng(0)
    for it in range(gen):
        samples = mean + sigma * g.standard_normal((pop, dim))
        scored = []
        for s in samples:
            c, info = sim.rollout(s)
            scored.append((c, s, info))
        scored.sort(key=lambda r: r[0])
        if scored[0][0] < best[0]:
            best = (scored[0][0], scored[0][1].copy(), scored[0][2])
        elite = np.array([s for _, s, _ in scored[:n_elite]])
        mean = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 0.02
        bi = best[2]
        print(f"gen {it:2d}  best_cost={best[0]:6.3f}  pz={bi['pz']:.3f} "
              f"tilt(final {abs(bi['f_roll'])+abs(bi['f_pitch']):.0f}, max {bi['max_tilt']:.0f}) "
              f"toppled={bi['toppled']}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="/tmp/mjm/unitree_g1/scene.xml")
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "assets", "g1_sitdown_traj.npz"))
    args = ap.parse_args()

    sim = Sim(args.scene)
    print(f"optimizing G1 sit-down: dim=12, gen={args.gen}, pop={args.pop} "
          f"(~{args.gen*args.pop*0.09:.0f}s)")
    cost, params, info = cem(sim, 12, args.gen, args.pop)
    print(f"\nBEST cost={cost:.3f}: pz={info['pz']:.3f} final_tilt="
          f"{abs(info['f_roll'])+abs(info['f_pitch']):.1f} max_tilt={info['max_tilt']:.1f} "
          f"toppled={info['toppled']}")
    success = (info['pz'] < 0.22 and not info['toppled']
               and abs(info['f_roll']) < 25 and abs(info['f_pitch']) < 30)
    print(f"RESULT: {'SIT-DOWN FOUND ✓' if success else 'not yet — tune cost/params or more gens'}")
    if success or True:
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, params=params, seated_ch=SEATED_CH,
                 t_desc=T_DESC, t_settle=T_SETTLE)
        print(f"saved trajectory params -> {out}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
