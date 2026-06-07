"""Optimize a G1 stand -> sit-ON-A-CHAIR descent, with a COLLIDABLE seat that catches the robot.

STATUS (2026-06-07): NOT SOLVED — sitting onto an elevated seat is a balance-critical support
transfer that open-loop control can't do. Verified across approaches: CEM topples, a backrest
doesn't save it, and pure squats only stay upright down to knee~1.6 (pelvis ~0.58 m, above chair
height) — deeper squats topple. The stable seated pose remains the CEM FLOOR sit-down
(g1_sitdown). A real chair-sit needs a learned / closed-loop balance policy (training = GPU). This
file is kept as the honest, reproducible attempt.


The earlier free-space chair-sit toppled (balance-critical support transfer). The key change here:
a physical seat box is present and collidable, so the descent can REST on it — the CEM only has to
find a knot trajectory (stand -> k1 -> k2-held) that lowers the pelvis onto the seat upright,
without toppling, while the seat bears the weight.

6 symmetric channels per knot (hip_pitch, hip_roll, knee, ankle_pitch, waist_pitch, shoulder_pitch),
2 knots = 12 params, CEM. Cost rewards: pelvis resting at seat height, over the seat, torso upright,
low tilt throughout, settled. Deterministic, CPU-only, NVIDIA-free.

Usage: python g1_chairsit_optimize.py [scene.xml] [--seat-h 0.40] [--seat-x -0.02] [--gen 20] [--pop 40] [--seed 0]
"""
import os
import argparse
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
CH_LO = np.array([-1.9, 0.0, 0.0, -1.0, -0.3, -1.2])
CH_HI = np.array([0.2, 0.4, 2.6, 0.6, 0.8, 1.5])
T_DESC, T_SETTLE = 4.0, 2.5
SEAT_HW = 0.16                                   # seat half-width


def pose_from_channels(home, ch):
    hp, hr, kn, an, wp, sp = ch
    t = home.copy()
    t[0], t[1], t[3], t[4] = hp, +hr, kn, an
    t[6], t[7], t[9], t[10] = hp, -hr, kn, an
    t[14] = wp
    t[15] = t[22] = sp
    return t


def smooth(a, b, u):
    u = np.clip(u, 0, 1)
    return a + (b - a) * (u * u * (3 - 2 * u))


def build(scene, seat_h, seat_x):
    sp = mujoco.MjSpec.from_file(scene)
    seat = sp.worldbody.add_body(name="seat", pos=[seat_x, 0.0, seat_h - 0.03])
    g = seat.add_geom()                                        # collidable seat — bears the weight
    g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [SEAT_HW, SEAT_HW, 0.03]
    g.rgba = [0.45, 0.32, 0.22, 1]; g.group = 5
    back = sp.worldbody.add_body(name="backrest",             # collidable backrest — catches the lean-back
                                 pos=[seat_x - SEAT_HW + 0.02, 0.0, seat_h + 0.22])
    bg = back.add_geom()
    bg.type = mujoco.mjtGeom.mjGEOM_BOX; bg.size = [0.03, SEAT_HW, 0.25]
    bg.rgba = [0.45, 0.32, 0.22, 1]; bg.group = 5
    return sp.compile()


class Sim:
    def __init__(self, scene, seat_h, seat_x):
        self.m = build(scene, seat_h, seat_x)
        self.d = mujoco.MjData(self.m)
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.kid = sid if sid >= 0 else 0
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.kid)
        self.home = self.d.qpos[7:].copy()
        self.pid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.dt = self.m.opt.timestep
        self.seat_h, self.seat_x = seat_h, seat_x

    def rollout(self, params, record=False):
        m, d = self.m, self.d
        k1 = pose_from_channels(self.home, np.clip(params[:6], CH_LO, CH_HI))
        k2 = pose_from_channels(self.home, np.clip(params[6:], CH_LO, CH_HI))
        wps = [self.home, k1, k2]
        T = [0.0, T_DESC * 0.5, T_DESC]
        mujoco.mj_resetDataKeyframe(m, d, self.kid)
        n = int((T_DESC + T_SETTLE) / self.dt)
        max_tilt = 0.0
        traj = [] if record else None
        for i in range(n):
            t = min(i * self.dt, T_DESC)
            tgt = wps[-1]
            for k in range(len(T) - 1):
                if t <= T[k + 1]:
                    tgt = smooth(wps[k], wps[k + 1], (t - T[k]) / (T[k + 1] - T[k])); break
            d.ctrl[:] = tgt
            mujoco.mj_step(m, d)
            w, x, y, z = d.qpos[3:7]
            roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
            pitch = np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
            max_tilt = max(max_tilt, abs(roll), abs(pitch))
            if record:
                traj.append(tgt.copy())
        px, py, pz = d.xpos[self.pid]
        w, x, y, z = d.qpos[3:7]
        f_roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
        f_pitch = np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
        settle = np.linalg.norm(d.qvel[:6])
        toppled = max_tilt > 75
        height_err = abs(pz - (self.seat_h + 0.06))         # pelvis just above seat top
        off_seat = max(0.0, abs(px - self.seat_x) - SEAT_HW)  # pelvis horizontally over the seat
        cost = (9.0 * height_err
                + 6.0 * off_seat
                + 0.06 * (abs(f_roll) + abs(f_pitch))
                + 0.02 * max(0, max_tilt - 40)
                + 0.2 * settle
                + (50.0 if toppled else 0.0))
        info = dict(pz=pz, px=px, f_roll=f_roll, f_pitch=f_pitch, max_tilt=max_tilt,
                    settle=settle, toppled=toppled, height_err=height_err, off_seat=off_seat, traj=traj)
        return cost, info


def cem(sim, gen, pop, seed=0, elite_frac=0.25):
    dim = 12
    mean = np.tile((CH_LO + CH_HI) / 2, 2)
    sigma = np.tile((CH_HI - CH_LO) * 0.3, 2)
    n_elite = max(2, int(pop * elite_frac))
    best = (1e9, None, None)
    g = np.random.default_rng(seed)
    for it in range(gen):
        samples = mean + sigma * g.standard_normal((pop, dim))
        results = [sim.rollout(s) for s in samples]
        scored = [(c, s, info) for (c, info), s in zip(results, samples)]
        scored.sort(key=lambda r: r[0])
        if scored[0][0] < best[0]:
            best = (scored[0][0], scored[0][1].copy(), scored[0][2])
        elite = np.array([s for _, s, _ in scored[:n_elite]])
        mean = elite.mean(axis=0); sigma = elite.std(axis=0) + 0.02
        bi = best[2]
        print(f"gen {it:2d} best={best[0]:6.3f} pz={bi['pz']:.3f} px={bi['px']:+.2f} "
              f"h_err={bi['height_err']:.3f} off={bi['off_seat']:.2f} "
              f"tilt(f{abs(bi['f_roll'])+abs(bi['f_pitch']):.0f},mx{bi['max_tilt']:.0f}) topp={bi['toppled']}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="/tmp/mjm/unitree_g1/scene.xml")
    ap.add_argument("--seat-h", type=float, default=0.40)
    ap.add_argument("--seat-x", type=float, default=-0.02)
    ap.add_argument("--gen", type=int, default=18)
    ap.add_argument("--pop", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sim = Sim(args.scene, args.seat_h, args.seat_x)
    print(f"chair-sit CEM: seat_h={args.seat_h} seat_x={args.seat_x} gen={args.gen} pop={args.pop} seed={args.seed}")
    cost, params, info = cem(sim, args.gen, args.pop, args.seed)
    ok = (not info['toppled'] and info['height_err'] < 0.08 and info['off_seat'] < 0.05
          and abs(info['f_roll']) < 25 and abs(info['f_pitch']) < 30)
    print(f"\nBEST cost={cost:.3f} pz={info['pz']:.3f} h_err={info['height_err']:.3f} "
          f"off_seat={info['off_seat']:.2f} final_tilt={abs(info['f_roll'])+abs(info['f_pitch']):.1f} toppled={info['toppled']}")
    print(f"RESULT: {'SAT ON CHAIR ✓' if ok else 'not yet'}")
    if args.out and params is not None:
        out = os.path.abspath(args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, params=params, seat_h=args.seat_h, seat_x=args.seat_x, t_desc=T_DESC, t_settle=T_SETTLE)
        print(f"saved -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
