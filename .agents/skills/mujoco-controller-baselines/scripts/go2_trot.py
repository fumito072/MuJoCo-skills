"""GO2 trot via a CPG-style foot trajectory + analytic 2-link leg IK + software PD.

Model-based, NVIDIA-free, CPU-only (Apple Silicon verified). No RL, no mjpc.
Pipeline per physics step (500 Hz):
  global phase -> per-leg foot target (stance push-back / swing lift)
  -> sagittal 2-link IK (thigh,calf) -> tau = kp*(q_des-q) - kd*qd, clip to ctrlrange.

Trot = diagonal pairs (FL,RR) and (FR,RL) half a cycle out of phase.
Leg geometry (from Menagerie unitree_go2): L1=L2=0.213 m. IK verified to reproduce
the home pose (thigh=0.9, calf=-1.8) at foot (x=0, z=-0.265).

Usage: python go2_trot.py [scene.xml] [--secs 5] [--freq 2.5] [--video out.mp4]
"""
import sys
import argparse
import mujoco
import numpy as np

L1 = L2 = 0.213
LEGS = ["FL", "FR", "RL", "RR"]
PHASE_OFFSET = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}  # diagonal trot


def leg_ik(x, z, L1=L1, L2=L2):
    """Sagittal 2-link IK. x fwd, z up (foot below hip => z<0). Returns (thigh, calf)."""
    r2 = x * x + z * z
    c2 = np.clip((r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1.0, 1.0)
    q2 = -np.arccos(c2)                       # knee bends backward (negative)
    k1 = L1 + L2 * np.cos(q2)
    k2 = L2 * np.sin(q2)
    den = k1 * k1 + k2 * k2
    s1 = (-k1 * x + k2 * z) / den
    c1 = (-k2 * x - k1 * z) / den
    q1 = np.arctan2(s1, c1)
    return q1, q2


def foot_target(phase, h0, x_amp, lift, duty):
    """Foot (x,z) in hip sagittal frame for a leg at the given normalized phase."""
    p = phase % 1.0
    if p < duty:                              # stance: on ground, push backward
        s = p / duty
        return x_amp * (1 - 2 * s), -h0
    s = (p - duty) / (1 - duty)               # swing: lift and swing forward
    return x_amp * (-1 + 2 * s), -h0 + lift * np.sin(np.pi * s)


def trot(m, d, cmd, steps, freq=2.0, xamp=0.08, h0=0.26, lift=0.10, duty=0.5,
         kp=80.0, kd=3.0, turn_gain=0.7, vx_nom=0.23, log=None):
    """Steerable GO2 trot. `cmd` is [vx, vy, wz] or a callable(t)->[vx,vy,wz].
    vx scales stride length; wz turns via a left/right stride differential (skid-steer-like).
    This is the velocity-command interface an obstacle-avoidance planner drives.
    """
    flim = m.actuator_ctrlrange[:, 1].copy()
    for c in range(steps):
        t = c * m.opt.timestep
        vx, vy, wz = (cmd(t) if callable(cmd) else cmd)
        gphase = t * freq
        stride = xamp * float(np.clip(abs(vx) / vx_nom, 0.15, 1.4))
        fwd = 1.0 if vx >= 0 else -1.0
        q_des = np.zeros(12)
        for i, leg in enumerate(LEGS):
            side = 1.0 if leg in ("FL", "RL") else -1.0      # left vs right
            # wz>0 => CCW/left (matches the G1 walk convention so one planner fits both):
            xa = stride * (1.0 - side * turn_gain * wz)       # differential -> yaw
            x, z = foot_target(gphase + PHASE_OFFSET[leg], h0, fwd * xa, lift, duty)
            thigh, calf = leg_ik(x, z)
            q_des[3 * i + 1] = thigh
            q_des[3 * i + 2] = calf
        d.ctrl[:] = np.clip(kp * (q_des - d.qpos[7:]) - kd * d.qvel[6:], -flim, flim)
        mujoco.mj_step(m, d)
        if log is not None:
            log(d)


def heading_cmd(d, speed=0.3, heading=0.0, kyaw=2.5, wz_max=0.6):
    """Closed-loop heading hold: corrects the open-loop trot's yaw drift so the GO2 walks a
    straight line (or any commanded heading) at a commanded forward speed. Pass this as the
    `cmd` to trot(); it reads the live MjData yaw each step and steers wz to null the error.
    """
    def yaw(dd):
        w, x, y, z = dd.qpos[3:7]
        return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def cmd(_t=0.0):
        err = (heading - yaw(d) + np.pi) % (2*np.pi) - np.pi
        return [speed, 0.0, float(np.clip(kyaw * err, -wz_max, wz_max))]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="/tmp/mjm/unitree_go2/scene.xml")
    # Defaults tuned on M5 Max: clean forward trot ~0.22 m/s. The key lesson:
    # duty=0.5 (true trot, no stance overlap) + enough lift (0.10) so feet actually
    # clear the ground. duty=0.6/lift=0.06 gave 99% ground contact (dragging) -> backward.
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--freq", type=float, default=2.0)
    ap.add_argument("--h0", type=float, default=0.26)
    ap.add_argument("--xamp", type=float, default=0.08)
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--kp", type=float, default=80.0)
    ap.add_argument("--kd", type=float, default=3.0)
    ap.add_argument("--video", type=str, default="")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    flim = m.actuator_ctrlrange[:, 1].copy()

    # IK self-check against the home pose
    q1h, q2h = leg_ik(0.0, -0.2648)
    print(f"[IK check] home foot(0,-0.2648) -> thigh={q1h:.3f} calf={q2h:.3f} "
          f"(expect ~0.900,-1.800)")

    def control(t):
        gphase = t * args.freq
        q_des = np.zeros(12)
        for i, leg in enumerate(LEGS):
            x, z = foot_target(gphase + PHASE_OFFSET[leg], args.h0,
                               args.xamp, args.lift, args.duty)
            thigh, calf = leg_ik(x, z)
            q_des[3 * i + 0] = 0.0      # hip abduction
            q_des[3 * i + 1] = thigh
            q_des[3 * i + 2] = calf
        tau = args.kp * (q_des - d.qpos[7:]) - args.kd * d.qvel[6:]
        return np.clip(tau, -flim, flim)

    renderer = frames = None
    if args.video:
        renderer = mujoco.Renderer(m, height=480, width=640)
        frames, fps = [], 50

    n = int(args.secs / m.opt.timestep)
    x0 = d.qpos[0]
    zs, rolls, pitches, vxs = [], [], [], []
    for k in range(n):
        t = k * m.opt.timestep
        d.ctrl[:] = control(t)
        mujoco.mj_step(m, d)
        zs.append(d.qpos[2])
        w, x, y, z = d.qpos[3:7]
        rolls.append(np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))))
        pitches.append(np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1))))
        vxs.append(d.qvel[0])
        if renderer is not None and k % (int(1/(50*m.opt.timestep))) == 0:
            renderer.update_scene(d, camera=-1)
            frames.append(renderer.render())

    zs = np.array(zs); vxs = np.array(vxs)
    dx = d.qpos[0] - x0
    fell = not (zs[-1] > 0.18 and abs(rolls[-1]) < 35 and abs(pitches[-1]) < 35)
    print(f"\ngait: freq={args.freq}Hz xamp={args.xamp} lift={args.lift} duty={args.duty} "
          f"kp={args.kp} kd={args.kd}")
    print(f"forward: dx={dx:+.3f} m in {args.secs}s -> {dx/args.secs:+.3f} m/s "
          f"(mean vx={vxs[len(vxs)//5:].mean():+.3f} m/s)")
    print(f"body: z end={zs[-1]:.3f} min={zs.min():.3f}  roll={rolls[-1]:+.1f} "
          f"pitch={pitches[-1]:+.1f}  lateral drift={d.qpos[1]:+.3f} m")
    walked = (not fell) and dx > 0.15
    print(f"RESULT: {'TROTS FORWARD ✓' if walked else ('STAYS UP but little progress' if not fell else 'FELL ✗')}")

    if args.video and frames:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(args.video, frames, fps=50)
            print(f"video -> {args.video} ({len(frames)} frames)")
        except Exception as e:
            np.save(args.video + ".frames.npy", np.array(frames))
            print(f"imageio unavailable ({e}); saved frames to {args.video}.frames.npy")
    return 0 if walked else 1


if __name__ == "__main__":
    sys.exit(main())
