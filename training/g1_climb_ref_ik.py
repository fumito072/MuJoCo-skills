"""IK post-process for the climb reference (2026-06-17). Diagnosis: during the
single-support phase the robot's CoM sits ~6-7 cm BEHIND the lead (R) foot, so the
one-leg rise tips backward and the climb stalls one-foot / z0.87. Hand-tuning the
keyframes failed (flexing the hip moves the FOOT, not the CoM, because foot and CoM
are coupled). This solves it properly: for each single-support frame, a damped
least-squares IK adjusts [base_y, R hip-pitch, R knee, R ankle-pitch] so that
  (a) the R foot STAYS planted at its original spot, and
  (b) the robot CoM_y is brought OVER the support foot.
i.e. move the pelvis forward over the lead foot while extending the leg to keep the
foot down. Refines g1_climb_reference.npz in place (single-support phase only).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco  # noqa: E402
import g1_sit_env  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "training", "g1_climb_reference.npz")
PH_LO, PH_HI = 0.48, 0.93        # single-support phase (R planted, L swinging)
J6, J9, J10 = 6, 9, 10           # R hip-pitch, R knee, R ankle-pitch

z = np.load(REF)
legs, base = z["legs"].copy(), z["base"].copy()
N = len(legs)
YAW = float(z["yaw"])

m = g1_sit_env.build_fbx_chair_model(0.002)
d = mujoco.MjData(m)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
rf = m.geom("right_foot").id


def fk(base_y, base_z, joints):
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.qpos[0:3] = (0.0, base_y, base_z)
    d.qpos[3:7] = (np.cos(YAW / 2), 0, 0, np.sin(YAW / 2))
    d.qpos[7:19] = joints
    mujoco.mj_forward(m, d)
    return d.geom_xpos[rf].copy(), d.subtree_com[pelvis].copy()


base_orig = base.copy()      # keep originals so we can taper the correction at edges
legs_orig = legs.copy()
refined = 0
for fr in range(N):
    ph = fr / N
    if not (PH_LO <= ph <= PH_HI):
        continue
    joints = legs[fr].copy()
    v_base_z = base[fr][1]                         # base z (held; foot_z objective covers height)
    # initial foot target = current R foot position (keep it planted there)
    foot0, com0 = fk(base[fr][0], v_base_z, joints)
    foot_tgt = foot0.copy()
    v = np.array([base[fr][0], joints[J6], joints[J9], joints[J10]], float)

    def res(v):
        j = joints.copy()
        j[J6], j[J9], j[J10] = v[1], v[2], v[3]
        foot, com = fk(v[0], v_base_z, j)
        # foot-planting is a HARD constraint (weight 6x): the foot must stay on the
        # platform. CoM-over-support is the softer objective it satisfies within that.
        return np.array([6.0 * (foot[1] - foot_tgt[1]), 6.0 * (foot[2] - foot_tgt[2]),
                         com[1] - foot[1]])

    for _ in range(30):
        r = res(v)
        if np.linalg.norm(r) < 1e-3:
            break
        J = np.zeros((3, 4))
        for i in range(4):
            dv = v.copy(); dv[i] += 1e-4
            J[:, i] = (res(dv) - r) / 1e-4
        lam = 1e-3
        delta = -np.linalg.solve(J.T @ J + lam * np.eye(4), J.T @ r)
        delta = np.clip(delta, -0.15, 0.15)        # step cap for stability
        v = v + delta
    base[fr][0] = v[0]
    joints[J6], joints[J9], joints[J10] = v[1], v[2], v[3]
    legs[fr] = joints
    refined += 1

# TAPER the IK correction to zero at the band edges so base_y/joints stay continuous
# with the un-refined frames outside the band (a hard edge injected a 124mm/frame =
# 6 m/s base-velocity spike that would wreck tracking). smoothstep ramp over RAMP frames.
band = [fr for fr in range(N) if PH_LO <= fr / N <= PH_HI]
if band:
    blo, bhi = band[0], band[-1]
    RAMP = 14
    for fr in band:
        e = min(fr - blo, bhi - fr) / RAMP
        w = np.clip(e, 0.0, 1.0)
        w = w * w * (3 - 2 * w)                     # smoothstep edge weight
        base[fr][0] = base_orig[fr][0] + w * (base[fr][0] - base_orig[fr][0])
        for j in (J6, J9, J10):
            legs[fr][j] = legs_orig[fr][j] + w * (legs[fr][j] - legs_orig[fr][j])
    # light 5-tap smoothing across the refined band to kill residual IK jitter
    k = np.ones(5) / 5.0
    base[band, 0] = np.convolve(np.pad(base[band, 0], 2, "edge"), k, "valid")
    for j in (J6, J9, J10):
        legs[np.array(band), j] = np.convolve(
            np.pad(legs[np.array(band), j], 2, "edge"), k, "valid")

np.savez_compressed(REF, legs=legs, base=base, dt=float(z["dt"]),
                    yaw=YAW, duration=float(z["duration"]))
print(f"IK-refined {refined} single-support frames (phase {PH_LO}-{PH_HI}) -> {REF}")
