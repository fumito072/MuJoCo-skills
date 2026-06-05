"""Inspect the Unitree GO2 MuJoCo model: DOF, joints, actuators, keyframes, feet.

Usage: python inspect_go2.py [path/to/scene.xml]
NVIDIA-free, CPU-only. Verified on Apple Silicon (mujoco 3.9.0).
"""
import sys
import mujoco
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mjm/unitree_go2/scene.xml"
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)

print(f"model: {path}")
print(f"nq={m.nq} nv={m.nv} nu={m.nu}  timestep={m.opt.timestep}  gravity={m.opt.gravity}")

print("\n--- joints (name, type, qposadr, dofadr) ---")
JT = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
for j in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "<unnamed>"
    print(f"  [{j}] {name:18s} {JT[m.jnt_type[j]]:6s} qposadr={m.jnt_qposadr[j]:2d} dofadr={m.jnt_dofadr[j]:2d}")

print("\n--- actuators (name, ctrlrange, forcerange, gaintype) ---")
GT = {0: "fixed", 1: "affine", 2: "muscle", 3: "user"}
for a in range(m.nu):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "<unnamed>"
    jid = m.actuator_trnid[a, 0]
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    print(f"  [{a}] {name:14s} -> joint {jname:16s} ctrl={tuple(m.actuator_ctrlrange[a])} "
          f"force={tuple(m.actuator_forcerange[a])} gain={GT[m.actuator_gaintype[a]]}")

print("\n--- keyframes ---")
for k in range(m.nkey):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k)
    print(f"  [{k}] {name}: qpos={np.array2string(m.key_qpos[k], precision=3, max_line_width=200)}")
if m.nkey == 0:
    print("  (none)")

print("\n--- foot geoms (class 'foot' typically named FL/FR/RL/RR) ---")
for gname in ["FL", "FR", "RL", "RR"]:
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
    if gid >= 0:
        bid = m.geom_bodyid[gid]
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        print(f"  geom '{gname}' id={gid} on body '{bname}'")

# settle from keyframe 0 (or default) to see standing height
if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d, 0)
else:
    mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
print(f"\nbase height at reset: z={d.qpos[2]:.3f}")
print(f"actuated joint nominal angles (qpos[7:]): "
      f"{np.array2string(d.qpos[7:], precision=3)}")
