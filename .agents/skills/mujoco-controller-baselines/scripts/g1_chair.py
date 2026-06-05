"""Shared helper: build a G1 MuJoCo model with a CHAIR added (via MjSpec), and measure
whether the robot is actually sitting on it (seat contact force).

A thick floating seat slab + a backrest (no legs, so the feet never collide with chair legs
during the sit-down). The seat is a static world geom that bears weight. NVIDIA-free, CPU.
"""
import numpy as np
import mujoco

CHAIR_CX = 0.0
SEAT_TOP = 0.50
# chair-sit leg/waist/shoulder channels (verified: chair bears ~75% body weight, upright)
CHAIR_SIT_CH = np.array([-1.1, 0.0, 1.4, -0.35, 0.0, 0.2])


def build_chaired_model(scene="/tmp/mjm/unitree_g1/scene.xml", cx=CHAIR_CX, seat_top=SEAT_TOP):
    spec = mujoco.MjSpec.from_file(scene)
    seat = spec.worldbody.add_geom()
    seat.name = "chair_seat"
    seat.type = mujoco.mjtGeom.mjGEOM_BOX
    seat.size = [0.17, 0.18, 0.06]                 # thick slab (avoids penetration spikes)
    seat.pos = [cx, 0, seat_top - 0.06]
    seat.rgba = [0.55, 0.4, 0.25, 1]
    back = spec.worldbody.add_geom()
    back.name = "chair_back"
    back.type = mujoco.mjtGeom.mjGEOM_BOX
    back.size = [0.02, 0.18, 0.17]
    back.pos = [cx - 0.17, 0, seat_top + 0.17]
    back.rgba = [0.5, 0.36, 0.22, 1]
    return spec.compile()


def seat_contact_force(m, d):
    """Sum of |normal force| on the chair seat geom (N). >~120 N => really sitting."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "chair_seat")
    tot = 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 == sid or c.geom2 == sid:
            mujoco.mj_contactForce(m, d, i, f6)
            tot += abs(f6[0])
    return tot


def base_tilt(d):
    w, x, y, z = d.qpos[3:7]
    roll = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
    return roll, pitch
