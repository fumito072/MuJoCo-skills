import mujoco, numpy as np

XML = """
<mujoco model="rf_test">
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" pos="0 0 0"/>
    <!-- a wall 2 m in front along +x -->
    <geom name="wall" type="box" pos="2 0 0.5" size="0.05 1 0.5" rgba="1 0 0 1"/>

    <body name="base" pos="0 0 0.5">
      <freejoint/>
      <geom name="basegeom" type="sphere" size="0.1" rgba="0 0 1 1"/>
      <!-- site whose +x axis points forward (default frame) -->
      <site name="rf_fwd" pos="0 0 0" zaxis="1 0 0"/>
      <!-- a fan via replicate: 5 rays sweeping in yaw -->
      <replicate count="5" euler="0 0 20">
        <site name="ray" pos="0 0 0" zaxis="1 0 0" size="0.005"/>
      </replicate>
    </body>
  </worldbody>
  <sensor>
    <rangefinder name="rf_fwd" site="rf_fwd"/>
    <rangefinder name="ray0" site="ray0"/>
    <rangefinder name="ray1" site="ray1"/>
    <rangefinder name="ray2" site="ray2"/>
    <rangefinder name="ray3" site="ray3"/>
    <rangefinder name="ray4" site="ray4"/>
  </sensor>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print("nsensor:", m.nsensor)
for i in range(m.nsensor):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
    adr = m.sensor_adr[i]
    print(f"  {name}: {d.sensordata[adr]:.4f}")

# Now test mj_ray directly (programmatic raycast, no sensor needed)
pnt = np.array([0.,0.,0.5])
vec = np.array([1.,0.,0.])
geomid = np.zeros(1, dtype=np.int32)
dist = mujoco.mj_ray(m, d, pnt, vec, None, 1, -1, geomid)
print("mj_ray forward dist:", dist, "hit geomid:", geomid[0],
      "name:", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geomid[0]))
