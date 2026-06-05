import mujoco, numpy as np, time

# Canonical pattern: rangefinder sensor INSIDE replicate
XML = """
<mujoco>
  <worldbody>
    <geom name="wall" type="box" pos="2 0 0.5" size="0.05 1 0.5"/>
    <body name="base" pos="0 0 0.5">
      <freejoint/>
      <geom name="basegeom" type="sphere" size="0.1"/>
      <replicate count="36" euler="0 0 10">
        <site name="ray" pos="0 0 0" zaxis="1 0 0" size="0.005"/>
      </replicate>
    </body>
  </worldbody>
  <sensor>
    <replicate count="36" euler="0 0 10">
      <rangefinder name="r" site="ray"/>
    </replicate>
  </sensor>
</mujoco>
"""
try:
    m = mujoco.MjModel.from_xml_string(XML)
    print("OK sensor-in-replicate. nsensor:", m.nsensor)
    print("sensor names:", [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SENSOR,i) for i in range(m.nsensor)][:5], "...")
except Exception as e:
    print("FAIL sensor-in-replicate:", e)

# Alt: explicit sensors referencing ray0..ray35
XML2 = """
<mujoco>
  <worldbody>
    <geom name="wall" type="box" pos="2 0 0.5" size="0.05 1 0.5"/>
    <body name="base" pos="0 0 0.5">
      <freejoint/>
      <geom name="basegeom" type="sphere" size="0.1"/>
      <replicate count="36" euler="0 0 10">
        <site name="ray" pos="0 0 0" zaxis="1 0 0" size="0.005"/>
      </replicate>
    </body>
  </worldbody>
  <sensor>
""" + "".join(f'    <rangefinder name="r{i}" site="ray{i}"/>\n' for i in range(36)) + "</sensor></mujoco>"
try:
    m2 = mujoco.MjModel.from_xml_string(XML2)
    print("OK explicit-sensors. nsensor:", m2.nsensor)
except Exception as e:
    print("FAIL explicit-sensors:", e)
