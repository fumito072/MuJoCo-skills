import mujoco

# Pattern A: explicit sites (no replicate) + explicit sensors  -- most robust
XML_A = """
<mujoco>
  <worldbody>
    <geom name="wall" type="box" pos="2 0 0.5" size="0.05 1 0.5"/>
    <body name="base" pos="0 0 0.5">
      <freejoint/>
      <geom name="basegeom" type="sphere" size="0.1"/>
"""
N=12
for i in range(N):
    deg = -90 + 180*i/(N-1)
    XML_A += f'      <site name="ray{i}" pos="0 0 0" euler="0 0 {deg}" size="0.005"/>\n'
# NOTE: default site zaxis is +z (up). We need ray along +z too. Use xyaxes or zaxis.
XML_A = XML_A.replace('size="0.005"/>', '')  # rebuild properly below
XML_A = """
<mujoco>
  <worldbody>
    <geom name="wall" type="box" pos="2 0 0.5" size="0.05 1 0.5"/>
    <body name="base" pos="0 0 0.5">
      <freejoint/>
      <geom name="basegeom" type="sphere" size="0.1"/>
"""
import numpy as np
for i in range(N):
    deg = -90 + 180*i/(N-1)
    a = np.deg2rad(deg)
    # ray direction in xy-plane, put it on site +z axis
    zx, zy = np.cos(a), np.sin(a)
    XML_A += f'      <site name="ray{i}" pos="0 0 0" zaxis="{zx:.4f} {zy:.4f} 0" size="0.005"/>\n'
XML_A += "    </body>\n  </worldbody>\n  <sensor>\n"
for i in range(N):
    XML_A += f'    <rangefinder name="r{i}" site="ray{i}"/>\n'
XML_A += "  </sensor>\n</mujoco>"

m = mujoco.MjModel.from_xml_string(XML_A)
d = mujoco.MjData(m)
mujoco.mj_forward(m,d)
print("Pattern A (explicit sites+sensors) WORKS. nsensor:", m.nsensor)
for i in range(N):
    deg = -90 + 180*i/(N-1)
    print(f"  ray {deg:+6.1f} deg -> {d.sensordata[m.sensor_adr[i]]:.3f}")
