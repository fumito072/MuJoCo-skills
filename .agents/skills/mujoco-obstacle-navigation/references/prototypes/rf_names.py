import mujoco
XML = """
<mujoco>
  <worldbody>
    <geom name="wall" type="box" pos="2 0 0.5" size="0.05 1 0.5"/>
    <body name="base" pos="0 0 0.5">
      <freejoint/>
      <geom name="basegeom" type="sphere" size="0.1"/>
      <replicate count="5" euler="0 0 20">
        <site name="ray" pos="0 0 0" zaxis="1 0 0" size="0.005"/>
      </replicate>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(XML)
print("site names:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)])
