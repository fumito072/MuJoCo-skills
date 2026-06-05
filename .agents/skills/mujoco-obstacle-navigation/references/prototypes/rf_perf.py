import mujoco, numpy as np, time
N=24
XML = """<mujoco><option timestep="0.002"/>
  <worldbody>
    <geom name="wall" type="box" pos="2 0 0.5" size="0.05 1 0.5"/>
    <body name="base" pos="0 0 0.5"><freejoint/>
      <geom name="basegeom" type="sphere" size="0.1"/>
"""
for i in range(N):
    a=np.deg2rad(-90+180*i/(N-1)); XML+=f'<site name="ray{i}" zaxis="{np.cos(a):.4f} {np.sin(a):.4f} 0" size="0.005"/>\n'
XML+="</body></worldbody><sensor>"
for i in range(N): XML+=f'<rangefinder name="r{i}" site="ray{i}"/>'
XML+="</sensor></mujoco>"
m=mujoco.MjModel.from_xml_string(XML); d=mujoco.MjData(m)
# confirm parent-body skipped: nearest hit ~1.95 not 0.1
mujoco.mj_forward(m,d)
vals=[d.sensordata[m.sensor_adr[i]] for i in range(N)]
print("min positive (parent skipped if ~1.95):", round(min(v for v in vals if v>0),3))
t0=time.time()
for _ in range(10000): mujoco.mj_step(m,d)
dt=time.time()-t0
print(f"10000 mj_step w/ {N} rangefinders @2ms: {dt:.3f}s -> {10000/dt:.0f} steps/s (realtime={10000*0.002/dt:.1f}x)")
