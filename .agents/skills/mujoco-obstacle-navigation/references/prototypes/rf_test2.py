import mujoco, numpy as np, time

# Test: does rangefinder ignore its parent body? And mj_ray bodyexclude.
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
"""
for i in range(36):
    XML += f'    <rangefinder name="r{i}" site="ray{i}"/>\n'
XML += "</sensor></mujoco>"

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
mujoco.mj_forward(m,d)
vals = [d.sensordata[m.sensor_adr[i]] for i in range(m.nsensor)]
print("36-ray ring (deg:val) sample:", {i*10: round(vals[i],3) for i in range(0,36,3)})

# base geom is radius 0.1 at origin; rangefinder did NOT report 0.1 -> it skips parent body
print("min positive:", min(v for v in vals if v>0), " (should be ~1.9, NOT 0.1)")

# mj_ray with bodyexclude = base body id
baseid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
gid = np.zeros(1, np.int32)
dist = mujoco.mj_ray(m,d,np.array([0.,0.,.5]),np.array([1.,0.,0.]),None,1,baseid,gid)
print("mj_ray bodyexclude=base:", round(dist,3), mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,gid[0]))

# Performance: step + read 36 rangefinders
mujoco.mj_resetData(m,d)
t0=time.time()
for _ in range(5000):
    mujoco.mj_step(m,d)
dt=time.time()-t0
print(f"5000 steps w/ 36 rangefinders: {dt:.3f}s = {5000/dt:.0f} steps/s")

# Pure mj_ray ring of 36, 5000 times (planner could call this directly)
def ray_ring(m,d,origin,n=36):
    out=np.empty(n)
    for k in range(n):
        a=2*np.pi*k/n
        v=np.array([np.cos(a),np.sin(a),0.])
        g=np.zeros(1,np.int32)
        out[k]=mujoco.mj_ray(m,d,origin,v,None,1,baseid,g)
    return out
t0=time.time()
for _ in range(5000):
    ray_ring(m,d,np.array([0.,0.,.5]))
print(f"5000x manual 36-ray ring: {time.time()-t0:.3f}s")
