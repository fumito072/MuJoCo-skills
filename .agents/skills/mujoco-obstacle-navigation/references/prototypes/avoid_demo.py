import mujoco, numpy as np

N=24; FOV=np.deg2rad(220); RMAX=4.0
def build():
    s=f"""<mujoco model="avoid_demo">
 <option timestep="0.01"/>
 <worldbody>
  <geom name="floor" type="plane" size="0 0 .1"/>
  <geom name="obs1" type="cylinder" pos="2.0 0.3 0.5" size="0.3 0.5" rgba=".8 .2 .2 1"/>
  <geom name="obs2" type="cylinder" pos="3.5 -0.6 0.5" size="0.3 0.5" rgba=".8 .2 .2 1"/>
  <geom name="obs3" type="box"      pos="3.0 1.0 0.5" size="0.3 0.3 0.5" rgba=".8 .2 .2 1"/>
  <site name="goal" pos="5 0 0.1" size="0.15" rgba="0 1 0 .5"/>
  <body name="base" pos="0 0 0.5">
   <joint name="x" type="slide" axis="1 0 0"/>
   <joint name="y" type="slide" axis="0 1 0"/>
   <joint name="yaw" type="hinge" axis="0 0 1"/>
   <geom name="basegeom" type="cylinder" size="0.2 0.3" rgba=".2 .4 .8 1"/>
"""
    for i in range(N):
        a=-FOV/2+FOV*i/(N-1)
        s+=f'   <site name="ray{i}" zaxis="{np.cos(a):.4f} {np.sin(a):.4f} 0" size="0.005"/>\n'
    s+="  </body>\n </worldbody>\n <sensor>\n"
    for i in range(N): s+=f'  <rangefinder name="r{i}" site="ray{i}"/>\n'
    s+=" </sensor>\n</mujoco>"
    return s

m=mujoco.MjModel.from_xml_string(build()); d=mujoco.MjData(m)
gx,gy=5.0,0.0
angles=np.array([-FOV/2+FOV*i/(N-1) for i in range(N)])

def planner(d, base_xy, base_yaw):
    # read ranges (-1 -> RMAX)
    rng=np.array([d.sensordata[m.sensor_adr[i]] for i in range(N)])
    rng=np.where(rng<0, RMAX, rng)
    # goal heading in body frame
    dx,dy=gx-base_xy[0], gy-base_xy[1]
    goal_dir=np.arctan2(dy,dx)-base_yaw
    goal_dir=(goal_dir+np.pi)%(2*np.pi)-np.pi
    # VFH-style: build binary polar histogram, blocked if range < threshold
    SAFE=1.2
    blocked = rng < SAFE
    # widen blocked sectors (robot radius)
    blk=blocked.copy()
    for i in range(N):
        if blocked[i]:
            for j in (i-2,i-1,i+1,i+2):
                if 0<=j<N: blk[j]=True
    # candidate sectors = free, pick one closest to goal heading
    free=np.where(~blk)[0]
    if len(free)==0:
        return 0.0, 1.5  # spin to search
    cand_ang=angles[free]
    best=free[np.argmin(np.abs(cand_ang-goal_dir))]
    steer=angles[best]
    vx=0.6*np.clip(rng[best]/RMAX,0.2,1.0)
    wz=2.0*steer
    return vx, wz

# Simple kinematic base integrator driven by planner (stand-in for locomotion layer)
mujoco.mj_forward(m,d)
qadr={n:m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in ("x","y","yaw")}
log=[]
for step in range(1500):
    mujoco.mj_forward(m,d)  # refresh sensors at current pose
    bx,by=d.qpos[qadr["x"]],d.qpos[qadr["y"]]
    yaw=d.qpos[qadr["yaw"]]
    if np.hypot(gx-bx,gy-by)<0.3:
        print(f"GOAL reached at step {step}, pos=({bx:.2f},{by:.2f})"); break
    vx,wz=planner(d,(bx,by),yaw)
    dt=0.01
    d.qpos[qadr["x"]]+=vx*np.cos(yaw)*dt
    d.qpos[qadr["y"]]+=vx*np.sin(yaw)*dt
    d.qpos[qadr["yaw"]]+=wz*dt
    if step%150==0: log.append((step,round(bx,2),round(by,2),round(np.degrees(yaw),1),round(vx,2)))
print("trajectory (step,x,y,yawdeg,vx):")
for r in log: print("  ",r)
print("final pos:", round(d.qpos[qadr['x']],2), round(d.qpos[qadr['y']],2))
