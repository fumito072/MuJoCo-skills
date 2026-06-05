"""Launch the MuJoCo interactive viewer for a robot — macOS requires mjpython.

On macOS the GUI must run on the main thread, so the passive viewer (mujoco.viewer) only works
under `mjpython`, NOT plain `python`. This script is therefore meant to be run as:
    mjpython scripts/view.py [unitree_go2|unitree_g1]

Do NOT also create an offscreen mujoco.Renderer in the same process — combining the mjpython
viewer with offscreen rendering raises an NSWindow main-thread exception (MuJoCo issue #798).
For offscreen frames/video use the separate mujoco-offscreen-render skill (plain python3).
"""
import os
import sys
import time
import mujoco
import mujoco.viewer

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "models")
robot = sys.argv[1] if len(sys.argv) > 1 else "unitree_go2"
scene = os.path.join(_MODELS, robot, "scene.xml")
if not os.path.exists(scene):
    sys.exit(f"model not found: {scene}\nRun the mujoco-env-setup skill first.")

if "mjpython" not in sys.executable and not os.environ.get("MJPYTHON"):
    print("WARNING: on macOS the viewer needs mjpython. Run: mjpython", __file__, robot)

m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m)
if m.nkey:
    mujoco.mj_resetDataKeyframe(m, d, 0)
print(f"viewing {robot} (nq={m.nq}, nu={m.nu}); close the window or Ctrl-C to quit")
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)            # passive physics; replace with a controller if desired
        viewer.sync()
        time.sleep(m.opt.timestep)
