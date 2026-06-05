"""Make a Mac ready to run every MuJoCo-skills skill, and fetch the robot models.

Does, in order:
  1. report the host (arch, macOS, CPU/RAM) and confirm it is Apple-Silicon / no-NVIDIA;
  2. check Python deps (mujoco, numpy, torch) and print versions;
  3. ensure robot models exist under <repo>/models/ (the path all skills resolve):
     copy from a local /tmp/mjm dev clone if present, else use the robot_descriptions
     package (auto-downloads MuJoCo Menagerie, then we copy unitree_go2 + unitree_g1);
  4. a smoke test: load GO2, step CPU physics, and render one offscreen frame (CGL);
  5. an agent-visibility check (works around the skills-CLI global-install issue #851):
     verify the installed skills are visible to Claude Code / Codex.

CPU-only, NVIDIA-free. Run with: python scripts/setup.py
"""
import os
import sys
import shutil
import platform
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
MODELS = os.path.join(ROOT, "models")
ROBOTS = ["unitree_go2", "unitree_g1"]


def section(t):
    print(f"\n=== {t} ===")


def host():
    section("Host")
    print(f"platform : {platform.platform()}")
    print(f"machine  : {platform.machine()} (Apple Silicon: {platform.machine() == 'arm64'})")
    try:
        import multiprocessing
        print(f"cpu cores: {multiprocessing.cpu_count()}")
    except Exception:
        pass
    nv = shutil.which("nvidia-smi")
    print(f"NVIDIA   : {'present (not used)' if nv else 'none — good, this stack is NVIDIA-free'}")


def deps():
    section("Python dependencies")
    ok = True
    for mod in ("mujoco", "numpy"):
        try:
            m = __import__(mod)
            print(f"  {mod:8s} {getattr(m, '__version__', '?')}")
        except Exception:
            print(f"  {mod:8s} MISSING  -> pip install {mod}")
            ok = False
    for mod in ("torch",):  # needed only for the pretrained G1 walk
        try:
            m = __import__(mod)
            cuda = getattr(getattr(m, "cuda", None), "is_available", lambda: False)()
            print(f"  {mod:8s} {getattr(m, '__version__', '?')} (CUDA={cuda}; CPU path used)")
        except Exception:
            print(f"  {mod:8s} missing (only needed for mujoco-pretrained-deploy walk)")
    print(f"  mjpython : {'found' if shutil.which('mjpython') else 'MISSING (needed by mujoco-viewer)'}")
    return ok


def ensure_models():
    section("Robot models -> models/")
    os.makedirs(MODELS, exist_ok=True)
    for r in ROBOTS:
        dst = os.path.join(MODELS, r)
        if os.path.exists(os.path.join(dst, "scene.xml")):
            print(f"  {r}: already present")
            continue
        src = os.path.join("/tmp/mjm", r)
        if os.path.exists(os.path.join(src, "scene.xml")):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  {r}: copied from local dev clone {src}")
            continue
        try:
            mod = {"unitree_go2": "go2_mj_description", "unitree_g1": "g1_mj_description"}[r]
            desc = __import__("robot_descriptions." + mod, fromlist=[mod])
            shutil.copytree(os.path.dirname(desc.MJCF_PATH), dst, dirs_exist_ok=True)
            print(f"  {r}: fetched via robot_descriptions")
        except Exception as e:
            print(f"  {r}: NOT available ({type(e).__name__}). Fix: pip install robot_descriptions")


def link_tmp():
    """Skill scripts resolve models at /tmp/mjm/<robot>; point that at the persistent models/.
    (/tmp is cleared on reboot — re-run this skill after a restart.)"""
    section("Wire /tmp/mjm -> models/ (so all skill scripts find the models)")
    os.makedirs("/tmp/mjm", exist_ok=True)
    for r in ROBOTS:
        src = os.path.join(MODELS, r)
        dst = os.path.join("/tmp/mjm", r)
        if os.path.exists(os.path.join(dst, "scene.xml")):
            print(f"  /tmp/mjm/{r}: already present")
            continue
        if not os.path.exists(os.path.join(src, "scene.xml")):
            print(f"  {r}: no model in models/ to link")
            continue
        try:
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
            print(f"  linked /tmp/mjm/{r} -> models/{r}")
        except Exception as e:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  copied models/{r} -> /tmp/mjm/{r} (symlink failed: {e})")


def smoke():
    section("Smoke test (CPU physics + offscreen render)")
    try:
        import mujoco
        import numpy as np
        scene = os.path.join(MODELS, "unitree_go2", "scene.xml")
        m = mujoco.MjModel.from_xml_path(scene)
        d = mujoco.MjData(m)
        for _ in range(200):
            mujoco.mj_step(m, d)
        r = mujoco.Renderer(m, 120, 160)
        r.update_scene(d)
        img = r.render()
        print(f"  GO2 loaded (nq={m.nq}, nu={m.nu}), stepped 200x, offscreen RGB {img.shape} OK")
        print("  -> physics + CGL offscreen rendering work on this Mac.")
    except Exception as e:
        print(f"  smoke test FAILED: {type(e).__name__}: {e}")


def visibility():
    section("Agent visibility (skills-CLI #851 workaround)")
    for label, p in [("Claude Code", os.path.expanduser("~/.claude/skills")),
                     ("Codex", os.path.expanduser("~/.agents/skills"))]:
        if os.path.isdir(p):
            n = len([x for x in os.listdir(p) if os.path.isdir(os.path.join(p, x))])
            print(f"  {label}: {p} has {n} skill(s)")
        else:
            print(f"  {label}: {p} not found (install with: npx skills add <repo> -a {'claude-code' if 'Claude' in label else 'codex'} --copy)")
    print("  NOTE: if a global install left skills invisible, prefer project-scope or --copy.")


if __name__ == "__main__":
    host()
    deps()
    ensure_models()
    link_tmp()
    smoke()
    visibility()
    print("\nSetup complete. Try: python ../mujoco-controller-baselines/scripts/go2_trot.py --secs 8")
