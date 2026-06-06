"""Stage A: language -> navigation -> the pretrained G1 walk, in MuJoCo, NVIDIA-free.

A typed (or, later, spoken) instruction is parsed into a navigation intent and executed by the
EXISTING pretrained G1 walk via its (vx,vy,wz) command hook + heading-hold — no VLA action
transfer, no IK, no fine-tuning. This is the safe first "language drives the humanoid" loop.

Bilingual (EN/JP) keyword parser; a small local LLM (MLX) could replace it for free-form later.
Voice front-end (Whisper on Apple Silicon) feeds text into parse() — see voice_nav.py.

Usage:
  python language_nav.py --say "forward; turn left; forward; stop" --video out.gif
"""
import os
import argparse
import importlib.util
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
gw_spec = importlib.util.spec_from_file_location(
    "g1_walk", os.path.normpath(os.path.join(HERE, "..", "..", "mujoco-pretrained-deploy", "scripts", "g1_walk.py")))
gw = importlib.util.module_from_spec(gw_spec)
gw_spec.loader.exec_module(gw)

# intent keywords (English + Japanese)
FWD = ["forward", "go", "ahead", "straight", "前", "進", "まっすぐ", "歩"]
BACK = ["back", "backward", "後ろ", "下が", "戻"]
LEFT = ["left", "左"]
RIGHT = ["right", "右"]
STOP = ["stop", "halt", "wait", "止ま", "停", "待", "ストッ", "とまっ", "やめ"]
FAST = ["fast", "quick", "速", "急"]
SLOW = ["slow", "ゆっくり", "遅"]


def parse(text):
    """text -> intent dict {kind,...} in {forward, back, turn, stop}, or None if not a command.
    Unrecognized speech returns None and is IGNORED (never defaults to forward) — so mic noise /
    misheard words can't make the robot move or override a stop."""
    t = text.lower().strip()
    has = lambda kws: any(k in t for k in kws)
    if has(STOP):                       # stop wins (safety first)
        return {"kind": "stop"}
    if has(LEFT):
        return {"kind": "turn", "deg": +90}
    if has(RIGHT):
        return {"kind": "turn", "deg": -90}
    if has(BACK):
        return {"kind": "back"}
    if has(FWD):
        speed = 0.55 if has(FAST) else (0.25 if has(SLOW) else 0.4)
        return {"kind": "forward", "speed": speed}
    return None                         # not a recognized command -> ignore


def build_segments(instructions, fwd_secs=4.0, turn_secs=3.5, stop_secs=2.0):
    """Turn a list of instructions into timed (speed, target_heading) segments (heading-hold)."""
    segs = []
    speed, heading, t = 0.0, 0.0, 0.0
    for ins in instructions:
        it = parse(ins)
        if it is None:
            continue
        if it["kind"] == "stop":
            speed = 0.0; dur = stop_secs
        elif it["kind"] == "turn":
            heading += np.deg2rad(it["deg"]); dur = turn_secs   # turn in place-ish then resume
        elif it["kind"] == "back":
            speed = -0.3; dur = fwd_secs
        else:
            speed = it["speed"]; dur = fwd_secs
        segs.append({"speed": speed, "heading": heading, "t0": t, "t1": t + dur, "ins": ins, "intent": it})
        t += dur
    return segs, t


def make_cmd(d, segs, kyaw=1.5):
    def yaw():
        w, x, y, z = d.qpos[3:7]
        return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def cmd(t):
        seg = segs[-1]
        for s in segs:
            if t < s["t1"]:
                seg = s; break
        err = (seg["heading"] - yaw() + np.pi) % (2*np.pi) - np.pi
        wz = float(np.clip(kyaw * err, -0.6, 0.6))
        # while turning, walk slowly so the turn completes before moving on
        vx = seg["speed"] if seg["intent"]["kind"] != "turn" else 0.1
        return [vx, 0.0, wz]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--say", default="forward; turn left; forward; stop",
                    help="instructions separated by ; (EN or JP)")
    ap.add_argument("--video", default="")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    instructions = [s for s in args.say.replace("\n", ";").split(";") if s.strip()]
    print("instructions:")
    for ins in instructions:
        print(f"  '{ins.strip()}' -> {parse(ins)}")
    segs, total = build_segments(instructions)

    m, d, policy = gw.make()
    cmd = make_cmd(d, segs)
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    renderer = cam = frames = None
    every = int(round(1.0 / (args.fps * gw.SIM_DT)))
    if args.video:
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = base_id; cam.distance, cam.azimuth, cam.elevation = 3.2, 130, -15
        renderer = mujoco.Renderer(m, 360, 480); frames = []

    st = {"k": 0}
    x0 = float(d.qpos[0]); y0 = float(d.qpos[1])

    def log(dd):
        if renderer is not None and st["k"] % every == 0:
            renderer.update_scene(dd, camera=cam)
            from PIL import Image
            frames.append(Image.fromarray(renderer.render()).convert("P", palette=Image.ADAPTIVE, colors=64))
        st["k"] += 1

    gw.walk(m, d, policy, cmd, int(total / gw.SIM_DT), log=log)
    w, x, y, z = d.qpos[3:7]
    yaw = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    print(f"executed {len(instructions)} instructions in {total:.0f}s")
    print(f"net move: dx={d.qpos[0]-x0:+.2f} dy={d.qpos[1]-y0:+.2f} m  final yaw={yaw:+.0f} deg  upright={d.qpos[2]>0.5}")
    if args.video and frames:
        out = os.path.abspath(args.video); os.makedirs(os.path.dirname(out), exist_ok=True)
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000/args.fps), loop=0, optimize=True)
        print(f"video -> {out} ({os.path.getsize(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
