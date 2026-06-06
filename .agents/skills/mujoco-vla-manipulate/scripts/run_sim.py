"""Watch it live: open the MuJoCo 3D viewer and drive the G1 by TEXT or your VOICE, NVIDIA-free.

This opens an interactive window (macOS needs mjpython) and runs the pretrained G1 walk in
real time, executing language/voice navigation instructions. No offscreen renderer is created in
this process (so it never hits the mjpython+offscreen crash, MuJoCo issue #798).

Run with the VLA venv's mjpython:
  .venv-vla/bin/mjpython run_sim.py --say "forward; turn left; forward; stop"
  .venv-vla/bin/mjpython run_sim.py --say "前に進んで; 右を向いて; 進んで; 止まれ"
  .venv-vla/bin/mjpython run_sim.py --mic 6        # speak into the mic, then watch the G1 do it
"""
import os
import sys
import time
import argparse
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_ln_spec = importlib.util.spec_from_file_location("language_nav", os.path.join(HERE, "language_nav.py"))
ln = importlib.util.module_from_spec(_ln_spec)
_ln_spec.loader.exec_module(ln)

import mujoco
import mujoco.viewer


def get_instructions(args):
    if args.mic:
        _vn = importlib.util.spec_from_file_location("voice_nav", os.path.join(HERE, "voice_nav.py"))
        vn = importlib.util.module_from_spec(_vn); _vn.loader.exec_module(vn)
        path = vn.record_mic(args.mic)
        print("transcribing (Whisper, Apple-Silicon, NVIDIA-free)...")
        text = vn.transcribe(path, args.model)
        print(f'heard: "{text}"')
        return vn.split_instructions(text)
    return [s for s in args.say.replace("\n", ";").split(";") if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--say", default="forward; turn left; forward; stop")
    ap.add_argument("--mic", type=float, help="record N seconds from the mic instead of --say")
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    args = ap.parse_args()

    if "mjpython" not in os.path.basename(sys.executable) and not os.environ.get("MJPYTHON"):
        print("NOTE: on macOS the live viewer needs mjpython. Re-run as:")
        print(f"  .venv-vla/bin/mjpython {os.path.relpath(__file__)} " + (f"--mic {args.mic}" if args.mic else f'--say "{args.say}"'))

    instructions = get_instructions(args)
    print("instructions:")
    for ins in instructions:
        print(f"  '{ins}' -> {ln.parse(ins)}")
    segs, total = ln.build_segments(instructions)

    m, d, policy = ln.gw.make()
    cmd = ln.make_cmd(d, segs)
    SIM_DT = ln.gw.SIM_DT

    print(f"\nopening viewer — the G1 will execute the {len(instructions)} instructions (~{total:.0f}s),")
    print("then keep standing until you close the window.\n")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        state = {"k": 0}

        def sync(dd):
            if state["k"] % 10 == 0:           # refresh the GUI at ~50 Hz
                viewer.sync()
                time.sleep(10 * SIM_DT)        # real-time pacing
            state["k"] += 1
            if not viewer.is_running():
                raise KeyboardInterrupt

        try:
            ln.gw.walk(m, d, policy, cmd, int(total / SIM_DT), log=sync)
            # after the sequence, hold a stand so the window stays interactive
            stand = lambda t: [0.0, 0.0, 0.0]
            while viewer.is_running():
                ln.gw.walk(m, d, policy, stand, 200, log=sync)
        except KeyboardInterrupt:
            pass
    print("viewer closed.")


if __name__ == "__main__":
    main()
