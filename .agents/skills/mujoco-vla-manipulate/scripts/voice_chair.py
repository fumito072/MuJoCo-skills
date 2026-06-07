"""End-to-end: ONE spoken command -> the G1 navigates to the chair (avoiding obstacles) and sits.

Wires the whole brain together, NVIDIA-free on a Mac:
  voice (mic/file) --Whisper(MLX)--> text --LLM NLU(Qwen2.5/MLX)--> plan --dispatch--> skills
The dispatcher reads the plan: if it contains a `goto` + `sit` (e.g. from "椅子まで行って座って"),
it runs the two-phase chair_goto_sit executor (VFH walk -> CEM floor sit-down). Directional-only
plans are pointed to voice_live (the real-time interactive loop).

Usage:
  .venv-llm/bin/python voice_chair.py --say "あの障害物を避けて椅子に行って座って" --video assets/voice_chair.gif
  .venv-llm/bin/python voice_chair.py --audio clip.wav --video out.gif
  .venv-llm/bin/python voice_chair.py --mic 6 --video out.gif
"""
import os
import sys
import argparse
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))


def _imp(name, fname):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--say", help="use this text directly (skip speech-to-text)")
    ap.add_argument("--audio", help="transcribe this audio file")
    ap.add_argument("--mic", type=float, help="record N seconds from the mic")
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--video", default="")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    # 1) get the command text
    if args.say:
        text = args.say
    else:
        vn = _imp("voice_nav", "voice_nav.py")
        if args.mic:
            path = vn.record_mic(args.mic)
        elif args.audio:
            path = args.audio
        else:
            sys.exit("give --say TEXT, --audio FILE, or --mic SECONDS")
        print("transcribing (Whisper, MLX, NVIDIA-free) ...")
        text = vn.transcribe(path, args.model)
    print(f'heard: "{text}"')

    # 2) understand -> plan
    nlu = _imp("nlu", "nlu.py")
    plan = nlu.understand(text)
    print(f"plan : {plan}")
    actions = [s.get("action") for s in plan]
    low = text.lower()
    # keyword guardrail: the LLM occasionally drops a step, so also read intent from the raw text
    wants_sit = ("sit" in actions) or any(k in text for k in ["座", "すわ"]) or "sit" in low
    wants_chair = ("goto" in actions) or ("椅子" in text) or "chair" in low

    # 3) dispatch
    if wants_sit or (wants_chair and wants_sit):   # "...go to the chair and sit"
        print("dispatch -> chair_goto_sit (navigate around obstacles to the chair, then sit down)")
        cgs = _imp("chair_goto_sit", "chair_goto_sit.py")
        frames = []
        arrived = cgs.phase1_walk(frames, args.fps)
        n1 = len(frames)
        seated = cgs.phase2_sit(frames, args.fps)
        print(f"RESULT: spoken command -> walked to chair={arrived}, sat down={seated}")
        if args.video and frames:
            from PIL import Image
            out = os.path.abspath(args.video); os.makedirs(os.path.dirname(out), exist_ok=True)
            frames[0].save(out, save_all=True, append_images=frames[1:],
                           duration=int(1000/args.fps), loop=0, optimize=True)
            print(f"video -> {out} ({os.path.getsize(out)/1e6:.2f} MB, {n1} walk + {len(frames)-n1} sit frames)")
    elif any(a in ("forward", "back", "turn", "stop") for a in actions):
        print("dispatch -> directional plan; for live control run voice_live.py --nlu")
        print(f"intents: {nlu.to_live_intents(plan)}")
    else:
        print("no executable plan recognized.")


if __name__ == "__main__":
    main()
