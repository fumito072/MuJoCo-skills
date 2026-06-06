"""Voice -> G1 navigation in MuJoCo, all on a Mac, NVIDIA-free.

Pipeline: microphone (or an audio file) -> Whisper speech-to-text on Apple Silicon (MLX, no
NVIDIA) -> split into instructions -> language_nav.parse -> drive the pretrained G1 walk in sim.
Whisper handles Japanese and English. The voice front-end is just text-in to the Stage-A
language->nav bridge.

Usage:
  python voice_nav.py --audio /path/clip.wav            # transcribe a file -> instructions
  python voice_nav.py --audio /path/clip.wav --run --video out.gif   # ...and drive G1 in sim
  python voice_nav.py --mic 6                            # record 6s from the mic, then drive
"""
import os
import re
import sys
import argparse
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_ln_spec = importlib.util.spec_from_file_location("language_nav", os.path.join(HERE, "language_nav.py"))
ln = importlib.util.module_from_spec(_ln_spec)
_ln_spec.loader.exec_module(ln)   # makes ln.parse / ln.build_segments / ln.gw available

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"   # NVIDIA-free Apple-Silicon STT; fast + great JP/EN


def transcribe(path, model=DEFAULT_MODEL):
    import mlx_whisper
    r = mlx_whisper.transcribe(path, path_or_hf_repo=model)
    return r["text"].strip()


def split_instructions(text):
    parts = re.split(r"[。、，,.;！!？?\n]|\bthen\b|\band\b|そして|また|つぎに|次に", text)
    return [p.strip() for p in parts if p.strip()]


def record_mic(secs, wav_path="/tmp/voice_cmd.wav", sr=16000):
    import sounddevice as sd
    import soundfile as sf
    print(f"recording {secs}s — speak now (e.g. 'go forward, turn left, stop' / '前に進んで、左を向いて、止まれ')...")
    audio = sd.rec(int(secs * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    sf.write(wav_path, audio, sr)
    return wav_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="transcribe this audio file")
    ap.add_argument("--mic", type=float, help="record N seconds from the microphone")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--run", action="store_true", help="drive the G1 in MuJoCo from the result")
    ap.add_argument("--video", default="")
    args = ap.parse_args()

    if args.mic:
        path = record_mic(args.mic)
    elif args.audio:
        path = args.audio
    else:
        sys.exit("give --audio FILE or --mic SECONDS")

    print(f"transcribing with {args.model} (Apple Silicon, NVIDIA-free) ...")
    text = transcribe(path, args.model)
    print(f"heard: \"{text}\"")
    instructions = split_instructions(text)
    print("instructions parsed:")
    for ins in instructions:
        print(f"  '{ins}' -> {ln.parse(ins)}")

    if args.run and instructions:
        import mujoco
        from PIL import Image
        segs, total = ln.build_segments(instructions)
        m, d, policy = ln.gw.make()
        cmd = ln.make_cmd(d, segs)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        renderer = cam = frames = None
        every = int(round(1.0 / (12 * ln.gw.SIM_DT)))
        if args.video:
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = bid; cam.distance, cam.azimuth, cam.elevation = 3.2, 130, -16
            renderer = mujoco.Renderer(m, 240, 320); frames = []
        st = {"k": 0}

        def log(dd):
            if renderer is not None and st["k"] % every == 0:
                renderer.update_scene(dd, camera=cam)
                frames.append(Image.fromarray(renderer.render()).convert("P", palette=Image.ADAPTIVE, colors=48))
            st["k"] += 1

        ln.gw.walk(m, d, policy, cmd, int(total / ln.gw.SIM_DT), log=log)
        print(f"drove G1 through {len(instructions)} spoken instructions; upright={d.qpos[2] > 0.5}")
        if args.video and frames:
            out = os.path.abspath(args.video); os.makedirs(os.path.dirname(out), exist_ok=True)
            frames[0].save(out, save_all=True, append_images=frames[1:], duration=83, loop=0, optimize=True)
            print(f"video -> {out} ({os.path.getsize(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
