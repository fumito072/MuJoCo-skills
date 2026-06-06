"""REAL-TIME voice control: talk to the G1 continuously and it reacts live, in MuJoCo, NVIDIA-free.

Unlike voice_nav.py (record N seconds, then act), this listens CONTINUOUSLY: a background mic
stream segments your speech by silence (VAD), Whisper (MLX, Apple Silicon, no NVIDIA) transcribes
each utterance, and a shared live command is updated on the fly. The main thread runs the
pretrained G1 walk continuously, reading that live command every policy step — so the moment you
say "turn left" / "左を向いて", the walking G1 starts turning. Say "stop" / "止まれ" to halt.

Threading: the GUI viewer must own the MAIN thread (mjpython), so mic capture + transcription run
on background threads and only mutate a small lock-guarded LiveCommand.

Run (live window):  .venv-vla/bin/mjpython voice_live.py
Run (headless, no window, prints reactions):  .venv-vla/bin/python voice_live.py --no-viewer
"""
import os
import sys
import time
import queue
import threading
import argparse
import importlib.util
from collections import deque

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
_vn = importlib.util.spec_from_file_location("voice_nav", os.path.join(HERE, "voice_nav.py"))
vn = importlib.util.module_from_spec(_vn)
_vn.loader.exec_module(vn)
ln = vn.ln              # language_nav: parse, gw (g1_walk), constants
gw = ln.gw

SR = 16000


def _yaw_of(d):
    w, x, y, z = d.qpos[3:7]
    return float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))


class LiveCommand:
    """The CURRENT desired action, written by the voice thread and read by the control loop.
    INTERRUPT semantics: each recognized command REPLACES the state (latest wins, no queue, no
    accumulation). 'stop' fully halts — zero speed AND freezes the heading where it is, so the
    robot stops translating *and* rotating. 'turn' is relative to the heading at the moment it is
    spoken, so saying it again is a fresh 90deg, not a pile-up."""
    def __init__(self):
        self.speed = 0.0
        self.target_heading = None   # absolute world heading to hold; None until first read
        self.d = None                # bound to MjData so apply() can read the current yaw
        self.lock = threading.Lock()

    def apply(self, intent):
        if intent is None:
            return
        with self.lock:
            yaw = _yaw_of(self.d) if self.d is not None else 0.0
            k = intent["kind"]
            if k == "stop":
                self.speed = 0.0; self.target_heading = yaw          # halt + hold here
            elif k == "forward":
                self.speed = intent["speed"]; self.target_heading = yaw  # go straight from now
            elif k == "back":
                self.speed = -0.3; self.target_heading = yaw
            elif k == "turn":
                self.speed = 0.0; self.target_heading = yaw + np.deg2rad(intent["deg"])  # 90deg from now

    def vxvywz(self, d):
        yaw = _yaw_of(d)
        with self.lock:
            if self.target_heading is None:
                self.target_heading = yaw
            err = (self.target_heading - yaw + np.pi) % (2*np.pi) - np.pi
            speed = self.speed
        wz = float(np.clip(1.5 * err, -0.6, 0.6))
        return [speed, 0.0, wz]


class MicVAD:
    """sounddevice callback: buffer speech, emit a segment after a silence gap."""
    def __init__(self, out_queue, thresh=0.012, block=1600, hang=6, min_blocks=2, preroll=3):
        self.q = out_queue
        self.thresh, self.hang, self.min_blocks = thresh, hang, min_blocks
        self.buf, self.recent = [], deque(maxlen=preroll)
        self.speaking, self.silence, self.nspeech = False, 0, 0

    def __call__(self, indata, frames, time_info, status):
        rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
        if rms > self.thresh:
            if not self.speaking:
                self.buf = list(self.recent); self.speaking = True; self.nspeech = 0
            self.buf.append(indata.copy()); self.silence = 0; self.nspeech += 1
        else:
            self.recent.append(indata.copy())
            if self.speaking:
                self.buf.append(indata.copy()); self.silence += 1
                if self.silence >= self.hang:
                    if self.nspeech >= self.min_blocks:
                        self.q.put(np.concatenate(self.buf, axis=0).reshape(-1).astype(np.float32))
                    self.buf, self.speaking, self.silence = [], False, 0


def transcription_worker(seg_q, live, model, stop_event, verbose=True):
    import mlx_whisper
    while not stop_event.is_set():
        try:
            seg = seg_q.get(timeout=0.25)
        except queue.Empty:
            continue
        try:
            text = mlx_whisper.transcribe(seg, path_or_hf_repo=model)["text"].strip()
        except Exception as e:
            print(f"(stt error: {e})"); continue
        if not text:
            continue
        if verbose:
            print(f'🎤 "{text}"', flush=True)
        applied = False
        for ins in vn.split_instructions(text):
            intent = ln.parse(ins)
            if intent is None:
                continue
            live.apply(intent)          # latest recognized command interrupts/overrides
            applied = True
            if verbose:
                print(f'   → {ins!r} = {intent}', flush=True)
        if not applied and verbose:
            print('   (no command recognized — ignored)', flush=True)


def start_listening(live, model, thresh, stop_event):
    import sounddevice as sd
    seg_q = queue.Queue()
    vad = MicVAD(seg_q, thresh=thresh)
    worker = threading.Thread(target=transcription_worker, args=(seg_q, live, model, stop_event), daemon=True)
    worker.start()
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=1600, callback=vad)
    stream.start()
    print("🎙️  listening — speak anytime (e.g. '前に進んで' / 'turn left' / '止まれ'). Ctrl-C to quit.")
    return stream


def run_control(m, d, policy, live, viewer=None, max_steps=None, on_step=None):
    """Continuous G1 walk reading the live command every policy step (no per-chunk reset)."""
    action = np.zeros(gw.NUM_ACT, dtype=np.float32)
    target = gw.DEFAULT_ANGLES.copy()
    obs = np.zeros(gw.NUM_OBS, dtype=np.float32)
    import torch
    c = 0
    while True:
        if max_steps is not None and c >= max_steps:
            break
        if viewer is not None and not viewer.is_running():
            break
        tau = (target - d.qpos[7:]) * gw.KPS + (0.0 - d.qvel[6:]) * gw.KDS
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        if c % gw.DECIMATION == 0:
            cmd_t = np.asarray(live.vxvywz(d), dtype=np.float32)
            qj = (d.qpos[7:] - gw.DEFAULT_ANGLES) * gw.DOF_POS_SCALE
            dqj = d.qvel[6:] * gw.DOF_VEL_SCALE
            grav = gw.gravity_orientation(d.qpos[3:7])
            omega = d.qvel[3:6] * gw.ANG_VEL_SCALE
            phase = (c * gw.SIM_DT) % gw.PERIOD / gw.PERIOD
            obs[:3] = omega; obs[3:6] = grav; obs[6:9] = cmd_t * gw.CMD_SCALE
            obs[9:9+gw.NUM_ACT] = qj
            obs[9+gw.NUM_ACT:9+2*gw.NUM_ACT] = dqj
            obs[9+2*gw.NUM_ACT:9+3*gw.NUM_ACT] = action
            obs[9+3*gw.NUM_ACT:9+3*gw.NUM_ACT+2] = [np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target = action * gw.ACTION_SCALE + gw.DEFAULT_ANGLES
        if viewer is not None and c % 10 == 0:
            viewer.sync(); time.sleep(10 * gw.SIM_DT)
        if on_step is not None:
            on_step(c, d)
        c += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--thresh", type=float, default=0.012, help="mic VAD energy threshold")
    ap.add_argument("--no-viewer", action="store_true", help="headless (no 3D window)")
    args = ap.parse_args()

    if not args.no_viewer and "mjpython" not in os.path.basename(sys.executable) and not os.environ.get("MJPYTHON"):
        print("NOTE: the live 3D window needs mjpython. Re-run as:")
        print(f"  .venv-vla/bin/mjpython {os.path.relpath(__file__)}")

    live = LiveCommand()
    stop_event = threading.Event()
    m, d, policy = gw.make()
    live.d = d                          # so the voice thread can read the current yaw on apply()
    stream = start_listening(live, args.model, args.thresh, stop_event)
    try:
        if args.no_viewer:
            run_control(m, d, policy, live)
        else:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(m, d) as viewer:
                run_control(m, d, policy, live, viewer=viewer)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set(); stream.stop(); stream.close()
    print("\nstopped.")


if __name__ == "__main__":
    main()
