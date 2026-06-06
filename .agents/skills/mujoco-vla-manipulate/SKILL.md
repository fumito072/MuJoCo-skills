---
name: mujoco-vla-manipulate
description: >-
  Run a pretrained Vision-Language-Action (VLA) model as the BRAIN of a MuJoCo robot on macOS
  Apple Silicon, NVIDIA-free — language instruction + camera image in, robot action out. Phase A
  of the Mac-native physical-AI stack: replay SmolVLA (450M, Apache-2.0) locally via PyTorch MPS
  to drive a standing Unitree G1's arm (manipulation) by IK, with locomotion intent decoupled into
  the existing (vx,vy,wz) walk command (same hierarchical split NVIDIA's GR00T also uses). Use to
  add a language-conditioned VLA brain to the G1/GO2 skills without any NVIDIA GPU or training.
license: Apache-2.0
compatibility: >-
  Requires a dedicated venv with lerobot[smolvla] (PyTorch >=2.4 with MPS, transformers). macOS
  Apple Silicon — MPS is required (CPU is ~57x slower). The SmolVLA weights are Apache-2.0
  (lerobot/smolvla_base, ~2GB, downloaded on first load). Inference only — no training (that needs
  a GPU and is off-Mac by design).
metadata:
  version: 0.1.0
  author: Colapis MuJoCo-skills
  tags: vla, vision-language-action, smolvla, lerobot, mlx, mps, manipulation, unitree-g1, apple-silicon, nvidia-free
---

# MuJoCo VLA Manipulate (a pretrained VLA brain on a Mac, NVIDIA-free)

The first step of the bigger mission: a Vision-Language-Action model running 100% locally on
Apple Silicon — **no NVIDIA** — so anyone with a Mac can give a humanoid language instructions.
Mirrors `mujoco-pretrained-deploy` (replay a pretrained checkpoint locally), scaled up from a
locomotion policy to a VLA.

## Phase-A gate result (verified on M5 Max) — the brain runs on a Mac
`lerobot/smolvla_base` (~450M, Apache-2.0), benchmarked on this machine:

| device | per action-chunk (50 steps) | effective control rate |
|---|---|---|
| CPU | ~9900 ms (0.10 chunks/s) | too slow |
| **MPS** | **~174 ms (5.76 chunks/s)** | **~288 steps/s** (chunk interpolated) |

**Verdict: YES — a pretrained VLA runs NVIDIA-free on Apple Silicon at usable speed.** Run the VLA
at a few Hz, interpolate its 50-step action chunk under the fast physics loop (same 50 Hz / 500 Hz
decimation discipline as the G1 walk). MPS is required; CPU is ~57x slower.

## Setup (dedicated venv, kept out of git)
```bash
python3 -m venv .venv-vla
.venv-vla/bin/pip install "lerobot[smolvla]"
.venv-vla/bin/python .agents/skills/mujoco-vla-manipulate/scripts/bench_smolvla.py   # the gate
```
(Separate venv because lerobot pins a newer torch; it must not disturb the mujoco-skills env.)

## Stage A (working): language & VOICE → navigation, in sim, NVIDIA-free
A typed or **spoken** instruction drives the pretrained G1 walk in MuJoCo — no VLA action
transfer, no IK, no fine-tuning (safest first loop). Bilingual EN/JP keyword parser →
heading-hold execution on the existing `(vx,vy,wz)` walk hook.
```bash
# text -> G1 navigates (EN or JP), render a GIF
.venv-vla/bin/python scripts/language_nav.py --say "forward; turn left; forward; stop" --video assets/language_nav.gif
.venv-vla/bin/python scripts/language_nav.py --say "前に進んで; 右を向いて; 進んで; 止まれ"

# VOICE -> G1: Whisper (MLX, Apple-Silicon, NVIDIA-free) transcribes, then drives the G1
.venv-vla/bin/pip install mlx-whisper sounddevice soundfile
.venv-vla/bin/python scripts/voice_nav.py --mic 6 --run --video assets/voice_nav.gif   # speak into the mic
.venv-vla/bin/python scripts/voice_nav.py --audio clip.wav --run                        # or from a file
```
Verified on M5 Max: a Japanese voice clip ("前に進んで、左を向いて、また進んで、止まれ") → Whisper
(`mlx-community/whisper-large-v3-turbo`) transcribed it exactly → the G1 walked the path in sim,
upright — 100% local, no NVIDIA. STT is just a text-in front-end to `language_nav.parse`. A small
local LLM (MLX) can later replace the keyword parser for free-form language.

## Architecture — why our existing skills are already the right shape
Every real G1 VLA stack (including NVIDIA's GR00T N1.7 + GEAR-SONIC) **decouples** the VLA
(arm manipulation) from locomotion: the VLA emits ~6–14 DoF arm + gripper chunks at low Hz; a
separate controller handles the legs. That is exactly our split:
- **Arm (manipulation):** VLA end-effector target → damped-least-squares IK on the two 7-DoF arms
  of `models/unitree_g1/g1_with_hands.xml` → joint targets → PD (same path as the walk). *(build)*
- **Locomotion:** route a navigate/approach intent into the **existing** velocity-command walk —
  `g1_walk.walk(m, d, policy, cmd, ...)` accepts a `callable(t) -> [vx,vy,wz]`; the VFH
  `make_planner` is the template. **No new locomotion learning.** *(reuse)*
- **Camera:** `mujoco-offscreen-render` head-camera RGB (CGL, plain python3). *(reuse)*

## SmolVLA I/O (from the loaded model)
- Inputs: `observation.images.camera1/2/3` (3×256×256), `observation.state` (6), and a tokenized
  language instruction (`observation.language.tokens` + `.attention_mask` via the SmolVLM tokenizer).
- Output: `action` (6) per step, emitted as flow-matching **chunks of 50**.
- **Critical glue:** the action **de-normalization** postprocess (and the gripper sign / EEF frame)
  — get it wrong and the arm moves nonsensically. Use `make_smolvla_pre_post_processors` (needs
  dataset stats) for correct normalization in the real loop.

## Honest scope (do not overclaim)
- ✅ The VLA **brain runs NVIDIA-free on a Mac at usable speed** (verified).
- ⚠️ SmolVLA is trained on tabletop arms (SO-100/SO-101), **not** the G1, and not on MuJoCo
  renders → zero-shot G1 driving is a **plumbing / IK demo, not robustness**. Closing the
  embodiment + visual gap = fine-tuning = GPU (off-Mac, one-time; inference stays local).
- 🔴 Whole-body, human-like humanoid VLA (walking while manipulating) is **frontier**: the only
  open one (GR00T) is CUDA-locked (flash-attn), so it runs as a cloud/Jetson brain with a Mac
  client — not locally. No NVIDIA-free whole-body humanoid VLA exists yet.

## Roadmap (A→E)
A. ✅ DONE — SmolVLA benchmarked on Mac + **language & voice → nav → walk** (Whisper STT, EN/JP).
B. arm IK manipulation on a standing G1 (drive the arm from SmolVLA chunks). C. arbiter (navigate-then-manipulate). D. fine-tune on G1 data (GPU once, off-Mac).
E. whole-body (cloud/Jetson brain, Mac client). The repo's decoupled architecture is already the
correct shape for all five — including GR00T's own design.
