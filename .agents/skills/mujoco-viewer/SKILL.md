---
name: mujoco-viewer
description: >-
  Open the MuJoCo interactive 3D viewer for a robot on macOS, NVIDIA-free. Encodes the one rule
  newcomers trip on: on macOS the passive viewer (mujoco.viewer) must run under `mjpython`, not
  plain `python`, because the GUI needs the main thread — and it must NOT share a process with an
  offscreen mujoco.Renderer (that crashes, MuJoCo issue #798). Use this to visually inspect a robot
  or watch a controller live; use mujoco-offscreen-render instead for headless frames/video.
license: Apache-2.0
compatibility: >-
  Requires Python 3.10+, mujoco>=3.9, and `mjpython` (ships with the mujoco wheel). macOS Apple
  Silicon, CPU-only, no NVIDIA GPU. Needs a GUI session (not headless/SSH). Models from
  mujoco-env-setup (<repo>/models/).
metadata:
  version: 0.1.0
  author: Colapis MuJoCo-skills
  tags: mujoco, viewer, mjpython, macos, apple-silicon, nvidia-free
---

# MuJoCo Viewer (macOS, mjpython)

Interactive 3D viewer for the robots, with the macOS-specific gotchas baked in.

## The rule that saves an hour
- **Use `mjpython`, not `python`.** On macOS the viewer's window must own the main thread.
  `python -m mujoco.viewer ...` / `python view.py` will not drive the GUI correctly; `mjpython` will.
- **Never create an offscreen `mujoco.Renderer` in the same process as the viewer.** Verified on
  Apple Silicon: doing so raises `NSInternalInconsistencyException ... Main Thread` (MuJoCo #798).
  Keep the interactive viewer (this skill, mjpython) and offscreen rendering
  (`mujoco-offscreen-render`, plain python3) in **separate processes**.

## Instructions
```bash
# requires a GUI session (not headless/SSH)
mjpython scripts/view.py unitree_go2     # or: unitree_g1
```
A window opens; the robot steps passive physics. Replace the step loop with a controller import to
watch a policy live (e.g. drive `go2_trot.trot` or the G1 walk).

## Scope
- ✅ Interactive inspection on a Mac with a display, NVIDIA-free.
- ❌ Headless/CI rendering — use `mujoco-offscreen-render` (CGL, plain python3) for that.
