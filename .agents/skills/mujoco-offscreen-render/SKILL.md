---
name: mujoco-offscreen-render
description: >-
  Render RGB / depth / segmentation frames of a MuJoCo robot offscreen (headless) on macOS,
  NVIDIA-free, CPU-only. Uses the built-in OpenGL renderer via CGL (no EGL/OSMesa, no GPU) from
  plain python3 — deliberately separate from the interactive mjpython viewer (combining them in one
  process crashes, MuJoCo issue #798). Use this to capture frames or build rollout videos for RL /
  evaluation; for a live interactive window use mujoco-viewer instead.
license: Apache-2.0
compatibility: >-
  Requires Python 3.10+, mujoco>=3.9, numpy, Pillow. macOS Apple Silicon, CPU-only, no NVIDIA GPU.
  Run with plain python3 (NOT mjpython). Models from mujoco-env-setup (<repo>/models/).
metadata:
  version: 0.1.0
  author: Colapis MuJoCo-skills
  tags: mujoco, offscreen, rendering, depth, segmentation, cgl, apple-silicon, nvidia-free
---

# MuJoCo Offscreen Render (macOS CGL, NVIDIA-free)

Headless RGB + depth + segmentation rendering — the basis for rollout videos and visual eval.

## The macOS rules
- **Plain `python3`, NOT mjpython.** Offscreen `mujoco.Renderer` works under regular python on
  macOS via CGL; running it under mjpython alongside the interactive viewer crashes (issue #798).
  So this skill (offscreen) and `mujoco-viewer` (interactive, mjpython) are separate processes.
- **No EGL/OSMesa needed.** macOS uses CGL by default — RGB/depth/segmentation all work CPU-side.
- **Depth precision is limited on macOS** (ARB_clip_control unavailable under CGL) — fine for
  RL/eval, not metrologically accurate.

## Instructions
```bash
python scripts/render.py unitree_go2 --out frame.png            # RGB
python scripts/render.py unitree_g1  --out g1.png --depth --seg # + depth map + segmentation
```
For rollout videos, render frames inside a control loop (see the GIF helpers in
mujoco-controller-baselines / mujoco-pretrained-deploy, which use this same CGL path with a
tracking camera and write a Pillow GIF).

## Scope
- ✅ Headless RGB/depth/seg on a Mac, NVIDIA-free; rollout video frames.
- ❌ Photorealistic / ray-traced output (that needs RTX/OptiX = NVIDIA, out of scope).
