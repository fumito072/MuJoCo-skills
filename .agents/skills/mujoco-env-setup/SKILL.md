---
name: mujoco-env-setup
description: >-
  Prepare a Mac to run the MuJoCo-skills suite and fetch the robot models, with NO NVIDIA GPU.
  Detects Apple Silicon / no-CUDA, checks the Python deps (mujoco, numpy, torch), populates
  <repo>/models/ with the Unitree GO2 + G1 MuJoCo Menagerie models (copying a local dev clone if
  present, else via the robot_descriptions package), runs a CPU physics + CGL offscreen-render
  smoke test, and checks that installed skills are visible to Claude Code / Codex (working around
  the skills-CLI global-install issue #851). Use this FIRST after cloning, or to diagnose a Mac.
license: Apache-2.0
compatibility: >-
  Requires Python 3.10+, mujoco>=3.9, numpy (torch optional, for the G1 walk). macOS Apple Silicon,
  CPU-only, no NVIDIA GPU. Fetches models via robot_descriptions or a local /tmp/mjm clone; needs
  network on first model fetch only.
metadata:
  version: 0.1.0
  author: Colapis MuJoCo-skills
  tags: mujoco, setup, apple-silicon, nvidia-free, robot-models, menagerie, diagnostics
---

# MuJoCo Env Setup (Mac, NVIDIA-free)

The first skill to run after cloning the repo. Makes "clone → it works" true by fetching the
robot models all other skills resolve from `<repo>/models/`, and verifying the Mac can run the
stack CPU-only.

## Why models are not committed
The Menagerie GO2 + G1 meshes are ~60 MB; committing them would bloat the repo and slow
`npx skills add`. Instead this skill fetches them into a git-ignored `models/` dir on first run —
small repo, fast install, fully local after setup.

## Instructions
```bash
python scripts/setup.py
```
It will:
1. **Host check** — Apple Silicon, no NVIDIA (good).
2. **Deps** — print mujoco/numpy/torch versions; flag anything missing (`pip install mujoco numpy`,
   and `pip install robot_descriptions` if no local model clone exists).
3. **Models** — populate `models/unitree_go2/` and `models/unitree_g1/` (copy from `/tmp/mjm` dev
   clone if present, else fetch via `robot_descriptions`). Every other skill resolves models here.
4. **Smoke test** — load GO2, step CPU physics, render one offscreen frame (CGL) — proves the
   no-NVIDIA core works.
5. **Agent visibility** — check `~/.claude/skills` and `~/.agents/skills`; if a global install left
   skills invisible (skills-CLI issue #851), it advises project-scope or `--copy`.

## Notes
- The robot_descriptions G1 ships a slightly different keyframe (`stand` vs the dev clone's
  `home`); the tuned G1 squat/sit-down were authored against the dev clone, so for exact fidelity
  keep a `/tmp/mjm` clone or pin the model version. Stand/walk/nav work with either.
- No GPU is ever required. If `mjpython` is missing, install it (ships with the `mujoco` wheel) —
  it is needed only by `mujoco-viewer`.
