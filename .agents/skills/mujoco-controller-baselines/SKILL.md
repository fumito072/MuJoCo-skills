---
name: mujoco-controller-baselines
description: >-
  Model-based locomotion & posture baselines for legged robots (Unitree GO2 / G1 / H1
  and other MuJoCo Menagerie models) that run 100% locally on macOS Apple Silicon with
  NO NVIDIA GPU, no CUDA, no cloud, and no reinforcement learning. Provides software-PD
  joint control, a CPG-style foot-trajectory + analytic 2-link leg IK quadruped trot,
  and standing/posture controllers. Use this when you need a WORKING controller to make
  a quadruped stand or walk forward in MuJoCo on a Mac, or a deterministic diagnostic
  baseline to sanity-check a robot model / actuator setup before (or instead of) RL.
license: Apache-2.0
compatibility: >-
  Requires Python 3.10+, mujoco>=3.9 (pip), numpy; Pillow for GIF export. Runs CPU-only
  on macOS Apple Silicon — no NVIDIA GPU required. Needs a MuJoCo MJCF model (e.g. MuJoCo
  Menagerie `unitree_go2`). Offscreen rendering uses plain python3 (NOT mjpython) via CGL.
metadata:
  version: 0.1.0
  author: Colapis MuJoCo-skills
  tags: mujoco, legged-robot, quadruped, unitree-go2, model-based-control, locomotion, apple-silicon, nvidia-free
---

# MuJoCo Controller Baselines (model-based, NVIDIA-free, Mac-native)

The **spine** of the behavior layer: deterministic, model-based controllers that make a
legged robot stand and walk **without RL, GPU, or cloud**. Verified on Apple Silicon
(M5 Max, mujoco 3.9.0): GO2 stands, and trots forward at ~0.23 m/s.

## When to use
- "Make the GO2 stand / walk in MuJoCo on my Mac" → use `go2_stand.py` / `go2_trot.py`.
- "I have a Menagerie quadruped, give me a working gait without training" → trot recipe below.
- "My RL policy misbehaves — is it the controller or the policy?" → these baselines are the oracle.

## Key model facts (verified, do not skip)
- **GO2 actuators are `<motor>` = direct-drive torque** (not position servos). You must compute
  PD in software: `tau = kp*(q_des - q) - kd*qd`, then write to `d.ctrl`.
- **`actuator_forcerange` is `(0,0)` = DISABLED, not zero torque.** The real torque limit is
  **`actuator_ctrlrange`** (±23.7 N·m hip/thigh, ±45.43 N·m calf). Clip `tau` to `ctrlrange`.
  (Clipping to forcerange gives zero torque → robot collapses identically for all gains.)
- GO2: `nq=19, nv=18, nu=12`, physics 500 Hz. Joint order `qpos[7:19]` = FL,FR,RL,RR ×
  (hip, thigh, calf). Home keyframe: base z=0.27, joints `[0, 0.9, -1.8]` per leg.
  Leg links `L1=L2=0.213 m`.

## Instructions

### 1. Inspect a model
```bash
python scripts/inspect_go2.py /path/to/unitree_go2/scene.xml
```
Prints DOF, joints, actuators (with ctrl/force ranges), keyframes, foot geoms, nominal stance.

### 2. Stand (PD to home stance)
```bash
python scripts/go2_stand.py /path/to/scene.xml --kp 60 --kd 3 --secs 3
```
Validates the control loop. kp≈60, kd≈3 hold the stance in sim (Unitree's real kp≈20 sags here).

### 3. Trot forward (CPG foot trajectory + 2-link IK + PD)
```bash
python scripts/go2_trot.py /path/to/scene.xml --secs 8
# tuned default: freq2.0 xamp0.08 h0=0.26 lift0.10 duty0.5 kp80 -> ~0.23 m/s forward
```
Pipeline per 500 Hz step: global phase → per-leg foot target (stance push-back / swing lift)
→ analytic sagittal 2-link IK → PD torque (clipped to ctrlrange). Diagonal trot:
(FL,RR) and (FR,RL) half a cycle out of phase.

**Critical gait lesson (why it walked):** `duty=0.5` (true trot, no stance overlap) **and**
enough swing `lift` (~0.10 m) so feet actually clear the ground. With `duty=0.6, lift=0.06`
the feet stayed ~99% in contact (dragging) and the robot drifted *backward*. See
`references/go2-trot-recipe.md`.

### 4. Render to GIF (offscreen, plain python3 — never mjpython)
```bash
python scripts/render_go2_trot.py assets/go2_trot.gif --secs 6
```

## Examples
- Stand still then hold: `go2_stand.py --kp 60 --kd 3` → `RESULT: STANDS ✓`.
- Walk ~1.8 m in 8 s: `go2_trot.py --secs 8` → `RESULT: TROTS FORWARD ✓` (~0.23 m/s).
- Tune speed: raise `--xamp` / `--freq` for faster, lower for gentler. Keep `--duty 0.5`.

## Scope & honesty
- ✅ Local on Mac, NVIDIA-free: GO2 stand + forward trot (shown), G1/H1 PD stand.
- ⚠️ Open-loop trot has slow yaw/lateral drift (no body-velocity feedback yet); a heading
  controller is future work. Dynamic G1/H1 *walking* via model-based control is a research
  bet (see project strategy II-5); humanoid walking ships via pretrained-policy replay.
- ❌ Out of scope (project decision: 100% local): running gaits, from-scratch humanoid RL.

## References
- `references/go2-trot-recipe.md` — full recipe, the 2-link IK derivation, and gotchas.
