# G1 floor sitting — what works, what doesn't, and the path forward

Verified on Apple Silicon (M5 Max, mujoco 3.9.0), MuJoCo Menagerie `unitree_g1` (29-DOF,
`<position>` actuators). Goal: floor-sit as a stepping stone toward chair-sitting.

## ✅ What works: the stable floor-sit (long-sit) POSE
Legs extended forward, torso upright, pelvis resting on the ground.
- Leg angles (per leg): `hip_pitch = -1.57, knee = 0.2, ankle_pitch = 0.0`; waist upright.
- Settles to **pelvis z ≈ 0.151**, roll ≈ 0, pitch ≈ -6° — stable and held indefinitely.
- The model has a pelvis collision geom, so the buttocks rest on the floor correctly.
- `python g1_sit.py --mode hold` → `SITS STABLY ✓`. Visually a clean long-sit.

Why it's stable: with the legs extended forward, the support polygon (buttocks + backs of
thighs + heels) is large and the CoM sits over it. This is the *target* of any sit-down.

## ❌ What does NOT work: the controlled STAND → FLOOR-SIT transition (open-loop)
~10 open-loop position-target trajectories were tried (direct slow interp, squat-then-seat,
buttocks-back, lean-then-sit-back, very-slow monotonic, bent-knee sit, kneel-first, legs-apart
for lateral stability, with/without torso lean). **Every one fails**, in one of two ways:
1. **Folds forward and stays on the feet** (knees too straight → robot bends at the hips,
   CoM goes forward, feet hold it; pelvis never reaches the floor; pitch 70–86°).
2. **Topples** when forced down (deep configs → the strong position servos throw the base;
   roll → ±180° or face-plant pitch ≈ -87°).

Also verified: **bent-knee floor-sit poses are themselves unstable** (CoM forward → face-plant);
only the legs-extended long-sit is stable. So the descent must end in the long-sit, which means
the legs have to leave foot-support and extend forward — an unavoidable support transfer.

## Why it's hard (root cause)
Floor sit-down is a **balance-critical, contact-rich support transfer** from feet to buttocks.
The free-floating base is under-actuated; mid-descent the robot must pass through configurations
that are not statically stable on the feet, and open-loop position targets cannot regulate the
CoM/ZMP through that transfer. This is the **same difficulty class as bipedal walking** — which is
exactly why G1 WALK is shipped via a pretrained policy, not hand-built control.

## ✅ SOLVED via offline trajectory optimization (approach B)
Hand-scripting fails because a *human* can't guess the narrow balanced path. **Letting a search
find it works.** `g1_sitdown_optimize.py` runs CEM (cross-entropy method) over the two
intermediate descent knots (12 symmetric params: hip_pitch, hip_roll, knee, ankle_pitch,
waist_pitch, shoulder_pitch × 2 knots), fixing the endpoints (stand → the verified stable
long-sit). Cost rewards ending seated (low pelvis), staying upright, and never toppling.

Result on M5 Max (~720 rollouts, ~65 s, CPU-only, NVIDIA-free): a non-toppling descent —
**final pelvis z = 0.151, final tilt 6.4°, max tilt 37° (vs. 80° topple threshold), toppled =
False.** The G1 stands, then sits down on the floor cleanly. Trajectory saved to
`assets/g1_sitdown_traj.npz`; replay with `g1_sitdown.py` (→ `SAT DOWN ON FLOOR ✓`).

This is a **sim-valid open-loop trajectory** (deterministic). For real-robot transfer it needs
added feedback; the intended loop is **optimize in sim → test on hardware → feed the error back →
re-optimize** (closing sim-to-real over iterations). Other hardening paths remain optional:
a pretrained sit/getup policy (mirrors the WALK solution, if a license-clean checkpoint exists),
or an online closed-loop CoM/capture-point regulator.

## Status
- `g1_sit.py --mode hold` → stable floor-sit POSE (verified).
- **`g1_sitdown.py` → stand → floor-sit descent (CEM-optimized, WORKS: `SAT DOWN ON FLOOR ✓`).**
- `g1_sitdown_optimize.py` → re-run/retune the search (e.g. after a hardware feedback round).
## Get-up (floor-sit → stand): the asymmetric hard case (NOT solved by open-loop CEM)
Sit-down and get-up are NOT symmetric. **Sit-down: solved** (gravity assists the descent).
**Get-up: the same CEM harness does NOT crack it** — three settings (slow quasi-static T=4 +
binary cost; slow + continuous cost + reversed-sit-down seed; fast T=2 for momentum) all topple
(robot collapses forward, pelvis stays ~0.06; best cost plateaus). Rising from a legs-extended
floor sit is a multi-phase, momentum/contact-dependent maneuver (sit up → fold to a crouch →
shift CoM over the feet → push up, often using the hands on the ground) that 2-knot open-loop
position trajectories can't express. This is a known-hard humanoid problem (RL "getup" tasks).
- `g1_getup_optimize.py` is kept as a starting harness (extend: more knots, arms-on-ground push,
  contact scheduling, or a dynamic optimizer like mjpc).
- **Reliable path = a pretrained getup policy** (mirrors the WALK solution — check MuJoCo
  Playground / Unitree humanoid getup checkpoints).
- **Important:** basic standing-up is ALREADY covered by `g1_squat.py` (sit-to-stand from a
  squat) and `g1_stand.py`. Only deep-floor-sit get-up is the open frontier.

## Chair-sitting: also a support-transfer problem (NOT the easy win we assumed)
We expected chair-sit to be EASIER than floor-sit (shallow descent onto a raised seat). It is
HARDER in open-loop control, verified directly: a stable squat (`g1_squat`, never topples) tips
over **the instant the buttocks contact the seat** — the contact force is a perturbation the
foot-balanced squat can't absorb. Plus the raised seat is a small target the drifting base keeps
missing/penetrating. CEM (`g1_chair_sitdown_optimize.py`, `g1_chair.py` builds a chaired model via
MjSpec) drives the cost down but stays toppled. So chair sit-down is the same hard SUPPORT-TRANSFER
class as floor get-up. (`g1_chair_sitdown_optimize.py` kept with honest STATUS.)

## Which humanoid transitions are open-loop-tractable (synthesis)
| transition | open-loop (CEM/script) | why |
|---|---|---|
| stand (hold) | ✅ | static, servos hold |
| squat sit-to-stand (`g1_squat`) | ✅ | up/down on feet, NO support transfer |
| **floor sit-DOWN** (`g1_sitdown`) | ✅ (CEM) | floor = large forgiving target, long-sit end-pose intrinsically stable, gravity assists |
| walk (`g1_walk`) | ✅ (pretrained) | policy encodes balance |
| **floor get-UP** | ❌ | rise against gravity, multi-phase support transfer |
| **chair sit-down / get-up** | ❌ | seat-contact perturbs foot balance → topples; small raised target |
**Rule of thumb:** open-loop CEM works when there's NO balance-critical support transfer onto a
small/raised support and gravity assists. The hard support-transfer transitions (get-up, chair)
need a **pretrained policy** (mirrors WALK — check MuJoCo Playground/Unitree getup&sit) or
**closed-loop CoM/ZMP control**. Note the robot already has a solid repertoire without these:
stand, squat sit-to-stand, walk (+ steer), and floor sit-down.
