# GO2 model-based trot — recipe, IK derivation, and gotchas

Verified on Apple Silicon (M5 Max, mujoco 3.9.0), MuJoCo Menagerie `unitree_go2`.
NVIDIA-free, CPU-only, no RL. Result: stable trot ~0.23 m/s forward, body z ≈ 0.23 m.

## The three gotchas that actually cost time

1. **Torque, not position.** GO2 Menagerie uses `<motor>` actuators (gear=1 → `ctrl` *is*
   joint torque, N·m). There is no built-in position servo; you implement PD yourself.

2. **`forcerange=(0,0)` means "no limit", not "zero force".** The torque ceiling lives in
   **`actuator_ctrlrange`** (±23.7 hip/thigh, ±45.43 calf). Early bug: clipping `tau` to
   `forcerange` → `clip(tau, 0, 0)` = always-zero torque → the robot collapsed *identically*
   for every kp/kd. Identical results across very different gains = "your control isn't applied."

3. **Duty + lift decide forward vs backward, not the x-sign.** With `duty=0.6` (stance
   overlap) and a small `lift=0.06`, foot ground-contact was ~99% — the feet dragged instead
   of stepping, and the robot crept *backward*. Switching to `duty=0.5` (true alternating
   trot) with `lift=0.10` dropped contact to ~68%, the feet cleared and stepped, and motion
   became cleanly forward. Diagnose with: contact fraction + foot slip speed, not by eye.

## Tuned parameters (forward ~0.23 m/s)
```
freq = 2.0 Hz        # gait cycles per second
xamp = 0.08 m        # half stride (foot x sweep amplitude in hip frame)
h0   = 0.26 m        # nominal foot depth below hip (stance height)
lift = 0.10 m        # swing apex clearance
duty = 0.5           # stance fraction (true trot — keep at 0.5)
kp   = 80, kd = 3    # software PD (per joint), clipped to ctrlrange
```
PD for *standing* holds at kp≈60, kd≈3 (Unitree's real kp≈20 sags in this sim).

## Foot trajectory (per leg, normalized phase p ∈ [0,1))
```
stance (p < duty):  s = p/duty;          x = xamp*(1-2s);   z = -h0          # planted, push back
swing  (p ≥ duty):  s = (p-duty)/(1-duty); x = xamp*(2s-1);  z = -h0 + lift*sin(pi*s)
```
Diagonal trot phase offsets: FL=0, FR=0.5, RL=0.5, RR=0.

## Sagittal 2-link leg IK (L1 = L2 = 0.213 m)
Hip abduction held at 0; thigh (q1) and calf (q2) move in the x–z plane.

Forward kinematics (foot relative to the hip/thigh joint; x forward, z up):
```
x = -L1*sin(q1) - L2*sin(q1+q2)
z = -L1*cos(q1) - L2*cos(q1+q2)
```
Inverse (target foot x, z):
```
r2 = x^2 + z^2
cos(q2) = (r2 - L1^2 - L2^2) / (2*L1*L2)      ;  q2 = -arccos(...)   # knee bends backward
k1 = L1 + L2*cos(q2);  k2 = L2*sin(q2)
s1 = (-k1*x + k2*z) / (k1^2 + k2^2)
c1 = (-k2*x - k1*z) / (k1^2 + k2^2)
q1 = atan2(s1, c1)
```
Verified: IK(0, -0.2648) → (q1, q2) = (0.900, -1.800) = the GO2 home stance. Always
round-trip-check IK against the known home pose before trusting it.

## Known limitations / next steps
- Open-loop: slow yaw drift and lateral creep (no body-velocity/heading feedback). Add a
  simple Raibert foot-placement / heading P-controller to hold a straight line and a
  commanded forward speed.
- No terrain adaptation, no push recovery. Next rung is convex-MPC (CasADi+OSQP+Pinocchio,
  all arm64 wheels) for a more robust, disturbance-rejecting trot.
- Same skeleton (foot trajectory + IK + PD) extends to Go1/A1 and, with sagittal+lateral IK,
  toward humanoid stand/step — but humanoid dynamic walking ships via pretrained replay
  (project strategy II-5/II-6), not this baseline.
```
