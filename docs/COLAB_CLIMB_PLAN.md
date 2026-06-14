# Colab/GPU climb training plan — synthesis of the 2026-06-15 Mac-CPU campaign

We spent ~52 RL runs + a scripted-demo attempt trying to learn the **floor → 0.22 m
platform climb** on Mac CPU. All three approaches (PPO, SAC, open-loop script)
converged on the **same wall**: the climb needs *closed-loop single-support balance
during a forward step-up*, which is an underactuated feedback problem that CPU-scale
RL couldn't explore and open-loop scripting can't hold. This matches the project's
prior result (the existing MJX Colab run got **floor 0/20**). GPU/MJX (thousands of
parallel envs) is the realistic path. **phase2 "sit from atop the step" is already
DELIVERED (g1_real_chair_sit.py, 20/20).** This doc is what to run on Colab so the
next GPU run does better than 0/20 — every item below is a hard-won lesson.

## What the existing MJX infra already has (don't redo)
`training/g1_climb_mjx_env.py` (`G1ClimbBox` extends Playground `Joystick`) + the
brax PPO pipeline in `training/g1_climb_colab.ipynb` already provide:
- the 0.22 m platform with **explicit foot/shin/thigh→platform contact pairs**
  (feet are contype=0; pairs are mandatory — we re-confirmed this gotcha on CPU);
- RSI reset modes: 4-point bridge 25%, **one-leg curriculum 15%**, on-platform 25%,
  floor/approach 35%;
- the full 29-DOF action, reward scaffold in `climb_config()`.

## The changes that matter (priority order — this is the new knowledge)

### 1. HEIGHT DOMAIN RANDOMIZATION + auto-curriculum  ← biggest lever
The prior 0/20 trained the **fixed 0.22 m** platform — too hard to discover from
scratch. We proved on CPU: 0.08 m direct is unlearnable but **0.02 m is learnable**,
and a **ladder catastrophically forgets**. The fix that avoids both:
- Sample platform height per episode from **[H_MIN, h_cap]** (start ~0.04) and **put
  the height in the observation**. Always randomizing the whole range keeps low
  heights practiced (no forgetting); MJX makes per-env height trivial (set the
  platform geom size/pos per env, or bucket envs by height).
- **Auto-curriculum**: raise `h_cap` by ~0.01–0.02 when floor-start success in the
  top band exceeds ~40% over enough episodes; ride it to 0.22 m. With thousands of
  envs this self-paces fast.

### 2. CLIMB-ONLY reframe (do NOT make RL learn to stand still)
Our biggest conceptual error: forcing RL to hold a still stand. Standing is **free**
via the SIT-mode stiff gains (zero-action holds the default pose, +reward) and the
mission FSM already holds a stiff stand after the climb. So:
- **Success = a BRIEF both-feet-upright arrival** on the platform near the target,
  not a long still hold. Then hand off to the stiff-hold / sit FSM.
- This sidesteps the limit cycles (march/bounce) that ate dozens of CPU runs.

### 3. Reward shaping — avoid the four traps we hit
- **Milking** (farm dense reward in a so-so pose): use **potential-based shaping**
  `r = Φ(s')−Φ(s)` (telescoping → staying earns ~0) for the climb progress, OR
  achievement-based + early termination. Don't pay per-step for an unfinished pose.
- **Leaning** (policy tips forward ~30°): put **uprightness in the reward with a
  NON-vanishing gradient** (penalty linear in the gravity error, not just exp() which
  goes flat at a big lean).
- **Walk-off / wander**: **anchor on the platform** (penalize base-y drift and leaving
  the platform footprint).
- **Thin-step contact artifact**: gate `on_platform` on the **contact-point height ≈
  platform top**, else a foot grazing the front face/edge false-positives.

### 4. Try SAC, not just PPO  ← new this session
On CPU, **PPO DIVERGED** on even the trivial on-step stand (peak-then-collapse,
ep_rew went negative while +reward was trivially available); `target_kl`/lr didn't
fix it. **SAC broke the posture wall PPO never could** (pitch 38°→2°). On GPU:
- brax PPO with thousands of envs + huge batch is far more stable than our CPU PPO
  (the divergence was partly tiny-batch), so PPO is still a fine first try **with**
  entropy + reward normalization + a KL guard;
- but **also run brax/MJX SAC** (off-policy, replay) as a parallel arm — it was the
  one thing that fixed the posture, and the replay handles "reach and hold" well.
Run both as a small sweep; keep the winner.

### 5. RSI coverage of the whole climb trajectory
The existing env seeds bridge/one-leg/on-platform. Add **mid-transfer states**
(one foot on the platform, weight shifting) so every phase is practiced — the gap
that blocked the floor→platform transition. **Harvest** them: as the policy starts
climbing at low heights, save its one-foot/on-platform frames and feed them back as
RSI (iterated, per height). The one-leg RSI mode is the right seed for the
single-support balance — keep/strengthen it (that balance is the core skill, and we
proved it's learnable in isolation: the v4 one-leg stand hit 20/20).

## Concrete recipe to run on Colab
1. **Env**: `G1ClimbBox` + changes 1–3, 5 above. Keep the contact pairs and one-leg
   RSI. Add `height` to obs; add `set_h_cap`; success = brief arrival.
2. **Algorithm sweep** (GPU is cheap): brax **PPO** (num_envs ≈ 4096–8192, large
   batch, entropy ≈ 1e-2 annealed, reward norm, KL guard) **and** MJX **SAC**.
3. **Curriculum**: start `h_cap≈0.04`, auto-expand to 0.22; ~100–300M env steps
   (minutes–hours on a T4/A100 at MJX throughput).
4. **Metric**: floor-start brief-arrival success **bucketed by height** (low/mid/high)
   — watch the high band cross 0, and the cap reach 0.22.
5. **Stop criterion**: floor-start success at 0.22 m ≳ 50%.

## Validation + deployment (train-on-GPU / run-on-Mac thesis)
- Eval floor-start climb success per height on GPU; render rollouts.
- **Export the policy to ONNX (torch/jax-free)** and run **inference on the Mac**
  (CPU) — exactly the project's deploy pattern (`deploy/export_climb_onnx.py`).
- **Chain**: navigate → back up to the chair → **climb (this policy)** → hand off to
  the **verified stiff-descent sit (g1_real_chair_sit.py, 20/20)**. That closes the
  full autonomous mission.

## One-line summary
Take the existing MJX env, and (1) randomize platform height + auto-curriculum to
0.22, (2) make success a brief arrival (let stiff gains hold the stand), (3) reward
with potential-based shaping + non-vanishing uprightness + on-platform anchor +
top-only contact, (4) sweep brax-PPO and MJX-SAC, (5) seed/harvest the full climb
trajectory incl. one-leg balance. Run 100–300M steps on GPU; export ONNX; run on Mac.
