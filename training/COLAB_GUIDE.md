# Training a G1 policy on free Google Colab → run it on your Mac

The plan, in one line: **train on Colab's free GPU (minutes), download the tiny policy, run it on
your Mac CPU/MPS — no NVIDIA on your side.** Same "train once on GPU, infer local" split as the
walk policy we already replay.

Why Colab and not your Mac: RL needs thousands of simulation environments in parallel (MuJoCo MJX
on a GPU); Apple's GPU path for that (`jax-metal`) is dead, so the *training* runs on Colab's
NVIDIA T4. The trained network is small and runs fine on the Mac afterward.

---

## The order — step by step

### 0. One-time prep
- A Google account (free).
- These files from this repo: `training/g1_train_colab.py` (runnable pipeline) and
  `training/g1_sit_env.py` (the sit-task scaffold). You'll paste/upload them in Colab.

### 1. Open Colab + turn on the free GPU
1. Go to **https://colab.research.google.com** → **New notebook**.
2. Menu **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**. (Free.)
3. Confirm you got a GPU — run a cell:
   ```python
   !nvidia-smi
   ```
   You should see a Tesla T4. If it says "no GPU", you've hit the free-tier cap — try again later
   (peak hours UTC 14:00–18:00 are worst) or use Colab Pro.

### 2. Install MuJoCo Playground (the RL framework)
Run in a cell:
```python
!pip install -q playground "jax[cuda12]"
```
Then sanity-check the GPU is visible to JAX:
```python
import jax; print(jax.devices())   # -> [CudaDevice(id=0)]
```
> Version note: Playground pins its own compatible `mujoco-mjx`/`jax`. If you hit a
> `make_data() got an unexpected keyword 'nconmax'`-type error (a version drift we saw locally),
> pin to the combo the Playground release expects, e.g. `!pip install -q playground==<ver>` and
> let it pull matching deps — or use the official tutorial notebook (link in §7).

### 3. Mount Google Drive (so checkpoints survive disconnects)
```python
from google.colab import drive; drive.mount('/content/drive')
```
Free sessions disconnect after ~12 h (and ~90 min idle), so always save to Drive.

### 4. FIRST validate the pipeline with the stock G1 walk (proven to converge)
Upload `g1_train_colab.py` (left sidebar 📁 → upload), then:
```python
!python g1_train_colab.py --task walk --steps 20_000_000 --out /content/drive/MyDrive/g1_walk
```
Watch the reward climb. On a T4 this is ~minutes–tens of minutes. This proves *your* Colab can
train + save a G1 policy end-to-end before we attempt the harder sit task.

### 5. Train the SIT task (the goal)
Upload `g1_sit_env.py` too, then:
```python
!python g1_train_colab.py --task sit --steps 40_000_000 --out /content/drive/MyDrive/g1_sit
```
This uses the custom sit reward (low pelvis + upright + planted feet + don't topple). **Expect to
iterate**: RL reward design is the real work — if it doesn't sit, we adjust the reward weights in
`g1_sit_env.py` and re-run. Save every run to Drive.

### 6. Download the trained policy to your Mac
- Grab the `g1_sit_params.pkl` the script saved (use the notebook's `files.download(...)` cell, or
  the left 📁 sidebar → right-click → Download). `g1_sit_config.json` is optional metadata.
- Put it in this repo at `models/policies/g1_sit_params.pkl`.

### 7. Run it on your Mac (inference, NVIDIA-free)
Replay the trained policy in plain MuJoCo (C engine, CPU — no MJX, no NVIDIA) with the runner. It
rebuilds the policy with the exact training network config and reproduces the G1 Joystick obs/action,
holding the command at zero (sit), then writes a GIF + sit metrics (pelvis height, uprightness):
```bash
.venv-rl/bin/python training/g1_sit_play.py \
    --params models/policies/g1_sit_params.pkl --video assets/g1_sit.gif --seconds 6
```
Look for `RESULT: LOWERED & UPRIGHT (sit-like)` and a final `pelvis_z` near the 0.42 target. If it
topples or just stands, that's reward-tuning feedback → adjust weights in `g1_sit_env.py` and re-train.

---

## Free-tier limits to plan around (2026)
- **T4 GPU, 16 GB** — not guaranteed; may be unavailable at peak.
- **~12 h max session, ~90 min idle timeout, ~15–30 GPU-hours/week.**
- → keep runs short, **checkpoint to Drive**, resume across sessions. Colab Pro (~$10/mo) lifts
  these if you want longer runs.

## Reference
- MuJoCo Playground: https://playground.mujoco.org/
- Official MJX training Colab (fallback if our script hits a version issue):
  https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
- Playground repo: https://github.com/google-deepmind/mujoco_playground

## Honest expectations
Free Colab makes the **compute** free. The **work** is the RL task design (reward, termination,
the chair) and iterating until the sit converges — that's a real ML loop, not one click. We start
from the stock G1 walk (known-good) to de-risk, then tune the sit task together.
