"""A custom G1 SIT task for MuJoCo Playground — learn (via RL) to lower into a stable seated/low
posture, the balance-critical transition open-loop control topples on.

It subclasses Playground's G1 Joystick env and changes only the *reward* and the *command*:
  - command is always zero (no walking) — the robot should hold position, not travel;
  - reward pulls the base down to SIT_HEIGHT while staying upright, still, and alive (not toppling).
RL can balance this descent (closed-loop) where the hand-scripted / CEM open-loop versions fell.

EXPERIMENTAL: the reward WEIGHTS below are a starting point — expect to tune them across Colab
runs (that's the real RL work). To literally sit ON a chair, add a chair geom to the XML and a
"pelvis over seat" reward term; this first version targets a stable low/seated posture.

Import this module before loading "G1Sit" — importing it registers the env.
"""
import jax.numpy as jp
from mujoco_playground import registry
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

SIT_HEIGHT = 0.42          # target base height (m): chair-ish, well below the ~0.74 stand height


def sit_config():
    cfg = g1_joystick.default_config()
    rc = cfg.reward_config
    rc.base_height_target = SIT_HEIGHT
    s = rc.scales
    # turn OFF all the locomotion / gait incentives
    for k in ("tracking_lin_vel", "tracking_ang_vel", "feet_phase", "feet_air_time",
              "feet_height", "feet_clearance", "feet_slip", "lin_vel_z", "ang_vel_xy"):
        if k in s:
            s[k] = 0.0
    # turn ON the sit incentives (base_height/orientation are squared-error COSTS -> negative scale)
    s.base_height = -10.0      # strongly pull the base to SIT_HEIGHT
    s.orientation = -2.0       # keep the torso upright
    if "stand_still" in s:
        s.stand_still = 1.0    # don't drift around
    s.alive = 1.0              # stay up the whole episode
    if "pose" in s:
        s.pose = -0.5          # mild posture regularization
    return cfg


class G1Sit(g1_joystick.Joystick):
    """G1 learns to sit down: lower to SIT_HEIGHT, upright, still, without toppling."""

    def sample_command(self, rng):
        del rng                # no randomness: always "stay / sit", never a walk command
        return jp.zeros(3)


registry.locomotion.register_environment("G1Sit", G1Sit, sit_config)
