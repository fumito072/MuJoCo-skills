"""Export the trained SB3 climb policy to ONNX with VecNormalize stats baked in.
The deploy bundle then needs ONLY onnxruntime (no torch, no SB3) on the robot.

    .venv-rl/bin/python deploy/export_climb_onnx.py [runs_climb/ppo_climb_latest]
"""
import os
import pickle
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "training"))

from stable_baselines3 import PPO  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "runs_climb", "ppo_climb_latest")
out = os.path.join(REPO, "deploy", "g1_climb_backstep.onnx")

model = PPO.load(path, device="cpu")
vn_path = path + "_vecnorm.pkl"
if not os.path.exists(vn_path):
    # CheckpointCallback naming: ppo_climb_vecnormalize_<steps>_steps.pkl
    import re
    mm = re.search(r"(\d+)_steps", path)
    vn_path = os.path.join(os.path.dirname(path),
                           f"ppo_climb_vecnormalize_{mm.group(1)}_steps.pkl")
with open(vn_path, "rb") as f:
    vn = pickle.load(f)
mean = torch.tensor(vn.obs_rms.mean, dtype=torch.float32)
std = torch.tensor(np.sqrt(vn.obs_rms.var + 1e-8), dtype=torch.float32)


class Deterministic(torch.nn.Module):
    """obs -> normalized -> policy mean action (tanh-free SB3 Gaussian head)."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, obs):
        o = torch.clip((obs - self.mean) / self.std, -10, 10)
        feats = self.policy.mlp_extractor.policy_net(
            self.policy.extract_features(o, self.policy.features_extractor))
        return torch.clip(self.policy.action_net(feats), -1.0, 1.0)


mod = Deterministic(model.policy).eval()
dummy = torch.zeros((1, model.observation_space.shape[0]), dtype=torch.float32)
with torch.no_grad():
    ref = mod(dummy).numpy()
torch.onnx.export(mod, dummy, out, input_names=["obs"], output_names=["action"],
                  opset_version=17, dynamo=False)

import onnxruntime as rt  # noqa: E402
sess = rt.InferenceSession(out, providers=["CPUExecutionProvider"])
got = sess.run(None, {"obs": dummy.numpy()})[0]
err = float(np.abs(got - ref).max())
print(f"exported {out}  ({os.path.getsize(out)} bytes)  torch-vs-onnx max err {err:.2e}")
assert err < 1e-5
