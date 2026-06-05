"""Phase-A gate: benchmark SmolVLA inference on this Mac (CPU vs MPS), NVIDIA-free.

Resolves the #1 unknown of the VLA-on-Mac program: how fast does a pretrained VLA run on Apple
Silicon with no NVIDIA? Loads lerobot/smolvla_base (~450M, Apache-2.0), builds a dummy observation
matching the policy's expected inputs, and times an action-chunk prediction on CPU and MPS.

Run with the dedicated venv:  .venv-vla/bin/python bench_smolvla.py
"""
import time
import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("loading lerobot/smolvla_base (downloads ~2GB on first run) ...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.eval()
cfg = policy.config
print(f"loaded. chunk_size={getattr(cfg,'chunk_size',None)} n_action_steps={getattr(cfg,'n_action_steps',None)}")
print("input_features:")
for k, v in cfg.input_features.items():
    print(f"  {k}: type={getattr(v,'type',None)} shape={getattr(v,'shape',None)}")
print("output_features:")
for k, v in cfg.output_features.items():
    print(f"  {k}: shape={getattr(v,'shape',None)}")


from transformers import AutoTokenizer
_tok = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
INSTRUCTION = "pick up the red cube"


def dummy_batch(device):
    b = {}
    for k, v in cfg.input_features.items():
        shape = tuple(v.shape)
        if "visual" in str(getattr(v, "type", "")).lower() or "image" in k:
            b[k] = torch.rand((1, *shape), dtype=torch.float32, device=device)   # (1,3,H,W) in [0,1]
        else:
            b[k] = torch.zeros((1, *shape), dtype=torch.float32, device=device)
    enc = _tok(INSTRUCTION, padding="max_length", max_length=cfg.tokenizer_max_length,
               truncation=True, return_tensors="pt")
    b["observation.language.tokens"] = enc["input_ids"].to(device)
    b["observation.language.attention_mask"] = enc["attention_mask"].to(dtype=torch.bool, device=device)
    return b


def run_chunk(b):
    with torch.no_grad():
        if hasattr(policy, "predict_action_chunk"):
            return policy.predict_action_chunk(b)
        policy.reset()
        return policy.select_action(b)


for device in ["cpu", "mps"]:
    try:
        if device == "mps" and not torch.backends.mps.is_available():
            print("\nMPS not available — skipping"); continue
        policy.to(device)
        b = dummy_batch(device)
        # warmup
        for _ in range(2):
            run_chunk(b)
        if device == "mps":
            torch.mps.synchronize()
        ts = []
        for _ in range(6):
            t0 = time.time()
            out = run_chunk(b)
            if device == "mps":
                torch.mps.synchronize()
            ts.append(time.time() - t0)
        ms = 1000 * np.median(ts)
        chunk = getattr(cfg, "chunk_size", None) or (out.shape[1] if hasattr(out, "shape") and out.ndim > 1 else 1)
        print(f"\n[{device.upper()}] chunk inference: median {ms:.0f} ms  -> {1000/ms:.2f} chunks/s")
        print(f"           chunk={chunk} steps -> effective control rate ~ {chunk * 1000/ms:.0f} steps/s after interpolation")
    except Exception as e:
        print(f"\n[{device.upper()}] FAILED: {type(e).__name__}: {str(e)[:200]}")

print("\nGATE: a few chunks/s on CPU or MPS => Phase A is comfortable (chunk interpolates to a high effective rate).")
