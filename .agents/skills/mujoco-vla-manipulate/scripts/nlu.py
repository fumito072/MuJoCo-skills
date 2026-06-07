"""Natural-language understanding for the robot — a local LLM replaces the keyword rules.

Instead of matching fixed words, a small instruction-tuned LLM (Qwen2.5-3B, 4-bit, via Apple MLX,
NVIDIA-free) reads a free-form / vague / casual command (Japanese or English) and emits a JSON
PLAN: an ordered list of steps. This understands paraphrases the keyword parser can't — "もっと
進めー", "そのまま行って", "もうええわ止まって" — and compound goals like "椅子まで行って座って"
-> [goto chair, sit]. ~150-220 ms/command on an M5 Max.

Usage:  python nlu.py "あの障害物を避けて椅子に行って座れ"
"""
import os
import sys
import json
import importlib.util

DEFAULT_LLM = "mlx-community/Qwen2.5-3B-Instruct-4bit"

SYS = (
    "You are a humanoid robot's command planner. Convert the spoken command (Japanese or English, "
    "casual/vague is fine) into a JSON PLAN: an ordered list of steps. "
    'Format: {"plan":[{"action":...},...]}. '
    'Allowed steps: {"action":"forward","speed":"slow|normal|fast"}, {"action":"back"}, '
    '{"action":"turn","direction":"left|right"}, {"action":"stop"}, '
    '{"action":"goto","target":"<noun>","avoid":"<noun or null>"}, {"action":"sit"}. '
    "Examples: 'もっと進めー'->{\"plan\":[{\"action\":\"forward\",\"speed\":\"normal\"}]} ; "
    "'もうええ止まって'->{\"plan\":[{\"action\":\"stop\"}]} ; "
    "'ゆっくり前進'->{\"plan\":[{\"action\":\"forward\",\"speed\":\"slow\"}]} ; "
    "'椅子まで行って座って'->{\"plan\":[{\"action\":\"goto\",\"target\":\"chair\"},{\"action\":\"sit\"}]} ; "
    "'赤い箱を避けて椅子へ行って座れ'->{\"plan\":[{\"action\":\"goto\",\"target\":\"chair\",\"avoid\":\"red box\"},{\"action\":\"sit\"}]}. "
    "Output ONLY the JSON."
)

SPEED = {"slow": 0.25, "normal": 0.4, "fast": 0.55}
_CACHE = {}


def _load(model):
    from mlx_lm import load
    if model not in _CACHE:
        _CACHE[model] = load(model)
    return _CACHE[model]


def understand(text, model=DEFAULT_LLM):
    """free-form text -> plan (list of step dicts). [] if it can't parse."""
    from mlx_lm import generate
    m, tok = _load(model)
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": text}],
        add_generation_prompt=True)
    out = generate(m, tok, prompt=prompt, max_tokens=128, verbose=False)
    js = out[out.find("{"):out.rfind("}") + 1]
    try:
        return json.loads(js).get("plan", [])
    except Exception:
        return []


def to_live_intents(plan):
    """Map plan steps to language_nav-style intents. Directional steps are executable now by the
    live walk; goto/sit are passed through (need the Stage-C navigate+sit executor)."""
    out = []
    for s in plan:
        a = s.get("action")
        if a == "forward":
            out.append({"kind": "forward", "speed": SPEED.get(s.get("speed", "normal"), 0.4)})
        elif a == "back":
            out.append({"kind": "back"})
        elif a == "stop":
            out.append({"kind": "stop"})
        elif a == "turn":
            out.append({"kind": "turn", "deg": 90 if s.get("direction") == "left" else -90})
        elif a in ("goto", "sit"):
            out.append({"kind": a, "target": s.get("target"), "avoid": s.get("avoid")})
    return out


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "椅子のところまで行って座って"
    plan = understand(text)
    print(f"command : {text}")
    print(f"plan    : {json.dumps(plan, ensure_ascii=False)}")
    print(f"intents : {json.dumps(to_live_intents(plan), ensure_ascii=False)}")
