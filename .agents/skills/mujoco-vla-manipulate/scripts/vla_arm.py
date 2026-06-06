"""Stage B: a real V+L+A loop on a standing G1 in MuJoCo, NVIDIA-free.

Closes the loop the voice demo skipped: the robot's HEAD CAMERA (V) + a language instruction (L)
go into the pretrained SmolVLA, which outputs an action chunk (A) that drives the standing G1's
right arm. Legs/torso are held in the stand pose; locomotion stays decoupled (the walk skill).

HONEST SCOPE: SmolVLA is trained on a tabletop SO-100 arm, not the G1, and on real photos, not
MuJoCo renders. So this is a *plumbing* demo — it proves the full V+L+A loop runs locally on a
Mac with no NVIDIA, and the arm moves in response to what the camera sees + what you say; it is
NOT a reliable grasp (that needs fine-tuning on G1 data = GPU, off-Mac).

Run:
  .venv-vla/bin/python vla_arm.py --probe                      # just save the camera + overview frames
  .venv-vla/bin/python vla_arm.py --instr "pick up the red block" --video assets/vla_arm.gif
"""
import os
import argparse
import numpy as np
import mujoco
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "models", "unitree_g1", "scene.xml"))

# right-arm actuators (verified): 22..28 = shoulder p/r/y, elbow, wrist r/p/y
RARM = list(range(22, 29))
DRIVE = RARM[:6]            # 6 actuators driven by SmolVLA's 6-dim action; wrist_yaw held


def build_scene():
    sp = mujoco.MjSpec.from_file(SCENE)
    table = sp.worldbody.add_body(name="table", pos=[0.5, 0.0, 0.35])
    g = table.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [0.22, 0.35, 0.35]; g.rgba = [0.55, 0.45, 0.35, 1]
    block = sp.worldbody.add_body(name="block", pos=[0.42, -0.12, 0.74])
    bg = block.add_geom(); bg.type = mujoco.mjtGeom.mjGEOM_BOX
    bg.size = [0.03, 0.03, 0.03]; bg.rgba = [0.9, 0.15, 0.12, 1]   # the red block
    return sp.compile()


def head_cam():
    c = mujoco.MjvCamera()
    c.lookat = [0.45, -0.05, 0.72]; c.distance = 0.75; c.azimuth = 180; c.elevation = -25
    return c


def view_cam():
    c = mujoco.MjvCamera()
    c.lookat = [0.35, 0.0, 0.8]; c.distance = 1.9; c.azimuth = 150; c.elevation = -20
    return c


def load_vla(device="mps"):
    import torch
    from transformers import AutoTokenizer
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").eval().to(device)
    tok = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    return policy, tok, policy.config, device


def vla_action(vla, rgb256, instruction, state6):
    import torch
    policy, tok, cfg, device = vla
    img = torch.from_numpy(rgb256).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0  # (1,3,256,256) in [0,1]
    b = {}
    for k, v in cfg.input_features.items():
        if "image" in k:
            b[k] = img
        elif "state" in k:
            b[k] = torch.from_numpy(state6.astype(np.float32)).unsqueeze(0).to(device)
    e = tok(instruction, padding="max_length", max_length=cfg.tokenizer_max_length, truncation=True, return_tensors="pt")
    b["observation.language.tokens"] = e["input_ids"].to(device)
    b["observation.language.attention_mask"] = e["attention_mask"].to(dtype=torch.bool, device=device)
    with torch.no_grad():
        chunk = policy.predict_action_chunk(b).float().cpu().numpy()[0]   # (50, 6)
    return chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr", default="pick up the red block")
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--video", default="")
    ap.add_argument("--probe", action="store_true", help="just save camera + overview frames and exit")
    ap.add_argument("--scale", type=float, default=1.0, help="VLA action -> arm-joint delta scale")
    args = ap.parse_args()

    m = build_scene(); d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(m, d, sid if sid >= 0 else 0)
    mujoco.mj_forward(m, d)
    stand_ctrl = m.key_qpos[sid][7:].copy()            # ctrl[a] = stand angle of the joint act a drives
    arm_lo = m.actuator_ctrlrange[DRIVE, 0]; arm_hi = m.actuator_ctrlrange[DRIVE, 1]
    stand_arm = stand_ctrl[DRIVE].copy()

    r_cam = mujoco.Renderer(m, 256, 256)               # the G1's head camera = the V input
    hcam = head_cam()

    if args.probe:
        r_cam.update_scene(d, camera=hcam); Image.fromarray(r_cam.render()).save("/tmp/vla_head.png")
        rv = mujoco.Renderer(m, 360, 480); rv.update_scene(d, camera=view_cam())
        Image.fromarray(rv.render()).save("/tmp/vla_view.png")
        print("saved /tmp/vla_head.png (VLA input) and /tmp/vla_view.png (overview)")
        print(f"block visible target; stand_arm={np.round(stand_arm,2)}")
        return

    vla = load_vla()
    r_view = mujoco.Renderer(m, 360, 480); vcam = view_cam()
    frames = []
    SIM_DT = m.opt.timestep
    VLA_EVERY = int(0.3 / SIM_DT)                        # re-infer ~3 Hz
    GIF_EVERY = int(round(1.0 / (10 * SIM_DT)))
    arm_target = stand_arm.copy()

    n = int(args.secs / SIM_DT)
    for c in range(n):
        if c % VLA_EVERY == 0:
            r_cam.update_scene(d, camera=hcam)
            rgb = r_cam.render()
            state6 = np.array(d.qpos[29:35])            # right-arm 6 joints (shoulder p/r/y, elbow, wrist r/p) as proprio
            chunk = vla_action(vla, rgb, args.instr, state6)
            act6 = chunk[0]                              # take the first action of the fresh chunk
            arm_target = np.clip(stand_arm + args.scale * act6, arm_lo, arm_hi)
            print(f"t={c*SIM_DT:4.1f}s  VLA action={np.round(act6,2)}", flush=True)
        d.ctrl[:] = stand_ctrl
        # smoothly move the driven arm joints toward the VLA target
        cur = d.ctrl[DRIVE]
        d.ctrl[DRIVE] = cur + np.clip(arm_target - cur, -0.02, 0.02)
        mujoco.mj_step(m, d)
        if args.video and c % GIF_EVERY == 0:
            r_view.update_scene(d, camera=vcam); over = Image.fromarray(r_view.render())
            r_cam.update_scene(d, camera=hcam); head = Image.fromarray(r_cam.render()).resize((360, 360))
            canvas = Image.new("RGB", (head.width + over.width, 360), (18, 18, 26))
            canvas.paste(head, (0, 0)); canvas.paste(over, (head.width, 0))   # [what the robot SEES | the robot]
            frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=64))

    upright = d.qpos[2] > 0.5
    print(f"done. upright={upright}  V+L+A loop ran on MPS, NVIDIA-free.")
    if args.video and frames:
        out = os.path.abspath(args.video); os.makedirs(os.path.dirname(out), exist_ok=True)
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
        print(f"video -> {out} ({os.path.getsize(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
