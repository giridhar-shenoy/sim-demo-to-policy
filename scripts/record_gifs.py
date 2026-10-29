"""Record the README GIF: one success and one ACT failure, side by side, labelled.

Seeds are taken from the eval CSV (first nominal success and first nominal failure,
falling back to the shifted rows), replayed with the same checkpoint, and checked
against the CSV label so the GIF shows episodes that are in the published table.
If the CSV has no ACT success, the left panel is the scripted expert on the failure's
seed, labelled as the reference, so the GIF shows what the policy should have done.
Frames come from the offscreen front camera (visual meshes); the inset in the top
right corner is the 96 px wrist image the policy actually sees.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sdp.envs import PandaPickPlaceEnv
from sdp.eval.act_policy import ACTPolicyAdapter
from sdp.eval.harness import read_csv
from sdp.expert.policy import ExpertPolicy

PANEL_HW = (240, 320)


def pick_seeds(csv_path: str) -> dict[str, tuple[int, str, str]]:
    """{'success': (seed, distribution, label), 'failure': (...)} from the eval CSV."""
    rows = read_csv(csv_path)
    picked: dict[str, tuple[int, str, str]] = {}
    for dist in ("nominal", "shifted"):
        for r in rows:
            key = "success" if r.success else "failure"
            if r.distribution == dist and key not in picked:
                picked[key] = (r.seed, r.distribution, r.label)
    return picked


def annotate(frame: np.ndarray, wrist: np.ndarray, text: str, colour: tuple) -> np.ndarray:
    img = Image.fromarray(frame)
    img.paste(Image.fromarray(wrist), (PANEL_HW[1] - wrist.shape[1] - 4, 26))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, PANEL_HW[1], 21], fill=(20, 20, 20))
    draw.text((5, 4), text, fill=colour, font=ImageFont.load_default())
    return np.asarray(img)


def rollout(env, policy, seed: int, dist: str) -> tuple[list[np.ndarray], dict]:
    obs, info = env.reset(seed=seed, options={"distribution": dist})
    policy.reset()
    frames = [(env.render(), obs["wrist"])]
    while True:
        obs, _, terminated, truncated, info = env.step(policy.act(obs, env))
        frames.append((env.render(), obs["wrist"]))
        if terminated or truncated:
            return frames, info


def main() -> None:
    from sdp.eval.taxonomy import label_episode

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="outputs/train/act_cpu_v2_scripted50")
    ap.add_argument("--csv", default="results/eval_act_cpu_v2_scripted50_s10100.csv")
    ap.add_argument("--out", default="docs/assets/act_success_failure.gif")
    ap.add_argument("--every", type=int, default=2, help="keep every n-th frame (20 Hz -> 10 fps)")
    ap.add_argument("--policy-label", default="ACT run 2", help="name shown on the policy panels")
    args = ap.parse_args()

    picked = pick_seeds(args.csv)
    policy = ACTPolicyAdapter(args.checkpoint)
    env = PandaPickPlaceEnv(
        render_mode="rgb_array", render_size=PANEL_HW, image_size=policy.image_size
    )
    panels = []
    try:
        if "success" not in picked and "failure" in picked:
            seed, dist, _ = picked["failure"]
            frames, info = rollout(env, ExpertPolicy(), seed, dist)
            label = label_episode(info["episode"], info)
            text = f"Scripted expert (reference) seed {seed}: {label.upper()}"
            panels.append([annotate(f, w, text, (150, 190, 255)) for f, w in frames[:: args.every]])
            print(f"no ACT success in {args.csv}; left panel = expert on seed {seed} -> {label}")
        for key in ("success", "failure"):
            if key not in picked:
                continue
            seed, dist, csv_label = picked[key]
            frames, info = rollout(env, policy, seed, dist)
            label = label_episode(info["episode"], info)
            if label != csv_label:
                raise RuntimeError(f"seed {seed}: replay gave {label}, CSV says {csv_label}")
            colour = (110, 230, 110) if key == "success" else (255, 120, 100)
            text = f"{args.policy_label}, seed {seed}: {label.upper()} in {len(frames) - 1} steps"
            panels.append([annotate(f, w, text, colour) for f, w in frames[:: args.every]])
            print(f"{key}: seed {seed} {dist} -> {label}, {len(frames) - 1} steps")
    finally:
        env.close()

    n = max(len(p) for p in panels)
    panels = [p + [p[-1]] * (n - len(p)) for p in panels]  # hold the last frame
    gap = np.full((PANEL_HW[0], 4, 3), 255, dtype=np.uint8)
    frames = [np.concatenate(sum(([p[i], gap] for p in panels), [])[:-1], axis=1) for i in range(n)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, frames, duration=1000 // (20 // args.every), loop=0)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out}: {n} frames, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
