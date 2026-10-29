"""Train the smallbc policy on recorded shards (CPU).

Adam, linear warmup then cosine decay to 0, batch 64, L1 on normalised K-step chunks,
random-shift augmentation, 10 % of episodes held out for validation loss. Logs to
``<out>/train_log.csv`` and checkpoints to ``<out>/ckpt_<step>.pt`` and ``<out>/last.pt``.

    python experiments/smallbc/train.py --data experiments/smallbc/data/demo5 \
        --out experiments/smallbc/runs/smoke --steps 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from data import PROPRIO_DIM, TARGET_DIM, Demos, norm_stats  # noqa: E402
from model import SmallBC  # noqa: E402


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return f


def to_t(*arrs):
    return [torch.from_numpy(np.ascontiguousarray(a)) for a in arrs]


@torch.no_grad()
def evaluate(model: SmallBC, demos: Demos, idx: np.ndarray, bs: int = 256) -> dict:
    model.eval()
    l1, xy, z, yaw, grip, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    xy_app, n_app = 0.0, 0  # frames in the expert's above_cube / descend phases
    for i in range(0, len(idx), bs):
        b = idx[i : i + bs]
        f, w = demos.images(b)
        ft, wt, pt, yt = to_t(f, w, demos.proprio[b], demos.labels[b])
        pred_n = model(ft, wt, pt)
        y_n = (yt - model.y_mean) / model.y_std
        l1 += float((pred_n - y_n).abs().mean()) * len(b)
        err = (pred_n * model.y_std + model.y_mean - yt).abs()  # raw units, (B, K, 5)
        xy += float(err[..., :2].mean()) * len(b)
        z += float(err[..., 2].mean()) * len(b)
        yaw += float(err[..., 3].mean()) * len(b)
        grip += float(err[..., 4].mean()) * len(b)
        app = torch.from_numpy(demos.phase[b] <= 1)
        if app.any():
            xy_app += float(err[app][..., :2].mean()) * int(app.sum())
            n_app += int(app.sum())
        n += len(b)
    model.train()
    return {
        "val_l1": l1 / n,
        "val_xy_mm": 1e3 * xy / n,
        "val_xy_mm_approach": 1e3 * xy_app / n_app if n_app else float("nan"),
        "val_z_mm": 1e3 * z / n,
        "val_yaw_deg": math.degrees(yaw / n),
        "val_grip": grip / n,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", required=True, nargs="+", help="one or more record.py output dirs")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--target", choices=("abs", "rel"), default="abs")
    p.add_argument("--no-proprio", action="store_true", help="cameras only")
    p.add_argument("--width", type=int, default=32, help="conv channels (layers use w, 2w, 2w, 2w)")
    p.add_argument("--num-kp", type=int, default=32)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--shift-pad", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-max-frames", type=int, default=2048)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-channels-last", action="store_true", help="channels_last is ~10 %% faster on CPU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-minutes", type=float, default=0.0, help="stop (and checkpoint) after this wall time; 0 = off")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    demos = Demos(args.data, chunk=args.chunk, mode=args.target)
    tr, va, val_seeds = demos.split(args.val_frac, seed=args.seed)
    y_stats = norm_stats(demos.labels[tr])
    p_stats = norm_stats(demos.proprio[tr])
    model = SmallBC(
        image_size=demos.image_size,
        width=args.width,
        num_kp=args.num_kp,
        use_proprio=not args.no_proprio,
        proprio_dim=PROPRIO_DIM,
        chunk=args.chunk,
        target_dim=TARGET_DIM,
        hidden=args.hidden,
        shift_pad=args.shift_pad,
    )
    model.set_stats(y_stats["mean"], y_stats["std"], p_stats["mean"], p_stats["std"])
    if not args.no_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    va_eval = np.sort(rng.choice(va, size=min(len(va), args.val_max_frames), replace=False)) if len(va) else va
    info = {
        "args": vars(args),
        "episodes": len(demos.episodes),
        "frames": demos.n_frames,
        "train_frames": int(len(tr)),
        "val_frames": int(len(va)),
        "val_seeds": val_seeds,
        "image_size": demos.image_size,
        "map_size": model.enc_front.map_size,
        "params": n_params,
        "load_s": round(time.perf_counter() - t_load, 2),
    }
    (out / "config.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "args"}), flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(min(args.warmup, args.steps // 10 + 1), args.steps))

    def save(step: int, name: str) -> None:
        ckpt = {
            "model": model.state_dict(),
            "model_cfg": model.cfg,
            "target_mode": args.target,
            "step": step,
            "args": vars(args),
            "val_seeds": val_seeds,
        }
        torch.save(ckpt, out / f"{name}.tmp")
        (out / f"{name}.tmp").replace(out / name)

    fields = ["step", "loss", "lr", "steps_per_s", "data_ms", "compute_ms", "elapsed_s",
              "val_l1", "val_xy_mm", "val_xy_mm_approach", "val_z_mm", "val_yaw_deg", "val_grip"]
    log_f = (out / "train_log.csv").open("w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=fields)
    writer.writeheader()

    t0 = time.perf_counter()
    win_t, win_loss, win_n, win_data, win_comp = time.perf_counter(), 0.0, 0, 0.0, 0.0
    step = 0
    try:
        for step in range(1, args.steps + 1):
            td = time.perf_counter()
            idx = tr[rng.integers(0, len(tr), args.batch)]
            f, w = demos.images(idx)
            ft, wt, pt, yt = to_t(f, w, demos.proprio[idx], demos.labels[idx])
            tc = time.perf_counter()
            loss = model.loss(ft, wt, pt, yt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            te = time.perf_counter()
            win_data += tc - td
            win_comp += te - tc
            win_loss += loss.item()
            win_n += 1

            do_val = len(va_eval) and (step % args.val_every == 0 or step == args.steps)
            if step % args.log_every == 0 or do_val or step == args.steps:
                now = time.perf_counter()
                row = {
                    "step": step,
                    "loss": round(win_loss / win_n, 5),
                    "lr": f"{sched.get_last_lr()[0]:.3e}",
                    "steps_per_s": round(win_n / (now - win_t), 3),
                    "data_ms": round(1e3 * win_data / win_n, 2),
                    "compute_ms": round(1e3 * win_comp / win_n, 2),
                    "elapsed_s": round(now - t0, 1),
                }
                if do_val:
                    row.update({k: round(v, 4) for k, v in evaluate(model, demos, va_eval).items()})
                writer.writerow(row)
                log_f.flush()
                print(row, flush=True)
                win_t, win_loss, win_n, win_data, win_comp = time.perf_counter(), 0.0, 0, 0.0, 0.0
            if step % args.ckpt_every == 0:
                save(step, f"ckpt_{step:06d}.pt")
                save(step, "last.pt")
            if args.max_minutes and time.perf_counter() - t0 > 60 * args.max_minutes:
                print(f"max-minutes reached at step {step}", flush=True)
                break
    finally:
        if step:
            save(step, "last.pt")
        log_f.close()


if __name__ == "__main__":
    main()
