"""Build the result tables from ``results/`` and write them into the docs.

Tables are written between marker comments, e.g.::

    <!-- sdp-report:success:begin -->
    ...generated...
    <!-- sdp-report:success:end -->

so every number in README.md and docs/results.md comes from a committed results file
and ``sdp report`` regenerates all of them. Unknown markers are left untouched.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sdp.eval.taxonomy import LABELS

# Commands for results files whose JSON predates the "command" field.
SOURCES = {
    "expert_pickplace.json": "python scripts/eval_expert.py",
    "eval_act_cpu_scripted50_s10000.json": (
        "sdp eval --checkpoint outputs/train/act_cpu_scripted50 "
        "--condition act_cpu_scripted50 --trials 100 --distributions nominal shifted"
    ),
}


COMBINED = "eval_act_pickplace"


def _eval_files(results: Path) -> list[Path]:
    """The expert file plus every per-call ``sdp eval`` output (not the combined file)."""
    evals = [f for f in sorted(results.glob("eval_*.json")) if f.stem != COMBINED]
    files = [results / "expert_pickplace.json"] + evals
    return [f for f in files if f.is_file()]


def write_combined(results: Path) -> None:
    """results/eval_act_pickplace.{csv,json}: every ACT episode row and summary in one place."""
    per_call = [f for f in _eval_files(results) if f.stem.startswith("eval_act_")]
    if not per_call:
        return
    rows = []
    for f in per_call:
        rows += f.with_suffix(".csv").read_text().splitlines()[(1 if rows else 0) :]
    (results / f"{COMBINED}.csv").write_text("\n".join(rows) + "\n")
    summary = {
        "sources": [f.name for f in per_call],
        "conditions": [
            {**c, "source": f.name}
            for f in per_call
            for c in json.loads(f.read_text())["conditions"]
        ],
    }
    (results / f"{COMBINED}.json").write_text(json.dumps(summary, indent=2) + "\n")


def source_command(path: Path) -> str:
    meta = json.loads(path.read_text()).get("meta", {})
    return meta.get("command") or SOURCES.get(path.name, "scripts/eval_all.sh")


def _pct(x: float) -> str:
    return f"{100 * x:.0f} %"


# Training runs in order: table label and how each differs from the spec'd ACT CPU
# config (dim_model 256, 2 enc / 1 dec layers, chunk 20, 96x96, batch 8, 3000 steps).
RUNS = {
    "act_cpu_scripted50": ("run 1", "spec'd config; lr 1e-4, backbone lr 1e-5, CVAE on"),
    "act_cpu_v2_scripted50": ("run 2", "run 1 + backbone lr 1e-4, CVAE off"),
    "act_cpu_v3_scripted50": ("run 3", "run 2 + 8000 steps instead of the spec'd 3000"),
    "act_cpu_v4_scripted50": ("run 4", "run 2 + proprioceptive state withheld (cameras only)"),
}


def _policy_name(condition: str) -> str:
    if condition in RUNS:
        return f"ACT {RUNS[condition][0]} (`{condition}`)"
    return condition


def _load_conditions(results: Path) -> list[tuple[str, dict]]:
    rows = []
    for path in _eval_files(results):
        for cond in json.loads(path.read_text())["conditions"]:
            rows.append((path.name, cond))
    return rows


def success_table(results: Path) -> str:
    lines = [
        "| Policy / data | Cube poses | Seeds | Trials | Success | Wilson 95 % CI "
        "| Mean length, successes (steps) | Per-episode CSV |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in _load_conditions(results):
        lo, hi = c["wilson95"]
        mean_ok = c["mean_steps_success"]
        mean_txt = f"{mean_ok:.1f}" if mean_ok is not None else "–"
        csv_name = name.replace(".json", ".csv")
        lines.append(
            f"| {_policy_name(c['condition'])} | {c['distribution']} "
            f"| {c['seed_min']}–{c['seed_max']} "
            f"| {c['trials']} | {c['successes']}/{c['trials']} ({_pct(c['success_rate'])}) "
            f"| {_pct(lo)}–{_pct(hi)} | {mean_txt} "
            f"| [`results/{csv_name}`](results/{csv_name}) |"
        )
    return "\n".join(lines)


def taxonomy_table(results: Path) -> str:
    header = "| Policy / data | Cube poses | " + " | ".join(LABELS) + " |"
    lines = [header, "|---|---|" + "---|" * len(LABELS)]
    for _name, c in _load_conditions(results):
        counts = " | ".join(str(c["labels"].get(label, 0)) for label in LABELS)
        lines.append(f"| {_policy_name(c['condition'])} | {c['distribution']} | {counts} |")
    return "\n".join(lines)


def _final_row(csv_path: Path) -> dict[str, str]:
    lines = csv_path.read_text().strip().splitlines()
    return dict(zip(lines[0].split(","), lines[-1].split(","), strict=False))


def training_table(results: Path) -> str:
    lines = [
        "| Run | Change from the spec'd CPU config | Steps | Wall time | Final loss | Config |",
        "|---|---|---|---|---|---|",
    ]
    order = {run: i for i, run in enumerate(RUNS)}
    paths = results.glob("train_*.json")
    for path in sorted(paths, key=lambda p: order.get(p.stem.removeprefix("train_"), len(order))):
        run = path.stem.removeprefix("train_")
        meta = json.loads(path.read_text())
        steps, loss = "–", "–"
        if path.with_suffix(".csv").is_file():
            row = _final_row(path.with_suffix(".csv"))
            steps = str(int(float(row["step"])))
            loss = f"{float(row['loss']):.3f}"
        label, change = RUNS.get(run, (run, "–"))
        lines.append(
            f"| {label} (`{run}`) | {change} | {steps} | {meta['wall_time_s'] / 60:.1f} min "
            f"| {loss} | `{meta.get('config', '?')}` |"
        )
    return "\n".join(lines)


def dataset_table(results: Path) -> str:
    lines = [
        "| Dataset | Episodes | Successful | Frames | Mean length | Demo seeds | Format |",
        "|---|---|---|---|---|---|---|",
    ]
    for path in sorted(results.glob("dataset_*.json")):
        d = json.loads(path.read_text())
        lines.append(
            f"| {d['dataset']} | {d['episodes']} | {d['successful_episodes']} | {d['frames']} "
            f"| {d['mean_episode_length_steps']:.1f} steps | {d['seeds'][0]}–{d['seeds'][1]} "
            f"| LeRobot {d['codebase_version']} |"
        )
    return "\n".join(lines)


def speed_table(results: Path) -> str:
    path = results / "env_speed.json"
    if not path.is_file():
        return "_not measured_"
    d = json.loads(path.read_text())
    return "\n".join(
        [
            "| Env | With both 96 px cameras | Physics + IK only | GL backend |",
            "|---|---|---|---|",
            f"| {d['env']} | {d['steps_per_s_with_images']:.0f} steps/s "
            f"| {d['steps_per_s_without_images']:.0f} steps/s | {d['mujoco_gl']} |",
        ]
    )


def _test_evals(results: Path) -> dict[str, dict[str, dict]]:
    """Each run's test-seed eval: its highest-seed eval file (run 1's dev file comes first)."""
    out = {}
    for run in RUNS:
        files = sorted(results.glob(f"eval_{run}_s*.json"))
        if files:
            out[run] = {
                c["distribution"]: c for c in json.loads(files[-1].read_text())["conditions"]
            }
    return out


def _best_run(evals: dict[str, dict[str, dict]]) -> str:
    """Most nominal successes, then most shifted successes; ties go to the later run."""
    order = list(RUNS)
    return max(
        evals,
        key=lambda r: (
            evals[r]["nominal"]["successes"],
            evals[r].get("shifted", {}).get("successes", -1),
            order.index(r),
        ),
    )


def _rate(c: dict) -> str:
    lo, hi = c["wilson95"]
    return (
        f"{c['successes']}/{c['trials']} ({_pct(c['success_rate'])}, "
        f"Wilson 95 % {_pct(lo)}–{_pct(hi)})"
    )


def headline(results: Path) -> str:
    evals = {r: e for r, e in _test_evals(results).items() if "nominal" in e}
    if not evals:
        return "_ACT evaluation pending._"
    run = _best_run(evals)
    act = evals[run]
    c = act["nominal"]
    if c["successes"] == 0 and all(
        e.get("shifted", {}).get("successes", 0) == 0 for e in evals.values()
    ):
        runs = "; ".join(
            f"{RUNS[r][0]}, seeds {e['nominal']['seed_min']}–{e['nominal']['seed_max']}: "
            f"{RUNS[r][1]}"
            for r, e in evals.items()
        )
        return (
            f"None of the {len(evals)} ACT CPU runs succeeds: each scores {_rate(c)} on 100 "
            f"held-out nominal test episodes and 0/100 with the cube's start box shifted by "
            f"8 cm ({runs})."
        )
    train = results / f"train_{run}.json"
    details = []
    if train.with_suffix(".csv").is_file():
        details.append(
            f"{int(float(_final_row(train.with_suffix('.csv'))['step']))} training steps"
        )
    if train.is_file():
        details.append(
            f"{json.loads(train.read_text())['wall_time_s'] / 60:.1f} min on the laptop CPU"
        )
    details.append("50 scripted demonstrations")
    text = (
        f"Best of {len(evals)} ACT runs: {RUNS[run][0]} ({RUNS[run][1]}; {', '.join(details)}) "
        f"succeeds in {_rate(c)} held-out test episodes (seeds {c['seed_min']}–{c['seed_max']})"
    )
    if "shifted" in act:
        text += f", and in {_rate(act['shifted'])} with the cube's start box shifted by 8 cm"
    others = ", ".join(
        f"{RUNS[r][0]} {e['nominal']['successes']}/{e['nominal']['trials']}"
        for r, e in evals.items()
    )
    return text + f". All runs, nominal, on their own test seeds: {others}."


def index_table(results: Path) -> str:
    lines = [
        "| Benchmark | Metric | Value | Regenerate with |",
        "|---|---|---|---|",
    ]
    speed = results / "env_speed.json"
    if speed.is_file():
        d = json.loads(speed.read_text())
        lines.append(
            f"| Env throughput | steps/s, both 96 px cameras | "
            f"{d['steps_per_s_with_images']:.0f} | `python scripts/bench_env_speed.py` |"
        )
    for name, c in _load_conditions(results):
        if name.startswith("expert") and c["condition"] != "expert_demo_seeds":
            continue
        lines.append(
            f"| {c['condition']} ({c['distribution']}) | success, seeds "
            f"{c['seed_min']}–{c['seed_max']} | {_rate(c)} | `{source_command(results / name)}` |"
        )
    for path in sorted(results.glob("train_*.json")):
        meta = json.loads(path.read_text())
        lines.append(
            f"| Training {path.stem.removeprefix('train_')} | wall time | "
            f"{meta['wall_time_s'] / 60:.1f} min | `scripts/train_cpu.sh` |"
        )
    for path in sorted(results.glob("dataset_*.json")):
        d = json.loads(path.read_text())
        lines.append(
            f"| Dataset {d['dataset']} | episodes (successful) / mean length | "
            f"{d['episodes']} ({d['successful_episodes']}) / {d['mean_episode_length_steps']:.1f} "
            f"steps | `scripts/record_demos.sh` |"
        )
    return "\n".join(lines)


def diagnostics_table(results: Path) -> str:
    lines = [
        "| Run | First-step d_x / d_y / d_yaw std, policy (data) | Max change, images swapped "
        "| Max change, black images | Own demo seeds 0–9 |",
        "|---|---|---|---|---|",
    ]
    for path in sorted(results.glob("openloop_*.json")):
        run = path.stem.removeprefix("openloop_")
        d = json.loads(path.read_text())
        f = d["first_action_across_episodes"]
        dims = ("d_x", "d_y", "d_yaw")
        pred = " / ".join(f"{f[k]['pred_std']:.4f}" for k in dims)
        data = " / ".join(f"{f[k]['data_std']:.2f}" for k in dims)
        swap = max(d.get("image_swap_max_abs_change", {"-": float("nan")}).values())
        black = max(d.get("black_images_abs_change", {"-": float("nan")}).values())
        seeds = results / f"trainseeds_{run}.json"
        own = "–"
        if seeds.is_file():
            t = json.loads(seeds.read_text())
            labels = ", ".join(f"{v} {k}" for k, v in t["labels"].items())
            own = f"{t['successes']}/{len(t['seeds'])} ({labels})"
        lines.append(f"| {run} | {pred} ({data}) | {swap:.4f} | {black:.2f} | {own} |")
    return "\n".join(lines)


TABLES = {
    "diagnostics": diagnostics_table,
    "headline": headline,
    "index": index_table,
    "success": success_table,
    "taxonomy": taxonomy_table,
    "training": training_table,
    "dataset": dataset_table,
    "speed": speed_table,
}


def render(text: str, results: Path) -> str:
    for name, build in TABLES.items():
        pattern = re.compile(
            rf"(<!-- sdp-report:{name}:begin -->).*?(<!-- sdp-report:{name}:end -->)",
            re.DOTALL,
        )
        if pattern.search(text):
            table = build(results)
            text = pattern.sub(lambda m, t=table: f"{m.group(1)}\n{t}\n{m.group(2)}", text)
    return text


def update_docs(results: str | Path, targets: list[str]) -> None:
    results = Path(results)
    write_combined(results)
    for target in targets:
        path = Path(target)
        if not path.is_file():
            continue
        new = render(path.read_text(), results)
        if target != "README.md" and path.parent.name == "docs":
            # docs/ links are relative to docs/
            new = new.replace("](results/", "](../results/")
        path.write_text(new)
        print(f"updated {path}")
