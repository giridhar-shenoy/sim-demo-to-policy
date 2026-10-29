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
            f"| {c['condition']} | {c['distribution']} | {c['seed_min']}–{c['seed_max']} "
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
        lines.append(f"| {c['condition']} | {c['distribution']} | {counts} |")
    return "\n".join(lines)


def training_table(results: Path) -> str:
    lines = [
        "| Run | Config | Steps | Wall time | Final loss | Machine |",
        "|---|---|---|---|---|---|",
    ]
    for path in sorted(results.glob("train_*.json")):
        meta = json.loads(path.read_text())
        csv_path = path.with_suffix(".csv")
        steps, loss = "–", "–"
        if csv_path.is_file():
            last = csv_path.read_text().strip().splitlines()
            header, final = last[0].split(","), last[-1].split(",")
            row = dict(zip(header, final, strict=False))
            steps = str(int(float(row["step"])))
            loss = f"{float(row['loss']):.3f}"
        wall = meta["wall_time_s"]
        lines.append(
            f"| {path.stem.removeprefix('train_')} | `{meta.get('config', '?')}` | {steps} "
            f"| {wall / 60:.1f} min | {loss} | {meta.get('machine', '?')} |"
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


HEADLINE_FILE = "eval_act_cpu_v2_scripted50_s10100.json"


def _act(results: Path) -> dict[str, dict]:
    path = results / HEADLINE_FILE
    if not path.is_file():
        return {}
    return {c["distribution"]: c for c in json.loads(path.read_text())["conditions"]}


def _rate(c: dict) -> str:
    lo, hi = c["wilson95"]
    return (
        f"{c['successes']}/{c['trials']} ({_pct(c['success_rate'])}, "
        f"Wilson 95 % {_pct(lo)}–{_pct(hi)})"
    )


def headline(results: Path) -> str:
    act = _act(results)
    if "nominal" not in act:
        return "_ACT evaluation pending._"
    c = act["nominal"]
    train = results / "train_act_cpu_v2_scripted50.json"
    minutes = json.loads(train.read_text())["wall_time_s"] / 60 if train.is_file() else None
    budget = f", {minutes:.0f} min on the laptop CPU" if minutes else ""
    text = (
        f"ACT (run 2: 3000 training steps{budget}, 50 scripted demonstrations) succeeds "
        f"in {_rate(c)} held-out test episodes (seeds {c['seed_min']}–{c['seed_max']})"
    )
    if "shifted" in act:
        text += f", and in {_rate(act['shifted'])} with the cube's start box shifted by 8 cm"
    return text + "."


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


TABLES = {
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
