"""Build the result tables from ``results/`` and write them into the docs.

Tables are written between marker comments, e.g.::

    <!-- sdp-report:success:begin -->
    ...generated...
    <!-- sdp-report:success:end -->

so every number in README.md and docs/results.md comes from a committed results file
and ``sdp report`` regenerates all of them. Unknown markers are left untouched.
"""

from __future__ import annotations

import csv
import json
import math
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
    data = json.loads(path.read_text())
    meta = data.get("meta", {})
    if meta.get("command"):
        return meta["command"]
    if path.name in SOURCES:
        return SOURCES[path.name]
    if path.name.startswith("eval_smallbc_") and meta.get("checkpoint"):
        c = data["conditions"][0]
        return (
            f"python experiments/smallbc/eval.py --ckpt {meta['checkpoint']} "
            f"--seed-start {c['seed_min']} --n {c['trials']} --distribution {c['distribution']} "
            f"--condition {c['condition']} --out results/{path.stem}"
        )
    return "scripts/eval_all.sh"


def _pct(x: float) -> str:
    return f"{100 * x:.0f} %"


# Training runs in order: table label and how each differs from the spec'd ACT CPU
# config (dim_model 256, 2 enc / 1 dec layers, chunk 20, 96x96, batch 8, 3000 steps).
RUNS = {
    "act_cpu_scripted50": ("run 1", "spec'd config; lr 1e-4, backbone lr 1e-5, CVAE on"),
    "act_cpu_v2_scripted50": ("run 2", "run 1 + backbone lr 1e-4, CVAE off"),
    "act_cpu_v3_scripted50": ("run 3", "run 2 + 8000 steps instead of the spec'd 3000"),
    "act_cpu_v4_scripted50": ("run 4", "run 2 + proprioceptive state withheld (cameras only)"),
    "act_cpu_v5_scripted50_128px": ("run 5", "run 4 + 128 px cameras (demos re-recorded)"),
    "act_cpu_v6_scripted50": ("run 6", "run 2 + lr 1e-5 and backbone lr 1e-5 (LeRobot defaults)"),
    "act_cpu_v7_abs": ("run 7", "run 2 + absolute TCP-target actions"),
    "act_cpu_v8_abs_dart": ("run 8", "run 7 + 200 demos, half with DART noise"),
    "smallbc_abs1000_25k": (
        "smallbc",
        "from-scratch CNN + spatial-softmax keypoints, MLP, absolute targets, "
        "1000 demo seeds (half DART), batch 64",
    ),
}
# The headline policy, fixed on dev seeds before any test eval (NOTES.md, session 8).
HEADLINE_RUN = "smallbc_abs1000_25k"
# Dataset each run was trained on (results/dataset_<name>.json).
RUN_DATASET = {run: "pickplace_scripted_50" for run in RUNS} | {
    "act_cpu_v5_scripted50_128px": "pickplace_scripted_50_128px",
    "act_cpu_v7_abs": "pickplace_abs_50",
    "act_cpu_v8_abs_dart": "pickplace_abs_dart_200",
    "smallbc_abs1000_25k": "smallbc_main1000",
}


def _label(run: str) -> str:
    """'ACT run N' for the ACT runs, the plain label otherwise (e.g. 'smallbc')."""
    label = RUNS[run][0]
    return f"ACT {label}" if label.startswith("run ") else label


def _policy_name(condition: str) -> str:
    if condition in RUNS:
        return f"{_label(condition)} (`{condition}`)"
    return condition


def _seed_groups(results: Path, run: str, ext: str = "json") -> dict[int, list[Path]]:
    """A run's eval files grouped by their first seed (one or several distributions each)."""
    groups: dict[int, list[Path]] = {}
    for f in sorted(results.glob(f"eval_{run}_s*.{ext}")):
        m = re.match(r"_s(\d+)(?:_[a-z]+)?$", f.stem[len(f"eval_{run}") :])
        if m:
            groups.setdefault(int(m.group(1)), []).append(f)
    return groups


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
        wall = f"{meta['wall_time_s'] / 60:.1f} min"
        if "checkpoint_step" in meta:  # the policy used is an intermediate checkpoint
            steps = f"{meta['checkpoint_step']} of {steps}"
            wall = f"{meta['wall_time_to_checkpoint_s'] / 60:.1f} min of {wall}"
            rows = csv.DictReader(path.with_suffix(".csv").open())
            at = next(r for r in rows if int(float(r["step"])) == meta["checkpoint_step"])
            loss = f"{float(at['loss']):.3f}"
        label, change = RUNS.get(run, (run, "–"))
        lines.append(
            f"| {label} (`{run}`) | {change} | {steps} | {wall} "
            f"| {loss} | `{meta.get('config', '?')}` |"
        )
    return "\n".join(lines)


def dataset_table(results: Path) -> str:
    lines = [
        "| Dataset | Image size | Episodes | Successful | Frames | Mean length | Demo seeds "
        "| Format |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for path in sorted(results.glob("dataset_*.json")):
        d = json.loads(path.read_text())
        lines.append(
            f"| {d['dataset']} | {d['image_size']} px | {d['episodes']} "
            f"| {d['successful_episodes']} | {d['frames']} "
            f"| {d['mean_episode_length_steps']:.1f} steps | {d['seeds'][0]}–{d['seeds'][1]} "
            f"| {d.get('format', 'LeRobot ' + d['codebase_version'])} |"
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
    """Each run's test-seed eval: its highest seed group (run 1's dev group comes first)."""
    out = {}
    for run in RUNS:
        groups = _seed_groups(results, run)
        if groups:
            files = groups[max(groups)]
            conds = [c for f in files for c in json.loads(f.read_text())["conditions"]]
            out[run] = {c["distribution"]: c for c in conds}
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


def _run_details(results: Path, run: str) -> str:
    details = []
    train = results / f"train_{run}.json"
    meta = json.loads(train.read_text()) if train.is_file() else {}
    if train.with_suffix(".csv").is_file():
        final = int(float(_final_row(train.with_suffix(".csv"))["step"]))
        if "checkpoint_step" in meta:
            details.append(f"checkpoint {meta['checkpoint_step']} of {final} training steps")
        else:
            details.append(f"{final} training steps")
    if "checkpoint_step" in meta:
        details.append(
            f"{meta['wall_time_to_checkpoint_s'] / 60:.1f} min on the laptop CPU to that checkpoint"
        )
    elif meta:
        details.append(f"{meta['wall_time_s'] / 60:.1f} min on the laptop CPU")
    ds = results / f"dataset_{RUN_DATASET.get(run, '')}.json"
    if ds.is_file():
        details.append(f"{json.loads(ds.read_text())['episodes']} demonstrations")
    return f"{_label(run)} ({RUNS[run][1]}; {', '.join(details)})"


def _run_list(runs: list[str]) -> str:
    nums = [int(RUNS[r][0].split()[-1]) for r in runs]
    if nums == list(range(nums[0], nums[0] + len(nums))) and len(nums) > 2:
        return f"runs {nums[0]}–{nums[-1]}"
    return "runs " + ", ".join(str(n) for n in nums)


def headline(results: Path) -> str:
    evals = {r: e for r, e in _test_evals(results).items() if "nominal" in e}
    if not evals:
        return "_ACT evaluation pending._"
    run = HEADLINE_RUN if HEADLINE_RUN in evals else _best_run(evals)
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
    text = (
        f"{_run_details(results, run)} succeeds in {_rate(c)} held-out test episodes "
        f"(seeds {c['seed_min']}–{c['seed_max']})"
    )
    if "shifted" in act:
        text += f", and in {_rate(act['shifted'])} with the cube's start box shifted by 8 cm"
    text += "."
    others = [r for r, e in evals.items() if r != run and e["nominal"]["successes"]]
    for r in reversed(others):
        e = evals[r]
        where = (
            "On the same seeds"
            if e["nominal"]["seed_min"] == c["seed_min"]
            else f"On its own test seeds {e['nominal']['seed_min']}–{e['nominal']['seed_max']}"
        )
        text += (
            f" {where}, {_run_details(results, r)} succeeds in {_rate(e['nominal'])} "
            f"nominal and {_rate(e['shifted'])} shifted episodes."
        )
    zero = [
        r
        for r, e in evals.items()
        if e["nominal"]["successes"] == 0 and e.get("shifted", {}).get("successes", 0) == 0
    ]
    if zero:
        text += (
            f" The earlier {_run_list(zero)}, with per-step delta actions, score 0/100 each on "
            f"their own test seeds."
        )
    return text


def mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test: a binomial(n, 1/2) test on the discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_table(results: Path) -> str:
    """Per-seed comparison of runs evaluated on exactly the same test seeds."""
    outcomes: dict[str, dict[int, dict[tuple[str, str], bool]]] = {}
    for run in RUNS:
        for seed, files in _seed_groups(results, run, "csv").items():
            rows = [r for f in files for r in csv.DictReader(f.open())]
            outcomes.setdefault(run, {})[seed] = {
                (r["distribution"], r["seed"]): r["success"] == "True" for r in rows
            }
    lines = [
        "| Run A | Run B | Seeds | Cube poses | Both succeed | Only A | Only B | Neither "
        "| McNemar exact p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    runs = list(outcomes)
    for i, a in enumerate(runs):
        for b in runs[i + 1 :]:
            for seed in sorted(set(outcomes[a]) & set(outcomes[b])):
                oa, ob = outcomes[a][seed], outcomes[b][seed]
                if set(oa) != set(ob) or not (any(oa.values()) or any(ob.values())):
                    continue
                seeds = sorted(int(k[1]) for k in oa)
                for dist in sorted({d for d, _ in oa}):
                    keys = [k for k in oa if k[0] == dist]
                    both = sum(oa[k] and ob[k] for k in keys)
                    only_a = sum(oa[k] and not ob[k] for k in keys)
                    only_b = sum(ob[k] and not oa[k] for k in keys)
                    lines.append(
                        f"| {_label(a)} | {_label(b)} | {seeds[0]}–{seeds[-1]} | {dist} | {both} "
                        f"| {only_a} | {only_b} | {len(keys) - both - only_a - only_b} "
                        f"| {mcnemar_exact_p(only_a, only_b):.4g} |"
                    )
    return "\n".join(lines)


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


def criterion_met(d: dict, threshold: float = 0.1) -> bool | None:
    """The pass criterion fixed before runs 4-6: swapping images moves the first-step
    d_x or d_yaw by >= 0.1, and their std across training episodes is >= 0.1.

    Defined on normalised delta actions only; returns None for other action modes.
    """
    first = d["first_action_across_episodes"]
    if "d_x" not in first:
        return None
    swap = d.get("image_swap_max_abs_change", {})
    moved = max(swap.get("d_x", 0.0), swap.get("d_yaw", 0.0)) >= threshold
    varies = max(first["d_x"]["pred_std"], first["d_yaw"]["pred_std"]) >= threshold
    return moved and varies


def diagnostics_table(results: Path) -> str:
    lines = [
        "| Run / checkpoint | First-step d_x / d_y / d_yaw std, policy (data) "
        "| Swap change d_x / d_yaw | Max change, black images | Own demo seeds 0–9 "
        "| Criterion |",
        "|---|---|---|---|---|---|",
    ]
    for path in sorted(results.glob("openloop_*.json")):
        run = path.stem.removeprefix("openloop_")
        d = json.loads(path.read_text())
        f = d["first_action_across_episodes"]
        dims = ("d_x", "d_y", "d_yaw") if "d_x" in f else ("target_x", "target_y", "target_yaw")
        pred = " / ".join(f"{f[k]['pred_std']:.4f}" for k in dims)
        data = " / ".join(f"{f[k]['data_std']:.2f}" for k in dims)
        swap = d.get("image_swap_max_abs_change")
        swap_txt = f"{swap[dims[0]]:.4f} / {swap[dims[2]]:.4f}" if swap else "–"
        black = d.get("black_images_abs_change")
        black_txt = f"{max(black.values()):.2f}" if black else "–"
        seeds = results / f"trainseeds_{run}.json"
        own = "–"
        if seeds.is_file():
            t = json.loads(seeds.read_text())
            labels = ", ".join(f"{v} {k}" for k, v in t["labels"].items())
            own = f"{t['successes']}/{len(t['seeds'])} ({labels})"
        met = criterion_met(d) if swap else None
        verdict = "–" if met is None else ("met" if met else "not met")
        lines.append(f"| {run} | {pred} ({data}) | {swap_txt} | {black_txt} | {own} | {verdict} |")
    return "\n".join(lines)


def timing_table(results: Path) -> str:
    """Wall time of each stage of a run, where it was recorded (from run 5 on)."""
    lines = [
        "| Run | Demo recording (50 episodes) | Training | Eval (200 episodes) |",
        "|---|---|---|---|",
    ]
    for run, (label, _) in RUNS.items():
        train = results / f"train_{run}.json"
        if not train.is_file():
            continue
        rec = "–"
        ds = results / f"dataset_{RUN_DATASET[run]}.json"
        if ds.is_file() and "recording_wall_time_s" in json.loads(ds.read_text()):
            rec = f"{json.loads(ds.read_text())['recording_wall_time_s']:.0f} s"
        ev = "–"
        groups = _seed_groups(results, run)
        if groups:
            metas = [json.loads(f.read_text()).get("meta", {}) for f in groups[max(groups)]]
            walls = [m.get("wall_time_s", m.get("wall_s")) for m in metas]
            if all(w is not None for w in walls):
                ev = f"{sum(walls) / 60:.1f} min"
        wall = json.loads(train.read_text())["wall_time_s"] / 60
        lines.append(f"| {label} (`{run}`) | {rec} | {wall:.1f} min | {ev} |")
    return "\n".join(lines)


TABLES = {
    "paired": paired_table,
    "timing": timing_table,
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
