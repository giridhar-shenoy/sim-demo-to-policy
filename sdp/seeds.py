"""The seed split. Demonstrations and evaluation never share a seed.

A seed fully determines the initial cube pose (``env.reset(seed=...)``), so disjoint
seed ranges mean no eval episode starts from a pose a demonstration started from.
"""

DEMO_SEEDS = range(0, 1000)
EVAL_SEEDS = range(10000, 20000)


def demo_seeds(n: int, start: int = 0) -> list[int]:
    seeds = list(range(DEMO_SEEDS.start + start, DEMO_SEEDS.start + start + n))
    check_demo(seeds)
    return seeds


def eval_seeds(n: int, start: int = 0) -> list[int]:
    seeds = list(range(EVAL_SEEDS.start + start, EVAL_SEEDS.start + start + n))
    check_eval(seeds)
    return seeds


def check_demo(seeds) -> None:
    bad = [s for s in seeds if s not in DEMO_SEEDS]
    if bad:
        raise ValueError(f"demo seeds outside {DEMO_SEEDS}: {bad[:5]}")


def check_eval(seeds) -> None:
    bad = [s for s in seeds if s not in EVAL_SEEDS]
    if bad:
        raise ValueError(f"eval seeds outside {EVAL_SEEDS}: {bad[:5]}")
