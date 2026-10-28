import pytest

from sdp.seeds import DEMO_SEEDS, EVAL_SEEDS, check_demo, check_eval, demo_seeds, eval_seeds


def test_demo_and_eval_seed_ranges_do_not_overlap():
    assert set(DEMO_SEEDS).isdisjoint(EVAL_SEEDS)
    assert (DEMO_SEEDS.start, DEMO_SEEDS.stop) == (0, 1000)
    assert (EVAL_SEEDS.start, EVAL_SEEDS.stop) == (10000, 20000)


def test_helpers_stay_in_their_ranges():
    assert demo_seeds(50) == list(range(50))
    assert eval_seeds(50)[0] == 10000 and eval_seeds(50)[-1] == 10049
    assert set(demo_seeds(1000)).isdisjoint(eval_seeds(10000))


def test_out_of_range_seeds_are_rejected():
    with pytest.raises(ValueError):
        check_demo([999, 1000])
    with pytest.raises(ValueError):
        check_eval([5])
    with pytest.raises(ValueError):
        demo_seeds(10, start=995)
