import pytest

from sdp.eval.stats import wilson_interval


def test_wilson_known_values():
    # 50/100 -> 0.404..0.596 (standard table value)
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_edges_stay_in_unit_interval():
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and 0.05 < hi < 0.08
    lo, hi = wilson_interval(50, 50)
    assert hi == 1.0 and 0.92 < lo < 0.95


def test_wilson_rejects_bad_input():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(6, 5)
