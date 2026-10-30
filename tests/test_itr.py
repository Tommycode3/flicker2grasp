import numpy as np
import pytest

from ssvep_robot.itr import bits_per_selection, itr


def test_perfect_accuracy_is_log2n():
    assert bits_per_selection(2, 1.0) == pytest.approx(1.0)
    assert bits_per_selection(12, 1.0) == pytest.approx(np.log2(12))
    # 40 targets, 100 %, 1 s per selection -> 60*log2(40) = 319.32 bits/min
    assert itr(40, 1.0, 1.0) == pytest.approx(319.316, abs=1e-3)


def test_chance_gives_zero():
    assert bits_per_selection(12, 1 / 12) == pytest.approx(0.0, abs=1e-12)
    assert bits_per_selection(4, 0.1) == 0.0


def test_hand_computed_value():
    # N=4, P=0.75: 2 + .75 log2 .75 + .25 log2(.25/3) = 0.792481...
    assert bits_per_selection(4, 0.75) == pytest.approx(0.7924812503605781, abs=1e-12)
    assert itr(4, 0.75, 2.0) == pytest.approx(0.7924812503605781 * 30, abs=1e-9)


def test_monotonic_in_accuracy():
    vals = [bits_per_selection(12, p) for p in np.linspace(1 / 12, 1, 50)]
    assert np.all(np.diff(vals) > 0)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        itr(12, 1.2, 1.0)
    with pytest.raises(ValueError):
        itr(12, 0.9, 0.0)
