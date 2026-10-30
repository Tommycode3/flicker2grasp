import numpy as np
import pytest

from ssvep_robot.data import crop


def test_crop_window_alignment():
    X = np.arange(1114)[None, None, :].repeat(2, 0)
    w = crop(X, 1.0)
    assert w.shape[-1] == 256
    assert w[0, 0, 0] == 38 + 33      # onset (index 38) + 0.13 s latency
    with pytest.raises(ValueError):
        crop(X, 4.5)
