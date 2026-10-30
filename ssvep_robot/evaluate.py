"""Leave-one-block-out (LOBO) evaluation, one subject at a time.

For each of the 15 blocks: calibrated methods (TRCA, eTRCA) are fit on the
other 14 blocks and tested on the held-out block; training-free methods (CCA,
FBCCA) simply score every trial (no fitting, so folds are equivalent to
scoring all trials once). Every prediction is therefore out-of-sample.
"""

from __future__ import annotations

import numpy as np

from .cca import CCA
from .data import FREQS, SubjectData, crop
from .fbcca import FBCCA
from .trca import TRCA

METHODS = ("CCA", "FBCCA", "TRCA", "eTRCA")


def make_method(name: str, freqs, fs: float):
    if name == "CCA":
        return CCA(freqs, fs, n_harmonics=5)
    if name == "FBCCA":
        return FBCCA(freqs, fs, n_harmonics=5, n_bands=5)
    if name == "TRCA":
        return TRCA(fs, n_bands=5, ensemble=False)
    if name == "eTRCA":
        return TRCA(fs, n_bands=5, ensemble=True)
    raise KeyError(name)


def lobo_predict(sd: SubjectData, method: str, window_s: float,
                 classes=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-sample predictions for every trial of one subject.

    ``classes`` restricts the problem to a subset of targets (labels are
    re-indexed 0..len(classes)-1). Returns ``(y_true, y_pred, block)``.
    """
    classes = np.arange(len(FREQS)) if classes is None else np.asarray(classes)
    keep = np.isin(sd.y, classes)
    remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[c] for c in sd.y[keep]])
    block = sd.block[keep]
    X = crop(sd.X[keep], window_s, sd.fs)
    model = make_method(method, FREQS[classes], sd.fs)

    pred = np.empty_like(y)
    if not model.requires_training:
        pred[:] = model.predict(X)
    else:
        for b in np.unique(block):
            tr, te = block != b, block == b
            pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
    return y, pred, block
