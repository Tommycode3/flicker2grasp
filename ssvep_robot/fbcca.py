r"""Filter-bank CCA (Chen, Wang, Gao, Jung & Gao, 2015, J. Neural Eng. 12:046008).

Each test trial is decomposed into :math:`N` sub-bands
:math:`X^{(1)}, \dots, X^{(N)}` (sub-band :math:`m` starts at roughly
:math:`8m` Hz, so higher sub-bands keep only the higher harmonics). Standard
CCA gives :math:`\rho_k^{(m)}` per sub-band and the scores are fused as

.. math::
    \tilde\rho_k = \sum_{m=1}^{N} w(m)\,\big(\rho_k^{(m)}\big)^2,
    \qquad w(m) = m^{-a} + b,

with :math:`a = 1.25,\ b = 0.25,\ N = 5` (Chen et al. 2015). Squared
correlations follow the paper; the authors' MATLAB code sums :math:`w\rho`
instead - the ranking differences are negligible in practice.
Training-free.
"""

from __future__ import annotations

import numpy as np

from .cca import cca_reference, cca_scores
from .filters import fb_weights, filter_bank


class FBCCA:
    requires_training = False

    def __init__(self, freqs, fs: float, n_harmonics: int = 5, n_bands: int = 5,
                 a: float = 1.25, b: float = 0.25):
        self.freqs = np.asarray(freqs, float)
        self.fs = fs
        self.n_harmonics = n_harmonics
        self.n_bands = n_bands
        self.w = fb_weights(n_bands, a, b)

    def fit(self, X=None, y=None):
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xf = filter_bank(X, self.fs, self.n_bands)            # band, trial, ch, sample
        refs = cca_reference(self.freqs, X.shape[-1], self.fs, self.n_harmonics)
        rho = np.stack([[cca_scores(x, refs) for x in band] for band in Xf])  # band, trial, class
        return np.einsum("m,mtk->tk", self.w, rho ** 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X).argmax(axis=1)
