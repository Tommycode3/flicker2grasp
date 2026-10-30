r"""Task-related component analysis (Nakanishi et al., 2018, IEEE TBME 65(1):104-112).

For class :math:`k`, let :math:`X_h \in \mathbb{R}^{N_c \times N_s}`,
:math:`h = 1..N_t`, be centred training trials. TRCA seeks a spatial filter
:math:`w` maximising the inter-trial covariance of :math:`y_h = w^\top X_h`:

.. math::
    \sum_{h_1 \neq h_2} \mathrm{Cov}(y_{h_1}, y_{h_2})
      = w^\top S w,
    \qquad
    S = \sum_{h_1 \neq h_2} X_{h_1} X_{h_2}^\top
      = \Big(\sum_h X_h\Big)\Big(\sum_h X_h\Big)^\top - \sum_h X_h X_h^\top,

subject to a variance constraint :math:`w^\top Q w = 1` with
:math:`Q = \bar U \bar U^\top`, :math:`\bar U = [X_1, \dots, X_{N_t}]`
(concatenated, centred). The Rayleigh quotient
:math:`\hat w = \arg\max_w \frac{w^\top S w}{w^\top Q w}` is the leading
eigenvector of the **generalized eigenproblem**

.. math::
    S w = \lambda Q w.

Classification (per sub-band :math:`m`): correlate the spatially filtered test
trial with the filtered class template :math:`\bar X_k` (training mean),

.. math::
    r_k^{(m)} = \mathrm{corr}\big(W^\top X^{(m)},\, W^\top \bar X_k^{(m)}\big),
    \qquad \rho_k = \sum_m w(m)\, r_k^{(m)},

where :math:`W = w_k` (TRCA) or :math:`W = [w_1, \dots, w_{N_f}]`
(**ensemble TRCA**: filters of all classes stacked; the 2-D Pearson
correlation is taken over the flattened matrices). Sub-band weights as in FBCCA.
This mirrors the authors' reference code (github.com/mnakanishi/TRCA-SSVEP).
Requires calibration data (here: the other 14 blocks).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .filters import fb_weights, filter_bank


def trca_filter(trials: np.ndarray) -> np.ndarray:
    """Leading TRCA spatial filter for ``trials[n_trials, channel, sample]``."""
    Xc = trials - trials.mean(axis=-1, keepdims=True)
    s = Xc.sum(axis=0)
    S = s @ s.T - np.einsum("tcs,tds->cd", Xc, Xc)
    U = np.concatenate(list(Xc), axis=1)
    Q = U @ U.T
    # Symmetric-definite generalized eigenproblem; eigh returns ascending order.
    _, V = eigh(S, Q)
    return V[:, -1]


def _corr_rows(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of ``A`` with vector ``b``."""
    A = A - A.mean(axis=1, keepdims=True)
    b = b - b.mean()
    return (A @ b) / np.sqrt(np.einsum("ij,ij->i", A, A) * (b @ b))


class TRCA:
    requires_training = True

    def __init__(self, fs: float, n_bands: int = 5, ensemble: bool = False):
        self.fs = fs
        self.n_bands = n_bands
        self.ensemble = ensemble
        self.w = fb_weights(n_bands)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        Xf = filter_bank(X, self.fs, self.n_bands)            # band, trial, ch, sample
        self.templates_ = np.stack([[Xf[m, y == k].mean(axis=0) for k in self.classes_]
                                    for m in range(self.n_bands)])   # band, class, ch, s
        self.filters_ = np.stack([[trca_filter(Xf[m, y == k]) for k in self.classes_]
                                  for m in range(self.n_bands)])     # band, class, ch
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xf = filter_bank(X, self.fs, self.n_bands)            # band, trial, ch, sample
        n_k = len(self.classes_)
        r = np.zeros((self.n_bands, X.shape[0], n_k))
        for m in range(self.n_bands):
            for k in range(n_k):
                W = self.filters_[m].T if self.ensemble else self.filters_[m, k][:, None]
                proj = np.einsum("cf,tcs->tfs", W, Xf[m]).reshape(X.shape[0], -1)
                tmpl = (W.T @ self.templates_[m, k]).ravel()
                r[m, :, k] = _corr_rows(proj, tmpl)
        return np.einsum("m,mtk->tk", self.w, r)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.decision_function(X).argmax(axis=1)]
