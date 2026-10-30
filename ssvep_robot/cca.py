r"""Standard CCA for SSVEP detection (Lin et al., 2006, IEEE TBME 53(12)).

For a test segment :math:`X \in \mathbb{R}^{N_s \times N_c}` and the sine-cosine
reference of stimulus frequency :math:`f_k`,

.. math::
    Y_k(t) = [\sin 2\pi f_k t,\ \cos 2\pi f_k t,\ \dots,\
              \sin 2\pi N_h f_k t,\ \cos 2\pi N_h f_k t]
    \in \mathbb{R}^{N_s \times 2N_h},

CCA finds :math:`w_x, w_y` maximising

.. math::
    \rho_k = \max_{w_x, w_y}
    \frac{w_x^\top X^\top Y_k w_y}
         {\sqrt{w_x^\top X^\top X w_x \; w_y^\top Y_k^\top Y_k w_y}}

(columns centred). The target is :math:`\hat k = \arg\max_k \rho_k`.

Implementation: with thin QR factorisations :math:`X = Q_x R_x`,
:math:`Y = Q_y R_y`, the canonical correlations are the singular values of
:math:`Q_x^\top Q_y` (Björck & Golub, 1973). This avoids explicit covariance
inversion and is numerically stable.
"""

from __future__ import annotations

import numpy as np


def cca_reference(freqs, n_samples: int, fs: float, n_harmonics: int,
                  phases=None) -> np.ndarray:
    """Sine-cosine references, shape ``[n_freqs, n_samples, 2*n_harmonics]``.

    Time index starts at ``1/fs`` as in the authors' reference code. ``phases``
    is accepted for completeness; CCA is invariant to it because sin/cos pairs
    span every phase of each harmonic.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    phases = np.zeros_like(freqs) if phases is None else np.asarray(phases, float)
    t = np.arange(1, n_samples + 1) / fs
    Y = np.empty((len(freqs), n_samples, 2 * n_harmonics))
    for i, (f, ph) in enumerate(zip(freqs, phases)):
        for h in range(1, n_harmonics + 1):
            Y[i, :, 2 * h - 2] = np.sin(2 * np.pi * h * f * t + h * ph)
            Y[i, :, 2 * h - 1] = np.cos(2 * np.pi * h * f * t + h * ph)
    return Y


def _orth(A: np.ndarray) -> np.ndarray:
    A = A - A.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(A, mode="reduced")
    return q


def canonical_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """Largest canonical correlation between ``X[n, p]`` and ``Y[n, q]``."""
    s = np.linalg.svd(_orth(X).T @ _orth(Y), compute_uv=False)
    return float(min(s[0], 1.0))


def cca_scores(X: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """:math:`\\rho_k` for one trial ``X[channel, sample]`` against all references."""
    qx = _orth(X.T)
    return np.array([min(np.linalg.svd(qx.T @ _orth(Yk), compute_uv=False)[0], 1.0)
                     for Yk in refs])


class CCA:
    """Training-free standard CCA classifier.

    Parameters
    ----------
    freqs : stimulus frequencies (Hz), one per class.
    fs : sampling rate.
    n_harmonics : :math:`N_h`.

    The trial is band-passed with sub-band 1 of the filter bank (6-90 Hz)
    before CCA, i.e. this is FBCCA with a single band.
    """

    requires_training = False

    def __init__(self, freqs, fs: float, n_harmonics: int = 5):
        self.freqs = np.asarray(freqs, float)
        self.fs = fs
        self.n_harmonics = n_harmonics

    def fit(self, X=None, y=None):
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """``X[trial, channel, sample]`` -> scores ``[trial, class]``."""
        from .filters import filter_bank
        Xf = filter_bank(X, self.fs, 1)[0]
        refs = cca_reference(self.freqs, X.shape[-1], self.fs, self.n_harmonics)
        return np.stack([cca_scores(x, refs) for x in Xf])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X).argmax(axis=1)
