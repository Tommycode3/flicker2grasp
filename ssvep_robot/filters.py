"""Band-pass and filter-bank design.

Filter bank follows the authors' reference implementation
(github.com/mnakanishi/TRCA-SSVEP, ``filterbank.m``) used for FBCCA/TRCA:
sub-band ``m`` (1-based) has passband ``[8m - 2, 90]`` Hz and stopband edges
``[stop_m, 100]`` Hz, Chebyshev type I, order from ``cheb1ord(Wp, Ws, 3 dB, 40 dB)``,
0.5 dB ripple, zero-phase (forward-backward) filtering.

We use second-order sections + ``sosfiltfilt`` for numerical stability; the
magnitude response is identical to the (b, a) form.

Filtering is applied *after* cropping the analysis window, so a short window
never sees samples outside itself (no look-ahead leakage).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import cheb1ord, cheby1, sosfiltfilt

PASSBAND = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
STOPBAND = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]


@lru_cache(maxsize=32)
def subband_sos(m: int, fs: float) -> np.ndarray:
    """SOS coefficients of sub-band ``m`` (1-based)."""
    nyq = fs / 2.0
    wp = [PASSBAND[m - 1] / nyq, 90.0 / nyq]
    ws = [STOPBAND[m - 1] / nyq, 100.0 / nyq]
    order, wn = cheb1ord(wp, ws, 3, 40)
    return cheby1(order, 0.5, wn, btype="bandpass", output="sos")


def filter_bank(X: np.ndarray, fs: float, n_bands: int) -> np.ndarray:
    """Apply sub-bands 1..n_bands. ``X[..., channel, sample]`` -> ``[band, ..., channel, sample]``."""
    return np.stack([sosfiltfilt(subband_sos(m, fs), X, axis=-1)
                     for m in range(1, n_bands + 1)])


def fb_weights(n_bands: int, a: float = 1.25, b: float = 0.25) -> np.ndarray:
    r"""Sub-band weights :math:`w(m) = m^{-a} + b` (Chen et al. 2015, Eq. 7)."""
    m = np.arange(1, n_bands + 1, dtype=float)
    return m ** (-a) + b
