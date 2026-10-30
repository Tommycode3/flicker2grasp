"""Nakanishi et al. (2015) 12-class JFPM SSVEP dataset.

Download goes through MOABB (``moabb.datasets.Nakanishi2015.data_path``), which
fetches ``s{n}.mat`` from https://github.com/mnakanishi/12JFPM_SSVEP and caches
it under ``~/mne_data``. We then read the raw tensor with SciPy instead of using
MOABB's continuous-Raw wrapper, because leave-one-block-out evaluation needs the
block index, which the raw tensor keeps explicitly.

Tensor layout (dataset README): ``eeg[target, channel, sample, block]`` =
12 x 8 x 1114 x 15, fs = 256 Hz, stimulus onset at the 39th sample (0.15 s of
pre-stimulus data). Channels: PO7, PO3, POz, PO4, PO8, O1, Oz, O2.

Only one subject is held in memory at a time (~8.6 MB as float64).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 256.0
CHANNELS = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
# Order of targets in the tensor (dataset README).
FREQS = np.array([9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                  10.25, 12.25, 14.25, 10.75, 12.75, 14.75])
# JFPM phases in units of pi (Nakanishi 2015): 0, 0.5, 1, 1.5 per keypad column.
PHASES = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1.5]) * np.pi
ONSET_SAMPLE = 38          # README: onset at the "39th sample point" (1-based) -> index 38
LATENCY_S = 0.13           # visual latency, as in the authors' TRCA reference code
N_SUBJECTS = 9
N_BLOCKS = 15


@dataclass
class SubjectData:
    """One subject: ``X[trial, channel, sample]`` (full 1114-sample epochs), labels, blocks."""

    X: np.ndarray
    y: np.ndarray
    block: np.ndarray
    subject: int
    fs: float = FS


def load_subject(subject: int) -> SubjectData:
    """Load one subject via MOABB's cached download. Returns all 180 trials."""
    from moabb.datasets import Nakanishi2015
    from scipy.io import loadmat

    path = Nakanishi2015().data_path(subject)
    eeg = loadmat(path, variable_names=["eeg"])["eeg"].astype(np.float64)  # 12,8,1114,15
    n_t, n_c, n_s, n_b = eeg.shape
    X = eeg.transpose(0, 3, 1, 2).reshape(n_t * n_b, n_c, n_s)
    y = np.repeat(np.arange(n_t), n_b)
    block = np.tile(np.arange(n_b), n_t)
    return SubjectData(X=X, y=y, block=block, subject=subject)


def crop(X: np.ndarray, window_s: float, fs: float = FS,
         latency_s: float = LATENCY_S) -> np.ndarray:
    """Crop ``[onset + latency, onset + latency + window)`` from full epochs."""
    start = ONSET_SAMPLE + int(round(latency_s * fs))
    n = int(round(window_s * fs))
    if start + n > X.shape[-1]:
        raise ValueError(f"window {window_s}s exceeds epoch length")
    return X[..., start:start + n]
