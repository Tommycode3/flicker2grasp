r"""Wolpaw information transfer rate (Wolpaw et al., 1998/2002).

.. math::
    B = \log_2 N + P \log_2 P + (1 - P)\log_2\frac{1 - P}{N - 1}
    \quad \text{[bits/selection]},
    \qquad
    \mathrm{ITR} = B \cdot \frac{60}{T} \quad \text{[bits/min]},

where :math:`N` is the number of targets, :math:`P` the accuracy and
:math:`T = T_\text{window} + T_\text{gaze shift}` seconds per selection.
Assumes equiprobable targets and errors uniformly spread over the other
:math:`N-1` targets. For :math:`P < 1/N` we return 0 (as the authors' code).
"""

from __future__ import annotations

import numpy as np


def bits_per_selection(n: int, p: float) -> float:
    if n < 2:
        raise ValueError("need at least 2 targets")
    if not 0.0 <= p <= 1.0:
        raise ValueError("accuracy must be in [0, 1]")
    if p < 1.0 / n:
        return 0.0
    b = np.log2(n)
    if p > 0:
        b += p * np.log2(p)
    if p < 1:
        b += (1 - p) * np.log2((1 - p) / (n - 1))
    return float(b)


def itr(n: int, p: float, t: float) -> float:
    """ITR in bits/min for ``n`` targets, accuracy ``p`` and ``t`` seconds per selection."""
    if t <= 0:
        raise ValueError("t must be positive")
    return bits_per_selection(n, p) * 60.0 / t
