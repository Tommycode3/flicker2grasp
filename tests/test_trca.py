import numpy as np

from ssvep_robot.trca import TRCA, trca_filter

FS = 256.0


def test_trca_filter_recovers_mixing_direction():
    """With a single reproducible source, TRCA's filter isolates it from noise."""
    rng = np.random.default_rng(0)
    t = np.arange(256) / FS
    s = np.sin(2 * np.pi * 10 * t)
    a = rng.normal(size=8)
    trials = np.array([np.outer(a, s) + rng.normal(size=(8, 256)) for _ in range(20)])
    w = trca_filter(trials)
    y = np.einsum("c,tcs->ts", w, trials).mean(0)
    assert abs(np.corrcoef(y, s)[0, 1]) > 0.95


def test_etrca_classifies_synthetic_templates():
    rng = np.random.default_rng(1)
    t = np.arange(256) / FS
    freqs = [9.0, 10.0, 11.0, 12.0]
    X, y = [], []
    for k, f in enumerate(freqs):
        a = rng.normal(size=(8, 1))
        for _ in range(10):
            X.append(0.3 * a * np.sin(2 * np.pi * f * t + k) + rng.normal(size=(8, 256)))
            y.append(k)
    X, y = np.array(X), np.array(y)
    tr = np.arange(len(y)) % 10 < 8
    for ens in (False, True):
        m = TRCA(FS, n_bands=2, ensemble=ens).fit(X[tr], y[tr])
        assert (m.predict(X[~tr]) == y[~tr]).mean() == 1.0
