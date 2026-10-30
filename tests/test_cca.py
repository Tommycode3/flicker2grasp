import numpy as np

from ssvep_robot.cca import CCA, canonical_correlation, cca_reference
from ssvep_robot.fbcca import FBCCA

FS = 256.0
FREQS = np.array([9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75])


def synth_trials(freqs, n_per=3, dur=2.0, snr_amp=0.5, n_ch=8, seed=0):
    """Multi-channel sinusoid (fundamental + 2nd harmonic, random phases and mixing) in white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * FS)) / FS
    X, y = [], []
    for k, f in enumerate(freqs):
        for _ in range(n_per):
            src = (np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
                   + 0.5 * np.sin(2 * np.pi * 2 * f * t + rng.uniform(0, 2 * np.pi)))
            mix = rng.normal(size=(n_ch, 1))
            X.append(snr_amp * mix * src + rng.normal(size=(n_ch, len(t))))
            y.append(k)
    return np.array(X), np.array(y)


def test_canonical_correlation_identity_and_bounds():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(500, 3))
    assert np.isclose(canonical_correlation(A, A @ rng.normal(size=(3, 3))), 1.0)
    r = canonical_correlation(A, rng.normal(size=(500, 2)))
    assert 0.0 <= r < 0.3


def test_reference_shape_and_orthogonality():
    Y = cca_reference([10.0], 256, FS, 3)[0]
    assert Y.shape == (256, 6)
    # integer number of cycles -> columns are (nearly) orthogonal
    G = Y.T @ Y / 128
    assert np.allclose(G, np.eye(6), atol=1e-6)


def test_cca_recovers_frequency_in_noise():
    X, y = synth_trials(FREQS)
    acc = (CCA(FREQS, FS, n_harmonics=3).predict(X) == y).mean()
    assert acc == 1.0


def test_fbcca_recovers_frequency_in_noise():
    X, y = synth_trials(FREQS, seed=3)
    acc = (FBCCA(FREQS, FS, n_harmonics=3, n_bands=3).predict(X) == y).mean()
    assert acc == 1.0


def test_cca_at_chance_on_pure_noise():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 8, 256))
    y = rng.integers(0, len(FREQS), 200)
    acc = (CCA(FREQS, FS, n_harmonics=3).predict(X) == y).mean()
    assert acc < 0.25          # chance = 1/8
