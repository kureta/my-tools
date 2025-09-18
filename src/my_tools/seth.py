# pyright: basic

"""
Python translation of http://sethares.engr.wisc.edu/comprog.html
"""

import einops as eo
import numpy as np
from scipy.signal import find_peaks


def get_harmonic_spectrum(f0=440.0, n_harmonics=20, decay=0.88):
    harmonics = f0 * np.arange(1, n_harmonics + 1)
    amplitudes = decay ** np.arange(0, n_harmonics)

    return harmonics, amplitudes


"""
- When creating a dissonance curve of two sets of partials, the expected
  input shapes are: f1s, n_pairs, 2; f1s, n_pairs, 2
- When calculating a single roughness value, they are:
  2; 2
- When calculating roughness curve, they are:
  f1s, 2; 2
"""


def f_dissonance(fvec, axis):
    # Used to stretch dissonance curve for different freqs:
    Dstar = 0.24  # Point of maximum dissonance
    S1 = 0.0207
    S2 = 18.96

    C1 = 5
    C2 = -5

    # Plomp-Levelt roughness curve:
    A1 = -3.51
    A2 = -5.75

    Fmin = np.min(fvec, axis=axis)
    S = Dstar / (S1 * Fmin + S2)
    Fdif = np.max(fvec, axis=axis) - np.min(fvec, axis=axis)

    SFdif = S * Fdif
    D = C1 * np.exp(A1 * SFdif) + C2 * np.exp(A2 * SFdif)

    return D


def dissonance(fdiss, amp, axis=-1):
    a = np.min(amp, axis=axis)
    D = np.sum(a * fdiss, axis=axis)

    return D


def roughness(frequency_pairs, axis):
    pass


def prepare_sweep(
    f0,
    spectrum0,
    amp0,
    f1,
    spectrum1,
    amp1,
    start_cents=0,
    end_cents=1200,
    resolution_cents=1.0,
):
    n_points = int(np.round((end_cents - start_cents) / resolution_cents))
    # End point excluded to easily concatenate curves
    cents_sweep = np.linspace(start_cents, end_cents, n_points, endpoint=False)

    tiled_spectrum0 = eo.repeat(spectrum0, "a -> b a", b=n_points)

    normalized_spectrum1 = spectrum1 / f1
    swept_spectrum1 = f0 * eo.einsum(
        2 ** (cents_sweep / 1200), normalized_spectrum1, "a, b -> a b"
    )

    entire_spectrum = np.concatenate([tiled_spectrum0, swept_spectrum1], axis=-1)
    entire_amps = np.concatenate([amp0, amp1], axis=-1)

    # get all pairs of indices
    idx = np.stack(np.triu_indices(len(spectrum0) * 2, k=1), axis=-1)
    # select all pairs of partial frequencies and amplitudes
    bin_pairs = entire_spectrum[..., idx]
    amp_pairs = entire_amps[..., idx]

    return bin_pairs, amp_pairs, cents_sweep


# TODO: Use a combination of the 2nd derivative and dissonance value.
# Higher octaves are more consonant than lower octaves but they are
# not selected because their 2nd derivaties are relatively smaller
# but their dissonance values are way lower.
def get_peaks(x_axis, curve, height=0.2):
    second_derivative = np.gradient(np.gradient(curve, x_axis), x_axis)
    # second_derivative /= curve
    second_derivative -= second_derivative.min()
    second_derivative /= second_derivative.max()
    # second_derivative = np.log(second_derivative) + 3.1
    # second_derivative /= 3.1

    dpeaks, _ = find_peaks(second_derivative, height=height)

    return dpeaks, second_derivative
