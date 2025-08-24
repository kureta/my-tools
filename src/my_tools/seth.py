# pyright: basic

"""
Python translation of http://sethares.engr.wisc.edu/comprog.html
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy.signal import find_peaks


def dissonance(fvec, amp, model="min"):
    """
    Given a list of partials in fvec, with amplitudes in amp, this routine
    calculates the dissonance by summing the roughness of every sine pair
    based on a model of Plomp-Levelt's roughness curve.
    The older model (model='product') was based on the product of the two
    amplitudes, but the newer model (model='min') is based on the minimum
    of the two amplitudes, since this matches the beat frequency amplitude.
    """

    # Used to stretch dissonance curve for different freqs:
    Dstar = 0.24  # Point of maximum dissonance
    S1 = 0.0207
    S2 = 18.96

    C1 = 5
    C2 = -5

    # Plomp-Levelt roughness curve:
    A1 = -3.51
    A2 = -5.75

    Fmin = np.min(fvec, axis=-1)
    S = Dstar / (S1 * Fmin + S2)
    Fdif = np.max(fvec, axis=-1) - np.min(fvec, axis=-1)

    if model == "min":
        a = np.amin(amp, axis=-1)
    elif model == "product":
        a = np.prod(amp, axis=-1)  # Older model
    else:
        raise ValueError('model should be "min" or "product"')
    SFdif = S * Fdif
    D = np.sum(a * (C1 * np.exp(A1 * SFdif) + C2 * np.exp(A2 * SFdif)), axis=-1)

    return D


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

    tiled_spectrum0 = np.tile(spectrum0[None, :], (n_points, 1))
    tiled_amp0 = np.tile(amp0[None, :], (n_points, 1))

    swept_f1 = f0 * 2 ** (cents_sweep / 1200)
    normalized_spectrum1 = spectrum1 / f1
    swept_spectrum1 = swept_f1[:, None] * normalized_spectrum1[None, :]
    tiled_amp1 = np.tile(amp1[None, :], (n_points, 1))

    entire_spectrum = np.concatenate([tiled_spectrum0, swept_spectrum1], axis=1)
    entire_amps = np.concatenate([tiled_amp0, tiled_amp1], axis=1)

    i, j = np.triu_indices(entire_spectrum.shape[1], k=1)
    idx = np.stack((i, j), axis=1)
    bin_pairs = entire_spectrum[:, idx]
    amp_pairs = entire_amps[:, idx]

    return bin_pairs, amp_pairs, cents_sweep


def get_peaks(x_axis, curve, height=0.2):
    second_derivative = np.gradient(np.gradient(curve, x_axis), x_axis)
    second_derivative -= second_derivative.min()
    second_derivative /= second_derivative.max()

    dpeaks, _ = find_peaks(second_derivative, height=height)

    return x_axis[dpeaks]


def plot_curve(x_axis, curve, height=0.2, figsize=(12, 4), dpi=100):
    second_derivative = np.gradient(np.gradient(curve, x_axis), x_axis)
    second_derivative -= second_derivative.min()
    second_derivative /= second_derivative.max()

    dpeaks, _ = find_peaks(second_derivative, height=height)

    fig = Figure(figsize=figsize, dpi=dpi)
    ax1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax2 = ax1.twinx()

    ax1.plot(x_axis, curve, color="blue")
    ax2.plot(x_axis, second_derivative, color="gray", alpha=0.6)
    ax1.plot(x_axis[dpeaks], curve[dpeaks], "ro", label="minima")

    for xii in x_axis[dpeaks]:
        ax1.axvline(x=xii, color="b", linestyle="-", alpha=0.3)

    ax1.grid(axis="y", which="major", linestyle="--", color="gray", alpha=0.7)

    ax1.set_xlabel("interval in cents")
    ax1.set_ylabel("sensory dissonance")

    ax2.set_ylabel("peak strength (normalized)")
    ax1.set_xticks(
        x_axis[dpeaks],
        [f"{int(np.round(t))}" for t in x_axis[dpeaks]],
    )
    ax1.tick_params(axis="x", rotation=45, labelsize=8)

    return fig
