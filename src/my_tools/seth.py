# pyright: basic

"""
Python translation of http://sethares.engr.wisc.edu/comprog.html
"""
import matplotlib.pyplot as plt
import numpy as np


# interval edges (inclusive at left, exclusive at right except last)
edges = np.array(
    [
        20,
        100,
        400,
        510,
        630,
        770,
        920,
        1080,
        1270,
        1480,
        1720,
        2000,
        2320,
        2700,
        3150,
        3700,
        4400,
        5300,
        6400,
        7700,
        9500,
        12000,
        15500,
    ]
)

# bandwidths for each interval
bandwidths = np.array(
    [
        80,
        100,
        110,
        120,
        140,
        150,
        160,
        190,
        210,
        240,
        280,
        320,
        380,
        450,
        550,
        700,
        900,
        1100,
        1300,
        1800,
        2500,
        3500,
    ]
)


def get_critical_bandwidth(f):
    # check bounds
    if f.min() < edges[0] or f.max() >= edges[-1]:
        raise ValueError(f"Frequencies must be in [20, 15500) {f=}")

    # digitize returns indices 1..len(edges); subtract 1 to index bandwidths
    idx = np.digitize(f, edges) - 1
    return bandwidths[idx][:, 0]


def diso(f0, f1, a0, a1):
    k = 5
    cbw = get_critical_bandwidth(f0)
    abs_delta_cbw = np.abs((f1 - f0) / cbw)
    plomp = k * abs_delta_cbw
    plomp *= np.exp(1 - k * abs_delta_cbw)
    # TODO: can also get minimum of amplitudes.
    # plomp *= a0 * a1
    plomp *= np.minimum(a0, a1)

    return plomp


def dissmeasure(fvec, amp, model="min"):
    """
    Given a list of partials in fvec, with amplitudes in amp, this routine
    calculates the dissonance by summing the roughness of every sine pair
    based on a model of Plomp-Levelt's roughness curve.
    The older model (model='product') was based on the product of the two
    amplitudes, but the newer model (model='min') is based on the minimum
    of the two amplitudes, since this matches the beat frequency amplitude.
    """
    # Sort by frequency
    sort_idx = np.argsort(fvec)
    am_sorted = np.asarray(amp)[sort_idx]
    fr_sorted = np.asarray(fvec)[sort_idx]

    # Used to stretch dissonance curve for different freqs:
    Dstar = 0.24  # Point of maximum dissonance
    S1 = 0.0207
    S2 = 18.96

    C1 = 5
    C2 = -5

    # Plomp-Levelt roughness curve:
    A1 = -3.51
    A2 = -5.75

    # Generate all combinations of frequency components
    idx = np.transpose(np.triu_indices(len(fr_sorted), 1))
    fr_pairs = fr_sorted[idx]
    am_pairs = am_sorted[idx]

    Fmin = fr_pairs[:, 0]
    S = Dstar / (S1 * Fmin + S2)
    Fdif = fr_pairs[:, 1] - fr_pairs[:, 0]

    if model == "min":
        a = np.amin(am_pairs, axis=1)
    elif model == "product":
        a = np.prod(am_pairs, axis=1)  # Older model
    else:
        raise ValueError('model should be "min" or "product"')
    SFdif = S * Fdif
    D = np.sum(a * (C1 * np.exp(A1 * SFdif) + C2 * np.exp(A2 * SFdif)))

    return D


def plot_this():
    """
    Reproduce Sethares Figure 3
    http://sethares.engr.wisc.edu/consemi.html#anchor15619672
    """
    freq = 500 * np.arange(1, 16)
    amp = 0.88 ** np.arange(0, 15)
    r_low = 1
    alpharange = 2.3
    method = "min"

    n = 3000
    diss = np.empty(n)
    a = np.concatenate((amp, amp))
    for i, alpha in enumerate(np.linspace(r_low, alpharange, n)):
        f = np.concatenate((freq, alpha * freq))
        d = dissmeasure(f, a, method)
        diss[i] = d

    plt.figure(figsize=(7, 3))
    plt.plot(np.linspace(r_low, alpharange, len(diss)), diss)
    plt.xlim(r_low, alpharange)

    plt.xlabel("frequency ratio")
    plt.ylabel("sensory dissonance")

    intervals = [(1, 1), (6, 5), (5, 4), (4, 3), (3, 2), (5, 3), (2, 1), (45, 32)]

    for n, d in intervals:
        plt.axvline(n / d, color="silver")

    plt.yticks([])
    plt.minorticks_off()
    plt.xticks(
        [n / d for n, d in intervals], ["{}/{}".format(n, d) for n, d in intervals]
    )
    plt.tight_layout()
    return plt.show()
