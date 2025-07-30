import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    import numpy as np
    import numba as nb
    from scipy.signal import find_peaks
    return find_peaks, mo, nb, np, plt


@app.cell
def _(np):
    """
    Python translation of http://sethares.engr.wisc.edu/comprog.html
    """


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
    return (dissmeasure,)


@app.cell
def _(dissmeasure, find_peaks, np, plt):
    n_harm = 21
    freq1 = 261.63 * (np.array(range(1, n_harm + 1)) ** 1.05)
    freq2 = 261.63 * (np.array(range(1, n_harm + 1)) ** 1.05)
    amp = 0.88 ** np.array(range(0, n_harm))
    r_low = 1
    alpharange = 2.1
    method = "min"

    n = 3000
    diss = np.empty(n)
    a = np.concatenate((amp, amp))
    for i, alpha in enumerate(np.linspace(r_low, alpharange, n)):
        f = np.concatenate((freq1, alpha * freq2))
        d = dissmeasure(f, a, method)
        diss[i] = d

    # 2) find local minima: a point i is a local minimum if y[i-1] > y[i] < y[i+1]
    #    so we look for sign changes in the discrete derivative
    peaks, props = find_peaks(-diss, prominence=0.1)

    x = np.linspace(r_low, alpharange, len(diss))

    plt.figure(figsize=(12, 6))
    plt.plot(x, diss)
    plt.plot(x[peaks], diss[peaks], "ro", label="minima")
    plt.xscale("log")
    plt.xlim(r_low, alpharange)

    plt.xlabel("frequency ratio")
    plt.ylabel("sensory dissonance")

    # 1) draw vertical dashed lines at each minima
    for xi in x[peaks]:
        plt.axvline(x=xi, color="b", linestyle="-", alpha=0.3)

    plt.grid(axis="y", which="major", linestyle="--", color="gray", alpha=0.7)

    # 2) add ticks at those x‐positions and label them with their numerical values
    plt.minorticks_off()
    plt.xticks(x[peaks], [f"{int(np.round(t))}" for t in np.log2(x[peaks]) * 1200])

    plt.tight_layout()
    plt.gca()
    return (method,)


@app.function
def generate_scale(generator, octave, length):
    tmp = 0
    scale = []

    for idx in range(length - 1):
        tmp = (tmp + generator) % octave
        scale.append(tmp)

    scale.sort()

    return scale


@app.cell
def _():
    # 12-EDO for 1.05
    asd = []
    current = 0
    for _ in range(12):
        current = (current + 735) % 1260
        asd.append(current)

    asd.sort()
    print(asd)
    return


@app.cell
def _():
    # Pythagorian for 1.05
    print(generate_scale(737, 1260, 12))
    return


@app.cell
def _():
    # Pythagorian for 0.95
    print(generate_scale(667, 1140, 12))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""## My attempt at a cleaner and faster version of the dissonance curve calculation"""
    )
    return


@app.cell
def _(nb, np):
    @nb.njit
    def dissonance(freq1: float, freq2: float, amp1: float, amp2: float) -> float:
        b1 = 3.5
        b2 = 5.75
        x_star = 0.24
        s1 = 0.021
        s2 = 19

        min_freq = np.minimum(freq1, freq2)
        min_amp = np.minimum(amp1, amp2)
        delta_freq = np.abs(freq1 - freq2)
        s = x_star / (s1 * min_freq + s2)

        diss = min_amp * (
            np.exp(-b1 * s * (delta_freq)) - np.exp(-b2 * s * delta_freq)
        )

        return diss
    return (dissonance,)


@app.cell
def _(dissonance, np):
    n_harmonics = 16
    harmonics = np.arange(1, n_harmonics + 1)
    amps = 0.88 ** np.arange(1, n_harmonics + 1)

    base_freq = 440.0
    f1 = base_freq
    f2 = base_freq * 2 ** (1 / 12)

    spec1 = f1 * harmonics
    spec2 = f2 * harmonics

    specs = np.concatenate([spec1, spec2])
    ampiks = np.concatenate([amps, amps])

    indices = np.arange(len(specs))
    i1, i2 = np.meshgrid(indices, indices)
    i1, i2 = i1.ravel(), i2.ravel()

    np.sum(dissonance(specs[i1], specs[i2], ampiks[i1], ampiks[i2]))
    return ampiks, specs


@app.cell
def _(ampiks, dissmeasure, method, specs):
    dissmeasure(specs, ampiks, method)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
