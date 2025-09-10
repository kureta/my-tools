import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import einops as eo
    import librosa
    from matplotlib.figure import Figure

    from itertools import combinations
    import timeit
    return Figure, eo, np


@app.cell
def _(np):
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
    return (f_dissonance,)


@app.cell
def _(np):
    def dissonance(fdiss, amp):
        a = np.min(amp, axis=-1)
        D = np.sum(a * fdiss, axis=-1)

        return D
    return (dissonance,)


@app.cell
def _(dissonance, eo, f_dissonance, np):
    n_points = 1000
    n_partials = 6
    f0 = 1100

    ratio = np.linspace(1, 2, n_points)
    p1 = eo.repeat(f0 * np.arange(1, 1 + n_partials), "a -> b a", b=n_points)
    p2 = f0 * eo.einsum(ratio, np.arange(1, 1 + n_partials), "a, b -> a b")
    p = np.concatenate((p1, p2), axis=-1)

    amp1 = 1 / np.arange(1, 1 + n_partials)
    amp2 = 1 / np.arange(1, 1 + n_partials)
    amp = np.concatenate((amp1, amp2), axis=-1)

    idx = np.stack(np.triu_indices(n_partials * 2, k=1), axis=-1)
    fvec = p[..., idx]
    ampvec = amp[..., idx]

    values = dissonance(eo.reduce(fvec, "a b 2 -> a b", f_dissonance), ampvec)
    return ratio, values


@app.cell
def _(Figure, ratio, values):
    fig = Figure(figsize=(12, 4))
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.plot(ratio, values)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
