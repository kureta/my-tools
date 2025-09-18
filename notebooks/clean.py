import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import einx as ex
    from matplotlib.figure import Figure
    return Figure, ex, np


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


    def dissonance(fdiss, amp, axis=-1):
        a = np.min(amp, axis=axis)
        D = np.sum(a * fdiss, axis=axis)

        return D
    return dissonance, f_dissonance


@app.cell
def _(dissonance, ex, f_dissonance, np):
    n_points = 1000  # horizontal resolution of the dissonance curve
    n_partials = 16  # number of partials
    f0 = 1100  # fundamental frequency of the base voice

    # define x-axis
    ratio = np.linspace(1, 2, n_points)
    # setup frequencies of partials of the base voice
    p1 = ex.rearrange("a -> b a", f0 * np.arange(1, 1 + n_partials), b=n_points)
    # setup frequencies of partials of the 2nd voice and sweep through f0 * [1, 2]
    # from unison to octave of the base voice
    p2 = f0 * ex.multiply("a, b -> a b", ratio, np.arange(1, 1 + n_partials))
    # stack all partials
    # p.shape = (n_points, 2*n_partials)
    p = np.concatenate((p1, p2), axis=-1)

    # setup amplitudes of corresponding partials with exponential decay
    # amp.shape = (2*n_partials) because we assume amplitudes of partials always the same
    amp1 = 0.88 ** np.arange(n_partials)
    amp2 = 0.88 ** np.arange(n_partials)
    amp = np.concatenate((amp1, amp2), axis=-1)

    # get all pairs of indices
    idx = np.stack(np.triu_indices(n_partials * 2, k=1), axis=-1)
    # select all pairs of partial frequencies and amplitudes
    fvec = p[..., idx]
    ampvec = amp[..., idx]

    # calculate the dissonance curve
    values = dissonance(ex.reduce("a... 2 -> a...", fvec, op=f_dissonance), ampvec)
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
