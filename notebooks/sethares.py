import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import my_tools.seth as st
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import numpy as np
    from scipy.signal import find_peaks
    return Figure, find_peaks, np, plt, st


@app.cell
def _():
    f0 = 440
    amp0 = 1.0
    n_partials = 8
    decay_factor = 0.9
    stretch_factor1 = 1.05
    stretch_factor2 = 1.0
    sweep_range = (-50, 1450)
    cents_per_sweep = 0.125
    return (
        amp0,
        cents_per_sweep,
        decay_factor,
        f0,
        n_partials,
        stretch_factor1,
        stretch_factor2,
        sweep_range,
    )


@app.cell
def _(
    amp0,
    cents_per_sweep,
    decay_factor,
    f0,
    find_peaks,
    n_partials,
    normalize,
    np,
    st,
):
    def interpolation():
        interps = []
        for i in range(3, 10):
            stretch_factor1 = 1.0
            stretch_factor2 = 1.0 + i * 0.005
            sweep_range = (-200, 1300)

            partials = st.generate_partial_freqs(f0, n_partials, stretch_factor1)
            amplitudes = st.generate_partial_amps(amp0, n_partials, decay_factor)
            swept_partials = st.sweep_partials(
                st.generate_partial_freqs(f0, n_partials, stretch_factor2),
                *sweep_range,
                cents_per_sweep,
            )
            dissonance_curve = st.dissonance(
                partials, amplitudes, swept_partials, amplitudes
            )
            d2 = np.gradient(np.gradient(dissonance_curve))
            measure = np.minimum(normalize(d2), (1 - normalize(dissonance_curve)))
            peaks, _ = find_peaks(
                measure,
                height=measure.mean() + measure.std() * 0.7,
                distance=5 * 4,
            )

            x_axis = np.linspace(*sweep_range, len(dissonance_curve))
            x = x_axis[peaks]
            interps.append(x)
        return interps
    return


@app.cell
def _(
    amp0,
    cents_per_sweep,
    decay_factor,
    f0,
    n_partials,
    st,
    stretch_factor1,
    stretch_factor2,
    sweep_range,
):
    partials = st.generate_partial_freqs(f0, n_partials, stretch_factor1)
    amplitudes = st.generate_partial_amps(amp0, n_partials, decay_factor)

    swept_partials = st.sweep_partials(
        st.generate_partial_freqs(f0, n_partials, stretch_factor2),
        *sweep_range,
        cents_per_sweep,
    )
    return amplitudes, partials, swept_partials


@app.cell
def _(amplitudes, partials, st, swept_partials):
    dissonance_curve = st.dissonance(
        partials, amplitudes, swept_partials, amplitudes
    )
    return (dissonance_curve,)


@app.cell
def _(Figure, dissonance_curve, find_peaks, np, sweep_range):
    def normalize(x):
        x -= x.min()
        x /= x.max()
        return x


    d2 = np.gradient(np.gradient(dissonance_curve))
    measure = np.minimum(normalize(d2), (1 - normalize(dissonance_curve)))
    peaks, _ = find_peaks(
        measure, height=measure.mean() + measure.std() * 0.7, distance=5 * 4
    )

    x_axis = np.linspace(*sweep_range, len(dissonance_curve))

    fig = Figure(figsize=(12, 5), dpi=300)
    ax = fig.add_axes((0.05, 0.15, 0.9, 0.8))

    ax.plot(x_axis, dissonance_curve)

    x = x_axis[peaks]
    y = dissonance_curve[peaks]
    ax.scatter(x, y, color="C0")

    for xi, yi in zip(x, y):
        ax.annotate(
            f"{int(np.round(xi))}",
            xy=(xi, yi),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    list(np.round(x) / 100), len(x), fig
    return measure, normalize, peaks, x, x_axis


@app.cell
def _(Figure, measure, x_axis):
    fig3 = Figure(figsize=(12, 5), dpi=300)
    ax3 = fig3.add_axes((0.05, 0.15, 0.9, 0.8))

    ax3.plot(x_axis, measure)
    return


@app.cell
def _(np, x):
    subdominant = []
    tonic = []
    dominant = []

    for vv in x:
        subdominant.append(int(np.round((vv)) - 857) % 1380)
        tonic.append(int(np.round(vv)) % 1260)
        dominant.append(int(np.round((vv)) + 857) % 1380)

    subdominant.sort()
    tonic.sort()
    dominant.sort()

    total = sorted(list(set(subdominant + tonic + dominant)))
    total
    return dominant, subdominant, tonic


@app.cell
def _(dominant, subdominant, tonic):
    for q, r, t in zip(subdominant, tonic, dominant):
        print(f"{q}\t{r}\t{t}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Harmonic scale

    ```
    0    0   0
    112  0   85
    182  231 204
    267  267 316
    498  316 386
    498  386 471
    729  498 702
    765  583 702
    814  702 933
    884  814 969
    996  884 1018
    1081 969 1088
    ```

    Stretched scale

    ```
    0    0    0
    117  0    89
    192  243  214
    280  280  331
    523  331  406
    523  406  494
    766  523  737
    803  612  737
    854  737  980
    929  854  1017
    1046 929  1068
    1135 1017 1143
    ```
    """
    )
    return


@app.cell
def _(np, x):
    scale = np.array(
        list(
            set(
                [
                    int(z) % 1260
                    for z in (
                        list(np.round(x))
                        + list(np.round(x - 737))
                        + list(np.round(x + 737))
                    )
                ]
            )
        )
        + [1200]
    )
    scale.sort()
    return (scale,)


@app.cell
def _(Figure, measure, np, peaks, plt, scale):
    powers = np.concatenate(3 * [measure[peaks]])
    sorti = np.argsort(scale)
    powala = powers[sorti]
    scala = scale[sorti]

    delta = np.diff(scala)

    fig2 = Figure(figsize=(12, 5), dpi=300)
    ax2 = fig2.add_axes((0.05, 0.15, 0.9, 0.8))

    ax2.plot(delta)
    plt.close(fig2)
    fig2
    return


@app.cell
def _():
    # dc = st.dissonance(swept_partials, amplitudes, swept_partials, amplitudes)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
