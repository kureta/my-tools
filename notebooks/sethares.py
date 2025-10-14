import marimo

__generated_with = "0.16.5"
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
    n_partials = 7
    stretch_factor = 1.0
    decay_factor = 0.88
    sweep_range = (-100, 1300)
    cents_per_sweep = 0.25
    return (
        amp0,
        cents_per_sweep,
        decay_factor,
        f0,
        n_partials,
        stretch_factor,
        sweep_range,
    )


@app.cell
def _(
    amp0,
    cents_per_sweep,
    decay_factor,
    f0,
    n_partials,
    st,
    stretch_factor,
    sweep_range,
):
    partials = st.generate_partial_freqs(f0, n_partials, stretch_factor)
    amplitudes = st.generate_partial_amps(amp0, n_partials, decay_factor)

    swept_partials = st.sweep_partials(partials, *sweep_range, cents_per_sweep)
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
        measure, height=measure.mean() + measure.std(), distance=5 * 4
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
    fig
    return measure, peaks, x, x_axis


@app.cell
def _(Figure, measure, x_axis):
    fig3 = Figure(figsize=(12, 5), dpi=300)
    ax3 = fig3.add_axes((0.05, 0.15, 0.9, 0.8))

    ax3.plot(x_axis, measure)
    return


@app.cell
def _(np, x):
    scale = np.array(
        list(
            set(
                [
                    int(z) % 1200
                    for z in (
                        list(np.round(x))
                        + list(np.round(x - 702))
                        + list(np.round(x + 702))
                    )
                ]
            )
        )
        + [1200]
    )

    scale
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


if __name__ == "__main__":
    app.run()
