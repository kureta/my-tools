import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import my_tools.seth as st
    import matplotlib.pyplot as plt
    return plt, st


@app.cell
def _():
    f0 = 440
    amp0 = 1.0
    n_partials = 6
    stretch_factor = 1.0
    decay_factor = 0.88
    return amp0, decay_factor, f0, n_partials, stretch_factor


@app.cell
def _(amp0, decay_factor, f0, n_partials, st, stretch_factor):
    partials = st.generate_partial_freqs(f0, n_partials, stretch_factor)
    amplitudes = st.generate_partial_amps(amp0, n_partials, decay_factor)

    swept_partials = st.sweep_partials(partials, 0, 1200, 1)
    return amplitudes, partials, swept_partials


@app.cell
def _(amplitudes, partials, st, swept_partials):
    dissonance_curve = st.dissonance(
        partials, amplitudes, swept_partials, amplitudes
    )
    return (dissonance_curve,)


@app.cell
def _(dissonance_curve, plt):
    plt.plot(dissonance_curve)
    return


@app.cell
def _(amplitudes, st, swept_partials):
    dc = st.dissonance(swept_partials, amplitudes, swept_partials, amplitudes)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
