from my_tools.page_one.state import state
from my_tools.seth import (
    dissonance,
    get_harmonic_spectrum,
    get_peaks,
    plot_curve,
    prepare_sweep,
)


# TODO: separate plotting and calculation
def show_plot():
    if not state.has_figure():
        return
    spectrum_1, amplitudes_1 = get_harmonic_spectrum(
        state.f1, state.n_harmonics, state.amp_decay  # pyright: ignore
    )
    spectrum_2, amplitudes_2 = get_harmonic_spectrum(
        state.f2, state.n_harmonics, state.amp_decay
    )

    overtone_pairs, amplitude_pairs, cents = prepare_sweep(
        state.f1,
        spectrum_1,
        amplitudes_1,
        state.f2,
        spectrum_2,
        amplitudes_2,
        state.start_delta_cents,
        state.start_delta_cents + state.delta_cents_range,
    )

    curve = dissonance(overtone_pairs, amplitude_pairs, state.method)
    peaks, d2curve = get_peaks(cents, curve, height=state.peak_cutoff)
    state.n_peaks = len(peaks)

    plot_curve(cents, curve, d2curve, peaks, state.figure)
    state.figure.element.update()
