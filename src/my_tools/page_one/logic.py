from my_tools.page_one.state import Config
from my_tools.seth import (
    dissonance,
    get_harmonic_spectrum,
    get_peaks,
    plot_curve,
    prepare_sweep,
)

conf = Config()


# TODO: separate plotting and calculation
def show_plot():
    if not conf.has_figure():
        return
    spectrum_1, amplitudes_1 = get_harmonic_spectrum(
        conf.f1, conf.n_harmonics, conf.amp_decay  # pyright: ignore
    )
    spectrum_2, amplitudes_2 = get_harmonic_spectrum(
        conf.f2, conf.n_harmonics, conf.amp_decay
    )

    overtone_pairs, amplitude_pairs, cents = prepare_sweep(
        conf.f1,
        spectrum_1,
        amplitudes_1,
        conf.f2,
        spectrum_2,
        amplitudes_2,
        conf.start_delta_cents,
        conf.start_delta_cents + conf.delta_cents_range,
    )

    curve = dissonance(overtone_pairs, amplitude_pairs, conf.method)
    peaks, d2curve = get_peaks(cents, curve, height=conf.peak_cutoff)
    conf.n_peaks = len(peaks)

    plot_curve(cents, curve, d2curve, peaks, conf.figure)
    conf.figure.element.update()
