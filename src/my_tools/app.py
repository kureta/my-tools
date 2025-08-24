# pyright: basic

from matplotlib.figure import Figure
from nicegui import ui

from my_tools.seth import (
    dissonance,
    get_harmonic_spectrum,
    get_peaks,
    plot_curve,
    prepare_sweep,
)

f1 = 440
f2 = 330
spectrum_1, amplitudes_1 = get_harmonic_spectrum(f1)
spectrum_2, amplitudes_2 = get_harmonic_spectrum(f2)

overtone_pairs, amplitude_pairs, cents = prepare_sweep(
    f1, spectrum_1, amplitudes_1, f2, spectrum_2, amplitudes_2, 0, 1300
)

curve = dissonance(overtone_pairs, amplitude_pairs)
peaks, d2curve = get_peaks(cents, curve, height=0.2)


with ui.row():
    ui.markdown(
        """## Thesis

A presentation page for my masters thesis proposal.
"""
    )

    with ui.column():
        ui.markdown("## Synthetic spectra")
        with ui.card().style():
            with ui.matplotlib(figsize=(12, 4)).figure as fig:
                plot_curve(cents, curve, d2curve, peaks, fig)

ui.run()
