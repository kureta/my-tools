# pyright: basic

from dataclasses import dataclass

from nicegui import ui

from my_tools.seth import (
    dissonance,
    get_harmonic_spectrum,
    get_peaks,
    plot_curve,
    prepare_sweep,
)


@dataclass
class Config:
    n_harmonics = 8
    f1 = 330
    f2 = 440
    amp_decay = 0.88
    start_delta_cents = 0
    delta_cents_range = 1300
    peak_cutoff = 0.2


conf = Config()


def show_plot():
    spectrum_1, amplitudes_1 = get_harmonic_spectrum(
        conf.f1, conf.n_harmonics, conf.amp_decay
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

    curve = dissonance(overtone_pairs, amplitude_pairs)
    peaks, d2curve = get_peaks(cents, curve, height=conf.peak_cutoff)

    plot_curve(cents, curve, d2curve, peaks, fig)
    fig.canvas.draw_idle()
    fig.element.update()

    return fig


# ======= GUI Code =======

with ui.row():
    with ui.column():
        ui.markdown(
            """## Thesis

    A presentation page for my masters thesis proposal.
    """
        )
        with ui.card():
            slider = (
                ui.slider(
                    min=27.5,
                    max=4186.01,
                    step=1,
                    on_change=lambda e: show_plot(),
                )
                .classes("w-96")
                .bind_value(conf, "f1")
            )
            with ui.row():
                ui.label("base frequency:")
                ui.label("").bind_text(slider, "value")

        with ui.card():
            slider = (
                ui.slider(
                    min=1,
                    max=32,
                    step=1,
                    on_change=lambda e: show_plot(),
                )
                .classes("w-96")
                .bind_value(conf, "n_harmonics")
            )
            with ui.row():
                ui.label("number of partials:")
                ui.label("").bind_text(slider, "value")

        with ui.card():
            slider = (
                ui.slider(
                    min=0,
                    max=1,
                    step=0.01,
                    on_change=lambda e: show_plot(),
                )
                .classes("w-96")
                .bind_value(conf, "amp_decay")
            )
            with ui.row():
                ui.label("amplitude decay:")
                ui.label("").bind_text(slider, "value")

        with ui.card():
            slider = (
                ui.slider(
                    min=0,
                    max=2400,
                    step=1,
                    on_change=lambda e: show_plot(),
                )
                .classes("w-96")
                .bind_value(conf, "start_delta_cents")
            )
            with ui.row():
                ui.label("start interval (cents)")
                ui.label("").bind_text(slider, "value")

        with ui.card():
            slider = (
                ui.slider(
                    min=1200,
                    max=2600,
                    step=1,
                    on_change=lambda e: show_plot(),
                )
                .classes("w-96")
                .bind_value(conf, "delta_cents_range")
            )
            with ui.row():
                ui.label("interval range (cents):")
                ui.label("").bind_text(slider, "value")

        with ui.card():
            slider = (
                ui.slider(
                    min=0,
                    max=1,
                    step=0.01,
                    on_change=lambda e: show_plot(),
                )
                .classes("w-96")
                .bind_value(conf, "peak_cutoff")
            )
            with ui.row():
                ui.label("peak cutoff:")
                ui.label("").bind_text(slider, "value")

    with ui.column():
        ui.markdown("## Synthetic spectra")
        with ui.card():
            with ui.matplotlib(figsize=(12, 4), dpi=300).figure as fig:
                show_plot()

ui.run()
