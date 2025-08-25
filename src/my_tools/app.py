from dataclasses import dataclass

import librosa
from nicegui import ui

from my_tools.file_picker import local_file_picker
from my_tools.seth import (
    dissonance,
    get_harmonic_spectrum,
    get_peaks,
    plot_curve,
    prepare_sweep,
)


@dataclass
class Config:
    n_harmonics = 20
    midi1 = 69
    f2 = 440
    amp_decay = 0.88
    start_delta_cents = 0
    delta_cents_range = 1300
    peak_cutoff = 0.2
    method = "min"
    n_peaks = 0
    audio_path = ""

    @property
    def f1(self):
        return librosa.midi_to_hz(self.midi1)


conf = Config()


async def pick_file():
    result = (await local_file_picker("~"))[0]
    conf.audio_path = result
    ui.notify(f"You chose {result}")


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

    curve = dissonance(overtone_pairs, amplitude_pairs, conf.method)
    peaks, d2curve = get_peaks(cents, curve, height=conf.peak_cutoff)
    conf.n_peaks = len(peaks)

    # we can just call fig.update() after plot_curve instead of using the context manager
    with fig:
        plot_curve(cents, curve, d2curve, peaks, fig)

    return fig


def create_slider(min, max, step, on_change, value, label):
    with ui.row():
        ui.label(label)
        ui.label("").bind_text(conf, value)
    ui.slider(
        min=min,
        max=max,
        step=step,
        on_change=on_change,
    ).classes(
        "w-96"
    ).bind_value(conf, value)


# ======= GUI Code =======

with ui.row():
    with ui.column():
        ui.markdown("## Control Parameters")
        ui.button("Choose lower voice", on_click=pick_file, icon="folder")
        ui.audio(
            "/home/kureta/Music/IRCAM/Orchidea_tam_tam_0.6/CSOL_tam_tam/Percussion/tympani-sticks/middle/tt-tymp_mid-ff-N.wav"
        ).bind_source_from(conf, "audio_path")

        with ui.card():
            ui.markdown("**Parameters of synthetic partials**")
            ui.separator()
            create_slider(21, 108, 1, show_plot, "midi1", "base midi #:")
            create_slider(1, 32, 1, show_plot, "n_harmonics", "number of partials:")
            create_slider(0, 1, 0.01, show_plot, "amp_decay", "amplitude decay:")
        with ui.card():
            ui.markdown("**Parameters of dissonance curve calculation**")
            ui.separator()
            with ui.row():
                ui.label("Calculation method:")
                ui.select(
                    ["min", "product"], value="min", on_change=show_plot
                ).bind_value(conf, "method")
            create_slider(
                0, 2400, 100, show_plot, "start_delta_cents", "start interval (cents)"
            )
            create_slider(
                1200,
                2600,
                100,
                show_plot,
                "delta_cents_range",
                "interval range (cents)",
            )
            create_slider(0, 1, 0.01, show_plot, "peak_cutoff", "peak cutoff:")

    with ui.column():
        ui.markdown("## Synthetic spectra")
        with ui.card():
            with ui.row():
                ui.label("n peaks detected:")
                ui.label("").bind_text(conf, "n_peaks")
            with ui.matplotlib(figsize=(14, 4)).figure as fig:
                show_plot()

ui.run()
