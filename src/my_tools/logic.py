from dataclasses import dataclass, field

import librosa
from nicegui import ui
from nicegui.elements.pyplot import MatplotlibFigure

from my_tools.file_picker import local_file_picker
from my_tools.seth import (
    dissonance,
    get_harmonic_spectrum,
    get_peaks,
    plot_curve,
    prepare_sweep,
)


# TODO: move all values to State (rename Config to State)
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
    audio_path = "/home/kureta/Music/IRCAM/Orchidea_tam_tam_0.6/CSOL_tam_tam/Percussion/tympani-sticks/middle/tt-tymp_mid-ff-N.wav"
    figure: MatplotlibFigure = field(init=False)

    @property
    def f1(self):
        return librosa.midi_to_hz(self.midi1)

    # TODO: this is a hack. get proper file name
    @property
    def file_name(self):
        return self.audio_path.split("/")[-1]


conf = Config()


async def pick_file():
    result = (await local_file_picker("~/Music/"))[0]
    conf.audio_path = result


# TODO: separate plotting and calculation
def show_plot():
    if conf.figure is None:
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
