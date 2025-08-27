from dataclasses import dataclass, field

import librosa
from nicegui.elements.pyplot import MatplotlibFigure


# TODO: state and logic are too coupled
@dataclass
class State:
    n_harmonics = 20
    midi1 = 69
    f2 = 440
    amp_decay = 0.88
    start_delta_cents = 0
    delta_cents_range = 1300
    peak_cutoff = 0.2
    method = "min"
    n_peaks = 0
    figure: MatplotlibFigure = field(init=False)

    @property
    def f1(self):
        return librosa.midi_to_hz(self.midi1)

    def has_figure(self):
        return self.__dict__.get("figure", False)


state = State()
