# pyright: basic

import librosa
import numpy as np
from nicegui import binding


# TODO: state and logic are too coupled
@binding.bindable_dataclass
class State:
    n_harmonics: int = 20
    stretch_1: float = 1.0
    stretch_2: float = 1.0
    midi1: int = 69
    f2: float = 440
    amp_decay: float = 0.88
    start_delta_cents: int = 0
    delta_cents_range: int = 1300
    peak_cutoff: float = 0.2
    method: str = "min"
    n_peaks: int = 0
    figure = None  # pyright: ignore[reportUnannotatedClassAttribute]

    @property
    def f1(self) -> np.floating:
        return librosa.midi_to_hz(self.midi1)

    def has_figure(self) -> bool:
        return self.__dict__.get("figure", None) is not None


state = State()
