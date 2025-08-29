import librosa
from nicegui import binding

from my_tools.seth import get_harmonic_spectrum


# TODO: state and logic are too coupled
@binding.bindable_dataclass
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
    figure = None

    @property
    def f1(self):
        return librosa.midi_to_hz(self.midi1)

    def has_figure(self):
        return self.__dict__.get("figure", False)

    # Should be calculated only if partials parameters change
    @property
    def spectrum1(self):
        return get_harmonic_spectrum(
            self.f1,  # pyright: ignore
            self.n_harmonics,
            self.amp_decay,  # pyright: ignore
        )

    @property
    def spectrum2(self):
        return get_harmonic_spectrum(
            self.f2,  # pyright: ignore
            self.n_harmonics,
            self.amp_decay,  # pyright: ignore
        )


state = State()
