import librosa
from nicegui import binding

from my_tools.seth import get_harmonic_spectrum


# TODO: state and logic are too coupled
@binding.bindable_dataclass
class State:
    n_harmonics: int = 20
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
    def f1(self):
        return librosa.midi_to_hz(self.midi1)

    def has_figure(self) -> bool:
        return self.__dict__.get("figure", None) is not None

    # Should be calculated only if partials parameters change
    @property
    def spectrum1(self):
        return get_harmonic_spectrum(
            self.f1,
            self.n_harmonics,
            self.amp_decay,
        )

    @property
    def spectrum2(self):
        return get_harmonic_spectrum(
            self.f2,
            self.n_harmonics,
            self.amp_decay,
        )


state = State()
