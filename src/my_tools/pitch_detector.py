# pyright: basic

import typing
from enum import Enum

import librosa
import numpy as np


class WindowType(Enum):
    HANN = "hann"
    HAMMING = "hamming"
    TRIANGLE = "triangle"


class SpectrogramNormalization(Enum):
    WINDOW = "window"
    FRAME_LENGTH = "frame_length"
    NONE = False


class SampleRate(Enum):
    SR_8192 = 8192
    SR_44100 = 44100
    SR_48000 = 48000
    SR_88200 = 88200
    SR_96000 = 96000


class FFTSize(Enum):
    FFT_512 = 512
    FFT_1024 = 1024
    FFT_2048 = 2048


class HopRatio(Enum):
    HOP_2 = 2
    HOP_4 = 4


class SpectorgramPower(Enum):
    POW_1 = 1
    POW_2 = 2


OvertoneScaling = typing.Callable[[int], float]


def inverse_sqrt(idx: int) -> float:
    return 1 / np.sqrt(idx)


def inverse(idx: int) -> float:
    return 1 / idx


class FFTWindow:
    def __init__(self, name, size) -> None:
        self.name = name
        self.window = librosa.filters.get_window(name, size)


class PitchDetector:
    def __init__(
        self,
        sample_rate=SampleRate.SR_48000,
        n_fft=FFTSize.FFT_2048,
        hop_ratio=HopRatio.HOP_4,
        spectrogram_power=SpectorgramPower.POW_1,
        spectrogram_window=WindowType.TRIANGLE,
        spectrogram_normalization=SpectrogramNormalization.NONE,
        lowest_midi_note=60,
        highest_midi_note=96,
        cents_resolution=10,
        n_overtones=100,
        fn_overtone_decay: OvertoneScaling = inverse_sqrt,
    ) -> None:
        # User defined parameters
        self.sample_rate = sample_rate.value
        self.n_fft = n_fft.value
        self.hop_ratio = hop_ratio.value
        self.spectrogram_power = spectrogram_power.value
        self.spectrogram_window = FFTWindow(
            spectrogram_window.value, n_fft.value)
        self.spectrogram_normalization = spectrogram_normalization.value
        self.lowest_midi_note = lowest_midi_note
        self.highest_midi_note = highest_midi_note
        self.cents_resolution = cents_resolution
        self.n_overtones = n_overtones
        self.fn_overtone_decay = fn_overtone_decay

        # Derived/calculated parameters
        self.hop_length = self.n_fft // self.hop_ratio
        self.fft_freqs = librosa.fft_frequencies(
            sr=self.sample_rate, n_fft=self.n_fft)
        self.cents_factor = 100 // self.cents_resolution

        # Calculate overtone matrix
        self.factors = np.stack([
            self._harmonics(
                librosa.midi_to_hz(n / self.cents_factor), n=self.n_overtones
            )
            for n in range(
                self.lowest_midi_note * self.cents_factor,
                self.highest_midi_note * self.cents_factor + 1,
            )
        ])

    def detect_pitch(self, waveform):
        spectrogram = self._spectrogram(waveform)

        result = self.factors @ spectrogram
        pitch = np.argmax(result, axis=0) / \
            self.cents_factor + self.lowest_midi_note
        amplitude = spectrogram.sum(axis=0) / self.n_fft

        confidence = (
            np.max(result, axis=0) /
            np.sum(result, axis=0)
        )

        return pitch, confidence, amplitude, result

    def _pure_sine_bin(self, hz, amp=1.0):
        time = np.arange(self.n_fft) / self.sample_rate
        # TODO: adding random phase seems to improve the situation
        # random_phase = np.random.uniform(-np.pi, np.pi, 1)[0]
        # wave = amp * np.sin(2 * np.pi * hz * time + random_phase)
        wave = amp * np.sin(2 * np.pi * hz * time + np.pi)

        peak = self._spectrogram(wave)

        return peak[:, 0]

    def _harmonics(self, hz, n=20):
        result = np.zeros_like(self.fft_freqs)
        for idx in range(1, n + 1):
            if (hz * idx) > self.sample_rate // 2:
                break
            result += self._pure_sine_bin(hz * idx,
                                          self.fn_overtone_decay(idx))
        return result

    def _spectrogram(self, waveform):
        spectrogram = np.abs(librosa.stft(
            y=waveform,
            pad_mode='constant',
            window=self.spectrogram_window.window,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            center=False,
        ) ** self.spectrogram_power)

        return spectrogram
