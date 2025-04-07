from enum import Enum

import librosa
import numpy as np
import torch
import torchaudio


class WindowType(Enum):
    HANN = "hann"
    HAMMING = "hamming"
    TRIANGLE = "triangle"


class SpectrogramNormalization(Enum):
    WINDOW = "window"
    FRAME_LENGTH = "frame_length"
    NONE = False


class SampleRate(Enum):
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


def inverse_sqrt(idx):
    return 1 / np.sqrt(idx)


def inverse(idx):
    return 1 / idx


class OvertoneScaling(Enum):
    INVERSE = inverse
    INVERSE_SQRT = inverse_sqrt


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
        fn_overtone_decay=OvertoneScaling.INVERSE_SQRT,
    ) -> None:
        # User defined parameters
        self.sample_rate = sample_rate.value
        self.n_fft = n_fft.value
        self.hop_ratio = hop_ratio.value
        self.spectrogram_power = spectrogram_power.value
        self.spectrogram_window = FFTWindow(spectrogram_window.value, n_fft.value)
        self.spectrogram_normalization = spectrogram_normalization.value
        self.lowest_midi_note = lowest_midi_note
        self.highest_midi_note = highest_midi_note
        self.cents_resolution = cents_resolution
        self.n_overtones = n_overtones
        self.fn_overtone_decay = fn_overtone_decay

        # Derived/calculated parameters
        self.hop_length = self.n_fft // self.hop_ratio
        self.fft_freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        self.cents_factor = 100 // self.cents_resolution

        # Calculate overtone matrix
        self.factors = np.stack(
            [
                self._harmonics(
                    librosa.midi_to_hz(n / self.cents_factor), n=self.n_overtones
                )
                for n in range(
                    self.lowest_midi_note * self.cents_factor,
                    self.highest_midi_note * self.cents_factor + 1,
                )
            ]
        )

    def detect_pitch(self, waveform):
        spectrogram = torchaudio.functional.spectrogram(
            waveform=waveform,
            pad=0,
            window=torch.from_numpy(self.spectrogram_window.window),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            power=self.spectrogram_power,
            normalized=self.spectrogram_normalization,
            center=True,
        )

        result = self.factors @ spectrogram.numpy()
        pitch = np.argmax(result, axis=0) / self.cents_factor + self.lowest_midi_note
        # TODO: division by zero
        confidence = (
            np.max(result, axis=0) / spectrogram.sum(dim=0).numpy() / self.n_fft
        )
        amplitude = spectrogram.sum(dim=0) / self.n_fft

        return pitch, confidence, amplitude, result

    def _pure_sine_bin(self, hz, amp=1.0):
        time = np.arange(self.n_fft) / self.sample_rate
        # TODO: adding random phase seems to improve the situation
        # wave = amp * np.sin(2 * np.pi * hz * time + np.random.uniform(-np.pi, np.pi, 1)[0])
        wave = amp * np.sin(2 * np.pi * hz * time)

        peak = torchaudio.functional.spectrogram(
            waveform=torch.from_numpy(wave).unsqueeze(0),
            pad=0,
            window=torch.from_numpy(self.spectrogram_window.window),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            power=self.spectrogram_power,
            normalized=self.spectrogram_normalization,
            center=False,
        )

        return peak[0, :, 0].numpy()

    def _harmonics(self, hz, n=20):
        result = np.zeros_like(self.fft_freqs)
        for idx in range(1, n + 1):
            if (hz * idx) > self.sample_rate // 2:
                break
            result += self._pure_sine_bin(hz * idx, self.fn_overtone_decay(idx))
        return result
