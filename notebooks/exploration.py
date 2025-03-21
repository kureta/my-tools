import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    from IPython.display import Audio
    import torch
    import torchaudio
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa
    from scipy.interpolate import interp1d

    mo.md("# Pitch detection")
    return Audio, Path, interp1d, librosa, mo, np, plt, torch, torchaudio


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Some TODOS

        ⚠️ **Instead of making up a bell curve at a given frequency, use stft of a sine wave at that frequency with the same window that is used to generate the spectrogram of the target signal.**

        - [ ] If a pitch does not correspond exactly to an FFT bin it has to be normalized. We can normalize the area under the curve or we can redistribute the actual peaks value to the surrounding bins proportionally, and then normalize.
        - [ ] We are currently decaying the amplitudes of overtones by the squareroot of their index. This can be further investigated.
        """
    )
    return


@app.cell
def p_1(N_FFT, SAMPLE_RATE, d_HOP_LENGTH, librosa, np, plt, torch):
    def plot_waveform(waveform, SAMPLE_RATE):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / SAMPLE_RATE

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c + 1}")
        figure.suptitle("waveform")

        return figure


    def plot_with_title(x, title):
        num_frames = len(x)
        end_time = librosa.frames_to_time(
            num_frames, sr=SAMPLE_RATE, hop_length=d_HOP_LENGTH, n_fft=N_FFT
        )
        time_axis = (np.arange(0, num_frames) / num_frames) * end_time
        figure, axis = plt.subplots(1, 1)
        axis.plot(time_axis, x, linewidth=1)
        axis.grid(True)
        figure.suptitle(title)

        return figure
    return plot_waveform, plot_with_title


@app.cell
def _(Audio, SAMPLE_RATE, mo, np, plot_waveform, plt, spectrogram, waveform):
    mo.vstack(
        items=[
            mo.md("## Wave File"),
            mo.hstack(
                items=[
                    mo.vstack(
                        items=[
                            plot_waveform(waveform, SAMPLE_RATE),
                            Audio(data=waveform, rate=SAMPLE_RATE),
                        ]
                    ),
                    plt.matshow(
                        np.log(1e-8 + spectrogram.flip(0).numpy()),
                        cmap="viridis",
                    ),
                ]
            ),
        ]
    )
    return


@app.cell
def _(Path, np):
    SAMPLE_WAV = Path(
        "/home/kureta/Music/Flute Samples/14. 3 Oriental Pieces_ I. Bergere captive.wav"
    )

    N_FFT = 2048
    OVERLAP_RATIO = 4

    SPECTROGRAM_POWER = 1.0
    WIN_NAME = "hamming"  # "hann" "hamming" "triangle"
    SPECTROGRAM_NORMALIZATION = False  # "window" or "frame_length" or False

    LOWEST_MIDI_NOTE = 60
    HIGHEST_MIDI_NOTE = 96
    N_OVERTONES = 20

    CENTS_RESOLUTION = 5


    def OVERTONE_SCALING(idx):
        return 1 / np.sqrt(idx)
    return (
        CENTS_RESOLUTION,
        HIGHEST_MIDI_NOTE,
        LOWEST_MIDI_NOTE,
        N_FFT,
        N_OVERTONES,
        OVERLAP_RATIO,
        OVERTONE_SCALING,
        SAMPLE_WAV,
        SPECTROGRAM_NORMALIZATION,
        SPECTROGRAM_POWER,
        WIN_NAME,
    )


@app.cell
def _(mo):
    offset = mo.ui.slider(0.0, 60.0, 0.5, label="Start offset in seconds.")

    offset
    return (offset,)


@app.cell
def _(
    CENTS_RESOLUTION,
    N_FFT,
    OVERLAP_RATIO,
    SAMPLE_WAV,
    WIN_NAME,
    librosa,
    offset,
    torchaudio,
):
    waveform, SAMPLE_RATE = torchaudio.load(SAMPLE_WAV)
    waveform = waveform[
        :,
        int(offset.value * SAMPLE_RATE) : int((offset.value + 10.0) * SAMPLE_RATE),
    ]

    d_WIN_LENGTH = N_FFT
    d_HOP_LENGTH = N_FFT // OVERLAP_RATIO
    d_WINDOW = librosa.filters.get_window(WIN_NAME, d_WIN_LENGTH)
    d_FFT_FREQS = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    d_CENTS_FACTOR = 100 // CENTS_RESOLUTION
    return (
        SAMPLE_RATE,
        d_CENTS_FACTOR,
        d_FFT_FREQS,
        d_HOP_LENGTH,
        d_WINDOW,
        d_WIN_LENGTH,
        waveform,
    )


@app.cell
def _(
    HIGHEST_MIDI_NOTE,
    LOWEST_MIDI_NOTE,
    N_FFT,
    N_OVERTONES,
    OVERTONE_SCALING,
    SAMPLE_RATE,
    SPECTROGRAM_NORMALIZATION,
    SPECTROGRAM_POWER,
    d_CENTS_FACTOR,
    d_FFT_FREQS,
    d_HOP_LENGTH,
    d_WINDOW,
    d_WIN_LENGTH,
    librosa,
    np,
    torch,
    torchaudio,
    waveform,
):
    def pure_sine_bin(hz, amp=1.0):
        time = np.arange(N_FFT) / SAMPLE_RATE
        # TODO: adding random phase seems to improve the situation
        # wave = amp * np.sin(2 * np.pi * hz * time + np.random.uniform(-np.pi, np.pi, 1)[0])
        wave = amp * np.sin(2 * np.pi * hz * time)

        peak = torchaudio.functional.spectrogram(
            waveform=torch.from_numpy(wave).unsqueeze(0),
            pad=0,
            window=torch.from_numpy(d_WINDOW),
            n_fft=N_FFT,
            hop_length=d_HOP_LENGTH,
            win_length=d_WIN_LENGTH,
            power=SPECTROGRAM_POWER,
            normalized=SPECTROGRAM_NORMALIZATION,
            center=False,
        )

        return peak[0, :, 0].numpy()


    def harmonics(hz, n=20):
        result = np.zeros_like(d_FFT_FREQS)
        for idx in range(1, n + 1):
            result += pure_sine_bin(hz * idx, OVERTONE_SCALING(idx))
        return result


    factors = np.stack(
        [
            harmonics(librosa.midi_to_hz(n / d_CENTS_FACTOR), n=N_OVERTONES)
            for n in range(
                LOWEST_MIDI_NOTE * d_CENTS_FACTOR,
                HIGHEST_MIDI_NOTE * d_CENTS_FACTOR + 1,
            )
        ]
    )

    spectrogram = torchaudio.functional.spectrogram(
        waveform=waveform[0],
        pad=0,
        window=torch.from_numpy(d_WINDOW),
        n_fft=N_FFT,
        hop_length=d_HOP_LENGTH,
        win_length=d_WIN_LENGTH,
        power=SPECTROGRAM_POWER,
        normalized=SPECTROGRAM_NORMALIZATION,
        center=True,
    )

    result = factors @ spectrogram.numpy()
    pitch = np.argmax(result, axis=0) / d_CENTS_FACTOR + LOWEST_MIDI_NOTE
    confidence = np.max(result, axis=0) / spectrogram.sum(dim=0).numpy() / N_FFT
    amplitude = spectrogram.sum(dim=0) / N_FFT

    # Mask amplitude with confidence
    min_confidence = 0.05
    amplitude[confidence <= min_confidence] = 0.0
    return (
        amplitude,
        confidence,
        factors,
        harmonics,
        min_confidence,
        pitch,
        pure_sine_bin,
        result,
        spectrogram,
    )


@app.cell
def _(confidence, mo, pitch, plot_with_title):
    mo.hstack(
        items=[
            plot_with_title(pitch, "Pitch"),
            plot_with_title(confidence, "Confidence"),
        ]
    )
    return


@app.cell
def _(SAMPLE_RATE, amplitude, interp1d, librosa, np, pitch, waveform):
    duration = 10.0
    t_audio = np.linspace(0, duration, waveform.shape[1], endpoint=False)
    t_control = np.linspace(0, duration, pitch.shape[0], endpoint=False)

    # convert pitch information from midi, to cycles per second, to radians per sample
    cycles_per_second = librosa.midi_to_hz(pitch)
    radians_per_second = 2 * np.pi * cycles_per_second
    radians_per_sample = radians_per_second / SAMPLE_RATE

    # Interpolate control frequencies to match the audio sample rate
    f_interp_radians_per_sample = interp1d(
        t_control, radians_per_sample, kind="linear", fill_value="extrapolate"
    )
    audio_radians_per_sample = f_interp_radians_per_sample(t_audio)

    # accumulate them into phase
    audio_phase = np.cumsum(audio_radians_per_sample)

    interp_amps = interp1d(
        t_control, amplitude, kind="linear", fill_value="extrapolate"
    )
    audio_amps = interp_amps(t_audio)

    # # Generate the sine wave
    sine_wave = audio_amps * np.sin(audio_phase)
    return (
        audio_amps,
        audio_phase,
        audio_radians_per_sample,
        cycles_per_second,
        duration,
        f_interp_radians_per_sample,
        interp_amps,
        radians_per_sample,
        radians_per_second,
        sine_wave,
        t_audio,
        t_control,
    )


@app.cell
def _(Audio, SAMPLE_RATE, sine_wave):
    Audio(data=sine_wave, rate=SAMPLE_RATE)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
