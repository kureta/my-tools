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

    mo.md("# Pitch detection")
    return Audio, Path, librosa, mo, np, plt, torch, torchaudio


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Some TODOS
        - [ ] If a pitch does not correspond exactly to an FFT bin it has to be normalized. We can normalize the area under the curve or we can redistribute the actual peaks value to the surrounding bins proportionally, and then normalize.
        - [ ] We are currently decaying the amplitudes of overtones by the squareroot of their index. This can be further investigated.
        """
    )
    return


@app.cell
def p_1(
    N_FFT,
    SAMPLE_RATE,
    d_CENTS_FACTOR,
    d_HOP_LENGTH,
    librosa,
    np,
    plt,
    torch,
):
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
        axis.plot(time_axis, x / d_CENTS_FACTOR, linewidth=1)
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
                    plot_waveform(waveform, SAMPLE_RATE),
                    plt.matshow(
                        np.log(1e-8 + spectrogram.flip(0).numpy()), cmap="viridis"
                    ),
                ]
            ),
            Audio(data=waveform, rate=SAMPLE_RATE),
        ]
    )
    return


@app.cell
def _(mo):
    spec_power = mo.ui.slider(1.0, 4.0, 0.1, label="Spectrogram power")

    mo.vstack(items=[mo.md("## Adjustable Parameters"), spec_power])
    return (spec_power,)


@app.cell
def _(Path, spec_power):
    SAMPLE_WAV = Path("/home/kureta/Music/Flute Samples/01. Air.wav")

    N_FFT = 2048
    OVERLAP_RATIO = 2

    SPECTROGRAM_POWER = spec_power.value
    WIN_NAME = "hann"  # "hann" "hamming" "triangle"
    SPECTROGRAM_NORMALIZATION = False  # "window" or "frame_length" or False

    LOWEST_MIDI_NOTE = 36  # C 2
    HIGHEST_MIDI_NOTE = 84  # C 6
    ERROR_IN_CENTS = 7

    CENTS_RESOLUTION = 10
    return (
        CENTS_RESOLUTION,
        ERROR_IN_CENTS,
        HIGHEST_MIDI_NOTE,
        LOWEST_MIDI_NOTE,
        N_FFT,
        OVERLAP_RATIO,
        SAMPLE_WAV,
        SPECTROGRAM_NORMALIZATION,
        SPECTROGRAM_POWER,
        WIN_NAME,
    )


@app.cell
def _(
    CENTS_RESOLUTION,
    ERROR_IN_CENTS,
    N_FFT,
    OVERLAP_RATIO,
    SAMPLE_WAV,
    WIN_NAME,
    librosa,
    torchaudio,
):
    waveform, SAMPLE_RATE = torchaudio.load(SAMPLE_WAV)
    waveform = waveform[:, int(1.5 * SAMPLE_RATE) : int(12.5 * SAMPLE_RATE)]

    d_WIN_LENGTH = N_FFT
    d_HOP_LENGTH = N_FFT // OVERLAP_RATIO
    d_WINDOW = librosa.filters.get_window(WIN_NAME, d_WIN_LENGTH)
    d_ERROR = ERROR_IN_CENTS / 100
    d_FFT_FREQS = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    d_CENTS_FACTOR = 100 // CENTS_RESOLUTION
    return (
        SAMPLE_RATE,
        d_CENTS_FACTOR,
        d_ERROR,
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
    SPECTROGRAM_NORMALIZATION,
    SPECTROGRAM_POWER,
    d_CENTS_FACTOR,
    d_ERROR,
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
    def normal(mean, variance):
        factor = 1 / np.sqrt(variance * 2 * np.pi)

        def curve(x):
            return factor * np.exp(-((x - mean) ** 2) / variance)

        return curve


    def normalized_normal(hz):
        result = normal(
            hz, librosa.midi_to_hz(librosa.hz_to_midi(hz) + d_ERROR) - hz
        )(d_FFT_FREQS)
        if result.sum() == 0.0:
            return result
        return result / result.sum()


    def harmonics(hz, n=20):
        result = np.zeros_like(d_FFT_FREQS)
        for idx in range(1, n + 1):
            result += normalized_normal(hz * idx) / np.sqrt(idx)
        return result


    factors = np.stack(
        [
            harmonics(librosa.midi_to_hz(n / d_CENTS_FACTOR))
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
    return factors, harmonics, normal, normalized_normal, result, spectrogram


@app.cell
def _(mo, np, plot_with_title, result):
    mo.vstack(
        items=[
            mo.hstack(
                items=[
                    # plot_with_title(_WINDOW, WIN_NAME),
                    plot_with_title(np.argmax(result, axis=0), "Pitch"),
                    plot_with_title(np.log2(np.max(result, axis=0)), "Confidence"),
                ]
            ),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
