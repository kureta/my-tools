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
    return Audio, Path, librosa, mo, np, plt, torch, torchaudio


@app.cell
def _(Path):
    SAMPLE_WAV = Path("/home/kureta/Music/Flute Samples/01. Air.wav")

    N_FFT = 2048
    OVERLAP_RATIO = 4
    WIN_LENGTH = N_FFT
    HOP_LENGTH = N_FFT // OVERLAP_RATIO
    SPECTROGRAM_POWER = 1.0
    return (
        HOP_LENGTH,
        N_FFT,
        OVERLAP_RATIO,
        SAMPLE_WAV,
        SPECTROGRAM_POWER,
        WIN_LENGTH,
    )


@app.cell
def _(plt, torch):
    def plot_waveform(waveform, sample_rate):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

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
    return (plot_waveform,)


@app.cell
def _(Audio, SAMPLE_WAV, mo, plot_waveform, torchaudio):
    waveform, sample_rate = torchaudio.load(SAMPLE_WAV)
    waveform = waveform[:, int(1.5 * sample_rate) : int(12.5 * sample_rate)]
    Audio(data=waveform, rate=sample_rate)
    mo.vstack(
        items=[
            plot_waveform(waveform, sample_rate),
            Audio(data=waveform, rate=sample_rate),
        ]
    )
    return sample_rate, waveform


@app.cell
def _(WIN_LENGTH, librosa, plt):
    window = librosa.filters.get_window("triangle", WIN_LENGTH)
    figure, axis = plt.subplots(1, 1)
    axis.plot(window, linewidth=1)
    axis.grid(True)
    figure.suptitle("window")
    return axis, figure, window


@app.cell
def _(
    HOP_LENGTH,
    N_FFT,
    SPECTROGRAM_POWER,
    WIN_LENGTH,
    librosa,
    np,
    plt,
    torch,
    torchaudio,
    waveform,
):
    spectrogram = torchaudio.functional.spectrogram(
        waveform=waveform[0],
        pad=0,
        window=torch.from_numpy(
            librosa.filters.get_window("triangle", WIN_LENGTH)
        ),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        power=SPECTROGRAM_POWER,
        normalized="window",
        center=True,
    )
    plt.matshow(np.log(1e-8 + spectrogram.flip(0).numpy()), cmap="viridis")
    return (spectrogram,)


@app.cell
def _(N_FFT, librosa, np, sample_rate):
    C2 = 36
    C6 = 84
    error = 0.07
    fft_freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)


    def normal(mean, variance):
        factor = 1 / np.sqrt(variance * 2 * np.pi)

        def curve(x):
            return factor * np.exp(-((x - mean) ** 2) / variance)

        return curve


    def normalized_normal(hz):
        result = normal(hz, librosa.midi_to_hz(error))(fft_freqs)
        if result.sum() == 0.0:
            return result
        return result / result.sum()
    return C2, C6, error, fft_freqs, normal, normalized_normal


@app.cell
def _(C2, C6, fft_freqs, librosa, normalized_normal, np):
    def harmonics(hz, n=20):
        result = np.zeros_like(fft_freqs)
        for idx in range(1, n + 1):
            result += normalized_normal(hz * idx) / np.sqrt(idx)
        return result / np.log2(idx)


    factors = np.stack(
        [
            harmonics(librosa.midi_to_hz(n / 10))
            for n in range(C2 * 10, C6 * 10 + 1)
        ]
    )

    factors.shape
    return factors, harmonics


@app.cell
def _(factors, spectrogram):
    # normalized_spectrogram = spectrogram.numpy()
    # normalized_spectrogram = normalized_spectrogram / normalized_spectrogram.max(axis=0)
    # result = factors @ normalized_spectrogram
    result = factors @ spectrogram.numpy()
    # plt.matshow(result)
    return (result,)


@app.cell
def _(np, plt, result):
    plt.plot(np.argmax(result, axis=0))
    return


@app.cell
def _(np, plt, result):
    plt.plot(np.max(result, axis=0))
    return


if __name__ == "__main__":
    app.run()
