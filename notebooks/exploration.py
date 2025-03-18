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
    SAMPLE_WAV = (
        Path.home() / "Music" / "Violin Samples" / "yee_amazing_grace#2.wav"
    )
    return (SAMPLE_WAV,)


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
    # waveform = waveform[:, int(0.4 * sample_rate):int(1.2 * sample_rate)]

    mo.vstack(
        items=[
            plot_waveform(waveform, sample_rate),
            mo.md("## haiku.wav"),
            Audio(data=waveform, rate=sample_rate),
        ]
    )
    return sample_rate, waveform


@app.cell
def _(librosa, plt):
    window = librosa.filters.get_window("triangle", 1024)
    figure, axis = plt.subplots(1, 1)
    axis.plot(window, linewidth=1)
    axis.grid(True)
    figure.suptitle("window")
    return axis, figure, window


@app.cell
def _(np, plt, torch, torchaudio, waveform, window):
    spectrogram = torchaudio.functional.spectrogram(
        waveform[0], 0, torch.from_numpy(window), 1024, 256, 1024, 2.0, False
    ).flip(0)
    plt.matshow(np.log(1e-8 + spectrogram.numpy()), cmap="viridis")
    return (spectrogram,)


if __name__ == "__main__":
    app.run()
