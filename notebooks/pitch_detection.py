import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import torch
    import torchaudio
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa
    from scipy.interpolate import interp1d

    mo.md("# Pitch detection")
    return Path, interp1d, librosa, mo, np, plt, torch


@app.cell
def _():
    from my_tools import pitch_detector as pd
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Some TODOS

    ### THIS IS IMPORTANT

     - Synchrosqueezing STFT gives very sharp peaks at the actual frequencies of overtones
     - Synchrosqueezing CWT almost completely removes overtones and gives fundamental

    ### Some thigs to try:

    - [ ] Use squeezed STFT
    - [ ] Use squeezed CWT butr with a wavelet that is created from harmonic overtones

    ⚠️ **Instead of making up a bell curve at a given frequency, use stft of a sine wave at that frequency with the same window that is used to generate the spectrogram of the target signal.**

    - [ ] If a pitch does not correspond exactly to an FFT bin it has to be normalized. We can normalize the area under the curve or we can redistribute the actual peaks value to the surrounding bins proportionally, and then normalize.
    - [ ] Using stft at that frequency instead of the above method. But should revisit that.
    - [ ] We are currently decaying the amplitudes of overtones by the squareroot of their index. This can be further investigated.
    - [ ] Refactor synthesizer into its own class
    - [ ] Implement everything in torch/torchaudio and ditch numpy. numpy can only be used during initialization, not for any calculations.
    - [ ] Also, try to get rid of `librosa`. Its `llvmlite` dependency prevents us from using a newer version of python.
    """
    )
    return


@app.cell(hide_code=True)
def _(detector, librosa, np, plt, torch):
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
            num_frames,
            sr=detector.sample_rate,
            hop_length=detector.hop_length,
            n_fft=detector.n_fft,
        )
        time_axis = (np.arange(0, num_frames) / num_frames) * end_time
        figure, axis = plt.subplots(1, 1)
        axis.plot(time_axis, x, linewidth=1)
        axis.grid(True)
        figure.suptitle(title)

        return figure


    def plot_with_dual_y(x1, x2, label1, label2, title):
        num_frames1 = len(x1)
        end_time1 = librosa.frames_to_time(
            num_frames1,
            sr=detector.sample_rate,
            hop_length=detector.hop_length,
            n_fft=detector.n_fft,
        )
        time_axis1 = (np.arange(0, num_frames1) / num_frames1) * end_time1

        num_frames2 = len(x2)
        end_time2 = librosa.frames_to_time(
            num_frames2,
            sr=detector.sample_rate,
            hop_length=detector.hop_length,
            n_fft=detector.n_fft,
        )
        time_axis2 = (np.arange(0, num_frames2) / num_frames2) * end_time2

        figure, axis1 = plt.subplots(1, 1)
        axis1.plot(time_axis1, x1, "g-", label=label1, linewidth=1)
        axis1.set_ylabel(label1)

        axis2 = axis1.twinx()
        axis2.plot(time_axis2, x2, "b-", label=label2, linewidth=1)
        axis2.set_ylabel(label2)

        axis1.grid(True)
        figure.suptitle(title)

        return figure
    return plot_with_dual_y, plot_with_title


@app.cell
def _(pd):
    detector = pd.PitchDetector(
        lowest_midi_note=24,
        highest_midi_note=87,
        spectrogram_window=pd.WindowType.HAMMING,
        sample_rate=pd.SampleRate.SR_8192,
        n_fft=pd.FFTSize.FFT_512,
    )
    return (detector,)


@app.cell
def _(Path, detector, librosa):
    # SAMPLE_WAV = Path("/home/kureta/Music/Flute Samples/03. Sonata Appassionata, Op. 140.wav")
    SAMPLE_WAV = Path(
        "/home/kureta/Music/Violin Samples/yee_brahms#9.wav"
    )  # 18, 23, 25, 26, 27, 28
    # SAMPLE_WAV = Path("/home/kureta/Music/haiku.wav")

    uncut_waveform, _ = librosa.load(
        SAMPLE_WAV, sr=detector.sample_rate, mono=True, duration=10.0
    )
    return (uncut_waveform,)


@app.cell
def _(detector, mo, np, uncut_waveform):
    waveform = uncut_waveform / np.max(np.abs(uncut_waveform))
    # waveform = torch.from_numpy(waveform)

    mo.vstack(
        items=[
            # Audio(data=waveform, rate=detector.sample_rate),
            mo.audio(src=waveform, rate=detector.sample_rate, normalize=True),
        ]
    )
    return (waveform,)


@app.cell
def _():
    min_confidence = 3.77e-3
    return (min_confidence,)


@app.cell
def _(detector, interp1d, librosa, min_confidence, np, waveform):
    # ANALYZE
    pitch, confidence, amplitude, result = detector.detect_pitch(waveform)

    # Mask amplitude with confidence
    amplitude[confidence <= min_confidence] = 0.0

    # RESYNTHESIZE
    duration = 10.0
    t_audio = np.linspace(0, duration, waveform.shape[0], endpoint=False)
    t_control = np.linspace(0, duration, pitch.shape[0], endpoint=False)

    # convert pitch information from midi, to cycles per second, to radians per sample
    cycles_per_second = librosa.midi_to_hz(pitch)
    radians_per_second = 2 * np.pi * cycles_per_second
    radians_per_sample = radians_per_second / detector.sample_rate

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
    return amplitude, confidence, pitch, result, sine_wave


@app.cell
def _(
    amplitude,
    confidence,
    detector,
    mo,
    pitch,
    plot_with_dual_y,
    plot_with_title,
    plt,
    result,
    sine_wave,
):
    figure1, ax1 = plt.subplots(1, 1)
    # ax1.imshow(detector.factors.T, origin="lower")
    figure2 = plot_with_dual_y(
        pitch,
        amplitude,
        "Pitch",
        "Amplitude",
        "Pitches and Amplitudes",
    )
    figure3 = plot_with_title(confidence, "Confidence")
    figure4, ax4 = plt.subplots(1, 1)
    ax4.imshow(result, origin="lower")

    mo.vstack(
        items=[
            # figure1,
            mo.hstack(items=[figure2, figure3]),
            mo.hstack(
                items=[
                    # figure4,
                    mo.audio(
                        src=sine_wave,
                        rate=detector.sample_rate,
                        normalize=True,
                    ),
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
