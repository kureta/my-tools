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

    from my_tools.pitch_detector import PitchDetector

    mo.md("# Pitch detection")
    return (
        Audio,
        Path,
        PitchDetector,
        interp1d,
        librosa,
        mo,
        np,
        plt,
        torch,
        torchaudio,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Some TODOS

        ⚠️ **Instead of making up a bell curve at a given frequency, use stft of a sine wave at that frequency with the same window that is used to generate the spectrogram of the target signal.**

        - [ ] If a pitch does not correspond exactly to an FFT bin it has to be normalized. We can normalize the area under the curve or we can redistribute the actual peaks value to the surrounding bins proportionally, and then normalize.
        - [ ] Using stft at that frequency instead of the above method. But should revisit that.
        - [ ] We are currently decaying the amplitudes of overtones by the squareroot of their index. This can be further investigated.
        - [ ] Refactor synthesizer into its own class
        - [ ] Implement everything in torch/torchaudio and ditch numpy. numpy can only be used during initialization, not for any calculations.
        """
    )
    return


@app.cell
def _(PitchDetector):
    detector = PitchDetector()
    return (detector,)


@app.cell
def p_1(detector, librosa, np, plt, torch):
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
    return plot_waveform, plot_with_dual_y, plot_with_title


@app.cell
def _(mo):
    offset = mo.ui.slider(0.0, 60.0, 0.5, label="Start offset in seconds.")

    offset
    return (offset,)


@app.cell
def _(Audio, Path, detector, librosa, mo, offset, torch):
    SAMPLE_WAV = Path(
        "/home/kureta/Music/Flute Samples/14. 3 Oriental Pieces_ I. Bergere captive.wav"
    )
    waveform, _ = librosa.load(SAMPLE_WAV, sr=detector.sample_rate, mono=False)
    waveform = waveform[
        :,
        int(offset.value * detector.sample_rate) : int(
            (offset.value + 10.0) * detector.sample_rate
        ),
    ]
    waveform = torch.from_numpy(waveform)

    mo.vstack(
        items=[
            Audio(data=waveform, rate=detector.sample_rate),
        ]
    )
    return SAMPLE_WAV, waveform


@app.cell
def _(detector, mo, plot_with_dual_y, plot_with_title, plt, waveform):
    pitch, confidence, amplitude = detector.detect_pitch(waveform[0])

    # Mask amplitude with confidence
    min_confidence = 0.025
    amplitude[confidence <= min_confidence] = 0.0


    mo.vstack(
        items=[
            plt.matshow(detector.factors),
            mo.hstack(
                items=[
                    plot_with_dual_y(
                        pitch,
                        amplitude,
                        "Pitch",
                        "Amplitude",
                        "Pitches and Amplitudes",
                    ),
                    plot_with_title(confidence, "Confidence"),
                ]
            ),
        ]
    )
    return amplitude, confidence, min_confidence, pitch


@app.cell
def _(amplitude, detector, interp1d, librosa, np, pitch, waveform):
    duration = 10.0
    t_audio = np.linspace(0, duration, waveform.shape[1], endpoint=False)
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
def _(Audio, detector, sine_wave):
    Audio(data=sine_wave, rate=detector.sample_rate)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
