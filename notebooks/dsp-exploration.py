import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    import numpy as np
    import librosa
    from librosa.core.spectrum import __overlap_add as overlap_add
    from librosa.util import frame as make_frames
    return librosa, make_frames, mo, np, plt


@app.cell
def _(librosa, mo):
    SAMPLE_RATE = 8192
    SAMPLE_PATH = "/home/kureta/Music/Cello Samples/BrahmsSonata1-00018-.wav"

    signal, _ = librosa.load(SAMPLE_PATH, sr=SAMPLE_RATE, mono=True, duration=10.0)

    mo.audio(signal, rate=SAMPLE_RATE)
    return SAMPLE_RATE, signal


@app.cell
def _():
    windowing = "triangle"
    return (windowing,)


@app.cell
def _(librosa, make_frames, np, signal, windowing):
    signal_frames = make_frames(signal, frame_length=512, hop_length=128)
    window = librosa.filters.get_window(windowing, 512)
    signal_frames = signal_frames * window[:, None]

    stft = np.fft.rfft(
        signal_frames,
        axis=0,
        # window="hann",
        # center=True,
        # dtype="complex128",
        # pad_mode="reflect",
    )
    return stft, window


@app.cell
def _(np, plt, stft):
    fig1, fig1_ax1 = plt.subplots(1, 1)
    fig1_ax1.imshow(np.flipud(np.abs(stft)), aspect="auto")
    fig1
    return


@app.cell
def _(SAMPLE_RATE, np):
    def make_complex_signal(freq, n_samples):
        duration = n_samples / SAMPLE_RATE
        t = np.linspace(-duration / 2, duration / 2, n_samples)
        z_signal1 = np.exp(2 * np.pi * 1j * t * freq)

        return z_signal1
    return (make_complex_signal,)


@app.cell
def _(librosa, make_complex_signal, np, plt, stft, window):
    harmonic_signal = []
    interval = np.linspace(21, 109, (109 - 21) * 10)
    results = []
    slice = 150
    for midi in interval:
        for idx in range(1, 21):
            freq = librosa.midi_to_hz(midi) * idx
            if freq > 8192 / 2:
                break
            decay = np.sqrt(idx)
            harmonic_signal.append(
                make_complex_signal(freq, 512) / decay
            )  # np.sqrt(idx))
            harmonic_signal.append(
                make_complex_signal(-freq, 512) / decay
            )  # np.sqrt(idx))

        harmonic_signalx = sum(harmonic_signal) * window
        harmonic_signal = []
        # harmonic_signal = harmonic_signal / np.max(np.abs(harmonic_signal))
        z_stft = np.fft.fft(harmonic_signalx, axis=0)[:257] * 2
        # z_stft = z_stft / np.max(np.abs(z_stft))

        result = np.abs(z_stft * stft[:, slice])
        results.append(result.mean())

    fig2, fig2_ax1 = plt.subplots(1, 1, figsize=(12, 7))
    fig2_ax1.plot(np.abs(stft)[:, slice])
    fig2
    return interval, results, slice


@app.cell
def _(plt, results):
    fig3, fig3_ax1 = plt.subplots(1, 1, figsize=(12, 7))
    # fig3_ax1.plot(np.array(results) - results[0]*np.exp(-2 * np.pi * np.arange(len(results))/len(results)))
    fig3_ax1.plot(results)
    # fig3_ax1.imshow(np.stack(np.abs(stacks)))
    fig3
    return


@app.cell
def _(interval, librosa, make_complex_signal, np, results, window):
    cursor = np.argmax(results)
    hs = []
    for i in range(1, 21):
        f = librosa.midi_to_hz(interval[cursor]) * i
        if f > 8192 / 2:
            break
        hs.append(make_complex_signal(f, 512))
        hs.append(make_complex_signal(-f, 512))

    hsx = sum(hs) * window
    zs = np.fft.fft(hsx, axis=0)[:257] * 2
    return (zs,)


@app.cell
def _(np, plt, slice, stft, zs):
    plt.plot(np.abs(stft[:, slice]) / np.max(np.abs(stft[:, slice])))
    plt.plot(
        (np.abs(stft[:, slice]) / np.max(np.abs(stft[:, slice])))
        * np.abs(zs)
        / np.max(np.abs(zs))
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
