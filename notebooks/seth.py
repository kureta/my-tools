import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    # pyright: basic

    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    import numpy as np
    from scipy.signal import find_peaks

    import librosa

    from my_tools.seth import dissonance, sweep_partials, get_peaks
    return (
        Figure,
        dissonance,
        find_peaks,
        get_peaks,
        librosa,
        mo,
        np,
        plt,
        sweep_partials,
    )


@app.cell
def _(librosa, mo, np):
    tamtam_path = "/home/kureta/Music/IRCAM/Orchidea_tam_tam_0.6/CSOL_tam_tam/Percussion/tympani-sticks/middle/tt-tymp_mid-ff-N.wav"
    # tamtam_path = "/home/kureta/Music/easy_thunder__thundergong1.flac"
    sr = 44100
    tamtam, _ = librosa.load(tamtam_path, mono=True, sr=sr)
    tamtam /= np.abs(tamtam).max()

    mo.audio(tamtam, sr, normalize=True)
    return sr, tamtam


@app.cell
def _(librosa, mo, np, sr):
    cello_path = "/home/kureta/Music/IRCAM/OrchideaSOL2020/Strings/Violoncello/ordinario/Vc-ord-F2-mf-4c-T17u.wav"
    cello, _ = librosa.load(cello_path, mono=True, sr=sr)
    cello /= np.abs(cello).max()

    mo.audio(cello, sr, normalize=True)
    return (cello,)


@app.cell
def _(find_peaks, np, sr):
    def get_overtones(audio, h, min_delta_f=25.96, max_delta_db=24, max_f=4434.92):
        """Calculates peaks of the spectrum to take as overtones of a sound

        audio = audio
        min_delta_f = minimum distance between overtones (in Hz). 25.96 is a half-step below the lowest note on the piano
        max_delta_db = overtones with a loudness this much lower than the loudes overtone will be ignored
        max_f = frequencies above this will be ignored. 4434.92 is a half-step above the highest note on the piano
        """
        # absolute value of stft (amplitudes)
        spectrum = np.abs(np.fft.rfft(audio, norm="forward"))
        spectrum /= np.abs(spectrum).max()
        # frequency at index (bin to Hz array)
        spec_freqs = np.fft.rfftfreq(audio.shape[0]) * sr
        # cut frequencies above threshold
        filtered_freqs_idx = spec_freqs <= max_f
        fs = spec_freqs[filtered_freqs_idx]
        mags = spectrum[filtered_freqs_idx]
        # Convert amplitude to db
        # mags = librosa.amplitude_to_db(mags, ref=1.0, amin=1e-10, top_db=None)
        # TODO: temporary workaround for negative values messing up the dissonance curve calculation
        # mags += 100
        # calculate overtone loudness lower limit
        # highest = mags.max()
        # lower_limit = highest - max_delta_db

        # NOTE:
        # dissonance curve calculatiopn depends on the absolute (not relative) value of amplitudes
        # curve is completely different if all amplitudes are scaled equally by some factor
        # is it OK?
        peaks, _ = find_peaks(
            mags,
            distance=min_delta_f / spec_freqs[1],
            height=h,
        )

        return fs, mags, peaks
    return (get_overtones,)


@app.cell
def _(cello, get_overtones, plt):
    cello_fs, cello_mags, cello_peaks = get_overtones(cello, h=0.02)

    plt.figure(figsize=(12, 6))
    plt.plot(cello_fs, cello_mags)
    plt.plot(cello_fs[cello_peaks], cello_mags[cello_peaks], "ro", label="minima")
    plt.gca()
    return cello_fs, cello_mags, cello_peaks


@app.cell
def _(get_overtones, plt, tamtam):
    fs, mags, tamtam_peaks = get_overtones(tamtam, h=0.08)

    plt.figure(figsize=(12, 6))
    plt.plot(fs, mags)
    plt.plot(fs[tamtam_peaks], mags[tamtam_peaks], "ro", label="minima")
    plt.gca()
    return fs, mags, tamtam_peaks


@app.cell
def _(np):
    def plot_curve(x_axis, curve, d2curve, dpeaks, figure):
        ax1 = figure.add_axes((0.05, 0.15, 0.9, 0.8))
        ax2 = ax1.twinx()

        ax1.plot(x_axis, curve, color="blue")
        ax2.plot(x_axis, d2curve, color="gray", alpha=0.6)
        ax1.plot(x_axis[dpeaks], curve[dpeaks], "ro", label="minima")

        for xii in x_axis[dpeaks]:
            ax1.axvline(x=xii, color="b", linestyle="-", alpha=0.3)

        ax1.grid(axis="y", which="major", linestyle="--", color="gray", alpha=0.7)

        ax1.set_xlabel("interval in cents")
        ax1.set_ylabel("sensory dissonance")

        ax2.set_ylabel("peak strength (normalized)")
        ax1.set_xticks(
            x_axis[dpeaks],
            [f"{int(np.round(t))}" for t in x_axis[dpeaks]],
        )
        ax1.tick_params(axis="x", rotation=45, labelsize=8)

        return figure
    return (plot_curve,)


@app.cell
def _(
    Figure,
    cello_fs,
    cello_mags,
    cello_peaks,
    dissonance,
    fs,
    get_peaks,
    mags,
    np,
    plot_curve,
    sweep_partials,
    tamtam_peaks,
):
    freq1 = fs[tamtam_peaks]
    amp1 = mags[tamtam_peaks]
    tmp = cello_fs[cello_peaks]
    tmp = (tmp / tmp[0]) * freq1[0]
    freq2 = sweep_partials(tmp, 0, 1300, 1)
    amp2 = cello_mags[cello_peaks]

    cents = np.linspace(0, 1300, 1300)

    # calculate dissonance curve
    dissonance_curve = dissonance(freq1, amp1, freq2, amp2)
    peaks, d2curve = get_peaks(cents, dissonance_curve, height=0.2)

    # find peaks in the dissonance curve
    height = 0.70
    print(np.round(peaks).astype(int))
    print(freq1[0], cello_fs[cello_peaks][0], tmp[0])
    #  and plot the curve with the peaks marked
    fig = Figure(figsize=(12, 4), dpi=100)

    plot_curve(cents, dissonance_curve, d2curve, peaks, fig)
    return


@app.function
def generate_scale(generator, octave, length):
    tmp = 0
    scale = []

    for _ in range(length - 1):
        tmp = (tmp + generator) % octave
        scale.append(tmp)

    scale.sort()

    return scale


@app.cell
def _():
    # Pythagorian for 1.05
    print(generate_scale(737, 1260, 12))
    # Pythagorian for 0.95
    print(generate_scale(667, 1140, 12))
    # Tamtam
    print(generate_scale(291, 1207, 12))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
