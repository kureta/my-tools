import marimo

__generated_with = "0.15.0"
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

    from my_tools.seth import dissonance, prepare_sweep, plot_curve, get_peaks
    return (
        Figure,
        dissonance,
        find_peaks,
        get_peaks,
        librosa,
        mo,
        np,
        plot_curve,
        plt,
        prepare_sweep,
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
    cello_path = "/home/kureta/Music/IRCAM/orchideaSOL2020/_OrchideaSOL2020_release/OrchideaSOL2020/Strings/Violoncello/ordinario/Vc-ord-C2-ff-4c-N.wav"
    cello, _ = librosa.load(cello_path, mono=True, sr=sr)
    cello /= np.abs(cello).max()

    mo.audio(cello, sr, normalize=True)
    return (cello,)


@app.cell
def _(find_peaks, librosa, np, sr):
    def get_overtones(audio, min_delta_f=25.96, max_delta_db=24, max_f=4434.92):
        """Calculates peaks of the spectrum to take as overtones of a sound

        audio = audio
        min_delta_f = minimum distance between overtones (in Hz). 25.96 is a half-step below the lowest note on the piano
        max_delta_db = overtones with a loudness this much lower than the loudes overtone will be ignored
        max_f = frequencies above this will be ignored. 4434.92 is a half-step above the highest note on the piano
        """
        # absolute value of stft (amplitudes)
        spectrum = np.abs(np.fft.rfft(audio, norm="forward"))
        # frequency at index (bin to Hz array)
        spec_freqs = np.fft.rfftfreq(audio.shape[0]) * sr
        # cut frequencies above threshold
        filtered_freqs_idx = spec_freqs <= max_f
        fs = spec_freqs[filtered_freqs_idx]
        mags = spectrum[filtered_freqs_idx]
        # Convert amplitude to db
        mags = librosa.amplitude_to_db(mags, ref=1.0, amin=1e-10, top_db=None)
        # TODO: temporary workaround for negative values messing up the dissonance curve calculation
        mags += 100
        # calculate overtone loudness lower limit
        highest = mags.max()
        lower_limit = highest - max_delta_db

        # NOTE:
        # dissonance curve calculatiopn depends on the absolute (not relative) value of amplitudes
        # curve is completely different if all amplitudes are scaled equally by some factor
        # is it OK?
        peaks, _ = find_peaks(
            mags,
            distance=min_delta_f / spec_freqs[1],
            height=lower_limit,
        )

        return fs, mags, peaks
    return (get_overtones,)


@app.cell
def _(cello, get_overtones, plt):
    cello_fs, cello_mags, cello_peaks = get_overtones(cello)

    plt.figure(figsize=(12, 6))
    plt.plot(cello_fs, cello_mags)
    plt.plot(cello_fs[cello_peaks], cello_mags[cello_peaks], "ro", label="minima")
    plt.gca()
    return cello_fs, cello_mags, cello_peaks


@app.cell
def _(get_overtones, plt, tamtam):
    fs, mags, tamtam_peaks = get_overtones(tamtam)

    plt.figure(figsize=(12, 6))
    plt.plot(fs, mags)
    plt.plot(fs[tamtam_peaks], mags[tamtam_peaks], "ro", label="minima")
    plt.gca()
    return fs, mags, tamtam_peaks


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
    prepare_sweep,
    tamtam_peaks,
):
    freq1 = fs[tamtam_peaks]
    amp1 = mags[tamtam_peaks]
    f0 = fs[tamtam_peaks][0]
    amp2 = cello_mags[cello_peaks]

    bin_pairs, amp_pairs, cents = prepare_sweep(
        f0,
        freq1,
        amp1,
        cello_fs[cello_peaks][0],
        cello_fs[cello_peaks],
        amp2,
        0,
        1300,
        1,
    )
    # calculate dissonance curve
    dissonance_curve = dissonance(bin_pairs, amp_pairs, model="min")
    peaks, d2curve = get_peaks(cents, dissonance_curve, height=0.2)

    # find peaks in the dissonance curve
    height = 0.70
    print(np.round(peaks).astype(int))
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## My attempt at a cleaner and faster version of the dissonance curve calculation""")
    return


@app.cell
def _(Figure, dissonance, get_peaks, np, plot_curve, prepare_sweep):
    # prepare partials
    n_harmonics = 20
    f1 = 440.0
    harmonics = f1 * np.arange(1, n_harmonics + 1)
    amps = 0.88 ** np.arange(0, n_harmonics)

    # lolo
    # freq1 = fs[tamtam_peaks]
    # amp1 = mags[tamtam_peaks]
    # f0 = fs[tamtam_peaks][0]
    # freq2 = f0 * (cello_fs[cello_peaks] / cello_fs[cello_peaks][0])
    # amp2 = cello_mags[cello_peaks]
    # pairs_spekis, pairs_ampiks, x_axis = prepare_sweep(
    #     f0, freq1, amp1, cello_fs[cello_peaks][0], cello_fs[cello_peaks], amp2, 0, 1300, 1
    # )
    # dbs = librosa.amplitude_to_db(amps, ref=1.0, amin=1e-10, top_db=None)
    # dbs -= dbs.min()
    # dbs += librosa.A_weighting(harmonics)
    # prepare vectorized pairs of partials
    pairs_spekis, pairs_ampiks, x_axis = prepare_sweep(
        f1, harmonics, amps, f1, harmonics, amps, 0, 1300, 1
    )
    # calculate dissonance curve
    curve = dissonance(pairs_spekis, pairs_ampiks, model="min")
    ppeaks, id2curve = get_peaks(x_axis, curve, height=0.2)

    # find peaks in the dissonance curve
    k = 0.25
    print(x_axis[ppeaks])
    #  and plot the curve with the peaks marked
    plot_curve(x_axis, curve, id2curve, ppeaks, Figure(figsize=(12, 4)))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
