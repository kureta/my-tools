import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # pyright: basic

    import marimo as mo
    import matplotlib.pyplot as plt

    import numpy as np
    from scipy.signal import find_peaks

    import librosa

    from my_tools.seth import dissmeasure, diso, plot_this
    return diso, dissmeasure, find_peaks, librosa, mo, np, plot_this, plt


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
    cello_fs,
    cello_mags,
    cello_peaks,
    dissmeasure,
    find_peaks,
    fs,
    mags,
    np,
    plt,
    tamtam_peaks,
):
    freq1 = fs[tamtam_peaks]
    amp1 = mags[tamtam_peaks]
    f0 = fs[tamtam_peaks][0]
    freq2 = f0 * (cello_fs[cello_peaks] / cello_fs[cello_peaks][0])
    amp2 = cello_mags[cello_peaks]

    r_low = 1.0
    alpharange = 2.1
    method = "min"

    n = 3000
    diss = np.empty(n)
    a = np.concatenate((amp1, amp2))
    for i, alpha in enumerate(np.linspace(r_low, alpharange, n)):
        f = np.concatenate((freq1, alpha * freq2))
        d = dissmeasure(f, a, method)
        diss[i] = d

    # 2) find local minima: a point i is a local minimum if y[i-1] > y[i] < y[i+1]
    #    so we look for sign changes in the discrete derivative
    peaks, props = find_peaks(-diss, prominence=0.15)

    x = np.linspace(r_low, alpharange, len(diss))

    plt.figure(figsize=(12, 6))
    plt.plot(x, diss)
    plt.plot(x[peaks], diss[peaks], "ro", label="minima")
    plt.xscale("log")
    plt.xlim(r_low, alpharange)

    plt.xlabel("frequency ratio")
    plt.ylabel("sensory dissonance")

    # 1) draw vertical dashed lines at each minima
    for xi in x[peaks]:
        plt.axvline(x=xi, color="b", linestyle="-", alpha=0.3)

    plt.grid(axis="y", which="major", linestyle="--", color="gray", alpha=0.7)

    # 2) add ticks at those x‐positions and label them with their numerical values
    plt.minorticks_off()
    plt.xticks(x[peaks], [f"{int(np.round(t))}" for t in np.log2(x[peaks]) * 1200])

    plt.tight_layout()
    plt.gca()
    return peaks, x


@app.cell
def _(np, peaks, x):
    [f"{int(np.round(t))}" for t in np.log2(x[peaks]) * 1200]
    return


@app.cell
def _(fs, librosa, peaks, tamtam_peaks, x):
    print(librosa.hz_to_midi(fs[tamtam_peaks][1]))
    print()

    for ratio in x[peaks]:
        hz = fs[tamtam_peaks][0] * ratio
        midi = librosa.hz_to_midi(hz) - librosa.hz_to_midi(fs[tamtam_peaks][0])
        print(midi)
    return


@app.function
def generate_scale(generator, octave, length):
    tmp = 0
    scale = []

    for idx in range(length - 1):
        tmp = (tmp + generator) % octave
        scale.append(tmp)

    scale.sort()

    return scale


@app.cell
def _():
    # 12-EDO for 1.05
    asd = []
    current = 0
    for _ in range(12):
        current = (current + 735) % 1260
        asd.append(current)

    asd.sort()
    print(asd)
    return


@app.cell
def _():
    # Pythagorian for 1.05
    print(generate_scale(737, 1260, 12))
    return


@app.cell
def _():
    # Pythagorian for 0.95
    print(generate_scale(667, 1140, 12))
    return


@app.cell
def _():
    # Tamtam
    print(generate_scale(291, 1207, 12))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""## My attempt at a cleaner and faster version of the dissonance curve calculation"""
    )
    return


@app.cell
def _(diso, np, plt):
    f_zero = 100.0
    x_axis = (f_zero * np.linspace(1, 3, 1000))[None, :]
    y_axis = diso(np.array([[f_zero]]), x_axis, 1.0, 1.0)
    plt.plot(x_axis[0], y_axis[0])
    return


@app.cell
def _(diso, find_peaks, np, plt):
    n_harmonics = 20
    harmonics = np.arange(1, n_harmonics + 1)
    amps = 0.88 ** np.arange(0, n_harmonics)

    f1 = 220.0

    # TODO: number of points = number of cents
    num = 3000

    # axis 0: different frequency ratios
    # axis 1: different harmonics
    # all combinations of f1 * harmonics and f2 * harmonics
    f2 = f1 * np.linspace(1, 2.3, num)
    all_spec2s = f2[:, None] * harmonics[None, :]
    # match the dimension of harmonics b ystacking num of them at axis 0
    spec1s = np.tile(f1 * harmonics[None, :], (num, 1))
    # same for amps
    all_amps = np.tile(amps[None, :], (num, 1))
    all_spekis = np.concatenate([spec1s, all_spec2s], axis=1)
    all_ampiks = np.concatenate([all_amps, all_amps], axis=1)
    ia, ja = np.triu_indices(all_spekis.shape[1], k=1)
    left_spekis = all_spekis[:, ia]
    right_spekis = all_spekis[:, ja]
    left_ampiks = all_ampiks[:, ia]
    right_ampiks = all_ampiks[:, ja]
    curve = np.sum(
        diso(left_spekis, right_spekis, left_ampiks, right_ampiks), axis=1
    )

    xpeaks, xprops = find_peaks(-curve, prominence=0.15)

    xx = np.linspace(1, 2.3, num)

    plt.figure(figsize=(11, 3), constrained_layout=True)
    plt.plot(xx, curve)
    plt.plot(xx[xpeaks], curve[xpeaks], "ro", label="minima")
    plt.xscale("log")
    plt.xlim(1, 2.3)

    plt.xlabel("frequency ratio")
    plt.ylabel("sensory dissonance")

    # 1) draw vertical dashed lines at each minima
    for xii in xx[xpeaks]:
        plt.axvline(x=xii, color="b", linestyle="-", alpha=0.3)

    plt.grid(axis="y", which="major", linestyle="--", color="gray", alpha=0.7)

    # 2) add ticks at those x‐positions and label them with their numerical values
    plt.minorticks_off()
    plt.xticks(
        xx[xpeaks], [f"{int(np.round(t))}" for t in np.log2(xx[xpeaks]) * 1200]
    )

    plt.gca().tick_params(axis="x", rotation=45, labelsize=8)

    plt.gcf()
    return


@app.cell
def _(plot_this):
    plot_this()
    return


@app.cell
def _():
    import utils.plot_helpers as ph
    return (ph,)


@app.cell
def _(ph):
    ph.simple_plot()
    return


@app.cell
def _(ph):
    ph.dual_xaxis_plot()
    return


@app.cell
def _(ph):
    ph.grid_plot()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
