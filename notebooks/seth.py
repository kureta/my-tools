import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    import numpy as np
    import numba as nb
    from scipy.signal import find_peaks

    import librosa

    from my_tools.seth import dissmeasure, diso, plot_this
    return diso, dissmeasure, find_peaks, librosa, mo, np, plot_this, plt


@app.cell
def _(librosa, mo):
    tamtam_path = "/home/kureta/Music/IRCAM/Orchidea_tam_tam_0.6/CSOL_tam_tam/Percussion/tympani-sticks/middle/tt-tymp_mid-ff-N.wav"
    # tamtam_path = "/home/kureta/Music/easy_thunder__thundergong1.flac"
    sr = 44100
    tamtam, _ = librosa.load(tamtam_path, mono=True, sr=sr)

    mo.audio(tamtam, sr, normalize=False)
    return sr, tamtam


@app.cell
def _(librosa, mo, sr):
    cello_path = "/home/kureta/Music/IRCAM/orchideaSOL2020/_OrchideaSOL2020_release/OrchideaSOL2020/Strings/Violoncello/ordinario/Vc-ord-C2-ff-4c-N.wav"
    cello, _ = librosa.load(cello_path, mono=True, sr=sr)

    mo.audio(cello, sr, normalize=False)
    return (cello,)


@app.cell
def _(find_peaks, np, sr):
    def get_overtones(audio, prominence=0.05):
        spectrum = np.abs(np.fft.rfft(audio))
        spec_freqs = np.fft.rfftfreq(audio.shape[0]) * sr

        filtered_freqs_idx = spec_freqs <= 4000
        fs = spec_freqs[filtered_freqs_idx]
        mags = spectrum[filtered_freqs_idx]
        mags /= mags.max()

        # TODOs
        # Use Bark Scale to set the minimum distance between peaks
        # convert amplitudes to perceptual loudness for dissonance curve calculation
        # dissonance curve calculatiopn depends on the absolute (not relative) value of amplitudes
        # curve is completely different if all amplitudes are scaled equally by some factor
        # is it OK?
        peaks, _ = find_peaks(
            mags, prominence=prominence, distance=50 / spec_freqs[1], height=0.01
        )

        return fs, mags, peaks
    return (get_overtones,)


@app.cell
def _(cello, get_overtones, plt):
    cello_fs, cello_mags, cello_peaks = get_overtones(cello, prominence=0.01)

    plt.figure(figsize=(12, 6))
    plt.plot(cello_fs, cello_mags)
    plt.plot(cello_fs[cello_peaks], cello_mags[cello_peaks], "ro", label="minima")
    plt.gca()
    return cello_fs, cello_mags, cello_peaks


@app.cell
def _(get_overtones, plt, tamtam):
    fs, mags, tamtam_peaks = get_overtones(tamtam, prominence=0.01)
    tamtam_peaks = tamtam_peaks[1:]

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

    # n_harm = 21
    # freq2 = f0 * (np.array(range(1, n_harm + 1)) ** 1.05)
    # amp2 = 1 / np.array(range(1, n_harm + 1))

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
    peaks, props = find_peaks(-diss, prominence=0.05)

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
    plt.plot(
        np.linspace(440, 440 * 2.1, 1000),
        diso(440, np.linspace(440, 440 * 2.1, 1000), 1, 1),
    )
    return


@app.cell
def _(diso, np, plt):
    n_harmonics = 32
    harmonics = np.arange(1, n_harmonics + 1)
    amps = 0.88 ** np.arange(0, n_harmonics)

    base_freq = 500.0
    f1 = base_freq

    num = 5000
    curve = np.empty(num)
    for idx, factor in enumerate(np.linspace(1, 2.3, num)):
        f2 = base_freq * factor

        spec1 = f1 * harmonics
        spec2 = f2 * harmonics

        curve[idx] = np.sum(
            diso(spec1[:, None], spec2[None, :], amps[:, None], amps[None, :])
        )

    plt.figure(figsize=(8.3, 3.3))
    plt.plot(np.linspace(1, 2.3, num), curve)
    plt.show()
    return


@app.cell
def _(plot_this):
    plot_this()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
