import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    # pyright: basic

    import marimo as mo
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    import numpy as np
    from scipy.signal import find_peaks

    import librosa
    from pathlib import Path
    import polars as pl
    from collections import defaultdict

    from my_tools.seth import (
        dissonance,
        sweep_partials,
        get_peaks,
        generate_partial_freqs,
        generate_partial_amps,
        __f_dissonance
    )
    return (
        Figure,
        Path,
        defaultdict,
        dissonance,
        find_peaks,
        generate_partial_amps,
        generate_partial_freqs,
        get_peaks,
        librosa,
        mo,
        np,
        pl,
        plt,
        sweep_partials,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # TODOs

    * amplitude or db or something else?
    * having lots of partials makes the dissonance curve a jumbly mess
    * separate peak finding, getting top n according to some criterion, such that we can use all these both for finding partial peaks and dissonance dips.
    """
    )
    return


@app.cell
def _(Path, defaultdict, librosa, pl):
    def n_to_none(data):
        if data == "N":
            return None
        return data


    def convert_pitch(pitch: str):
        # this discards multiphonic pitches
        if pitch == "N" or "_" in pitch:
            return None
        return float(librosa.note_to_midi(pitch, round_midi=False))


    def parse_filepath(filepath: Path):
        category, instrument, technique = filepath.parent.parts

        mute = None
        if "+" in instrument:
            instrument, mute = instrument.split("+")

        inst, tech, pitch, dyn, instance, misc = filepath.stem.split("-")

        mt = None
        if "+" in inst:
            inst, mt = inst.split("+")

        return {
            "category": category,
            "instrument": instrument,
            "mute": mute,
            "technique": technique,
            "inst": inst,
            "mt": mt,
            "tech": tech,
            "pitch": convert_pitch(pitch),
            "dyn": n_to_none(dyn),
            "instance": n_to_none(instance),
            "misc": n_to_none(misc),
        }


    def load_sol(sol_path):
        data = defaultdict(list)
        for file_path in sol_path.rglob("*.wav"):
            for k, v in parse_filepath(file_path.relative_to(sol_path)).items():
                data[k].append(v)
            data["full_path"].append(file_path.absolute())

        return pl.DataFrame(data)
    return (load_sol,)


@app.cell
def _(Path, load_sol):
    sol_path = Path("~/Music/IRCAM/OrchideaSOL2020/").expanduser()
    df = load_sol(sol_path)
    return (df,)


@app.cell
def _(df, mo):
    table = mo.ui.table(df)
    table
    return (table,)


@app.cell
def _(librosa, mo, table):
    samplesz = []
    sr = 44100
    for path in table.value["full_path"]:
        selected_sample, _ = librosa.load(path, mono=True, sr=sr)
        samplesz.append(selected_sample)

    mo.vstack(mo.audio(ss, sr, normalize=False) for ss in samplesz)
    return (sr,)


@app.cell
def _(df, pl, table):
    filtered = df.filter((pl.col("inst") == "Vc") & pl.col("pitch").is_not_null())
    table.value
    return


@app.cell
def _(librosa, mo, sr):
    tamtam_path = "/home/kureta/Music/IRCAM/CSOL_tam_tam/Percussion/Tamtam/double-bass-bow/tt-bow-f-N.wav"
    # tamtam_path = "/home/kureta/Music/IRCAM/CSOL_multiphonics/Winds/Multiphonics-Cl-vo/MulClBb-mulvocl-N-N-mph10.wav"
    tamtam, _ = librosa.load(tamtam_path, mono=True, sr=sr)
    # tamtam = samplesz[1]
    # tamtam /= np.abs(tamtam).max()

    mo.audio(tamtam, sr, normalize=False)
    return (tamtam,)


@app.cell
def _(librosa, mo, sr):
    cello_path = "/home/kureta/Music/IRCAM/OrchideaSOL2020/Strings/Violoncello/ordinario/Vc-ord-F2-ff-4c-T17u.wav"
    cello, _ = librosa.load(cello_path, mono=True, sr=sr)
    # cello = librosa.effects.pitch_shift(
    #     samplesz[0], sr=sr, n_steps=41, bins_per_octave=1200
    # )
    # cello /= np.abs(cello).max()
    mo.audio(cello, sr, normalize=False)
    return (cello,)


@app.cell
def _(find_peaks, librosa, np, sr):
    def get_overtones(audio, h, min_delta_f=25.96, max_delta_db=24, max_f=4434.92):
        """Calculates peaks of the spectrum to take as overtones of a sound

        audio = audio
        min_delta_f = minimum distance between overtones (in Hz). 25.96 is a half-step below the lowest note on the piano
        max_delta_db = overtones with a loudness this much lower than the loudes overtone will be ignored
        max_f = frequencies above this will be ignored. 4434.92 is a half-step above the highest note on the piano
        """
        # absolute value of stft (amplitudes)
        spectrum = np.abs(np.fft.rfft(audio, norm="forward"))
        # spectrum /= np.abs(spectrum).max()
        # frequency at index (bin to Hz array)
        spec_freqs = np.fft.rfftfreq(audio.shape[0]) * sr
        # cut below audible range and above some arbitrary frequency
        filtered_freqs_idx = (spec_freqs <= 4000) & (spec_freqs > 25)
        # cut frequencies above threshold
        fs = spec_freqs[filtered_freqs_idx]
        mags = spectrum[filtered_freqs_idx]
        # Convert amplitude to db
        # THIS LINE
        mags = librosa.amplitude_to_db(mags, ref=1.0, amin=1e-20, top_db=None)
        mags += librosa.A_weighting(fs)
        # TODO: temporary workaround for negative values messing up the dissonance curve calculation
        # AND THIS LINE FOR dB
        mags += 180
        # calculate overtone loudness lower limit
        highest = mags.mean()

        # NOTE:
        # dissonance curve calculatiopn depends on the absolute (not relative) value of amplitudes
        # curve is completely different if all amplitudes are scaled equally by some factor
        # is it OK?
        peaks, _ = find_peaks(
            mags,
            distance=min_delta_f / spec_freqs[1],
            height=highest * h,
        )

        # get first n loudest partials
        # this = np.flip(np.argsort(mags[peaks]))
        # peaks = peaks[this][:6]

        return fs, mags, peaks[:6]
    return (get_overtones,)


@app.cell
def _(Figure, np):
    def downsample_to(arr, target_len):
        n = arr.size
        if n <= target_len:
            return arr
        idx = np.linspace(0, n - 1, target_len, dtype=int)
        return arr[idx]


    def downsample_mean(x, target_len):
        window = len(x) // target_len
        # truncate so length is divisible by window
        n = len(x) - len(x) % window
        x_cut = x[:n]
        # reshape into (n_windows, window)
        xw = x_cut.reshape(-1, window)
        # average each row
        return xw.mean(axis=1)


    def plot_curvez(x_axis, curve, dpeaks, downsamplez=4000):
        figure = Figure(figsize=(12, 5), dpi=300)

        ax1 = figure.add_axes((0.05, 0.15, 0.9, 0.8))

        ax1.plot(
            downsample_mean(x_axis, 1024),
            downsample_mean(curve, 1024),
            color="blue",
        )
        ax1.plot(x_axis[dpeaks], curve[dpeaks], "ro", label="minima")

        for xii in x_axis[dpeaks]:
            ax1.axvline(x=xii, color="b", linestyle="-", alpha=0.3)

        ax1.grid(axis="y", which="major", linestyle="--", color="gray", alpha=0.7)

        ax1.set_xlabel("frequency (Hz)")
        ax1.set_ylabel("loudness (dB)")
        orig_ticks = ax1.get_xticks()
        filtered_ticks = orig_ticks[
            (orig_ticks < x_axis[dpeaks].min())
            | (orig_ticks > x_axis[dpeaks].max())
        ][1:-1]
        x_ticks = np.concatenate((filtered_ticks, x_axis[dpeaks]))
        ax1.set_xticks(
            x_ticks,
            [f"{t:.2f}" for t in x_ticks],
        )
        ax1.tick_params(axis="x", rotation=45, labelsize=8)

        return figure
    return downsample_mean, plot_curvez


@app.cell
def _(cello, get_overtones, plot_curvez):
    cello_fs, cello_mags, cello_peaks = get_overtones(cello, h=1.4)
    plot_curvez(cello_fs, cello_mags, cello_peaks)
    return cello_fs, cello_mags, cello_peaks


@app.cell
def _(get_overtones, plot_curvez, tamtam):
    fs, mags, tamtam_peaks = get_overtones(tamtam, h=1.35)
    plot_curvez(fs, mags, tamtam_peaks)
    return fs, mags


@app.cell
def _(Figure, np):
    def plot_curve(x_axis, curve, d2curve, dpeaks):
        figure = Figure(figsize=(12, 4), dpi=300)

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
    cello_fs,
    cello_mags,
    cello_peaks,
    downsample_mean,
    fs,
    generate_partial_amps,
    generate_partial_freqs,
    librosa,
    mags,
    np,
    sweep_partials,
):
    span = 1200 * 1 + 100
    freq1 = generate_partial_freqs(
        librosa.midi_to_hz(42), 8, stretch_factor=1.05
    )  # fs[tamtam_peaks]
    amp1_ = generate_partial_amps(1.0, 8, decay_factor=0.88)  # mags[tamtam_peaks]
    amp1 = (
        librosa.amplitude_to_db(amp1_, ref=1.0, amin=1e-20, top_db=None)
        + librosa.A_weighting(freq1)
        + 180
    )

    tmp = cello_fs[cello_peaks]
    freq2 = sweep_partials(tmp, 0, span, 1)
    amp2 = cello_mags[cello_peaks]

    base_midi = librosa.hz_to_midi(freq1[0]) * 100

    top_midi = librosa.hz_to_midi(tmp[0]) * 100
    start = top_midi
    midi_cents = np.linspace(start, start + span, span)


    # calculate dissonance curve
    def minilt(x, y):
        idx = (x < 1000) & (x > 20)
        dfs = downsample_mean(x[idx], 512)
        dam = downsample_mean(y[idx], 512)
        return dfs, dam


    cf, cm = minilt(cello_fs, cello_mags)
    cf = sweep_partials(cf, 0, span, 1)
    mfs, mamp = minilt(fs, mags)
    # dissonance_curve = dissonance(*minilt(fs, mags), cf, cm)
    # figir = Figure(figsize=(12, 4), dpi=300)
    # bx = figir.add_axes((0.05, 0.15, 0.9, 0.8))
    # bx.plot(dissonance_curve)
    return amp1_, base_midi, cf, cm, freq1, mamp, mfs, midi_cents, start


@app.cell
def _(cf, cm, dissonance, mamp, mfs, mo, np):
    shit = []
    for dilim in mo.status.progress_bar(
        cf, title="Loading", subtitle="Please wait", show_eta=True, show_rate=True
    ):
        val = 0.0
        for left, right in zip(dilim, cm):
            for first, second in zip(mfs, mamp):
                val += dissonance(np.array([first, left]), np.array([second, right]))
        shit.append(val)
    return (shit,)


@app.cell
def _(cf, cm, mamp, mfs, plt):
    plt.plot(cf[0], cm)
    plt.plot(mfs, mamp)
    return


@app.cell
def _(np, plt, shit):
    curvat = np.array(shit)
    d2 = np.gradient(np.gradient(curvat))
    plt.plot(d2[10:])
    plt.plot(curvat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### **TODO**

    Looks like ordinary just intonation
    """
    )
    return


@app.cell
def _(dissonance_curve, find_peaks, get_peaks, midi_cents, np, plot_curve):
    peaks, d2curve = get_peaks(midi_cents, dissonance_curve, height=0.2)

    # find peaks in the dissonance curve
    height = 0.70
    mcs = [f"{mc:.2f}" for mc in midi_cents]
    amp_threshold = np.percentile(d2curve, 95)
    min_distance = 14
    #  and plot the curve with the peaks marked
    peaks, _ = find_peaks(d2curve, height=amp_threshold, distance=min_distance)
    plot_curve(midi_cents, dissonance_curve, d2curve, peaks)
    return (peaks,)


@app.cell
def _(base_midi, midi_cents, peaks, start):
    print(base_midi, start)
    print([f"{round(f - 4100) / 100}" for f in midi_cents[peaks]])
    return


@app.cell
def _(amp1_, freq1, librosa, np, sr):
    t = np.linspace(0, 6, 6 * sr)
    wave = np.zeros_like(t)

    for f, a in zip(freq1, librosa.db_to_amplitude(amp1_, ref=1.0)):
        wave += a * np.sin(2 * np.pi * f * t)

    wave /= np.abs(wave).max()
    return (wave,)


@app.cell
def _(Figure, wave):
    figa = Figure(figsize=(12, 4), dpi=300)
    axa = figa.add_axes((0.05, 0.15, 0.9, 0.8))
    axa.plot(wave[:2048])
    figa
    return


@app.cell
def _(mo, sr, wave):
    mo.audio(wave, sr, normalize=False)
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
