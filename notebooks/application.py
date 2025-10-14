import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# First sketch of the toolkit for my thesis""")
    return


@app.cell
def _():
    from collections import defaultdict
    from pathlib import Path

    import librosa
    import marimo as mo
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import numpy as np
    import einx as ex
    import polars as pl
    from scipy.signal import find_peaks
    from scipy.ndimage import median_filter

    from my_tools.seth import (
        dissonance,
        sweep_partials,
        generate_partial_freqs,
        generate_partial_amps,
    )
    return (
        Figure,
        Path,
        defaultdict,
        dissonance,
        find_peaks,
        generate_partial_amps,
        generate_partial_freqs,
        librosa,
        median_filter,
        mo,
        np,
        pl,
        plt,
        sweep_partials,
    )


@app.cell
def _():
    SOL_PATH = "~/Music/IRCAM/OrchideaSOL2020/"
    TAMTAM_PATH = "~/Music/IRCAM/CSOL_tam_tam/"
    SAMPLE_RATE = 44100
    return SAMPLE_RATE, SOL_PATH


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Loaded SOL database into a polars `DataFrame`

    ### **TODO**
    - Multiphonics samples do not follow the same naming standard.
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

        if instrument == "Tamtam":
            inst, tech, dyn, pitch = filepath.stem.split("-")
            instance, misc = None, None
        else:
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
        sol_path = Path(sol_path).expanduser()
        data = defaultdict(list)
        for file_path in sol_path.rglob("*.wav"):
            for k, v in parse_filepath(file_path.relative_to(sol_path)).items():
                data[k].append(v)
            data["file_name"].append(str(file_path.stem))
            data["full_path"].append(str(file_path.absolute()))

        return pl.DataFrame(data)
    return (load_sol,)


@app.cell
def _(SOL_PATH, load_sol, mo):
    sol = load_sol(SOL_PATH)
    # tamtam = load_sol(TAMTAM_PATH)
    # sol.extend(tamtam)
    sol_selection = mo.ui.table(sol)
    sol_selection
    return (sol_selection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Play the samples selected from the above table""")
    return


@app.cell(hide_code=True)
def _(SAMPLE_RATE, librosa, mo):
    def load_selected(selected):
        if n := len(selected.value) > 10:
            print(f"Too many samples were selected ({n})")
            print("Not creating any players.")
            raise ValueError("Too many samples!")

        samples = selected.value.rows(named=True)
        for row in samples:
            sample, _ = librosa.load(row["full_path"], mono=True, sr=SAMPLE_RATE)
            row["data"] = sample

        return samples


    def play_selected(selected):
        ui = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.md(f"**Sample ID:** `{row['file_name']}`"),
                        mo.audio(row["data"], SAMPLE_RATE, normalize=False),
                    ]
                )
                for row in selected
            ]
        )

        return ui
    return load_selected, play_selected


@app.cell
def _(load_selected, play_selected, sol_selection):
    samples = load_selected(sol_selection)
    play_selected(samples)
    return (samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Draw spectrum / partials graph""")
    return


@app.cell
def _(SAMPLE_RATE, librosa, np):
    def get_spectrum_data(audio, is_db=True, is_A_weighted=True):
        amplitudes = np.abs(np.fft.rfft(audio, norm="forward"))
        # frequency at index (bin to Hz array)
        frequencies = np.fft.rfftfreq(audio.shape[0]) * SAMPLE_RATE

        # drop dc
        amplitudes = amplitudes[1:]
        frequencies = frequencies[1:]

        # Convert amplitude to db
        if is_db:
            # convert to db
            amplitudes = librosa.amplitude_to_db(
                amplitudes, ref=1.0, amin=1e-20, top_db=None
            )
            # make all values positive (and +2 just in case)
            amplitudes += 180
            # perceptual weighing of frequencies
            if is_A_weighted:
                amplitudes += librosa.A_weighting(frequencies)

        return frequencies, amplitudes
    return (get_spectrum_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### **TODO**: We add `data` to `sample` `dict`, we should do the same for `fs`, `amps`, etc.

    - `fs` and `amps` that are direct results of fft (I thiink we can ignore these)
    - `fs` amd `amps` filtered for peak detection
    - `idx` peak indices for filtered values
    - Then remove `auto_partial_peaks` and `get_spectrum_data` from drawing functions.
    """
    )
    return


@app.cell
def _(find_peaks, median_filter, np):
    def auto_partial_peaks(freqs, amps, n_partials=-1, min_f=25.0, max_f=4434.92):
        # cut below audible range and above some arbitrary frequency
        unit = freqs[1] if freqs[0] == 0.0 else freqs[0]
        idx = (freqs <= max_f) & (freqs >= min_f)
        freqs = freqs[idx]
        amps = amps[idx]

        # THIS HELPS
        noise_floor = median_filter(amps, size=int((16.0 * min_f) / unit))
        filtered_amps = amps - noise_floor

        amp_threshold = np.percentile(filtered_amps, 99)
        min_distance = int(min_f / unit)

        peaks, _ = find_peaks(
            filtered_amps, distance=min_distance, height=amp_threshold
        )

        return freqs, amps, peaks[:n_partials]
    return (auto_partial_peaks,)


@app.cell
def _(Figure, auto_partial_peaks, get_spectrum_data, np):
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


    def downsample_median(x, target_len):
        window = len(x) // target_len
        # truncate so length is divisible by window
        n = len(x) - len(x) % window
        x_cut = x[:n]
        # reshape into (n_windows, window)
        xw = x_cut.reshape(-1, window)
        # average each row
        return np.median(xw, axis=1)


    def draw_peaks_(fs, amps):
        fs_peaks, amp_peaks, peaks = auto_partial_peaks(fs, amps)

        fig = Figure(figsize=(12, 5), dpi=300)
        ax = fig.add_axes((0.05, 0.15, 0.9, 0.8))
        ax.scatter(fs_peaks[peaks], amp_peaks[peaks], color="red")
        ax.plot(fs_peaks, amp_peaks)

        return fig


    def draw_peaks(selection):
        figures = []
        for s in selection:
            fs, amps = get_spectrum_data(s["data"])

            figures.append(draw_peaks_(fs, amps))

        return figures
    return downsample_median, draw_peaks


@app.cell
def _(draw_peaks, samples):
    draw_peaks(samples)
    return


@app.cell
def _(auto_partial_peaks, downsample_median, get_spectrum_data):
    def load_peaks(data):
        fs, amps = get_spectrum_data(data)
        fs, amps = downsample_median(fs, 1024), downsample_median(amps, 1024)
        fs_peaks, amp_peaks, peaks = auto_partial_peaks(fs, amps)

        return fs_peaks, amp_peaks, peaks
    return


@app.cell
def _(librosa, samples):
    librosa.midi_to_hz(samples[0]["pitch"])
    return


@app.cell
def _(
    Figure,
    SAMPLE_RATE,
    auto_partial_peaks,
    generate_partial_amps,
    generate_partial_freqs,
    get_spectrum_data,
    librosa,
    np,
    samples,
    sweep_partials,
):
    n_partials = 7

    tmp_fs_, tmp_amp_ = get_spectrum_data(
        librosa.load(
            "/home/kureta/Music/titanium-gong.wav", mono=True, sr=SAMPLE_RATE
        )[0]
    )
    # tmp_fs, tmp_amp, tmp_idx_ = auto_partial_peaks(tmp_fs_, tmp_amp_)
    # tmp_idx = tmp_idx_[: n_partials + 4]

    tmp_fs = generate_partial_freqs(
        librosa.midi_to_hz(samples[0]["pitch"]), n_partials, 1.02
    )

    # tmp_fs *= 2 ** (25.0 * np.random.randn(*tmp_fs.shape) / 1200)
    tmp_amp = librosa.amplitude_to_db(
        generate_partial_amps(1 / 32, n_partials, 0.88),
        ref=1.0,
        amin=1e-20,
        top_db=None,
    )
    tmp_amp += librosa.A_weighting(tmp_fs) + 180
    tmp_idx = np.arange(len(tmp_fs))

    tr_fs_, tr_amp_ = get_spectrum_data(samples[0]["data"])
    tr_fs, tr_amp, tr_idx_ = auto_partial_peaks(tr_fs_, tr_amp_)
    tr_idx = tr_idx_[:n_partials]
    s_tr_fs = sweep_partials(tr_fs[tr_idx], -100, 1300, 0.5)

    fig = Figure(figsize=(12, 5), dpi=300)
    ax = fig.add_axes((0.05, 0.15, 0.9, 0.8))
    ax.scatter(tmp_fs[tmp_idx], tmp_amp[tmp_idx], color="red")
    ax.scatter(tr_fs[tr_idx], tr_amp[tr_idx], color="blue")
    # ax.plot(tmp_fs, tmp_amp)
    fig
    return s_tr_fs, tmp_amp, tmp_fs, tmp_idx, tr_amp, tr_idx


@app.cell
def _(dissonance, np, plt, s_tr_fs, tmp_amp, tmp_fs, tmp_idx, tr_amp, tr_idx):
    curve = dissonance(tmp_fs[tmp_idx], tmp_amp[tmp_idx], s_tr_fs, tr_amp[tr_idx])
    plt.plot(np.linspace(-100, 1300, 1400 * 2), curve)
    return (curve,)


@app.cell
def _(curve, find_peaks, np, plt):
    def normalize(x):
        x -= x.min()
        x /= x.max()
        return x


    d2 = normalize(np.gradient(np.gradient(curve)))
    norm_curve = normalize(curve)
    kinks = d2 * (1 - norm_curve)
    peaks_, _ = find_peaks(
        kinks, distance=33 * 2, height=np.mean(kinks) + 2 * np.std(kinks)
    )
    # peaks_, _ = find_peaks(kinks, distance=33 * 2, height=kinks.mean() + 2.0)
    peaks = peaks_  # np.sort(peaks_[np.flip(np.argsort(kinks[peaks_]))][:11])

    plt.scatter(np.linspace(-100, 1300, 1400 * 2)[peaks], kinks[peaks])
    plt.plot(np.linspace(-100, 1300, 1400 * 2), kinks)
    return (peaks,)


@app.cell
def _(SAMPLE_RATE, librosa, np, tmp_amp, tmp_fs):
    t = np.linspace(0, 6, 6 * SAMPLE_RATE)

    # wave = samples[1]["data"]

    # wave = librosa.load(
    #     "/home/kureta/Music/titanium-gong.wav", mono=True, sr=SAMPLE_RATE
    # )[0]

    wave = np.zeros_like(t)

    for f, a in zip(
        tmp_fs,
        librosa.db_to_amplitude(
            tmp_amp - librosa.A_weighting(tmp_fs) - 180, ref=1.0
        ),
    ):
        wave += a * np.sin(2 * np.pi * f * t)
    return (wave,)


@app.cell
def _(SAMPLE_RATE, librosa, mo, np, peaks, samples, wave):
    # Shouldn't normalize!
    # Get the sample with nearest pitch and shift that

    ["original", mo.audio(samples[0]["data"], SAMPLE_RATE, normalize=True)] + [
        [
            f"{int(np.round(cents / 100) * 100)}",
            mo.audio(
                librosa.effects.pitch_shift(
                    samples[0]["data"],
                    sr=SAMPLE_RATE,
                    n_steps=int(np.round(cents / 100) * 100) * 2,
                    bins_per_octave=2400,
                ),
                SAMPLE_RATE,
                normalize=True,
            ),
            f"{cents:.0f}",
            mo.audio(
                librosa.effects.pitch_shift(
                    samples[0]["data"],
                    sr=SAMPLE_RATE,
                    n_steps=int(cents * 2),
                    bins_per_octave=2400,
                ),
                SAMPLE_RATE,
                normalize=True,
            ),
            mo.audio(wave, SAMPLE_RATE, normalize=True),
        ]
        for cents in np.linspace(-100, 1300, 1400 * 2)[peaks]
    ]
    return


@app.cell
def _(SAMPLE_RATE, mo, np):
    scala = [
        0,
        85,
        182,
        267,
        316,
        386,
        498,
        583,
        702,
        765,
        814,
        884,
        969,
        1018,
        1088,
        1200,
    ]


    def make_wave(f):
        t = np.linspace(0, 1, SAMPLE_RATE)
        wave = np.zeros_like(t)

        for f, a in zip(f * np.arange(7), 0.88 ** (np.arange(7) + 1)):
            wave += a * np.sin(2 * np.pi * f * t)
        return wave


    [
        mo.audio(make_wave(440 * (2 ** (s / 1200))), SAMPLE_RATE, normalize=True)
        for s in scala
    ]
    return (scala,)


@app.cell
def _(Figure, scala):
    fig2 = Figure(figsize=(12, 5), dpi=300)
    ax2 = fig2.add_axes((0.05, 0.15, 0.9, 0.8))

    ax2.plot([b - a for a, b in zip(scala[:-1], scala[1:])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Try using the entire spectra

    Seems like a dead-end for now.

    Maybe do stft of size 8196 or something and take the average over frames. Initial transient attack of a tamtam is very loud but is only heard for a short time, has disproportionate power in stationary FFT.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Real-time audio using Supriya/Supercollier""")
    return


if __name__ == "__main__":
    app.run()
