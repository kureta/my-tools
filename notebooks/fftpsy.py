import marimo

__generated_with = "0.17.0"
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
        librosa,
        median_filter,
        mo,
        np,
        pl,
        plt,
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


@app.cell
def _(SAMPLE_RATE, librosa, mo):
    def load_selected(selected):
        # if n := len(selected.value) > 10:
        #     print(f"Too many samples were selected ({n})")
        #     print("Not creating any players.")
        #     raise ValueError("Too many samples!")

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
    return (load_selected,)


@app.cell
def _(load_selected, sol_selection):
    samples = sorted(load_selected(sol_selection), key=lambda x: x["pitch"])
    # play_selected(samples)
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

        amp_threshold = np.percentile(filtered_amps, 98)
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
        redux = x.shape[0] % target_len
        x = x[:-redux]
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
    # draw_peaks(samples)
    draw_peaks(samples[:1])
    return


@app.cell
def _(SAMPLE_RATE, np):
    def make_wave(f):
        t = np.linspace(0, 1, SAMPLE_RATE)
        wave = np.zeros_like(t)

        for f, a in zip(f * np.arange(7), 0.88 ** (np.arange(7) + 1)):
            wave += a * np.sin(2 * np.pi * f * t)
        return wave
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


@app.cell
def _(Figure, downsample_median, get_spectrum_data, samples):
    def do_the_thing(win_size):
        fs, amps = get_spectrum_data(samples[0]["data"])
        idx = fs < 4000
        fs, amps = fs[idx], amps[idx]
        fs, amps = (
            downsample_median(fs, win_size),
            downsample_median(amps, win_size),
        )

        fig = Figure(figsize=(12, 5), dpi=300)
        ax = fig.add_axes((0.05, 0.15, 0.9, 0.8))
        ax.plot(fs, amps)

        return fig
    return (do_the_thing,)


@app.cell
def _(do_the_thing):
    do_the_thing(1024)
    return


@app.cell
def _(downsample_median, get_spectrum_data, np, samples):
    def do_all():
        fss = []
        ampss = []
        for sample in samples:
            fs, amps = get_spectrum_data(sample["data"])
            idx = fs < 4000
            fs, amps = fs[idx], amps[idx]
            fs, amps = downsample_median(fs, 1024), downsample_median(amps, 1024)
            fss.append(fs)
            ampss.append(amps)

        return np.vstack(fss), np.vstack(ampss)
    return (do_all,)


@app.cell
def _(do_all):
    f1, a1 = do_all()
    return a1, f1


@app.cell
def _(downsample_median, get_spectrum_data, samples):
    f0_, a0_ = get_spectrum_data(samples[0]["data"])
    idx0 = f0_ < 4000
    f0_, a0_ = f0_[idx0], a0_[idx0]
    f0, a0 = downsample_median(f0_, 1024), downsample_median(a0_, 1024)
    return a0, f0


@app.cell
def _(curva):
    curva.shape
    return


@app.cell
def _(a0, a1, dissonance, f0, f1):
    curva = dissonance(f0, a0, f1, a1)
    return (curva,)


@app.cell
def _(curva, plt, samples):
    plt.plot([s["pitch"] for s in samples][:12], curva.sum(axis=0)[:12])
    return


@app.cell
def _(f1):
    f1.shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
