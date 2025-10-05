import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# First sketch of the toolset for my thesis""")
    return


@app.cell
def _():
    from collections import defaultdict
    from pathlib import Path

    import librosa
    import marimo as mo
    from matplotlib.figure import Figure
    import numpy as np
    import polars as pl
    from scipy.signal import find_peaks
    from scipy.ndimage import median_filter
    return (
        Figure,
        Path,
        defaultdict,
        find_peaks,
        librosa,
        median_filter,
        mo,
        np,
        pl,
    )


app._unparsable_cell(
    r"""
        SOL_PATH = \"~/Music/IRCAM/OrchideaSOL2020/\"
    SAMPLE_RATE = 44100
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Loaded SOL database into a polars `DataFrame`

    ### TODO
    - Tamtam samples do not follow the same naming standard.
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
            data["file_name"].append(str(file_path.stem))
            data["full_path"].append(str(file_path.absolute()))

        return pl.DataFrame(data)
    return (load_sol,)


@app.cell
def _(Path, SOL_PATH, load_sol, mo):
    sol = load_sol(Path(SOL_PATH).expanduser())
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
            mo.hstack(
                [
                    mo.md(f"**Sample ID:** `{row['file_name']}`"),
                    mo.audio(row["data"], SAMPLE_RATE, normalize=False),
                ]
            )
            for row in selected
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
    ### TODO: We add `data` to `sample` `dict`, we should do the same for `fs`, `amps`, etc.

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

        amp_threshold = np.percentile(filtered_amps, 95)
        min_distance = int((2.0 * min_f) / unit)

        peaks, _ = find_peaks(
            filtered_amps, distance=min_distance, height=amp_threshold
        )

        return freqs, amps, peaks[:n_partials]
    return (auto_partial_peaks,)


@app.cell
def _(Figure, auto_partial_peaks, get_spectrum_data):
    def downsample_mean(x, target_len):
        window = len(x) // target_len
        # truncate so length is divisible by window
        n = len(x) - len(x) % window
        x_cut = x[:n]
        # reshape into (n_windows, window)
        xw = x_cut.reshape(-1, window)
        # average each row
        return xw.mean(axis=1)


    def _draw_peaks(fs, amps):
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

            figures.append(_draw_peaks(fs, amps))

        return figures
    return (draw_peaks,)


@app.cell
def _(draw_peaks, samples):
    draw_peaks(samples)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
