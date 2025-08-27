import librosa
import numpy as np
from matplotlib import pyplot as plt
from nicegui import ui
from scipy.signal import find_peaks

from my_tools.page_two.state import state
from my_tools.tools.file_picker import local_file_picker


async def pick_file1():
    result = await local_file_picker(
        "~/Music/IRCAM/Orchidea_tam_tam_0.6/CSOL_tam_tam/Percussion/"
    )
    state.audio1_path = result[0]
    state.audio1, _ = process_audio(state.audio1_path)
    state.spectrum1, state.amplitudes1, state.peaks1 = get_overtones(state.audio1)


async def pick_file2():
    result = await local_file_picker(
        "~/Music/IRCAM/SOL_0.9_HQ/Strings/Violoncello/ordinario/"
    )
    state.audio2_path = result[0]
    state.audio2, _ = process_audio(state.audio2_path)
    state.spectrum2, state.amplitudes2, state.peaks2 = get_overtones(state.audio2)


def process_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=state.sr, mono=True, duration=10)
    audio /= np.abs(audio).max()

    return audio


def get_overtones(audio, min_delta_f=25.96, max_delta_db=0.11, max_f=4434.92):
    """Calculates peaks of the spectrum to take as overtones of a sound

    audio = audio
    min_delta_f = minimum distance between overtones (in Hz). 25.96 is a half-step below the lowest note on the piano
    max_delta_db = overtones with a loudness this much lower than the loudes overtone will be ignored
    max_f = frequencies above this will be ignored. 4434.92 is a half-step above the highest note on the piano
    """
    # absolute value of stft (amplitudes)
    spectrum = np.abs(np.fft.rfft(audio, norm="forward"))
    # frequency at index (bin to Hz array)
    spec_freqs = np.fft.rfftfreq(audio.shape[0]) * state.sr
    # cut frequencies above threshold
    filtered_freqs_idx = spec_freqs <= max_f
    fs = spec_freqs[filtered_freqs_idx]
    mags = spectrum[filtered_freqs_idx]
    # Convert amplitude to db
    # mags = librosa.amplitude_to_db(mags, ref=1.0, amin=1e-10, top_db=None)
    # # TODO: temporary workaround for negative values messing up the dissonance curve calculation
    # mags += 100
    # # calculate overtone loudness lower limit
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


def create_page_two():
    state.audio1 = process_audio(state.audio1_path)
    state.spectrum1, state.amplitudes1, state.peaks1 = get_overtones(state.audio1)
    state.audio2 = process_audio(state.audio2_path)
    state.spectrum2, state.amplitudes2, state.peaks2 = get_overtones(state.audio2)
    with ui.row():
        with ui.column():
            ui.markdown("# Audio Page Placeholder")  # noqa: F821
            with ui.button_group():
                ui.button(
                    "Page One",
                    color="",
                    on_click=lambda: ui.navigate.to("/"),
                )
                ui.button(
                    "Two",
                    color="primary",
                    on_click=lambda: ui.navigate.to("/audio"),
                )

            with ui.card().tight().style("padding: 1.0rem; gap: 0.5rem"):
                ui.button("Choose lower voice", on_click=pick_file1, icon="folder")
                ui.label("").bind_text_from(state, "file1_name")
                ui.audio("").bind_source_from(state, "audio1_path")

            with ui.card().tight().style("padding: 1.0rem; gap: 0.5rem"):
                ui.button("Choose upper voice", on_click=pick_file2, icon="folder")
                ui.label("").bind_text_from(state, "file2_name")
                ui.audio("").bind_source_from(state, "audio2_path")
        with ui.column():
            with ui.pyplot(figsize=(12, 4)):
                if len(state.peaks1) <= 0 or len(state.peaks2) <= 0:
                    return
                plt.plot(state.spectrum1, state.amplitudes1)
                plt.plot(
                    state.spectrum1[state.peaks1],
                    state.amplitudes1[state.peaks1],
                    "ro",
                    label="minima",
                )
            with ui.pyplot(figsize=(12, 4)):
                if len(state.peaks1) <= 0 or len(state.peaks2) <= 0:
                    return
                plt.plot(state.spectrum2, state.amplitudes2)
                plt.plot(
                    state.spectrum2[state.peaks2],
                    state.amplitudes2[state.peaks2],
                    "ro",
                    label="minima",
                )
