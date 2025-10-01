import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import librosa
    import scipy.signal
    import matplotlib.pyplot as plt
    return librosa, np, plt, scipy


@app.cell(hide_code=True)
def _(librosa, np, plt, scipy):
    def auto_partial_peaks(
        audio_path,
        sr=44100,
    ):
        # Load audio (mono)
        y, sr = librosa.load(audio_path, sr=sr, mono=True)

        spectrum = np.abs(np.fft.rfft(y, norm="forward"))
        # spectrum = librosa.amplitude_to_db(spectrum, ref=1.0, amin=1e-10, top_db=None)

        freqs = np.fft.rfftfreq(y.shape[0]) * sr
        idx = (freqs <= 2000) & (freqs > 25)
        freqs = freqs[idx]
        spectrum = spectrum[idx]

        # Estimate noise floor, mean, std
        noise_floor = np.median(spectrum)
        mean = np.mean(spectrum)
        std = np.std(spectrum)
        max_val = np.max(spectrum)

        # use the spacing between strongest peaks as estimate of possible F0
        # get the indices of the N highest peaks for initial spacing estimate
        prelim_peaks, _ = scipy.signal.find_peaks(
            spectrum, height=noise_floor + 2 * std
        )
        if len(prelim_peaks) > 1:
            spacings = np.diff(prelim_peaks)
            mean_spacing = np.mean(spacings)
        else:
            mean_spacing = 1

        # Heuristics for detection
        min_distance = int(mean_spacing * 0.7) if mean_spacing > 5 else 5
        prominence = (mean + 2 * std) * 0.3
        height = noise_floor * std

        # Final peak detection
        peaks, props = scipy.signal.find_peaks(
            spectrum,
            height=height,
            distance=min_distance,
            prominence=prominence,
        )

        partial_freqs = freqs[peaks]
        partial_mags = spectrum[peaks]

        this = np.flip(np.argsort(partial_mags))
        partial_freqs = partial_freqs[this][:6]
        partial_mags = partial_mags[this][:6]

        # Plot for inspection
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, spectrum, label="Spectrum")
        plt.plot(partial_freqs, partial_mags, "rx", label="Detected partials")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("Detected Partials in Spectrum")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Print and return
        print("Detected partial frequencies (Hz):")
        print(np.round(partial_freqs, 2))
        print("Detected partial magnitudes:")
        print(np.round(partial_mags, 3))
        return partial_freqs, partial_mags
    return (auto_partial_peaks,)


@app.cell
def _(auto_partial_peaks):
    # Usage
    # Replace 'yourfile.wav' with your audio file path
    audio_file = "/home/kureta/Music/IRCAM/OrchideaSOL2020/Strings/Violoncello/ordinario/Vc-ord-A#2-mf-3c-T13u.wav"
    # audio_file = "/home/kureta/Music/IRCAM/CSOL_multiphonics/Winds/Multiphonics-Cl-vo/MulClBb-mulvocl-N-N-mph10.wav"

    partials, mags = auto_partial_peaks(audio_file, sr=44100)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
