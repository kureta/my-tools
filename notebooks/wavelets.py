import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt

    from ssqueezepy import ssq_cwt
    from ssqueezepy.experimental import scale_to_freq
    import librosa
    return librosa, mo, np, plt, ssq_cwt


@app.cell
def _(np):
    def morlet_manual(t, bandwidth, center_freq):
        """Generate complex Morlet wavelet.
        t: time vector (e.g., np.linspace)
        bandwidth: width of Gaussian envelope (B, dimensionless)
        center_freq: central frequency (C, dimensionless)
        """
        norm = 1.0 / np.sqrt(bandwidth * np.sqrt(np.pi))
        return (
            norm
            * np.exp(1j * 2 * np.pi * center_freq * t)
            * np.exp(-(t**2) / (2 * bandwidth**2))
        )


    def manual_cwt(signal, scales, bandwidth, center_freq, dt=1.0):
        n = len(signal)
        t = np.arange(-n // 2, n // 2) * dt
        cwt_matrix = np.zeros((len(scales), n), dtype=complex)
        for idx, scale in enumerate(scales):
            wavelet_data = morlet_manual(t / scale, bandwidth, center_freq)
            wavelet_data = np.conj(
                wavelet_data[::-1]
            )  # time-reverse and conjugate for convolution
            conv = np.convolve(signal, wavelet_data, mode="same")
            cwt_matrix[idx, :] = conv / np.sqrt(scale)  # normalize by sqrt(scale)
        return cwt_matrix
    return


@app.cell
def _(librosa, mo, ssq_cwt):
    SAMPLE_WAV = "/home/kureta/Music/Cello Samples/Romberg38-00026-.wav"

    signal, _ = librosa.load(SAMPLE_WAV, sr=8192, mono=True, duration=2.0)

    bandwidth = 1.5
    center_freq = 4.0

    # t = np.linspace(-2, 2, 400)
    # wavelet = morlet_manual(t, bandwidth, center_freq)
    # wavelets = []
    # for w in range(1, 11):
    #     wavelets.append(morlet_manual(t, bandwidth, w))

    # wavelet = np.sum(wavelets) / 10.0

    # scales = np.arange(1, 256)
    # cwt_coeffs = manual_cwt(signal, scales, bandwidth, center_freq)

    # %%# CWT + SSQ CWT ####################################
    Twxo, Wxo, *_ = ssq_cwt(signal)

    stft = librosa.stft(signal)

    mo.audio(signal, rate=8192)
    return Twxo, stft


@app.cell
def _(Twxo, np, plt, stft):
    figure, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7))

    # ax1.imshow(np.abs(cwt_coeffs), aspect='auto')
    # ax1.set_xlabel('Sample')
    # ax1.set_ylabel('Scale')
    # ax1.set_title('CWT')

    ax2.imshow(np.abs(Twxo) ** 0.5, aspect="auto")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Scale")
    ax2.set_title("Squeezed CWT")

    ax3.imshow(np.flipud(np.abs(stft)) ** 0.5, aspect="auto")
    ax3.set_xlabel("Sample")
    ax3.set_ylabel("Scale")
    ax3.set_title("STFT")

    plt.tight_layout()

    figure
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
