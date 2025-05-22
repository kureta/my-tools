import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    return np, plt


@app.cell
def _(np):
    def morlet_manual(t, bandwidth, center_freq):
        """Generate complex Morlet wavelet.
        t: time vector (e.g., np.linspace)
        bandwidth: width of Gaussian envelope (B, dimensionless)
        center_freq: central frequency (C, dimensionless)
        """
        norm = 1.0 / np.sqrt(bandwidth * np.sqrt(np.pi))
        return norm * np.exp(1j * 2 * np.pi * center_freq * t) * np.exp(-t**2 / (2 * bandwidth**2))

    def manual_cwt(signal, scales, bandwidth, center_freq, dt=1.0):
        n = len(signal)
        t = np.arange(-n//2, n//2) * dt
        cwt_matrix = np.zeros((len(scales), n), dtype=complex)
        for idx, scale in enumerate(scales):
            wavelet_data = morlet_manual(t / scale, bandwidth, center_freq)
            wavelet_data = np.conj(wavelet_data[::-1])  # time-reverse and conjugate for convolution
            conv = np.convolve(signal, wavelet_data, mode='same')
            cwt_matrix[idx, :] = conv / np.sqrt(scale)  # normalize by sqrt(scale)
        return cwt_matrix
    return manual_cwt, morlet_manual


@app.cell
def _(manual_cwt, morlet_manual, np):
    bandwidth = 1.5
    center_freq = 1.0

    t = np.linspace(-2, 2, 400)
    wavelet = morlet_manual(t, bandwidth, center_freq)

    signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 200))
    scales = np.arange(1, 50)
    cwt_coeffs = manual_cwt(signal, scales, bandwidth, center_freq)
    return cwt_coeffs, scales, signal


@app.cell
def _(Twxo, cwt_coeffs, np, plt, scales, signal):
    figure,  (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    ax1.imshow(np.abs(cwt_coeffs), aspect='auto', extent=[0, len(signal), scales[-1], scales[0]])
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Scale')
    ax1.set_title('CWT')

    ax2.imshow(np.abs(Twxo), aspect='auto', extent=[0, len(signal), scales[-1], scales[0]])
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Scale')
    ax2.set_title('Squeezed CWT')

    plt.tight_layout()

    figure
    return


@app.cell
def _(signal):
    from ssqueezepy import ssq_cwt
    from ssqueezepy.experimental import scale_to_freq

    # #%%# Define signal ####################################
    # N = 2048
    # t = np.linspace(0, 10, N, endpoint=False)
    # xo = np.cos(2 * np.pi * 2 * (np.exp(t / 2.2) - 1))
    # xo += xo[::-1]  # add self reflected
    # x = xo + np.sqrt(2) * np.random.randn(N)  # add noise

    #%%# CWT + SSQ CWT ####################################
    Twxo, Wxo, *_ = ssq_cwt(signal)
    return (Twxo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
