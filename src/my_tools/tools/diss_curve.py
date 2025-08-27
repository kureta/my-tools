import numpy as np
from nicegui import ui

from my_tools.seth import dissonance, get_peaks, prepare_sweep
from my_tools.tools.components import LabeledSlider


class DissCurve:
    def __init__(self, state):
        self.state = state

    def create_diss_curve_controls(self):
        with ui.card().tight().style("padding: 1.5rem; gap: 0.5rem"):
            ui.markdown("**Parameters of dissonance curve calculation**")
            ui.separator()
            with ui.row():
                ui.label("Calculation method:")
                ui.select(
                    ["min", "product"], value="min", on_change=self.calculate_diss_curve
                ).bind_value(self.state, "method")
            LabeledSlider(
                0,
                2400,
                100,
                self.calculate_diss_curve,
                "start interval (cents)",
            ).bind_value(self.state, "start_delta_cents")
            LabeledSlider(
                1200,
                2600,
                100,
                self.calculate_diss_curve,
                "interval range (cents)",
            ).bind_value(self.state, "delta_cents_range")
            LabeledSlider(
                0, 1, 0.01, self.calculate_diss_curve, "peak cutoff:"
            ).bind_value(self.state, "peak_cutoff")

    def create_diss_curve_display(self):
        with ui.card().style("width: 100%"):
            with ui.row():
                ui.label("n peaks detected:")
                ui.label("").bind_text(self.state, "n_peaks")
            self.state.figure = ui.matplotlib(figsize=(11, 4)).figure
            self.calculate_diss_curve()

    # TODO: separate plotting and calculation
    def calculate_diss_curve(self):
        if not self.state.has_figure():
            return

        # TODO: move  these out of here
        # these can also be generated from audio recordings or by some other method
        peaks_1, amplitudes_1 = self.state.spectrum1
        peaks_2, amplitudes_2 = self.state.spectrum2

        overtone_pairs, amplitude_pairs, cents = prepare_sweep(
            self.state.f1,
            peaks_1,
            amplitudes_1,
            self.state.f2,
            peaks_2,
            amplitudes_2,
            self.state.start_delta_cents,
            self.state.start_delta_cents + self.state.delta_cents_range,
        )

        curve = dissonance(overtone_pairs, amplitude_pairs, self.state.method)
        peaks, d2curve = get_peaks(cents, curve, height=self.state.peak_cutoff)
        self.state.n_peaks = len(peaks)

        self.plot_curve(cents, curve, d2curve, peaks)

    def plot_curve(self, x_axis, curve, d2curve, dpeaks):
        self.state.figure.clear()

        ax1 = self.state.figure.add_axes((0.05, 0.15, 0.9, 0.8))
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

        self.state.figure.element.update()
