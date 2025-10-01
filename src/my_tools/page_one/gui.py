# pyright: basic

from nicegui import ui

from my_tools.page_one.state import state
from my_tools.tools.components import LabeledSlider
from my_tools.tools.diss_curve import DissCurve


def create_page_one():
    diss_curve = DissCurve(state)
    # Top level row of the page
    with ui.row():
        # Side bar column
        with ui.column():
            # Navigation buttons
            with ui.button_group():
                _ = ui.button(
                    "Page One", color="primary", on_click=lambda: ui.navigate.to("/")
                )
                _ = ui.button(
                    "Two", color="", on_click=lambda: ui.navigate.to("/audio")
                )
            _ = ui.separator()

            # Sliders for setting parameters of synthetic partilas
            # TODO: Should we package these too or expand the diss. curve sliders?
            with ui.card().style("width: 100%"):
                _ = ui.markdown("## Control Parameters")
            with ui.card().tight().style("padding: 1.5rem; gap: 0.5rem"):
                _ = ui.markdown("**Parameters of synthetic partials**")
                _ = ui.separator()
                _ = LabeledSlider(
                    21, 108, 1, diss_curve.calculate_diss_curve, "base midi #:"
                ).bind_value(state, "midi1")
                _ = LabeledSlider(
                    1, 32, 1, diss_curve.calculate_diss_curve, "number of partials:"
                ).bind_value(state, "n_harmonics")
                _ = LabeledSlider(
                    0, 1, 0.01, diss_curve.calculate_diss_curve, "amplitude decay:"
                ).bind_value(state, "amp_decay")
                _ = LabeledSlider(
                    0.95,
                    1.05,
                    0.001,
                    diss_curve.calculate_diss_curve,
                    "spectral stretch 1:",
                ).bind_value(state, "stretch_1")
                _ = LabeledSlider(
                    0.95,
                    1.05,
                    0.001,
                    diss_curve.calculate_diss_curve,
                    "spectral stretch 2:",
                ).bind_value(state, "stretch_2")

            # Parameters for dissonance curve calculation
            # TODO: why are these packages like this?
            diss_curve.create_diss_curve_controls()

        # Main display. For dissonance curve, consonance peaks, partials, spectra, etc. as necessary
        with ui.column():
            with ui.row():
                _ = ui.markdown("## Synthetic spectra")

            # The dissonance curve itself
            diss_curve.create_diss_curve_display()
