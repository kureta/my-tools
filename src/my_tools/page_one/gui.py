from nicegui import ui

from my_tools.page_one.state import state
from my_tools.tools.components import LabeledSlider
from my_tools.tools.diss_curve import DissCurve


def create_page_one():
    diss_curve = DissCurve(state)
    with ui.row():
        with ui.column():
            with ui.button_group():
                ui.button(
                    "Page One",
                    color="primary",
                    on_click=lambda: ui.navigate.to("/"),
                )
                ui.button(
                    "Two",
                    color="",
                    on_click=lambda: ui.navigate.to("/audio"),
                )
            ui.separator()

            with ui.card().style("width: 100%"):
                ui.markdown("## Control Parameters")
            with ui.card().tight().style("padding: 1.5rem; gap: 0.5rem"):
                ui.markdown("**Parameters of synthetic partials**")
                ui.separator()
                LabeledSlider(
                    21, 108, 1, diss_curve.calculate_diss_curve, "base midi #:"
                ).bind_value(state, "midi1")
                LabeledSlider(
                    1, 32, 1, diss_curve.calculate_diss_curve, "number of partials:"
                ).bind_value(state, "n_harmonics")
                LabeledSlider(
                    0, 1, 0.01, diss_curve.calculate_diss_curve, "amplitude decay:"
                ).bind_value(state, "amp_decay")
                LabeledSlider(
                    0.9,
                    1.1,
                    0.001,
                    diss_curve.calculate_diss_curve,
                    "spectral stretch 1:",
                ).bind_value(state, "stretch_1")
                LabeledSlider(
                    0.95,
                    1.05,
                    0.001,
                    diss_curve.calculate_diss_curve,
                    "spectral stretch 2:",
                ).bind_value(state, "stretch_2")
            diss_curve.create_diss_curve_controls()

        with ui.column():
            with ui.row():
                ui.markdown("## Synthetic spectra")

            diss_curve.create_diss_curve_display()
