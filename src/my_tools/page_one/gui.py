from nicegui import ui

from my_tools.page_one.logic import show_plot
from my_tools.page_one.state import state
from my_tools.tools.components import (
    create_diss_curve_controls,
    create_diss_curve_display,
    create_slider,
)


def create_page_one():
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
                create_slider(21, 108, 1, show_plot, state, "midi1", "base midi #:")
                create_slider(
                    1, 32, 1, show_plot, state, "n_harmonics", "number of partials:"
                )
                create_slider(
                    0, 1, 0.01, show_plot, state, "amp_decay", "amplitude decay:"
                )
            create_diss_curve_controls(state, show_plot)

        with ui.column():
            with ui.row():
                ui.markdown("## Synthetic spectra")

            create_diss_curve_display(state, show_plot)
