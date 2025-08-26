from nicegui import ui

from my_tools.page_one.logic import conf, show_plot


# TODO: refactor out dissonance cirve calculation controls and
#       dissonance curve display as reusable ui components
def create_slider(min, max, step, on_change, value, label):
    with ui.row():
        ui.label(label)
        ui.label("").bind_text(conf, value)
    ui.slider(
        min=min,
        max=max,
        step=step,
        on_change=on_change,
    ).classes(
        "w-96"
    ).bind_value(conf, value)


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
                create_slider(21, 108, 1, show_plot, "midi1", "base midi #:")
                create_slider(1, 32, 1, show_plot, "n_harmonics", "number of partials:")
                create_slider(0, 1, 0.01, show_plot, "amp_decay", "amplitude decay:")
            with ui.card().tight().style("padding: 1.5rem; gap: 0.5rem"):
                ui.markdown("**Parameters of dissonance curve calculation**")
                ui.separator()
                with ui.row():
                    ui.label("Calculation method:")
                    ui.select(
                        ["min", "product"], value="min", on_change=show_plot
                    ).bind_value(conf, "method")
                create_slider(
                    0,
                    2400,
                    100,
                    show_plot,
                    "start_delta_cents",
                    "start interval (cents)",
                )
                create_slider(
                    1200,
                    2600,
                    100,
                    show_plot,
                    "delta_cents_range",
                    "interval range (cents)",
                )
                create_slider(0, 1, 0.01, show_plot, "peak_cutoff", "peak cutoff:")

        with ui.column():
            with ui.row():
                ui.markdown("## Synthetic spectra")

            with ui.card():
                with ui.row():
                    ui.label("n peaks detected:")
                    ui.label("").bind_text(conf, "n_peaks")
                conf.figure = ui.matplotlib(figsize=(14, 4)).figure
                show_plot()
