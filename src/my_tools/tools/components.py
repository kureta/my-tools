from nicegui import ui


# TODO: components may be too specific, not reusable
def create_slider(min, max, step, on_change, state, value, label):
    with ui.row():
        ui.label(label)
        ui.label("").bind_text(state, value)
    ui.slider(
        min=min,
        max=max,
        step=step,
        on_change=on_change,
    ).classes(
        "w-96"
    ).bind_value(state, value)


def create_diss_curve_controls(state, show_plot):
    with ui.card().tight().style("padding: 1.5rem; gap: 0.5rem"):
        ui.markdown("**Parameters of dissonance curve calculation**")
        ui.separator()
        with ui.row():
            ui.label("Calculation method:")
            ui.select(["min", "product"], value="min", on_change=show_plot).bind_value(
                state, "method"
            )
        create_slider(
            0,
            2400,
            100,
            show_plot,
            state,
            "start_delta_cents",
            "start interval (cents)",
        )
        create_slider(
            1200,
            2600,
            100,
            show_plot,
            state,
            "delta_cents_range",
            "interval range (cents)",
        )
        create_slider(0, 1, 0.01, show_plot, state, "peak_cutoff", "peak cutoff:")


def create_diss_curve_display(state, show_plot):
    with ui.card():
        with ui.row():
            ui.label("n peaks detected:")
            ui.label("").bind_text(state, "n_peaks")
        state.figure = ui.matplotlib(figsize=(11, 4)).figure
        show_plot()
