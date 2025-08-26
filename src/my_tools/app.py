from nicegui import ui

from my_tools.logic import conf, pick_file
from my_tools.page_one import create_page_one


@ui.page("/")
def main_page():
    create_page_one()


@ui.page("/audio")
def audio_page():
    ui.markdown("# Audio Page Placeholder")
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

    with ui.card():
        ui.button("Choose lower voice", on_click=pick_file, icon="folder")
        ui.label("").bind_text_from(conf, "file_name")
        ui.audio("").bind_source_from(conf, "audio_path")


ui.run()
