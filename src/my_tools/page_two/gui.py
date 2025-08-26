from nicegui import ui

from my_tools.page_two.state import Config
from my_tools.tools.file_picker import local_file_picker

conf = Config()


async def pick_file():
    result = (await local_file_picker("~/Music/"))[0]
    conf.audio_path = result


def create_page_two():
    ui.markdown("# Audio Page Placeholder")  # noqa: F821
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
