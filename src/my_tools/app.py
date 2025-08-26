from nicegui import ui

from my_tools.page_one.gui import create_page_one
from my_tools.page_two.gui import create_page_two


@ui.page("/")
def main_page():
    create_page_one()


@ui.page("/audio")
def audio_page():
    create_page_two()


ui.run()
