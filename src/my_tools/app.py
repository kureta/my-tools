from nicegui import binding, context, ui

from my_tools.page_one.gui import create_page_one
from my_tools.page_two.gui import create_page_two


@ui.page("/")
def main_page():
    create_page_one()


@ui.page("/audio")
def audio_page():
    create_page_two()


ui.timer(
    1.0,
    lambda: (
        print("bindings:", len(binding.bindings)),
        print("a. links:", len(binding.active_links)),
        print("b. props:", len(binding.bindable_properties)),
    ),
)

ui.run()
