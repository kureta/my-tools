# pyright: basic

from nicegui import binding, ui

from my_tools.page_one.gui import create_page_one
from my_tools.page_two.gui import create_page_two


@ui.page("/")
def main_page():
    create_page_one()


@ui.page("/audio")
def audio_page():
    create_page_two()


# Prints the number of active links if there are any.
# These reduce performance.
def bull():
    if n_active_links := len(binding.active_links) > 0:
        print(f"WARNING: There are {n_active_links} active links!")
        for b in binding.active_links:
            print(b)


_ = ui.timer(
    1.0,
    bull,
)

ui.run()
