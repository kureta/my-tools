# pyright: basic

from typing import Any, Callable, Optional, Self

from nicegui import ui


# TODO: components may be too specific, not reusable
class LabeledSlider(ui.slider):
    def __init__(self, min, max, step, on_change, label):

        with ui.row():
            ui.label(label)
            self.value_label = ui.label("")
        super().__init__(min=min, max=max, step=step, on_change=on_change)
        self.classes("w-96")

    def bind_value(
        self,
        target_object: Any,
        target_name: str = "value",
        *,
        forward: Optional[Callable[[Any], Any]] = None,
        backward: Optional[Callable[[Any], Any]] = None,
    ) -> Self:
        self.value_label.bind_text_from(target_object, target_name)
        return super().bind_value(
            target_object, target_name, forward=forward, backward=backward
        )
