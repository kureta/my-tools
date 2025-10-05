# pyright: basic

import json
import sys

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QMainWindow, QPushButton, QSlider, QVBoxLayout,
                             QWidget)


class AppState(QObject):
    filePathChanged = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._file_path = ""
        self.load_persistent()

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, value):
        if self._file_path != value:
            self._file_path = value
            self.filePathChanged.emit(value)

    def load_persistent(self):
        try:
            with open("appstate.json", "r") as f:
                data = json.load(f)
                self._file_path = data.get("file_path", "")
        except FileNotFoundError:
            pass

    def save_persistent(self):
        with open("appstate.json", "w") as f:
            json.dump({"file_path": self._file_path}, f)


class LabeledSlider(QWidget):
    def __init__(self, name, start, end, tick):
        super().__init__()
        outer = QVBoxLayout(self)
        label_container = QWidget()
        outer.addWidget(label_container)
        labels = QHBoxLayout(label_container)

        lbl_name = QLabel()
        lbl_name.setText(name)
        lbl_value = QLabel()
        labels.addWidget(lbl_name)
        labels.addWidget(lbl_value)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(start)
        slider.setMaximum(end)
        slider.setInvertedAppearance
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(tick)
        slider.valueChanged.connect(lambda x: lbl_value.setText(str(x)))
        lbl_value.setText(str(slider.value()))
        outer.addWidget(slider)
        self.slider = slider
        self.valueChanged = slider.valueChanged

    def value(self):
        return self.slider.value()

    def setValue(self, value):
        self.slider.setValue(value)


# Tell pyqtgraph to use OpenGL whenever possible
pg.setConfigOptions(useOpenGL=True, background="w", foreground="k")


class GLSinePlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.xx = np.linspace(-2 * np.pi, 2 * np.pi, 2000)
        self.phase = 0.0

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(
            elevation=90,  # look straight down
            azimuth=-90,  # rotate so X maps to screen‐right
        )

        # create the GLLinePlotItem with width=4
        pos = self._pos(self.phase)
        self.curve = gl.GLLinePlotItem(
            pos=pos, color="black", width=8.0, antialias=True
        )
        self.view.addItem(self.curve)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 360)
        self.slider.valueChanged.connect(self.update_phase)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        self.setMinimumSize(500, 400)

    def _pos(self, φ):
        y = np.sin(self.xx + φ)
        z = np.zeros_like(y)
        return np.vstack((self.xx, y, z)).T

    def update_phase(self, deg):
        phi = np.deg2rad(deg)
        self.curve.setData(pos=self._pos(phi), color="black", width=8.0, antialias=True)


class SinePlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.xx = np.linspace(0, 2 * np.pi, 2000)
        self.phase = 0.0

        # useOpenGL=True gives you a QOpenGLWidget canvas under the hood
        self.plot = pg.PlotWidget(useOpenGL=True, antialias=True)
        self.curve = self.plot.plot(self.xx, np.sin(self.xx + self.phase), pen="black")
        self.plot.setLabel("bottom", "x")
        self.plot.setLabel("left", "sin(x + φ)")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 360)
        self.slider.valueChanged.connect(self.update_phase)

        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        self.setMinimumSize(500, 300)

    def update_phase(self, deg):
        self.phase = np.deg2rad(deg)
        self.curve.setData(self.xx, np.sin(self.xx + self.phase))


class MainWindow(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setWindowTitle("State-driven File Chooser")

        central = QWidget()
        layout = QVBoxLayout(central)
        self.btn = QPushButton("Choose File")
        self.lbl = QLabel()
        self.slider = QSlider(orientation=Qt.Orientation.Horizontal)
        layout.addWidget(SinePlotter())
        layout.addWidget(self.btn)
        layout.addWidget(self.lbl)
        layout.addWidget(self.slider)
        self.setCentralWidget(central)

        slit = LabeledSlider("asd", 1, 10, 1)
        layout.addWidget(slit)

        # wire up UI → state
        self.btn.clicked.connect(self.on_choose_file)
        # wire up state → UI
        self.state.filePathChanged.connect(self.lbl.setText)

        # initialize UI from state
        if self.state.file_path:
            self.lbl.setText(self.state.file_path)

    def on_choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a file")
        if path:
            self.state.file_path = path


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_state = AppState()
    w = MainWindow(app_state)
    w.showMaximized()
    exit_code = app.exec()
    app_state.save_persistent()
    sys.exit(exit_code)
