from dataclasses import dataclass, field

import numpy as np
from nicegui.elements.pyplot import MatplotlibFigure


@dataclass
class State:
    audio1_path = "/home/kureta/Music/IRCAM/Orchidea_tam_tam_0.6/CSOL_tam_tam/Percussion/double-bass-bow/tt-bow-p-N.wav"
    audio2_path = "/home/kureta/Music/IRCAM/SOL_0.9_HQ/Strings/Violoncello/ordinario/Vc-ord-C2-mf-4c.wav"

    audio1 = np.empty(0)
    spectrum1 = np.empty(0)
    audio2 = np.empty(0)
    spectrum2 = np.empty(0)

    f1 = 440
    peaks1 = np.empty(0)
    amplitudes1 = np.empty(0)
    f2 = 440
    peaks2 = np.empty(0)
    amplitudes2 = np.empty(0)

    start_delta_cents = 0
    delta_cents_range = 1300
    peak_cutoff = 0.2
    method = "min"

    figure: MatplotlibFigure = field(init=False)

    def has_figure(self):
        return self.__dict__.get("figure", False)


state = State()
