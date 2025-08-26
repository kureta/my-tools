from dataclasses import dataclass


@dataclass
class State:
    audio_path = "/home/kureta/Music/IRCAM/Orchidea_tam_tam_0.6/CSOL_tam_tam/Percussion/tympani-sticks/middle/tt-tymp_mid-ff-N.wav"

    # TODO: this is a hack. get proper file name
    @property
    def file_name(self):
        return self.audio_path.split("/")[-1]


state = State()
