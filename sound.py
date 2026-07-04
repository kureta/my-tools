# pyright: basic

import threading
from typing import Any

import numpy as np
import sounddevice as sd
from pythonosc import osc_server
from pythonosc.dispatcher import Dispatcher


class SineOsc:
    def __init__(
        self, frequency: float = 440.0, amplitude: float = 0.2, samplerate: int = 44100
    ) -> None:
        self.frequency: float = frequency
        self.amplitude: float = amplitude
        self.sr: int = samplerate
        self.phase: float = 0.0
        self.lock: threading.Lock = threading.Lock()

    def set_frequency(self, _addr: str, _args: Any, freq: float) -> None:
        with self.lock:
            self.frequency = freq

    def set_amplitude(self, _addr: str, _args: Any, amp: float) -> None:
        with self.lock:
            self.amplitude = amp

    def audio_callback(
        self, outdata: np.ndarray, frames: int, _time: Any, _status: sd.CallbackFlags
    ) -> None:
        with self.lock:
            frequency: float = self.frequency
            amplitude: float = self.amplitude
        t: np.ndarray = np.arange(frames)
        increment: float = (2 * np.pi * frequency) / self.sr
        phases: np.ndarray = self.phase + increment * t
        outdata[:] = amplitude * np.sin(phases).reshape(-1, 1)
        self.phase = float((phases[-1] + increment) % (2 * np.pi))


class OSCController:
    def __init__(self, synth: SineOsc, ip: str = "0.0.0.0", port: int = 8000) -> None:
        self.dispatcher: Dispatcher = Dispatcher()
        self.dispatcher.map("/freq", synth.set_frequency, "freq")
        self.dispatcher.map("/amp", synth.set_amplitude, "amp")
        self.server: osc_server.ThreadingOSCUDPServer = (
            osc_server.ThreadingOSCUDPServer((ip, port), self.dispatcher)
        )
        self.thread: threading.Thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )

    def start(self) -> None:
        print(f"OSC server running on {self.server.server_address}")
        self.thread.start()


if __name__ == "__main__":
    sine: SineOsc = SineOsc()
    osc: OSCController = OSCController(sine)
    osc.start()
    with sd.OutputStream(channels=1, callback=sine.audio_callback, samplerate=sine.sr):
        print("Audio stream running. Control via OSC on port 8000 (/freq, /amp)")
        while True:
            sd.sleep(1000)
