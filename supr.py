# pyright: basic

import logging
import time

import supriya

logging.basicConfig(level=logging.DEBUG)

# server = supriya.Server().boot()
#
# sleep(10)
#
# server.quit()


def main() -> None:
    """
    The example entry-point function.
    """
    # opts = supriya.Options(
    #     output_bus_channel_count=2,
    #     output_device="Focusrite Scarlett 2i2 Analog Stereo",
    #     executable="/usr/bin/scsynth",
    # )
    # Create a server and boot it:
    # server = supriya.Server().boot(options=opts)
    opts = supriya.Options(
        output_device="Focusrite Scarlett 2i2 Analog Stereo",
        output_bus_channel_count=2,
        sample_rate=48000,
    )
    server = supriya.Server().boot(options=opts)
    time.sleep(3)
    # Define a C-major chord in Hertz
    frequencies = [261.63, 329.63, 392.00]
    # Create an empty list to store synths in:
    synths: list[supriya.Synth] = []
    # Start an OSC bundle to run immediately:
    with server.at():
        # Add the default synthdef to the server and open a "completion"
        # context manager to group further commands for when the synthdef
        # finishes loading:
        with server.add_synthdefs(supriya.default):
            # Loop over the frequencies:
            for frequency in frequencies:
                # Create a synth using the default synthdef and the frequency
                # and add it to the list of synths:
                synths.append(
                    server.add_synth(synthdef=supriya.default, frequency=frequency)
                )
    # Let the notes play for 4 seconds:
    time.sleep(4)
    # Loop over the synths and free them
    for synth in synths:
        synth.free()
    # Wait a second for the notes to fade out:
    time.sleep(1)
    # Quit the server:
    server.quit()


if __name__ == "__main__":
    main()


# import librosa
# from pythonosc import udp_client
#
# # load your file
# y, sr = librosa.load("/home/kureta/Music/titanium-gong.wav", mono=True)
# # make sure it’s a flat Python list of floats
# audio_data = y.astype(float).tolist()
#
# # set up an OSC client pointed at SC’s default port
# client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
#
# buffer_id = 0
# length_frames = len(audio_data)
#
# # tell SC to allocate a buffer of the right size
# client.send_message("/pythonBuffer/alloc", [buffer_id, length_frames])
#
# # send in chunks of up to 1024 samples (you can tune this)
# chunk_size = 2048
# for offset in range(0, length_frames, chunk_size):
#     chunk = audio_data[offset : offset + chunk_size]
#     # message: /pythonBuffer/setn bufferID offset samples...
#     client.send_message("/pythonBuffer/setn", [buffer_id, offset] + chunk)
#
