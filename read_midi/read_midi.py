"""
read_midi.py
============
Reads a MIDI file and prints every message as it plays back in real time.

Usage:
    python read_midi.py                   → plays berge.mid (default)
    python read_midi.py freude.mid        → plays the given file

Dependencies:
    pip install mido
"""

import sys
from mido import MidiFile


def play_midi(filename: str) -> None:
    """
    Open a MIDI file and print each message to the console
    with the correct real-time delay between events.

    Args:
        filename: Path to the .mid file to play back.
    """
    print(f"Playing: {filename}\n")

    # MidiFile.play() handles the timing for us — it yields
    # each message at the right moment, so we just print it.
    try:
        midi = MidiFile(filename)
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error opening MIDI file: {e}")
        sys.exit(1)

    try:
        for msg in midi.play():
            # Skip meta messages (tempo changes, track names, etc.)
            # so the output stays readable
            if not msg.is_meta:
                print(msg)
    except KeyboardInterrupt:
        # Ctrl+C — user stopped playback early, that's fine
        print("\n\nPlayback stopped by user.")


if __name__ == "__main__":
    # Allow the user to pass a filename on the command line,
    # otherwise fall back to the default sample file.
    midi_file = sys.argv[1] if len(sys.argv) > 1 else "berge.mid"
    play_midi(midi_file)
