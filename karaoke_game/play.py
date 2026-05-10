"""
play.py  (karaoke_game)
========================
Karaoke game — sing the notes of "Fly Me to the Moon" into
your microphone and earn points for hitting each pitch!

How it works:
  1. The song list defines each note as (frequency, duration, lyric).
  2. A pyglet clock schedules `next_note()` to advance the song.
  3. sounddevice captures the mic and runs HPS pitch detection.
  4. Every time a note elapses, we check how close your sung
     frequency was to the target and color the display green / red.

Controls:
    Close the window  →  quit

Dependencies:
    pip install numpy sounddevice pyglet mido
"""

import math
import sys
import numpy as np
import sounddevice as sd
import pyglet
from mido import MidiFile   # not used for playback here, but kept for future use


# ─────────────────────────────────────────────
# AUDIO SETTINGS
# ─────────────────────────────────────────────
RATE       = 44100   # sample rate (Hz)
CHUNK_SIZE = 2048    # frames per buffer — larger = more stable FFT, slight latency


# ─────────────────────────────────────────────
# STEP 1: PITCH DETECTION WITH HPS
# Harmonic Product Spectrum (HPS) is a robust way to find the
# fundamental frequency even when harmonics are louder than the root.
# ─────────────────────────────────────────────

def detect_frequency(frame: np.ndarray) -> float | None:
    """
    Find the dominant pitch in an audio frame using HPS.

    Args:
        frame: 1-D numpy array of audio samples (float32, -1..1).

    Returns:
        Detected frequency in Hz, or None if the input is too quiet.
    """
    # Gate: ignore near-silent frames to avoid spurious detections
    rms = np.sqrt(np.mean(frame ** 2))
    if rms < 0.01:
        return None

    # Apply a Hanning window to reduce spectral leakage at the edges
    windowed = frame * np.hanning(len(frame))

    # Real FFT — we only need the positive half of the spectrum
    fft_mag = np.abs(np.fft.rfft(windowed))
    freqs   = np.fft.rfftfreq(len(frame), d=1.0 / RATE)

    # Narrow the search to typical singing range (80–600 Hz)
    # This prevents the algorithm from locking onto background noise
    lo = np.searchsorted(freqs, 80)
    hi = np.searchsorted(freqs, 600)
    fft_mag = fft_mag[lo:hi]
    freqs   = freqs[lo:hi]

    # HPS: multiply the spectrum by downsampled copies of itself.
    # Harmonics reinforce at the fundamental — so the peak sharpens there.
    hps = fft_mag.copy()
    for h in range(2, 5):
        downsampled = fft_mag[::h]
        hps[:len(downsampled)] *= downsampled

    peak_idx = np.argmax(hps)
    return float(freqs[peak_idx])


def note_name(freq: float) -> str:
    """Convert a frequency (Hz) to a musical note name like 'A' or 'C#'."""
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi_num = round(12 * math.log2(freq / 440) + 69)
    return NOTE_NAMES[midi_num % 12]


def cents_diff(f1: float | None, f2: float | None) -> float:
    """
    Return the pitch distance between two frequencies in cents,
    also checking one octave up/down so octave errors don't penalize the player.

    100 cents = 1 semitone.  A difference < 80 cents is considered a "hit".
    Returns 999 if either frequency is None (silence).
    """
    if f1 is None or f2 is None:
        return 999

    # Check the exact pitch plus one octave up and one octave down
    diffs = [
        abs(1200 * math.log2(f1       / f2)),
        abs(1200 * math.log2(f1 * 2.0 / f2)),
        abs(1200 * math.log2(f1 / 2.0 / f2)),
    ]
    return min(diffs)


# ─────────────────────────────────────────────
# STEP 2: SONG DATA
# Each entry is (target_freq_Hz, duration_seconds, lyric_word).
# Frequencies taken from the "Fly Me to the Moon" melody.
# ─────────────────────────────────────────────
SONG = [
    (164.8, 0.34, "Fly"),
    (261.6, 0.34, "me"),
    (196.0, 0.34, "to"),
    (523.3, 0.40, "the"),
    (164.8, 0.34, "moon"),
    (196.0, 0.34, "let"),
    (261.6, 0.34, "me"),
    (493.9, 0.40, "play"),
    (261.6, 0.34, "a-"),
    (196.0, 0.34, "mong"),
    (164.8, 0.34, "the"),
    (440.0, 0.40, "stars"),
    (164.8, 0.34, "let"),
    (261.6, 0.34, "me"),
    (196.0, 0.34, "see"),
    (392.0, 0.40, "what"),
    (261.6, 0.34, "spring"),
    (174.6, 0.34, "is"),
    (349.2, 1.00, "like"),
    (174.6, 0.34, "on"),
    (261.6, 0.34, "Ju-"),
    (174.6, 0.34, "pi-"),
    (261.6, 0.34, "ter"),
    (523.3, 0.40, "and"),
    (261.6, 0.34, "Mars"),
]


# ─────────────────────────────────────────────
# STEP 3: MIC DEVICE SELECTION
# ─────────────────────────────────────────────

def select_input_device() -> int:
    """List all input devices and return the one the user chose."""
    print("Available input devices:\n")
    valid_ids = []
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            print(f"  {i}: {dev['name']}")
            valid_ids.append(i)

    while True:
        try:
            choice = int(input("\nSelect input device: "))
            if choice in valid_ids:
                return choice
            print(f"  ✗ Please choose from: {valid_ids}")
        except ValueError:
            print("  ✗ Please enter a number.")


# ─────────────────────────────────────────────
# STEP 4: PYGLET WINDOW & LABELS
# All text labels are created up-front; we just update .text / .color.
# ─────────────────────────────────────────────

window = pyglet.window.Window(600, 400, "🎤 Fly Me To The Moon")

# Big note name displayed in the center
label_note = pyglet.text.Label(
    "", font_name="Courier New", font_size=48,
    x=300, y=230, anchor_x="center", anchor_y="center",
    color=(255, 220, 50, 255),   # gold
)

# Current lyric word just below the note
label_lyric = pyglet.text.Label(
    "", font_name="Courier New", font_size=24,
    x=300, y=160, anchor_x="center", anchor_y="center",
    color=(200, 200, 255, 255),  # soft lavender
)

# "X / N" progress counter near the bottom
label_index = pyglet.text.Label(
    "", font_name="Courier New", font_size=14,
    x=300, y=50, anchor_x="center", anchor_y="center",
    color=(150, 150, 150, 255),  # grey
)

# Score in the top-left corner
label_score = pyglet.text.Label(
    "SCORE: 0", font_name="Courier New", font_size=20,
    x=10, y=390, anchor_x="left", anchor_y="top",
    color=(255, 200, 40, 255),
)

# Shows what note the player is currently singing
label_sung = pyglet.text.Label(
    "", font_name="Courier New", font_size=20,
    x=300, y=100, anchor_x="center", anchor_y="center",
    color=(100, 255, 200, 255),  # mint green
)


# ─────────────────────────────────────────────
# STEP 5: MICROPHONE STATE
# `current_freq` is written by the audio thread and read by the
# pyglet main thread — a simple float is safe enough here.
# ─────────────────────────────────────────────

current_freq: float | None = None


def audio_callback(indata, frames, time_info, status):
    """
    sounddevice calls this on every audio buffer (runs in a background thread).
    We just detect the pitch and store it — no heavy work here.
    """
    global current_freq
    if status:
        print(f"[audio] {status}")
    current_freq = detect_frequency(indata[:, 0])


# ─────────────────────────────────────────────
# STEP 6: SONG PROGRESSION & SCORING
# ─────────────────────────────────────────────

score               = 0
current_index       = 0
current_target_freq: float | None = None


def next_note(dt: float) -> None:
    """
    Called by the pyglet clock after each note's duration expires.
    Checks if the player hit the last note, then advances to the next one.

    Args:
        dt: Time elapsed since the last call (provided by pyglet clock).
    """
    global current_index, score, current_target_freq

    # ── Score the previous note ──────────────────────────────────────
    # We use `is not None` instead of a truthiness check so that a
    # frequency of 0.0 (extremely unlikely but possible) isn't skipped.
    if current_target_freq is not None and current_freq is not None:
        diff = cents_diff(current_freq, current_target_freq)
        if diff < 80:
            # Close enough — award points and flash green
            score            += 10
            label_score.text  = f"SCORE: {score}"
            label_note.color  = (50, 255, 100, 255)   # bright green = hit ✓
        else:
            # Too far off — flash red
            label_note.color  = (255, 60, 60, 255)    # red = miss ✗

    # ── Advance to the next note ─────────────────────────────────────
    if current_index < len(SONG):
        freq, dur, lyric    = SONG[current_index]
        current_target_freq = freq
        label_note.text     = note_name(freq)
        label_note.color    = (255, 220, 50, 255)      # reset to gold for new note
        label_lyric.text    = f'"{lyric}"'
        label_index.text    = f"{current_index + 1} / {len(SONG)}"
        current_index      += 1

        # Schedule ourselves again after the note's duration
        pyglet.clock.schedule_once(next_note, dur)
    else:
        # All notes done — show final result
        label_note.text  = "DONE! 🎉"
        label_note.color = (255, 220, 50, 255)
        label_lyric.text = f"Final score: {score} / {len(SONG) * 10}"
        label_index.text = ""


@window.event
def on_draw():
    """Pyglet calls this every frame to redraw the window."""
    window.clear()
    label_note.draw()
    label_lyric.draw()
    label_index.draw()
    label_score.draw()

    # Show the note the player is currently singing (or "..." for silence)
    # `is not None` is correct here — `if current_freq` would wrongly
    # treat 0.0 Hz as silence, causing the label to disappear unexpectedly.
    if current_freq is not None:
        label_sung.text = f"You: {note_name(current_freq)}  ({current_freq:.0f} Hz)"
    else:
        label_sung.text = "You: ..."
    label_sung.draw()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    input_device = select_input_device()

    # Open the mic stream — use a context manager so it's always closed
    # properly when the window is shut (even on exceptions).
    stream = sd.InputStream(
        device=input_device,
        channels=1,
        samplerate=RATE,
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
        latency='low',
    )

    with stream:
        # Give the player one second to get ready before the first note
        pyglet.clock.schedule_once(next_note, 1.0)
        pyglet.app.run()


if __name__ == "__main__":
    main()