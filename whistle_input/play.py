"""
play.py  (whistle_input)
=========================
Whistle controller with a live frequency plot — same arrow-key
logic as audio_sample.py but with a PyQtGraph window so you can
see the pitch history as a moving graph.

Use this version for debugging your whistle gesture, then switch
to audio_sample.py when you just want the key trigger without the GUI.

Controls:
    Close the Qt window  →  stop

Dependencies:
    pip install sounddevice numpy pyqtgraph PyQt6 pynput
"""

import time
import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from collections import deque
from pynput.keyboard import Controller, Key


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RATE         = 44100  # sample rate (Hz)
CHUNK_SIZE   = 2048   # frames per buffer
MAX_HISTORY  = 6      # rolling window size for chirp detection
COOLDOWN     = 0.8    # seconds between key presses
RMS_GATE     = 0.01   # minimum volume threshold (ignore silence)
CHIRP_THRESH = 5      # minimum average Hz/frame change to count as a chirp


keyboard = Controller()


# ─────────────────────────────────────────────
# PITCH DETECTION
# Uses a simple FFT peak — no HPS here because the frequency range
# (80–800 Hz) is narrow enough that the fundamental usually dominates.
# ─────────────────────────────────────────────

def detect_frequency(frame: np.ndarray) -> float | None:
    """
    Find the dominant frequency in an audio frame using FFT peak picking.

    The search is limited to 80–800 Hz to focus on voice / lower whistles
    and ignore background noise above that range.

    Args:
        frame: 1-D float32 audio samples.

    Returns:
        Frequency in Hz, or None if the frame is too quiet.
    """
    rms = np.sqrt(np.mean(frame ** 2))
    if rms < RMS_GATE:
        return None   # too quiet — skip

    # Hanning window to suppress spectral leakage at buffer edges
    windowed = frame * np.hanning(len(frame))
    fft_mag  = np.abs(np.fft.rfft(windowed))
    freqs    = np.fft.rfftfreq(len(frame), d=1.0 / RATE)

    # Restrict to the frequency range we care about
    lo = np.searchsorted(freqs, 80)
    hi = np.searchsorted(freqs, 800)
    fft_mag = fft_mag[lo:hi]
    freqs   = freqs[lo:hi]

    peak = np.argmax(fft_mag)
    return float(freqs[peak])


# ─────────────────────────────────────────────
# CHIRP DETECTION STATE
# Using deque instead of a plain list so appending and the automatic
# length-limiting are both O(1) — much faster than list.pop(0).
# ─────────────────────────────────────────────

freq_history: deque[float] = deque(maxlen=MAX_HISTORY)
last_action_time: float    = 0.0


# ─────────────────────────────────────────────
# KEY TRIGGER HELPERS
# ─────────────────────────────────────────────

def trigger_up() -> None:
    """Press and immediately release the UP arrow key."""
    print("UP ⬆️")
    keyboard.press(Key.up)
    keyboard.release(Key.up)


def trigger_down() -> None:
    """Press and immediately release the DOWN arrow key."""
    print("DOWN ⬇️")
    keyboard.press(Key.down)
    keyboard.release(Key.down)


# ─────────────────────────────────────────────
# CHIRP ANALYSIS
# ─────────────────────────────────────────────

def detect_chirp() -> None:
    """
    Check the recent frequency history for an upward or downward trend.
    Fires the corresponding arrow key if a consistent chirp is detected.

    Uses the average frame-to-frame difference rather than a majority vote
    (simpler and fast enough for this window size).
    """
    global last_action_time

    # Need a full window before we can judge the trend
    if len(freq_history) < MAX_HISTORY:
        return

    # Enforce cooldown so one long gesture doesn't spam keys
    now = time.time()
    if now - last_action_time < COOLDOWN:
        return

    diffs   = np.diff(list(freq_history))
    avg_diff = np.mean(diffs)

    if avg_diff > CHIRP_THRESH:
        trigger_up()
        last_action_time = now
        freq_history.clear()   # fresh start for the next gesture

    elif avg_diff < -CHIRP_THRESH:
        trigger_down()
        last_action_time = now
        freq_history.clear()


# ─────────────────────────────────────────────
# AUDIO CALLBACK
# Runs in a background thread — keep it fast!
# ─────────────────────────────────────────────

current_freq: float | None = None


def audio_callback(indata, frames, time_info, status) -> None:
    """
    Receive each audio buffer, detect pitch, and check for a chirp.

    IMPORTANT: the third parameter is named `time_info` here, NOT `time`.
    Naming it `time` would shadow the `time` module imported at the top,
    which would cause a subtle NameError when `time.time()` is called
    inside `detect_chirp()`.
    """
    global current_freq

    if status:
        print(f"[stream status] {status}")

    freq = detect_frequency(indata[:, 0])
    if freq is None:
        return   # silence — nothing to do

    current_freq = freq
    freq_history.append(freq)
    detect_chirp()


# ─────────────────────────────────────────────
# PYQTGRAPH LIVE PLOT
# Shows the rolling frequency history so you can see your chirp shape.
# This is optional — the controller works fine without the visual.
# ─────────────────────────────────────────────

app = pg.mkQApp("Whistle Input")

win = pg.GraphicsLayoutWidget(title="Whistle Input Monitor")
win.resize(600, 300)

plot = win.addPlot(title="Pitch History (Hz)")
plot.setLabel('left',   'Frequency (Hz)')
plot.setLabel('bottom', 'Recent frames')

curve = plot.plot(pen='c')   # cyan line

win.show()


def update_plot() -> None:
    """Refresh the chart with the latest frequency history."""
    if freq_history:
        curve.setData(list(freq_history))


# Redraw the plot every 50 ms (20 fps — lightweight)
timer = pg.QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(50)


# ─────────────────────────────────────────────
# DEVICE SELECTION & STREAM
# ─────────────────────────────────────────────

def select_input_device() -> int:
    """List all mic devices and return the user's chosen index."""
    print("Available input devices:")
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


def main():
    device = select_input_device()

    stream = sd.InputStream(
        device=device,
        channels=1,
        samplerate=RATE,
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
        latency='low',
    )

    # Keep the stream alive for the entire lifetime of the Qt window
    with stream:
        print("🎤 Whistle control running... (close the window to stop)")
        pg.exec()   # blocks until the window is closed


if __name__ == "__main__":
    main()