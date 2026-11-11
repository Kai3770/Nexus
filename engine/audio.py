"""
NEXUS-3D-EXTREME :: engine/audio.py
===================================
Procedural sound effects.

The game ships no audio assets: every effect is synthesised with NumPy the
first time the game starts (noise bursts shaped by envelopes and FFT filters,
pitch-swept sine bodies, simple chimes), written as 16-bit mono WAV files into
the cache directory and then played through Panda3D's audio manager.

The synthesis half is pure NumPy and is unit tested headlessly.  Playback is
optional: on a machine without an audio device Panda3D hands out null sounds
and every ``play`` call silently does nothing.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from . import config as C
from . import weapons as W

SAMPLE_RATE = 44100
AUDIO_VERSION = "3"
VOICES_PER_SOUND = 4


# --------------------------------------------------------------------------- #
#  Synthesis primitives (all vectorised)
# --------------------------------------------------------------------------- #
def _time(duration: float) -> np.ndarray:
    return np.arange(int(duration * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE


def _envelope(t: np.ndarray, attack: float, decay: float) -> np.ndarray:
    """Fast exponential attack followed by an exponential decay."""
    return (1.0 - np.exp(-t / max(attack, 1.0e-5))) * np.exp(-t / max(decay, 1.0e-5))


def _filter(signal: np.ndarray, low: float = 0.0, high: float = 0.0) -> np.ndarray:
    """Zero-phase FFT filter: 4th order style high-pass at ``low`` and low-pass at ``high`` (Hz)."""
    spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, 1.0 / SAMPLE_RATE)
    gain = np.ones_like(freq)
    if high > 0.0:
        gain /= np.sqrt(1.0 + (freq / high) ** 8)
    if low > 0.0:
        ratio = np.divide(low, freq, out=np.full_like(freq, 1.0e6), where=freq > 0.0)
        gain /= np.sqrt(1.0 + ratio ** 8)
    return np.fft.irfft(spectrum * gain, n=signal.size)


def _sweep(t: np.ndarray, f_start: float, f_end: float, curve: float = 30.0) -> np.ndarray:
    """Sine whose frequency glides exponentially from ``f_start`` to ``f_end``."""
    freq = f_end + (f_start - f_end) * np.exp(-t * curve)
    return np.sin(2.0 * np.pi * np.cumsum(freq) / SAMPLE_RATE)


def _normalise(signal: np.ndarray, peak: float = 0.9) -> np.ndarray:
    top = float(np.max(np.abs(signal)))
    return signal * (peak / top) if top > 1.0e-9 else signal


def gunshot(duration: float, body: float, crack: float, tail: float, weight: float, seed: int) -> np.ndarray:
    """
    Layered gunshot: a high-passed noise crack, a pitch-dropping low body and a
    longer band-limited noise tail.  ``weight`` (0..1) shifts energy to the body.
    """
    rng = np.random.default_rng(seed)
    t = _time(duration)
    noise = rng.uniform(-1.0, 1.0, t.size)
    snap = _filter(noise, low=crack) * _envelope(t, 0.0004, 0.012 + 0.02 * weight)
    thump = _sweep(t, body * 3.2, body, 55.0) * _envelope(t, 0.001, 0.05 + 0.10 * weight)
    echo = _filter(rng.uniform(-1.0, 1.0, t.size), low=180.0, high=2400.0 - 1200.0 * weight) * _envelope(t, 0.01, tail)
    return _normalise(snap * (1.0 - 0.4 * weight) + thump * (0.55 + 0.6 * weight) + echo * 0.45)


def click(duration: float, pitch: float, seed: int, tone: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _time(duration)
    body = _filter(rng.uniform(-1.0, 1.0, t.size), low=pitch * 0.5, high=pitch * 3.0) * _envelope(t, 0.0003, duration * 0.18)
    return _normalise(body + np.sin(2.0 * np.pi * pitch * t) * _envelope(t, 0.0005, duration * 0.3) * tone, 0.7)


def chime(notes, note_length: float, decay: float, harmonic: float = 0.35) -> np.ndarray:
    """Sequence of bell-like tones (used for hit confirmation, purchases, level ups)."""
    parts = []
    for frequency in notes:
        t = _time(note_length)
        tone = np.sin(2.0 * np.pi * frequency * t) + harmonic * np.sin(2.0 * np.pi * frequency * 2.01 * t)
        parts.append(tone * _envelope(t, 0.002, decay))
    step = int(note_length * SAMPLE_RATE * 0.55)
    out = np.zeros(step * (len(parts) - 1) + parts[-1].size)
    for index, part in enumerate(parts):
        out[index * step: index * step + part.size] += part
    return _normalise(out, 0.75)


def thud(duration: float, pitch: float, seed: int, air: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _time(duration)
    body = _sweep(t, pitch * 2.2, pitch, 40.0) * _envelope(t, 0.002, duration * 0.3)
    dust = _filter(rng.uniform(-1.0, 1.0, t.size), low=200.0, high=1800.0) * _envelope(t, 0.004, duration * 0.2)
    return _normalise(body + dust * air, 0.8)


def whoosh(duration: float, low: float, high: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _time(duration)
    swell = np.sin(np.pi * np.clip(t / duration, 0.0, 1.0)) ** 2
    return _normalise(_filter(rng.uniform(-1.0, 1.0, t.size), low=low, high=high) * swell, 0.6)


def explosion(duration: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _time(duration)
    boom = _sweep(t, 150.0, 38.0, 9.0) * _envelope(t, 0.002, 0.42)
    blast = _filter(rng.uniform(-1.0, 1.0, t.size), high=900.0) * _envelope(t, 0.001, 0.30)
    debris = _filter(rng.uniform(-1.0, 1.0, t.size), low=900.0, high=5000.0) * _envelope(t, 0.02, 0.55) * 0.25
    return _normalise(np.tanh((boom * 1.2 + blast + debris) * 1.6), 0.95)


def buzz(duration: float, pitch: float) -> np.ndarray:
    t = _time(duration)
    return _normalise(np.sign(np.sin(2.0 * np.pi * pitch * t)) * _envelope(t, 0.002, duration * 0.4), 0.45)


def rail_shot() -> np.ndarray:
    """Aurora Rail: a heavy body with a descending capacitor whine on top."""
    base = gunshot(0.9, 55.0, 3600.0, 0.28, 0.9, 107)
    t = _time(0.9)
    whine = _sweep(t, 5200.0, 600.0, 7.0) * _envelope(t, 0.002, 0.22) * 0.35
    return _normalise(base + whine)


# name -> zero argument generator.  Weapon shots are indexed by weapon id.
SOUND_RECIPES = {
    "shot_0": lambda: gunshot(0.45, 95.0, 2200.0, 0.11, 0.45, 101),      # VK-9 Vanguard
    "shot_1": lambda: gunshot(0.32, 130.0, 3000.0, 0.07, 0.25, 102),     # HX-4 Hornet
    "shot_2": lambda: gunshot(0.75, 62.0, 1400.0, 0.20, 0.95, 103),      # Breacher-12
    "shot_3": lambda: gunshot(0.65, 80.0, 2600.0, 0.18, 0.70, 104),      # Longbow DMR
    "shot_4": lambda: gunshot(0.34, 150.0, 2800.0, 0.08, 0.35, 105),     # P9 Sidekick
    "shot_5": lambda: gunshot(0.50, 78.0, 1800.0, 0.13, 0.65, 106),      # Bulwark LMG
    "shot_6": lambda: rail_shot(),                                           # Aurora Rail
    "hit": lambda: click(0.07, 2400.0, 201, 0.9),
    "headshot": lambda: chime((1568.0, 2349.0), 0.16, 0.06),
    "kill": lambda: chime((784.0, 1175.0, 1568.0), 0.20, 0.09),
    "hurt": lambda: thud(0.16, 180.0, 202, 0.8),
    "reload": lambda: np.concatenate([click(0.10, 1300.0, 203), np.zeros(int(0.16 * SAMPLE_RATE)), click(0.12, 900.0, 204)]),
    "empty": lambda: click(0.06, 3200.0, 205, 0.1),
    "swap": lambda: whoosh(0.18, 500.0, 4000.0, 206),
    "jump": lambda: whoosh(0.16, 250.0, 1600.0, 207),
    "land": lambda: thud(0.22, 85.0, 208, 0.6),
    "slide": lambda: whoosh(0.55, 700.0, 5200.0, 209),
    "step": lambda: thud(0.09, 120.0, 210, 0.9),
    "throw": lambda: whoosh(0.26, 300.0, 2600.0, 211),
    "explosion": lambda: explosion(1.5, 212),
    "respawn": lambda: chime((523.0, 784.0), 0.22, 0.10),
    "ui_click": lambda: click(0.05, 1800.0, 213, 0.8),
    "ui_buy": lambda: chime((880.0, 1320.0), 0.14, 0.07),
    "ui_error": lambda: buzz(0.18, 140.0),
    "level_up": lambda: chime((523.0, 659.0, 784.0, 1047.0), 0.22, 0.12),
    "victory": lambda: chime((523.0, 659.0, 784.0, 1047.0, 1319.0), 0.30, 0.18),
    "defeat": lambda: chime((392.0, 330.0, 262.0), 0.34, 0.20),
}


assert all(("shot_%d" % index) in SOUND_RECIPES for index in range(W.WEAPON_COUNT))


def write_wav(path: Path, signal: np.ndarray) -> None:
    """Store a float signal in [-1, 1] as 16-bit mono PCM."""
    pcm = (np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def ensure_sound_files(directory: Path | None = None, force: bool = False) -> dict:
    """
    Synthesise every effect that is missing from the cache directory.
    Returns ``{name: Path}``.  A version stamp invalidates old caches.
    """
    directory = Path(directory) if directory is not None else C.SFX_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = directory / "version.txt"
    current = stamp.read_text().strip() if stamp.exists() else ""
    rebuild = force or current != AUDIO_VERSION
    paths = {}
    for name, recipe in SOUND_RECIPES.items():
        path = directory / (name + ".wav")
        if rebuild or not path.exists():
            write_wav(path, recipe())
        paths[name] = path
    stamp.write_text(AUDIO_VERSION)
    return paths


class SoundBank:
    """
    Voice pool on top of Panda3D's audio manager (several voices per effect so
    automatic fire overlaps cleanly).  Construct it only after the Ursina
    application exists.
    """

    def __init__(self, volume: float = 0.8, directory: Path | None = None):
        self.volume = float(volume)
        self.voices: dict = {}
        self.cursor: dict = {}
        self.enabled = False
        self.rng = np.random.default_rng(7)
        try:
            from panda3d.core import Filename
            from ursina.audio import _audio_manager
            paths = ensure_sound_files(directory)
            for name, path in paths.items():
                pool = [_audio_manager.getSound(Filename.fromOsSpecific(str(path.resolve()))) for _ in range(VOICES_PER_SOUND)]
                self.voices[name] = pool
                self.cursor[name] = 0
            self.enabled = True
        except Exception as error:                 # audio must never take the game down
            print("audio disabled:", error)

    def set_volume(self, volume: float) -> None:
        self.volume = float(min(max(volume, 0.0), 1.0))

    def play(self, name: str, volume: float = 1.0, pitch: float = 1.0, balance: float = 0.0, vary: float = 0.0) -> None:
        """Play an effect on its next free voice.  ``vary`` adds random pitch variation."""
        pool = self.voices.get(name)
        if not self.enabled or pool is None or self.volume <= 0.0:
            return
        index = self.cursor[name]
        self.cursor[name] = (index + 1) % len(pool)
        sound = pool[index]
        if vary > 0.0:
            pitch *= 1.0 + float(self.rng.uniform(-vary, vary))
        try:
            sound.setVolume(float(min(max(volume, 0.0), 1.0)) * self.volume)
            sound.setPlayRate(float(pitch))
            sound.setBalance(float(min(max(balance, -1.0), 1.0)))
            sound.play()
        except Exception:
            self.enabled = False

    def play_at(self, name: str, position, listener, listener_right, volume: float = 1.0, reference: float = 9.0,
                vary: float = 0.0) -> None:
        """Positional playback: inverse-square style distance roll-off and left / right panning."""
        offset = np.asarray(position, dtype=np.float64) - np.asarray(listener, dtype=np.float64)
        distance = float(np.sqrt(np.dot(offset, offset)))
        gain = volume / (1.0 + (distance / reference) ** 2)
        if gain < 0.01:
            return
        pan = float(np.dot(offset, listener_right)) / max(distance, 1.0e-6)
        self.play(name, gain, 1.0, pan * 0.8, vary)

    def shot(self, weapon: int, volume: float = 1.0) -> None:
        self.play("shot_%d" % (int(weapon) % W.WEAPON_COUNT), volume, vary=0.035)

    def shot_at(self, weapon: int, position, listener, listener_right) -> None:
        self.play_at("shot_%d" % (int(weapon) % W.WEAPON_COUNT), position, listener, listener_right, 0.9, 14.0, 0.035)
