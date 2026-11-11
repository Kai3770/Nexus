"""Procedural audio synthesis and the rendered end-to-end smoke test."""
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

from helpers import ROOT

from engine import audio
from engine import weapons as W


def test_every_sound_is_synthesised_as_valid_pcm():
    folder = tempfile.mkdtemp()
    paths = audio.ensure_sound_files(folder)
    assert len(paths) == len(audio.SOUND_RECIPES) >= 25
    assert all(("shot_%d" % i) in paths for i in range(W.WEAPON_COUNT))
    for name, path in paths.items():
        with wave.open(str(path)) as handle:
            assert handle.getnchannels() == 1 and handle.getsampwidth() == 2 and handle.getframerate() == audio.SAMPLE_RATE
            samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        assert 0.03 < samples.size / audio.SAMPLE_RATE < 2.0, name
        assert 2000 < np.abs(samples).max() <= 32767, name
        assert abs(float(samples.mean())) < 600.0, name
    stamp = os.path.getmtime(paths["explosion"])
    audio.ensure_sound_files(folder)                                             # cached: nothing is rewritten
    assert os.path.getmtime(paths["explosion"]) == stamp


def test_shotgun_is_heavier_than_the_smg():
    def low_share(signal):
        spectrum = np.abs(np.fft.rfft(signal)) ** 2
        freq = np.fft.rfftfreq(signal.size, 1.0 / audio.SAMPLE_RATE)
        return spectrum[freq < 250.0].sum() / spectrum.sum()
    assert low_share(audio.SOUND_RECIPES["shot_2"]()) > low_share(audio.SOUND_RECIPES["shot_1"]())


def _display_command():
    """Command prefix that provides a display, or None when rendering is impossible here."""
    if os.environ.get("DISPLAY") or sys.platform in ("win32", "darwin"):
        return []
    if shutil.which("xvfb-run"):
        return ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24"]
    return None


def test_rendered_autoplay_match():
    """Runs the real client (window, shaders, HUD, listen server, bots) for 240 frames with the autopilot."""
    import importlib.util
    prefix = _display_command()
    if prefix is None or importlib.util.find_spec("ursina") is None:
        return "skipped"
    shots = tempfile.mkdtemp()
    saves = tempfile.mkdtemp()
    command = prefix + [sys.executable, os.path.join(ROOT, "main.py"), "--solo", "crucible", "--mode", "ffa", "--bots", "4",
                        "--autoplay", "--frames", "240", "--screenshots", shots, "--shot-every", "120",
                        "--size", "960x540", "--no-vsync", "--save-dir", saves]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240, cwd=ROOT)
    output = result.stdout + result.stderr
    assert "Traceback" not in output, output[-3000:]
    assert ":shader(error)" not in output and "GLSL" not in output.replace("supports glsl", ""), output[-3000:]
    images = sorted(os.listdir(shots))
    assert images == ["frame_00120.png", "frame_00240.png"], images
    assert all(os.path.getsize(os.path.join(shots, name)) > 20000 for name in images)   # not a blank frame
    assert os.path.exists(os.path.join(saves, "profile.sav"))                          # profile saved on exit
