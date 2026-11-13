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


def test_mixer_ceilings_master_clamp_and_inverse_square_falloff():
    from engine import config as C
    assert all(0.0 < ceiling <= 0.3 for ceiling in C.AUDIO_CEILING.values())
    assert C.AUDIO_CEILING["weapon"] <= 0.3 and C.AUDIO_CEILING["explosion"] <= 0.3 and C.AUDIO_CEILING["impact"] <= 0.3
    for name in list(audio.SOUND_RECIPES) + ["shot_3"]:
        assert audio.category_of(name) in C.AUDIO_CEILING, name
        assert audio.mix_gain(name, volume=5.0, master=9.0) <= 0.3 + 1e-12            # nothing can exceed the ceiling
    assert audio.mix_gain("shot_0", 1.0, 1.0) == C.AUDIO_CEILING["weapon"]
    assert audio.mix_gain("shot_0", 1.0, 0.5) == C.AUDIO_CEILING["weapon"] * 0.5      # master scales linearly
    assert audio.mix_gain("shot_0", 1.0, -3.0) == 0.0
    reference = C.AUDIO_REFERENCE_DISTANCE["weapon"]
    near = audio.mix_gain("shot_0", 1.0, 1.0, distance=reference * 0.3)
    at_ref = audio.mix_gain("shot_0", 1.0, 1.0, distance=reference)
    assert near == at_ref == C.AUDIO_CEILING["weapon"]                                # no boost inside the reference
    for factor in (2.0, 3.0, 5.0):
        far = audio.mix_gain("shot_0", 1.0, 1.0, distance=reference * factor)
        assert np.isclose(far, at_ref / factor ** 2)                                  # Volume = Base / Distance^2
    assert audio.mix_gain("shot_0", 1.0, 1.0, overlapping=2) == at_ref * C.AUDIO_DUCK_FACTOR ** 2


def _display_command():
    """Command prefix that provides a display, or None when rendering is impossible here."""
    if os.environ.get("DISPLAY") or sys.platform in ("win32", "darwin"):
        return []
    if shutil.which("xvfb-run"):
        return ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24"]
    return None


def _run(command, timeout=300):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    output = result.stdout + result.stderr
    assert "Traceback" not in output, output[-3000:]
    assert ":shader(error)" not in output and "GLSL" not in output.replace("supports glsl", ""), output[-3000:]
    return output


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
                        "--size", "960x540", "--no-vsync", "--quality", "1", "--fps-limit", "0", "--save-dir", saves]
    output = _run(command)
    images = sorted(os.listdir(shots))
    assert images == ["frame_00120.png", "frame_00240.png"], images
    assert all(os.path.getsize(os.path.join(shots, name)) > 20000 for name in images)   # not a blank frame
    assert os.path.exists(os.path.join(saves, "profile.sav"))                          # profile saved on exit
    summary = [line for line in output.splitlines() if line.startswith("frames 240")]
    assert summary and "quality" in summary[0] and "render scale" in summary[0], output[-2000:]


def test_render_smoke_exercises_every_renderer_feature():
    """Textures, shadow maps, sky dome, skyline, bloom, effect pools and the batched HUD on a new arena."""
    import importlib.util
    prefix = _display_command()
    if prefix is None or importlib.util.find_spec("ursina") is None:
        return "skipped"
    shots = tempfile.mkdtemp()
    output = _run(prefix + [sys.executable, os.path.join(ROOT, "dev", "render_smoke.py"), shots, "terminal", "2"])
    final = [line for line in output.splitlines() if line.startswith("final ")]
    assert final, output[-2000:]
    for expected in ("'shadows': True", "'bloom': True", "'post': True", "'quality': 'HIGH'", "'shadow_size': 2048"):
        assert expected in final[0], final[0]
    images = sorted(os.listdir(shots))
    assert images == ["terminal_%s.png" % name for name in ("09", "21", "27", "40", "board", "overview")], images
    assert all(os.path.getsize(os.path.join(shots, name)) > 100000 for name in images)  # textured frames compress badly
