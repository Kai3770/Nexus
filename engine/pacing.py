"""
NEXUS-3D-EXTREME :: engine/pacing.py
====================================
Frame pacing logic that needs no window: the dynamic resolution controller
used by ``engine/renderer.py`` and the preset fallback used by the client.

The simulation itself never depends on the frame rate: it advances in fixed
60 Hz ticks (client prediction and server rollback require identical steps on
both sides) and the renderer interpolates between ticks with the frame's real
delta time.  What *does* depend on the machine is how long a frame takes to
draw, and that is what this module regulates.
"""
from __future__ import annotations

from . import config as C


class ResolutionController:
    """
    Dynamic resolution: picks the 3D render scale from the measured frame time
    so that the frame rate holds ``TARGET_FPS``.  The scale drops one step as
    soon as the smoothed frame time misses the budget; it climbs back when
    there is clear headroom, or -- when a frame limiter or vsync hides the
    headroom -- by probing one step up after a quiet period that doubles each
    time the probe fails.
    """

    def __init__(self, ceiling: float, enabled: bool = True, fps_limit: float = 0.0):
        self.configure(ceiling, enabled, fps_limit)

    def configure(self, ceiling: float, enabled: bool, fps_limit: float) -> None:
        self.enabled = bool(enabled)
        self.ceiling = float(ceiling)
        steps = [s for s in C.DYNRES_STEPS if s < ceiling - 1.0e-6] + [float(ceiling)]
        self.steps = steps if self.enabled else [float(ceiling)]
        self.index = len(self.steps) - 1
        target = 1000.0 / C.TARGET_FPS
        cap = 1000.0 / fps_limit if fps_limit and fps_limit > 0 else 0.0
        self.drop_ms = max(C.DYNRES_DROP_MS, cap * 1.06)
        self.raise_ms = max(C.DYNRES_RAISE_MS, cap * 1.02) if cap < target * 0.72 else 0.0
        self.average_ms = target * 0.8
        self.cooldown = C.DYNRES_COOLDOWN
        self.quiet = 0.0
        self.probe_wait = 6.0
        self.probing = False
        self.changes = 0

    @property
    def scale(self) -> float:
        return self.steps[self.index]

    def settle(self) -> None:
        """Ignore the hitch of a level load: restart the averaging and hold the scale for a moment."""
        self.average_ms = min(self.average_ms, 1000.0 / C.TARGET_FPS)
        self.cooldown = C.DYNRES_COOLDOWN
        self.quiet = 0.0

    def update(self, frame_seconds: float) -> float:
        """Feed one frame time; returns the render scale to use from now on."""
        frame_seconds = min(max(frame_seconds, 1.0e-4), 0.05)
        self.average_ms += (frame_seconds * 1000.0 - self.average_ms) * 0.08
        if not self.enabled or len(self.steps) < 2:
            return self.scale
        self.cooldown -= frame_seconds
        if self.cooldown > 0.0:
            return self.scale
        if self.average_ms > self.drop_ms:
            if self.index > 0:
                self.index -= 1
                self.changes += 1
                self.cooldown = C.DYNRES_COOLDOWN
                if self.probing:
                    self.probe_wait = min(self.probe_wait * 2.0, 60.0)
            self.probing = False
            self.quiet = 0.0
            return self.scale
        self.quiet += frame_seconds
        if self.index < len(self.steps) - 1:
            headroom = self.raise_ms > 0.0 and self.average_ms < self.raise_ms
            if headroom or self.quiet >= self.probe_wait:
                self.index += 1
                self.changes += 1
                self.cooldown = C.DYNRES_COOLDOWN
                self.probing = not headroom
                self.quiet = 0.0
        elif self.probing and self.quiet >= self.probe_wait:
            self.probing = False
            self.probe_wait = 6.0
        return self.scale


class QualityGovernor:
    """
    Last line of defence for the frame rate: when the dynamic resolution is
    already at its lowest step and frames still miss the budget for
    ``QUALITY_FALLBACK_SECONDS``, ask for the next lower graphics preset.
    """

    def __init__(self):
        self.slow = 0.0

    def update(self, controller: ResolutionController, quality: int, frame_seconds: float) -> bool:
        """Returns True when the preset should be lowered by one step now."""
        if not controller.enabled or quality <= C.QUALITY_LOW:
            self.slow = 0.0
            return False
        frame_seconds = min(max(frame_seconds, 0.0), 0.05)
        if controller.index == 0 and controller.average_ms > controller.drop_ms:
            self.slow += frame_seconds
        else:
            self.slow = max(self.slow - frame_seconds * 0.5, 0.0)
        if self.slow >= C.QUALITY_FALLBACK_SECONDS:
            self.slow = 0.0
            return True
        return False
