"""Regression tests for defects found in the full code review (fuzzing, strict-numerics soak, reviewers)."""
import numpy as np

import helpers  # noqa: F401

from engine import config as C
from engine import simulation as S


def test_an_idle_neutral_uplink_drains_exactly_to_zero():
    # The drain used to multiply after clipping, so a neutral zone's charge halved every tick
    # and never reached zero (it ended in floating point underflow).
    game = S.GameWorld("crucible", mode=C.MODE_UPLINK, seed=3)
    for team in (0, 1):
        row = game.add_player("p%d" % team, team=team)
        game.store.move.pos[row] = (0.0, -500.0, 0.0)
    game.uplink_owner[:] = -1
    game.uplink_charge[:] = (0.004, -0.004, 0.3)
    with np.errstate(all="raise"):
        for _ in range(C.TICK_RATE):
            game.end_tick()
            game.store.move.pos[:2] = (0.0, -500.0, 0.0)
            game.store.move.vel[:2] = 0.0
    assert game.uplink_charge[0] == 0.0 and game.uplink_charge[1] == 0.0
    assert np.isclose(game.uplink_charge[2], 0.3 - 0.5 / C.UPLINK_CAPTURE_TIME, atol=1.0e-9)
