"""
NEXUS-3D-EXTREME :: engine/streaks.py
=====================================
Scorestreak rewards as structure-of-arrays tables built from
``config.STREAK_TABLE``, plus the small pure helpers the simulation, the bots
and the HUD share: which reward a streak earns, which ready reward ``X`` uses
next, and the lance's damage falloff.

A reward is earned once per life when the kill streak reaches its count; it
then waits (also through deaths) until it is used or the match ends.  Ready
and earned rewards are bit masks, bit ``i`` = reward ``i``.
"""
from __future__ import annotations

import numpy as np

from . import config as C

_ROWS = C.STREAK_TABLE
_COL = {name: i for i, name in enumerate(C.STREAK_COLUMNS)}

STREAK_KEYS = tuple(row[_COL["key"]] for row in _ROWS)
STREAK_NAMES = tuple(row[_COL["name"]] for row in _ROWS)
STREAK_DESCRIPTIONS = tuple(row[_COL["description"]] for row in _ROWS)
STREAK_COUNT = len(_ROWS)
STREAK_INDEX = {key: i for i, key in enumerate(STREAK_KEYS)}
KILLS = np.array([row[_COL["kills"]] for row in _ROWS], dtype=np.int64)
DURATION = np.array([row[_COL["duration"]] for row in _ROWS], dtype=np.float64)

SCAN = STREAK_INDEX["scan"]
AEGIS = STREAK_INDEX["aegis"]
LANCE = STREAK_INDEX["lance"]


def duration_ticks(index: int) -> int:
    return int(round(float(DURATION[index]) * C.TICK_RATE))


def earned_by(streak: int, earned_mask: int) -> list:
    """Rewards a streak of ``streak`` kills earns that were not earned in this life yet."""
    return [i for i in range(STREAK_COUNT) if streak >= KILLS[i] and not (int(earned_mask) >> i) & 1]


def next_ready(ready_mask: int) -> int:
    """The reward ``X`` uses next (the smallest ready one), or -1."""
    mask = int(ready_mask)
    return (mask & -mask).bit_length() - 1 if mask else -1


def next_goal(streak: int, earned_mask: int) -> int:
    """The next reward this life can still earn (for the HUD), or -1."""
    for i in np.argsort(KILLS, kind="stable").tolist():
        if not (int(earned_mask) >> i) & 1 and streak < KILLS[i]:
            return int(i)
    return -1


def is_streak_kill(weapon: int) -> bool:
    return C.STREAK_KILL_BASE <= int(weapon) < C.STREAK_KILL_BASE + STREAK_COUNT


def lance_damage(distance) -> np.ndarray:
    """Damage of the orbital lance at ``distance`` metres from the impact (linear falloff)."""
    d = np.asarray(distance, dtype=np.float64)
    return C.LANCE_DAMAGE * np.clip(1.0 - d / C.LANCE_RADIUS, 0.0, 1.0)


def validate_streaks() -> list:
    """Problems with the reward table (empty when sound)."""
    problems = []
    if not 1 <= STREAK_COUNT <= 8:
        problems.append("between one and eight rewards fit the bit masks, the table has %d" % STREAK_COUNT)
    if len(set(STREAK_KEYS)) != STREAK_COUNT:
        problems.append("reward keys must be unique")
    if np.any(KILLS < 2) or np.any(np.diff(KILLS) <= 0):
        problems.append("kill counts must start at 2 or more and rise strictly")
    if np.any(DURATION <= 0.0):
        problems.append("every reward needs a positive duration")
    taken = set(range(C.EQUIP_KILL_BASE, C.EQUIP_KILL_BASE + 16)) | {255}
    if any(C.STREAK_KILL_BASE + i in taken for i in range(STREAK_COUNT)):
        problems.append("STREAK_KILL_BASE overlaps the equipment kill ids or the fall pseudo weapon")
    for key in ("scan", "aegis", "lance"):
        if key not in STREAK_INDEX:
            problems.append("the simulation needs a %r reward" % key)
    return problems
