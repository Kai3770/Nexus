"""
NEXUS-3D-EXTREME :: engine/equipment.py
=======================================
Lethal and tactical equipment.

``config.EQUIPMENT_TABLE`` as arrays, the per-kind helpers the simulation
uses (fuses, splash damage, stun length, cooldowns) and the balance rules
every item has to satisfy (:func:`validate_equipment`, mirrored by the
test-suite).  Needs no window: the dedicated server and the tests import it.

Every player carries one LETHAL (key G) and one TACTICAL (key F).  Kinds:

``frag``    bouncing grenade, timed fuse, radial damage with knock-back
``sticky``  sticks to the first surface or player it touches, then detonates
``fire``    bursts on impact into a pool that burns for ``duration`` seconds
``axe``     thrown hatchet: a direct hit is lethal, there is no splash
``stun``    bouncing grenade that slows and disorients everyone it reaches
``smoke``   bouncing canister that fills a sphere with smoke
``stim``    used on yourself: heals ``damage`` points over ``duration`` seconds
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C

KIND_FRAG, KIND_STICKY, KIND_FIRE, KIND_AXE, KIND_STUN, KIND_SMOKE, KIND_STIM = range(len(C.EQUIPMENT_KINDS))

_ROWS = [dict(zip(C.EQUIPMENT_COLUMNS, row)) for row in C.EQUIPMENT_TABLE]
EQUIP_KEYS = tuple(row["key"] for row in _ROWS)
EQUIP_NAMES = tuple(row["name"] for row in _ROWS)
EQUIP_DESCRIPTIONS = tuple(row["description"] for row in _ROWS)
EQUIP_COUNT = len(_ROWS)
EQUIP_INDEX = {key: index for index, key in enumerate(EQUIP_KEYS)}


def _column(name: str, dtype=np.float64) -> np.ndarray:
    return np.array([row[name] for row in _ROWS], dtype=dtype)


SLOT = _column("slot", np.int64)
KIND = np.array([C.EQUIPMENT_KINDS.index(row["kind"]) for row in _ROWS], dtype=np.int64)
COOLDOWN = _column("cooldown")
SPEED = _column("speed")
GRAVITY = _column("gravity")
FUSE = _column("fuse")
RADIUS = _column("radius")
DAMAGE = _column("damage")
DURATION = _column("duration")
STRENGTH = _column("strength")
UNLOCK_COST = _column("unlock_cost")
UNLOCK_LEVEL = _column("unlock_level", np.int64)

LETHALS = tuple(i for i in range(EQUIP_COUNT) if SLOT[i] == C.EQUIP_LETHAL)
TACTICALS = tuple(i for i in range(EQUIP_COUNT) if SLOT[i] == C.EQUIP_TACTICAL)
DEFAULT_EQUIPMENT = (EQUIP_INDEX[C.DEFAULT_EQUIPMENT_KEYS[0]], EQUIP_INDEX[C.DEFAULT_EQUIPMENT_KEYS[1]])
BOUNCES = np.isin(KIND, (KIND_FRAG, KIND_STUN, KIND_SMOKE))           # bounce off walls until the fuse runs out
STICKS = np.isin(KIND, (KIND_STICKY, KIND_AXE))                       # stop dead on the first wall they touch
CONTACT = np.isin(KIND, (KIND_STICKY, KIND_AXE, KIND_FIRE))            # react to players they fly into
_STUN = np.flatnonzero(KIND == KIND_STUN)
STUN_FACTOR = float(STRENGTH[_STUN[0]]) if _STUN.size else 1.0        # movement multiplier while stunned


def sanitize(equipment) -> tuple:
    """``(lethal, tactical)`` ids; anything invalid for its slot is replaced by the default."""
    try:
        lethal, tactical = (int(v) for v in equipment)
    except (TypeError, ValueError):
        return DEFAULT_EQUIPMENT
    if not (0 <= lethal < EQUIP_COUNT and SLOT[lethal] == C.EQUIP_LETHAL):
        lethal = DEFAULT_EQUIPMENT[0]
    if not (0 <= tactical < EQUIP_COUNT and SLOT[tactical] == C.EQUIP_TACTICAL):
        tactical = DEFAULT_EQUIPMENT[1]
    return lethal, tactical


def cooldown_ticks(equip: int, reduction: float = 0.0) -> int:
    """Cooldown of an item in ticks after the operator's cooldown reduction (fraction)."""
    return max(int(round(float(COOLDOWN[equip]) * (1.0 - float(reduction)) * C.TICK_RATE)), C.TICK_RATE)


def flight_ticks(equip: int) -> int:
    """Fuse of a freshly thrown item in ticks (a sticky charge gets extra time to find something)."""
    seconds = float(FUSE[equip]) + (C.STICK_FLIGHT if KIND[equip] == KIND_STICKY else 0.0)
    return max(int(round(seconds * C.TICK_RATE)), 1)


def stick_ticks(equip: int) -> int:
    return max(int(round(float(FUSE[equip]) * C.TICK_RATE)), 1)


def splash_damage(equip: int, distance) -> np.ndarray:
    """Radial damage of a frag or sticky blast: ``damage * (1 - d / r)^2`` inside the radius."""
    distance = np.asarray(distance, dtype=np.float64)
    radius = max(float(RADIUS[equip]), 1.0e-6)
    return np.where(distance < radius, float(DAMAGE[equip]) * (1.0 - np.minimum(distance / radius, 1.0)) ** 2, 0.0)


def stun_ticks(equip: int, distance) -> np.ndarray:
    """Stun length in ticks: ``duration`` at the centre, ``STUN_EDGE_SCALE * duration`` at the edge."""
    fraction = np.clip(np.asarray(distance, dtype=np.float64) / max(float(RADIUS[equip]), 1.0e-6), 0.0, 1.0)
    seconds = float(DURATION[equip]) * (1.0 - (1.0 - C.STUN_EDGE_SCALE) * fraction)
    return np.round(seconds * C.TICK_RATE).astype(np.int64)


def kill_radius(equip: int, resist: float = 1.0) -> float:
    """Distance inside which one blast kills a full-health player (0 when it never does)."""
    peak = float(DAMAGE[equip]) * resist
    if KIND[equip] not in (KIND_FRAG, KIND_STICKY) or peak < C.BASE_HEALTH:
        return 0.0
    return float(RADIUS[equip]) * (1.0 - math.sqrt(C.BASE_HEALTH / peak))


def describe(equip: int) -> str:
    """One line of numbers for the equipment menu."""
    kind = int(KIND[equip])
    cooldown = "COOLDOWN %.0fs" % COOLDOWN[equip]
    if kind == KIND_FRAG:
        return "%s    DAMAGE %d    RADIUS %.1fm    FUSE %.1fs" % (cooldown, DAMAGE[equip], RADIUS[equip], FUSE[equip])
    if kind == KIND_STICKY:
        return "%s    DAMAGE %d (%d IF STUCK)    RADIUS %.1fm" % (cooldown, DAMAGE[equip],
                                                               DAMAGE[equip] * C.STICK_DAMAGE_MULT, RADIUS[equip])
    if kind == KIND_FIRE:
        return "%s    %d DAMAGE / S    RADIUS %.1fm    BURNS %.0fs" % (cooldown, DAMAGE[equip], RADIUS[equip],
                                                                    DURATION[equip])
    if kind == KIND_AXE:
        return "%s    DIRECT HIT %d    NO SPLASH" % (cooldown, DAMAGE[equip])
    if kind == KIND_STUN:
        return "%s    STUN UP TO %.1fs    MOVE x%.2f    RADIUS %.0fm" % (cooldown, DURATION[equip], STRENGTH[equip],
                                                                         RADIUS[equip])
    if kind == KIND_SMOKE:
        return "%s    CLOUD %.0fm    LASTS %.0fs" % (cooldown, RADIUS[equip], DURATION[equip])
    return "%s    HEALS %d OVER %.1fs" % (cooldown, DAMAGE[equip], DURATION[equip])


# --------------------------------------------------------------------------- #
#  Balance
# --------------------------------------------------------------------------- #
LETHAL_COOLDOWN_RANGE = (8.0, 20.0)
TACTICAL_COOLDOWN_RANGE = (12.0, 25.0)
MAX_PLATING_RESIST = 0.70                   # Blast Plating at its top level


def balance_report() -> list:
    """The numbers that decide whether an item is fair, one dictionary per item."""
    report = []
    for e in range(EQUIP_COUNT):
        kind = int(KIND[e])
        row = {"key": EQUIP_KEYS[e], "slot": C.EQUIP_SLOT_NAMES[int(SLOT[e])], "kind": C.EQUIPMENT_KINDS[kind],
               "cooldown": float(COOLDOWN[e]), "kill_radius": kill_radius(e),
               "kill_radius_plated": kill_radius(e, MAX_PLATING_RESIST)}
        if kind == KIND_STICKY:
            row["stuck_damage_plated"] = float(DAMAGE[e]) * C.STICK_DAMAGE_MULT * MAX_PLATING_RESIST
        if kind == KIND_FIRE:
            row["fire_ttk"] = C.BASE_HEALTH / float(DAMAGE[e])
            row["fire_total"] = float(DAMAGE[e] * DURATION[e])
        if kind == KIND_STIM:
            row["heal_rate"] = float(DAMAGE[e] / max(DURATION[e], 1.0e-6))
        report.append(row)
    return report


def validate_equipment() -> list:
    """Return the violated balance rules (an empty list when the table is sound)."""
    problems = []
    for slot in (C.EQUIP_LETHAL, C.EQUIP_TACTICAL):
        if not any(SLOT[e] == slot for e in range(EQUIP_COUNT)):
            problems.append("no %s equipment" % C.EQUIP_SLOT_NAMES[slot])
    for slot, e in enumerate(DEFAULT_EQUIPMENT):
        if SLOT[e] != slot or UNLOCK_COST[e] > 0.0:
            problems.append("default %s must be free and in its own slot" % EQUIP_KEYS[e])
    splash = [e for e in LETHALS if KIND[e] in (KIND_FRAG, KIND_STICKY, KIND_FIRE)]
    for e in range(EQUIP_COUNT):
        key, kind = EQUIP_KEYS[e], int(KIND[e])
        low, high = LETHAL_COOLDOWN_RANGE if SLOT[e] == C.EQUIP_LETHAL else TACTICAL_COOLDOWN_RANGE
        if not low <= COOLDOWN[e] <= high:
            problems.append("%s cooldown %.0fs is outside %.0f-%.0fs" % (key, COOLDOWN[e], low, high))
        if kind in (KIND_FRAG, KIND_STICKY, KIND_FIRE, KIND_AXE) and SLOT[e] != C.EQUIP_LETHAL:
            problems.append("%s deals damage and must be a lethal" % key)
        if kind in (KIND_STUN, KIND_SMOKE, KIND_STIM) and SLOT[e] != C.EQUIP_TACTICAL:
            problems.append("%s is a tactical" % key)
        if kind != KIND_STIM and not (10.0 <= SPEED[e] <= 30.0 and 0.5 <= FUSE[e] <= 3.0):
            problems.append("%s: throw speed or fuse out of range" % key)
        if kind in (KIND_FRAG, KIND_STICKY):
            if kill_radius(e) > 1.0:
                problems.append("%s one-shots players %.2f m from the blast" % (key, kill_radius(e)))
            if RADIUS[e] > 7.0:
                problems.append("%s blast radius above 7 m" % key)
        if kind == KIND_STICKY and DAMAGE[e] * C.STICK_DAMAGE_MULT * MAX_PLATING_RESIST < C.BASE_HEALTH:
            problems.append("%s does not kill the player it sticks to" % key)
        if kind == KIND_FIRE:
            if C.BASE_HEALTH / DAMAGE[e] < 2.0:
                problems.append("%s kills in under two seconds" % key)
            if DAMAGE[e] * DURATION[e] < C.BASE_HEALTH or RADIUS[e] > 4.0:
                problems.append("%s cannot finish a kill or burns too wide" % key)
        if kind == KIND_AXE and (DAMAGE[e] < C.BASE_HEALTH or RADIUS[e] > 0.0):
            problems.append("%s must kill on a direct hit and nothing else" % key)
        if kind == KIND_STUN and not (DURATION[e] <= 3.0 and 0.4 <= STRENGTH[e] <= 0.8 and DAMAGE[e] == 0.0):
            problems.append("%s stuns too long or too hard" % key)
        if kind == KIND_SMOKE and not (RADIUS[e] <= 6.0 and DURATION[e] <= 10.0 and DAMAGE[e] == 0.0):
            problems.append("%s smoke too large or too long" % key)
        if kind == KIND_STIM:
            rate = DAMAGE[e] / max(DURATION[e], 1.0e-6)
            if DAMAGE[e] > 0.6 * C.BASE_HEALTH or rate > 0.5 * C.BASE_HEALTH / (C.TTK_MAX_MS / 1000.0):
                problems.append("%s heals too much or too fast to lose a gunfight with" % key)
    for e in LETHALS:
        if KIND[e] == KIND_AXE and splash and COOLDOWN[e] >= min(COOLDOWN[s] for s in splash):
            problems.append("%s needs a direct hit and must recharge faster than every splash lethal" % EQUIP_KEYS[e])
        if KIND[e] in (KIND_STICKY, KIND_FIRE):
            frags = [f for f in LETHALS if KIND[f] == KIND_FRAG]
            if frags and COOLDOWN[e] < max(COOLDOWN[f] for f in frags):
                problems.append("%s is more reliable than a frag and must not recharge faster" % EQUIP_KEYS[e])
    order = sorted(range(EQUIP_COUNT), key=lambda e: (int(UNLOCK_LEVEL[e]), e))
    costs = [float(UNLOCK_COST[e]) for e in order]
    if costs != sorted(costs):
        problems.append("unlock costs must rise with the unlock level")
    return problems
