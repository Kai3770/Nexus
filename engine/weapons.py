"""
NEXUS-3D-EXTREME :: engine/weapons.py
=====================================
Weapon registry: turns the balance matrices of :mod:`engine.config` into
NumPy tables, derives upgraded statistics, measures time-to-kill and runs the
deterministic weapon state machine.

All tuning lives in ``config.py`` (``WEAPON_TABLE``, ``UPGRADE_TABLE``,
``OPERATOR_TABLE``, ``SPREAD_MATRIX``); this module contains no numbers of its
own.  :func:`validate_balance` checks the invariants of the balancing model
and is part of the test-suite:

* every weapon, stock and fully upgraded, kills a 100 health target with torso
  hits in 300 to 500 ms;
* sorted by damage per trigger pull, fire interval, recoil and reload time
  never decrease.

All timing is expressed in *command ticks* of the owning player (one tick per
processed input command), so client prediction and the server agree exactly.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C

# --------------------------------------------------------------------------- #
#  Weapon classes and movement states (rows / columns of the spread matrix)
# --------------------------------------------------------------------------- #
CLASS_AR, CLASS_SMG, CLASS_SHOTGUN, CLASS_MARKSMAN, CLASS_PISTOL, CLASS_LMG, CLASS_SNIPER = range(7)
CLASS_NAMES = C.WEAPON_CLASSES

MOVE_STAND, MOVE_WALK, MOVE_SPRINT, MOVE_CROUCH, MOVE_SLIDE, MOVE_AIR = range(6)
MOVE_STATE_NAMES = ("STANDING", "WALKING", "SPRINTING", "CROUCHED", "SLIDING", "AIRBORNE")
SPREAD_MATRIX = np.array(C.SPREAD_MATRIX, dtype=np.float64)

_DEG = np.pi / 180.0

WEAPON_DTYPE = np.dtype([
    ("klass", np.int32), ("auto", np.bool_), ("damage", np.float64), ("pellets", np.int32),
    ("interval", np.float64), ("mag", np.int32), ("reload", np.float64), ("swap", np.float64),
    ("spread_hip", np.float64), ("spread_ads", np.float64),
    ("bloom_shot", np.float64), ("bloom_max", np.float64), ("bloom_recover", np.float64),
    ("range_start", np.float64), ("range_end", np.float64), ("range_min", np.float64),
    ("recoil_pitch", np.float64), ("recoil_yaw", np.float64), ("recoil_recover", np.float64),
    ("mobility", np.float64), ("ads_zoom", np.float64),
    ("unlock_cost", np.float64), ("unlock_level", np.int32), ("tier", np.float64),
])


def _build_weapons():
    """Convert ``config.WEAPON_TABLE`` rows into the structured table used at run time."""
    rows = [dict(zip(C.WEAPON_COLUMNS, row)) for row in C.WEAPON_TABLE]
    table = np.zeros(len(rows), dtype=WEAPON_DTYPE)
    for index, row in enumerate(rows):
        burst = row["damage"] * row["pellets"]                       # damage per trigger pull
        climb = C.RECOIL_PITCH_PER_DAMAGE * burst
        table[index] = (
            row["klass"], row["auto"], row["damage"], row["pellets"], row["interval"], row["mag"],
            row["reload_s"] * C.TICK_RATE, row["swap"],
            row["hip_deg"] * _DEG, row["ads_deg"] * _DEG,
            row["bloom_shot_deg"] * _DEG, row["bloom_max_deg"] * _DEG, row["bloom_recover_deg"] * _DEG,
            row["range_start"], row["range_end"], row["range_min"],
            climb, climb * C.RECOIL_YAW_RATIO, row["recoil_recover"],
            row["mobility"], row["ads_zoom"], row["unlock_cost"], row["unlock_level"], row["tier"])
    return rows, table


_ROWS, WEAPONS = _build_weapons()
WEAPON_KEYS = tuple(row["key"] for row in _ROWS)
WEAPON_NAMES = tuple(row["name"] for row in _ROWS)
WEAPON_DESCRIPTIONS = tuple(row["description"] for row in _ROWS)
WEAPON_COUNT = len(_ROWS)
WEAPON_INDEX = {key: i for i, key in enumerate(WEAPON_KEYS)}
DEFAULT_LOADOUT = tuple(WEAPON_INDEX[key] for key in C.DEFAULT_LOADOUT_KEYS)

# --------------------------------------------------------------------------- #
#  Upgrade tracks (levels are bytes on the wire)
# --------------------------------------------------------------------------- #
UPG_DAMAGE, UPG_FIRE_RATE, UPG_MAGAZINE, UPG_RELOAD, UPG_STABILIZER, UPG_LASER, UPG_BARREL, UPG_STOCK = range(8)
UPGRADE_KEYS = tuple(row[0] for row in C.UPGRADE_TABLE)
UPGRADE_NAMES = tuple(row[1] for row in C.UPGRADE_TABLE)
UPGRADE_DESCRIPTIONS = tuple(row[6] for row in C.UPGRADE_TABLE)
UPGRADE_COUNT = len(C.UPGRADE_TABLE)
UPGRADE_MAX = np.array([row[2] for row in C.UPGRADE_TABLE], dtype=np.int64)
UPGRADE_EFFECT = np.array([row[3] for row in C.UPGRADE_TABLE], dtype=np.float64)
UPGRADE_BASE_COST = np.array([row[4] for row in C.UPGRADE_TABLE], dtype=np.float64)
UPGRADE_COST_MULT = np.array([row[5] for row in C.UPGRADE_TABLE], dtype=np.float64)

OP_PLATING, OP_CAPACITOR, OP_BOUNTY, OP_NANITES = range(4)
OPERATOR_KEYS = tuple(row[0] for row in C.OPERATOR_TABLE)
OPERATOR_NAMES = tuple(row[1] for row in C.OPERATOR_TABLE)
OPERATOR_DESCRIPTIONS = tuple(row[6] for row in C.OPERATOR_TABLE)
OPERATOR_COUNT = len(C.OPERATOR_TABLE)
OPERATOR_MAX = np.array([row[2] for row in C.OPERATOR_TABLE], dtype=np.int64)
OPERATOR_EFFECT = np.array([row[3] for row in C.OPERATOR_TABLE], dtype=np.float64)
OPERATOR_BASE_COST = np.array([row[4] for row in C.OPERATOR_TABLE], dtype=np.float64)
OPERATOR_COST_MULT = np.array([row[5] for row in C.OPERATOR_TABLE], dtype=np.float64)

LOADOUT_SLOTS = 2
LOADOUT_BLOB_SIZE = LOADOUT_SLOTS * (1 + UPGRADE_COUNT) + OPERATOR_COUNT + 1     # bytes on the wire


def clamp_levels(levels) -> np.ndarray:
    """Clamp ``(..., UPGRADE_COUNT)`` weapon upgrade levels to their legal range."""
    return np.clip(np.asarray(levels, dtype=np.int64), 0, UPGRADE_MAX)


def clamp_operator(levels) -> np.ndarray:
    return np.clip(np.asarray(levels, dtype=np.int64), 0, OPERATOR_MAX)


def encode_loadout(weapon_ids, levels, operator, prestige: int) -> bytes:
    """Pack a loadout into ``LOADOUT_BLOB_SIZE`` bytes (connect handshake)."""
    weapon_ids = np.clip(np.asarray(weapon_ids, dtype=np.int64), 0, WEAPON_COUNT - 1)
    levels = clamp_levels(levels).reshape(LOADOUT_SLOTS, UPGRADE_COUNT)
    blob = bytearray()
    for slot in range(LOADOUT_SLOTS):
        blob.append(int(weapon_ids[slot]))
        blob += levels[slot].astype(np.uint8).tobytes()
    blob += clamp_operator(operator).astype(np.uint8).tobytes()
    blob.append(int(np.clip(prestige, 0, 255)))
    return bytes(blob)


def decode_loadout(blob: bytes):
    """Inverse of :func:`encode_loadout`; every value is range checked."""
    if len(blob) != LOADOUT_BLOB_SIZE:
        raise ValueError("bad loadout blob size %d" % len(blob))
    raw = np.frombuffer(blob, dtype=np.uint8).astype(np.int64)
    weapon_ids = np.zeros(LOADOUT_SLOTS, dtype=np.int64)
    levels = np.zeros((LOADOUT_SLOTS, UPGRADE_COUNT), dtype=np.int64)
    cursor = 0
    for slot in range(LOADOUT_SLOTS):
        weapon_ids[slot] = min(int(raw[cursor]), WEAPON_COUNT - 1)
        levels[slot] = raw[cursor + 1: cursor + 1 + UPGRADE_COUNT]
        cursor += 1 + UPGRADE_COUNT
    operator = raw[cursor: cursor + OPERATOR_COUNT]
    prestige = int(raw[cursor + OPERATOR_COUNT])
    return weapon_ids, clamp_levels(levels), clamp_operator(operator), prestige


def derive_stats(weapon_ids, levels) -> dict:
    """
    Effective weapon statistics for arbitrary leading shape.

    ``weapon_ids`` is ``(...)`` and ``levels`` ``(..., UPGRADE_COUNT)``.  The
    result maps stat name -> array of shape ``(...)``.
    """
    weapon_ids = np.asarray(weapon_ids, dtype=np.int64)
    lv = clamp_levels(levels).astype(np.float64)
    base = WEAPONS[weapon_ids]
    eff = UPGRADE_EFFECT
    mag = np.maximum(np.rint(base["mag"] * (1.0 + eff[UPG_MAGAZINE] * lv[..., UPG_MAGAZINE])), 1.0).astype(np.int64)
    return {
        "weapon": weapon_ids.copy(),
        "klass": base["klass"].astype(np.int64),
        "auto": base["auto"].copy(),
        "damage": base["damage"] * (1.0 + eff[UPG_DAMAGE] * lv[..., UPG_DAMAGE]),
        "pellets": base["pellets"].astype(np.int64),
        "interval": np.maximum(base["interval"] * (1.0 - eff[UPG_FIRE_RATE] * lv[..., UPG_FIRE_RATE]), 1.0),
        "mag": mag,
        "reload_ticks": np.maximum(np.rint(base["reload"] * (1.0 - eff[UPG_RELOAD] * lv[..., UPG_RELOAD])), 6.0).astype(np.int64),
        "swap_ticks": np.rint(base["swap"]).astype(np.int64),
        "spread_hip": base["spread_hip"] * (1.0 - eff[UPG_LASER] * lv[..., UPG_LASER]),
        "spread_ads": base["spread_ads"].copy(),
        "bloom_shot": base["bloom_shot"].copy(),
        "bloom_max": base["bloom_max"].copy(),
        "bloom_recover": base["bloom_recover"].copy(),
        "range_start": base["range_start"] * (1.0 + eff[UPG_BARREL] * lv[..., UPG_BARREL]),
        "range_end": base["range_end"] * (1.0 + eff[UPG_BARREL] * lv[..., UPG_BARREL]),
        "range_min": base["range_min"].copy(),
        "recoil_pitch": base["recoil_pitch"] * (1.0 - eff[UPG_STABILIZER] * lv[..., UPG_STABILIZER]),
        "recoil_yaw": base["recoil_yaw"] * (1.0 - eff[UPG_STABILIZER] * lv[..., UPG_STABILIZER]),
        "recoil_recover": base["recoil_recover"].copy(),
        "mobility": base["mobility"] * (1.0 + eff[UPG_STOCK] * lv[..., UPG_STOCK]),
        "ads_zoom": base["ads_zoom"].copy(),
    }


def damage_falloff(distance, range_start, range_end, range_min):
    """Linear damage falloff multiplier between ``range_start`` and ``range_end``."""
    span = np.maximum(np.asarray(range_end, dtype=np.float64) - range_start, 1.0e-6)
    t = np.clip((np.asarray(distance, dtype=np.float64) - range_start) / span, 0.0, 1.0)
    return 1.0 + (np.asarray(range_min, dtype=np.float64) - 1.0) * t


def movement_state(mflags, horizontal_speed):
    """Classify bodies into the columns of :data:`SPREAD_MATRIX` (vectorised)."""
    mflags = np.asarray(mflags)
    speed = np.asarray(horizontal_speed, dtype=np.float64)
    state = np.where(speed > 0.9, MOVE_WALK, MOVE_STAND)
    state = np.where((mflags & C.MF_SPRINTING) != 0, MOVE_SPRINT, state)
    state = np.where((mflags & C.MF_CROUCHED) != 0, MOVE_CROUCH, state)
    state = np.where((mflags & C.MF_SLIDING) != 0, MOVE_SLIDE, state)
    state = np.where((mflags & C.MF_ON_GROUND) == 0, MOVE_AIR, state)
    return state


def spread_angle(klass, spread_hip, spread_ads, bloom, mflags, horizontal_speed):
    """
    Cone half-angle (radians) of the next shot: the accuracy metric that also
    drives the dynamic crosshair.
    """
    ads = (np.asarray(mflags) & C.MF_ADS) != 0
    base = np.where(ads, spread_ads, spread_hip)
    factor = SPREAD_MATRIX[np.asarray(klass, dtype=np.int64), movement_state(mflags, horizontal_speed)]
    factor = np.where(ads, 1.0 + (factor - 1.0) * 0.35, factor)
    return base * factor + np.where(ads, np.asarray(bloom) * 0.5, bloom)


# --------------------------------------------------------------------------- #
#  Balance model: time-to-kill and invariants
# --------------------------------------------------------------------------- #
def shots_to_kill(damage_per_trigger: float, health: float = C.BASE_HEALTH) -> int:
    """Trigger pulls needed to remove ``health`` when every round (pellet) lands."""
    return int(math.ceil(health / damage_per_trigger - 1.0e-9))


def time_to_kill_ms(weapon: int, levels=None, health: float = C.BASE_HEALTH, zone: int = C.ZONE_TORSO) -> float:
    """
    Milliseconds from the first round to the killing round at maximum cadence.
    Uses the same fractional-carry schedule as :func:`step_weapons`: round ``k``
    leaves on tick ``ceil(k * interval)``.
    """
    if levels is None:
        levels = np.zeros(UPGRADE_COUNT, dtype=np.int64)
    stats = derive_stats(int(weapon), levels)
    burst = float(stats["damage"]) * int(stats["pellets"]) * float(C.ZONE_MULTIPLIER[zone])
    rounds = shots_to_kill(burst, health)
    return math.ceil((rounds - 1) * float(stats["interval"]) - 1.0e-9) * C.TICK_MS


def balance_report() -> list:
    """One dictionary per weapon with the numbers a designer cares about (stock and fully upgraded)."""
    report = []
    for index in range(WEAPON_COUNT):
        stock = derive_stats(index, np.zeros(UPGRADE_COUNT, dtype=np.int64))
        burst = float(stock["damage"]) * int(stock["pellets"])
        report.append({
            "key": WEAPON_KEYS[index], "class": CLASS_NAMES[int(stock["klass"])], "burst_damage": burst,
            "rpm": 60.0 * C.TICK_RATE / float(stock["interval"]), "shots_to_kill": shots_to_kill(burst),
            "ttk_ms": time_to_kill_ms(index), "ttk_max_ms": time_to_kill_ms(index, UPGRADE_MAX),
            "ttk_head_ms": time_to_kill_ms(index, zone=C.ZONE_HEAD),
            "reload_s": float(stock["reload_ticks"]) / C.TICK_RATE,
            "recoil_deg": float(np.degrees(stock["recoil_pitch"])),
            "dps": burst * C.TICK_RATE / float(stock["interval"]),
        })
    return report


def validate_balance() -> list:
    """
    Check the invariants of the balancing model.  Returns a list of violations
    (empty when the tables are sound).
    """
    problems = []
    report = balance_report()
    for row in report:
        for label in ("ttk_ms", "ttk_max_ms"):
            if not C.TTK_MIN_MS - 1.0e-6 <= row[label] <= C.TTK_MAX_MS + 1.0e-6:
                problems.append("%s %s = %.0f ms is outside %.0f-%.0f ms" % (
                    row["key"], label, row[label], C.TTK_MIN_MS, C.TTK_MAX_MS))
    for index in range(WEAPON_COUNT):
        stock = float(WEAPONS["damage"][index]) * int(WEAPONS["pellets"][index])
        best = float(derive_stats(index, UPGRADE_MAX)["damage"]) * int(WEAPONS["pellets"][index])
        if shots_to_kill(stock) != shots_to_kill(best):
            problems.append("%s: damage upgrades change the shots-to-kill" % WEAPON_KEYS[index])
    order = sorted(report, key=lambda row: row["burst_damage"])
    for weaker, stronger in zip(order, order[1:]):
        if stronger["rpm"] > weaker["rpm"] + 1.0e-9:
            problems.append("%s hits harder than %s but also fires faster" % (stronger["key"], weaker["key"]))
        if stronger["recoil_deg"] < weaker["recoil_deg"] - 1.0e-9:
            problems.append("%s hits harder than %s but recoils less" % (stronger["key"], weaker["key"]))
        if stronger["reload_s"] < weaker["reload_s"] - 1.0e-9:
            problems.append("%s hits harder than %s but reloads faster" % (stronger["key"], weaker["key"]))
    return problems


# --------------------------------------------------------------------------- #
#  Weapon state (structure of arrays) and its deterministic step
# --------------------------------------------------------------------------- #
class WeaponState:
    """Ammunition, cadence and loadout statistics of ``capacity`` players."""

    STAT_FIELDS = ("weapon", "klass", "auto", "damage", "pellets", "interval", "mag", "reload_ticks",
                   "swap_ticks", "spread_hip", "spread_ads", "bloom_shot", "bloom_max", "bloom_recover",
                   "range_start", "range_end", "range_min", "recoil_pitch", "recoil_yaw", "recoil_recover",
                   "mobility", "ads_zoom")

    def __init__(self, capacity: int):
        n = int(capacity)
        self.capacity = n
        self.slot = np.zeros(n, dtype=np.int64)
        self.ammo = np.zeros((n, LOADOUT_SLOTS), dtype=np.int64)
        self.reload_end = np.zeros(n, dtype=np.int64)        # command tick when the reload completes (0 = idle)
        self.next_fire = np.zeros(n, dtype=np.float64)       # fractional command tick of the next allowed shot
        self.bloom = np.zeros(n, dtype=np.float64)
        self.prev_buttons = np.zeros(n, dtype=np.uint16)
        self.shots_fired = np.zeros(n, dtype=np.int64)       # lifetime counter, seeds the spread pattern
        self.levels = np.zeros((n, LOADOUT_SLOTS, UPGRADE_COUNT), dtype=np.int64)
        ids = np.tile(np.array(DEFAULT_LOADOUT, dtype=np.int64), (n, 1))
        stats = derive_stats(ids, self.levels)
        for name in self.STAT_FIELDS:
            setattr(self, name, stats[name])
        self.ammo[:] = self.mag

    def set_loadout(self, row: int, weapon_ids, levels) -> None:
        """Install a loadout for one player and recompute its derived statistics."""
        ids = np.clip(np.asarray(weapon_ids, dtype=np.int64), 0, WEAPON_COUNT - 1).reshape(LOADOUT_SLOTS)
        lv = clamp_levels(levels).reshape(LOADOUT_SLOTS, UPGRADE_COUNT)
        self.levels[row] = lv
        stats = derive_stats(ids, lv)
        for name in self.STAT_FIELDS:
            getattr(self, name)[row] = stats[name]
        self.reset_rows(np.array([row]))

    def reset_rows(self, rows) -> None:
        """Full magazines, slot 0, timers cleared (spawn / respawn)."""
        rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))
        self.slot[rows] = 0
        self.ammo[rows] = self.mag[rows]
        self.reload_end[rows] = 0
        self.next_fire[rows] = 0.0
        self.bloom[rows] = 0.0
        self.prev_buttons[rows] = 0

    def current(self, name: str, rows):
        """Statistic ``name`` of the weapon each of ``rows`` currently holds."""
        rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))
        return getattr(self, name)[rows, self.slot[rows]]

    def save_row(self, row: int) -> tuple:
        return (int(self.slot[row]), self.ammo[row].copy(), int(self.reload_end[row]), float(self.next_fire[row]),
                float(self.bloom[row]), int(self.prev_buttons[row]), int(self.shots_fired[row]))

    def load_row(self, row: int, saved: tuple) -> None:
        (self.slot[row], self.ammo[row], self.reload_end[row], self.next_fire[row], self.bloom[row],
         self.prev_buttons[row], self.shots_fired[row]) = saved


def step_weapons(ws: WeaponState, rows, buttons, cmd_tick, active, dt: float = C.TICK_DT) -> np.ndarray:
    """
    Advance the weapon state machine of ``rows`` by one command.

    ``buttons`` are this command's input masks, ``cmd_tick`` the command tick
    of each row and ``active`` a boolean mask (dead players do nothing).
    Returns the boolean mask of rows that released a round this tick.

    Order inside a tick: weapon swap -> reload completion -> trigger ->
    bloom -> reload start.  Fully automatic weapons fire while the trigger is
    held, the others on the press edge.  The cadence carries its fractional
    remainder (``next_fire`` is a float tick) so upgraded fire rates that are
    not a whole number of ticks still average out exactly.
    """
    rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))
    n = rows.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    buttons = np.broadcast_to(np.asarray(buttons, dtype=np.uint16), (n,))
    tick = np.broadcast_to(np.asarray(cmd_tick, dtype=np.int64), (n,))
    active = np.broadcast_to(np.asarray(active, dtype=bool), (n,))
    prev = ws.prev_buttons[rows]
    slot = ws.slot[rows]
    reload_end = ws.reload_end[rows]
    next_fire = ws.next_fire[rows]
    bloom = ws.bloom[rows]

    shoot_down = (buttons & C.BTN_SHOOT) != 0
    shoot_pressed = shoot_down & ((prev & C.BTN_SHOOT) == 0)
    swap_pressed = ((buttons & C.BTN_SWAP) != 0) & ((prev & C.BTN_SWAP) == 0) & active
    reload_pressed = ((buttons & C.BTN_RELOAD) != 0) & ((prev & C.BTN_RELOAD) == 0)

    # ---- weapon swap ------------------------------------------------------
    slot = np.where(swap_pressed, 1 - slot, slot)
    reload_end = np.where(swap_pressed, 0, reload_end)
    bloom = np.where(swap_pressed, 0.0, bloom)
    next_fire = np.where(swap_pressed, np.maximum(next_fire, tick + ws.swap_ticks[rows, slot]), next_fire)

    ammo = ws.ammo[rows, slot]
    mag = ws.mag[rows, slot]
    interval = ws.interval[rows, slot]

    # ---- reload completion ---------------------------------------------------
    done = (reload_end > 0) & (tick >= reload_end)
    ammo = np.where(done, mag, ammo)
    reload_end = np.where(done, 0, reload_end)

    # ---- trigger ---------------------------------------------------------------
    trigger = np.where(ws.auto[rows, slot], shoot_down, shoot_pressed)
    fired = trigger & active & (reload_end == 0) & (ammo > 0) & (tick >= next_fire - 1.0e-9)
    carry = (tick - next_fire) < 1.0
    next_fire = np.where(fired, np.where(carry, next_fire, tick) + interval, next_fire)
    ammo = np.where(fired, ammo - 1, ammo)
    bloom = np.where(fired, np.minimum(bloom + ws.bloom_shot[rows, slot], ws.bloom_max[rows, slot]),
                     np.maximum(bloom - ws.bloom_recover[rows, slot] * dt, 0.0))

    # ---- reload start (manual, or automatic on an empty magazine) -----------------
    start = active & (reload_end == 0) & (ammo < mag) & (reload_pressed | (ammo == 0))
    reload_end = np.where(start, tick + ws.reload_ticks[rows, slot], reload_end)

    ws.slot[rows] = slot
    ws.ammo[rows, slot] = ammo
    ws.reload_end[rows] = reload_end
    ws.next_fire[rows] = next_fire
    ws.bloom[rows] = bloom
    ws.prev_buttons[rows] = buttons
    ws.shots_fired[rows] += fired
    return fired
