"""
NEXUS-3D-EXTREME :: engine/weapons.py
=====================================
Weapon data tables, upgrade effects and the deterministic weapon state machine.

Everything is table driven: ``WEAPONS`` is one structured NumPy array, the
upgrade tracks are parallel arrays, and :func:`step_weapons` advances the
ammunition / reload / fire-cadence state of every selected row with boolean
masks.  The client runs the same function for its predicted player that the
server runs authoritatively, so a shot the client predicts is a shot the
server grants.

All timing is expressed in *command ticks* of the owning player (one tick per
processed input command), which keeps the cadence identical on both sides no
matter how packets are batched.
"""
from __future__ import annotations

import numpy as np

from . import config as C

# --------------------------------------------------------------------------- #
#  Weapon classes and movement states (rows / columns of the spread matrix)
# --------------------------------------------------------------------------- #
CLASS_AR, CLASS_SMG, CLASS_SHOTGUN, CLASS_MARKSMAN, CLASS_PISTOL, CLASS_LMG, CLASS_SNIPER = range(7)
CLASS_NAMES = ("ASSAULT RIFLE", "SMG", "SHOTGUN", "MARKSMAN", "PISTOL", "LMG", "SNIPER")

MOVE_STAND, MOVE_WALK, MOVE_SPRINT, MOVE_CROUCH, MOVE_SLIDE, MOVE_AIR = range(6)
MOVE_STATE_NAMES = ("STANDING", "WALKING", "SPRINTING", "CROUCHED", "SLIDING", "AIRBORNE")

# Bullet Spread Matrix: multiplier on the weapon's base cone,
# rows = weapon class, columns = movement state.
SPREAD_MATRIX = np.array([
    #  stand  walk  sprint crouch slide  air
    [1.00, 1.35, 1.90, 0.72, 1.85, 2.60],   # assault rifle
    [1.00, 1.15, 1.45, 0.80, 1.30, 1.90],   # smg
    [1.00, 1.05, 1.20, 0.90, 1.10, 1.30],   # shotgun
    [1.00, 1.70, 2.60, 0.60, 2.40, 3.40],   # marksman
    [1.00, 1.20, 1.50, 0.78, 1.35, 1.90],   # pistol
    [1.00, 1.55, 2.30, 0.62, 2.20, 3.10],   # lmg
    [1.00, 1.90, 3.00, 0.55, 2.80, 3.80],   # sniper
], dtype=np.float64)

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

# key, display name, short description
WEAPON_INFO = (
    ("vanguard", "VK-9 VANGUARD", "Balanced full-auto rifle. The corporate standard issue."),
    ("hornet", "HX-4 HORNET", "Feather-light SMG that shreds up close and on the move."),
    ("breacher", "BREACHER-12", "Pump shotgun. Eight pellets, one very bad day."),
    ("longbow", "LONGBOW DMR", "Semi-auto marksman rifle. Two taps at any range."),
    ("sidekick", "P9 SIDEKICK", "Reliable sidearm with a fast draw."),
    ("bulwark", "BULWARK LMG", "75 round belt. Hold the lane, never reload."),
    ("aurora", "AURORA RAIL", "Capacitor driven rail rifle. One beam, one elimination."),
)
WEAPON_KEYS = tuple(info[0] for info in WEAPON_INFO)
WEAPON_NAMES = tuple(info[1] for info in WEAPON_INFO)
WEAPON_DESCRIPTIONS = tuple(info[2] for info in WEAPON_INFO)
WEAPON_COUNT = len(WEAPON_INFO)
WEAPON_INDEX = {key: i for i, key in enumerate(WEAPON_KEYS)}
DEFAULT_LOADOUT = (WEAPON_INDEX["vanguard"], WEAPON_INDEX["sidekick"])

_T = C.TICK_RATE
WEAPONS = np.array([
    # klass          auto   dmg  pel  intvl  mag  reload     swap  hip          ads          bloom/shot   bloom max   recover     r0     r1    rmin  rec pitch    rec yaw      recov mob   zoom  cost     lvl tier
    (CLASS_AR,       True,  26.0, 1,  6.0,   30,  2.1 * _T,  18.0, 2.2 * _DEG,  0.25 * _DEG, 0.35 * _DEG, 2.6 * _DEG, 9.0 * _DEG,  28.0,  60.0, 0.60, 0.62 * _DEG, 0.28 * _DEG, 7.0, 1.00, 1.35, 0.0,     1,  1.0),
    (CLASS_SMG,      True,  19.0, 1,  4.0,   32,  1.7 * _T,  14.0, 2.6 * _DEG,  0.60 * _DEG, 0.30 * _DEG, 3.0 * _DEG, 11.0 * _DEG, 14.0,  34.0, 0.50, 0.42 * _DEG, 0.36 * _DEG, 9.0, 1.08, 1.20, 0.0,     1,  1.0),
    (CLASS_SHOTGUN,  False, 15.0, 8,  44.0,  6,   2.8 * _T,  20.0, 4.6 * _DEG,  3.60 * _DEG, 0.0,         0.0,        0.0,         7.0,   20.0, 0.20, 2.60 * _DEG, 0.80 * _DEG, 5.0, 0.98, 1.15, 2500.0,  3,  1.6),
    (CLASS_MARKSMAN, False, 58.0, 1,  16.0,  12,  2.4 * _T,  20.0, 3.4 * _DEG,  0.08 * _DEG, 0.90 * _DEG, 2.8 * _DEG, 6.0 * _DEG,  55.0, 110.0, 0.75, 1.50 * _DEG, 0.45 * _DEG, 5.5, 0.96, 2.20, 6000.0,  6,  2.4),
    (CLASS_PISTOL,   False, 30.0, 1,  7.0,   12,  1.3 * _T,  10.0, 1.6 * _DEG,  0.30 * _DEG, 0.55 * _DEG, 2.4 * _DEG, 10.0 * _DEG, 16.0,  38.0, 0.50, 0.90 * _DEG, 0.35 * _DEG, 9.0, 1.10, 1.15, 0.0,     1,  0.8),
    (CLASS_LMG,      True,  31.0, 1,  7.0,   75,  4.3 * _T,  26.0, 3.6 * _DEG,  0.50 * _DEG, 0.22 * _DEG, 3.4 * _DEG, 7.0 * _DEG,  34.0,  75.0, 0.65, 0.52 * _DEG, 0.42 * _DEG, 5.0, 0.90, 1.35, 15000.0, 10, 3.6),
    (CLASS_SNIPER,   False, 120.0, 1, 70.0,  5,   3.0 * _T,  28.0, 7.5 * _DEG,  0.0,         0.0,         0.0,        0.0,         120.0, 200.0, 0.85, 4.20 * _DEG, 0.60 * _DEG, 3.5, 0.92, 4.00, 40000.0, 15, 5.5),
], dtype=WEAPON_DTYPE)

# --------------------------------------------------------------------------- #
#  Per-weapon upgrade tracks (levels are bytes on the wire)
# --------------------------------------------------------------------------- #
UPG_DAMAGE, UPG_FIRE_RATE, UPG_MAGAZINE, UPG_RELOAD, UPG_STABILIZER, UPG_LASER, UPG_BARREL, UPG_STOCK = range(8)
UPGRADE_KEYS = ("damage", "fire_rate", "magazine", "reload", "stabilizer", "laser", "barrel", "stock")
UPGRADE_NAMES = ("DAMAGE", "FIRE RATE", "MAGAZINE", "RELOAD SPEED", "STABILIZER", "LASER MODULE",
                 "LONG BARREL", "FEATHER STOCK")
UPGRADE_DESCRIPTIONS = (
    "+5% bullet damage per level",
    "-2% time between shots per level",
    "+8% magazine capacity per level",
    "-4% reload time per level",
    "Attachment: -9% recoil per level",
    "Attachment: -8% hip-fire spread per level",
    "Attachment: +10% damage range per level",
    "Attachment: +1.5% movement speed per level",
)
UPGRADE_COUNT = len(UPGRADE_KEYS)
UPGRADE_MAX = np.array([20, 15, 10, 10, 5, 5, 5, 5], dtype=np.int64)
UPGRADE_BASE_COST = np.array([150.0, 220.0, 180.0, 160.0, 400.0, 350.0, 450.0, 300.0], dtype=np.float64)
UPGRADE_COST_MULT = np.array([1.32, 1.38, 1.45, 1.40, 1.85, 1.80, 1.90, 1.75], dtype=np.float64)
UPGRADE_EFFECT = np.array([0.05, 0.02, 0.08, 0.04, 0.09, 0.08, 0.10, 0.015], dtype=np.float64)

# --------------------------------------------------------------------------- #
#  Operator (global) upgrades
# --------------------------------------------------------------------------- #
OP_PLATING, OP_CAPACITOR, OP_BOUNTY, OP_NANITES = range(4)
OPERATOR_KEYS = ("plating", "capacitor", "bounty", "nanites")
OPERATOR_NAMES = ("REINFORCED PLATING", "PULSE CAPACITOR", "BOUNTY UPLINK", "MEDIC NANITES")
OPERATOR_DESCRIPTIONS = (
    "+5 maximum health per level",
    "-5% pulse grenade cooldown per level",
    "+10% combat credits per level",
    "-6% health regeneration delay per level",
)
OPERATOR_COUNT = len(OPERATOR_KEYS)
OPERATOR_MAX = np.array([10, 10, 20, 8], dtype=np.int64)
OPERATOR_BASE_COST = np.array([500.0, 650.0, 300.0, 450.0], dtype=np.float64)
OPERATOR_COST_MULT = np.array([1.60, 1.55, 1.42, 1.65], dtype=np.float64)
OPERATOR_EFFECT = np.array([5.0, 0.05, 0.10, 0.06], dtype=np.float64)

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
