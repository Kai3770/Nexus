"""
NEXUS-3D-EXTREME :: engine/secure_save.py              (MODULE D)
=================================================================
Math foundations of the incremental economy and tamper-evident profile
persistence.

Economy
-------
Every purchasable thing follows one law::

    Cost = BaseCost * Multiplier ** CurrentLevel

which is implemented once (:func:`exp_cost`) and reused, vectorised, for
weapon upgrades, attachments, operator upgrades, sponsor contracts and the
experience curve.  Bulk purchases use the closed form of the geometric series
instead of loops.

Passive income comes from *sponsor contracts*.  The "Corporate Sponsor Thread"
(:class:`CorporateSponsorThread`) wakes a few times per second, measures the
elapsed time with ``time.perf_counter_ns`` and credits
``income_per_second * dt`` to the profile, whether the player is in a match or
idling in a menu.

Persistence
-----------
A save is the canonical JSON of the profile (the *internal data string*).
The file stores that string sealed with the game's stream cipher and, as its
final line, ``SHA-256(data_string + SALT)`` in hexadecimal.  Loading
recomputes the signature; any edit to the file -- body or signature -- is
detected, the previous good save is restored if one exists and a strike is
recorded.

Scope note: the salt and the cipher key are constants in this file, as the
design brief requires.  That defeats hand-editing a save, not a person who
reads this source.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from . import config as C
from . import crypto
from . import weapons as W

# --------------------------------------------------------------------------- #
#  Cryptographic constants
# --------------------------------------------------------------------------- #
SAVE_SALT = "NX3D|corporate-ledger|7f3a9e21d4b86c05a1e9|integrity-salt|do-not-edit"
SAVE_KEY = b"NX3D-profile-vault-key-41c7a9d2e86f"
SAVE_HEADER = "NEXUS3D-PROFILE v3"
SIGNATURE_PREFIX = "#SHA256:"
SAVE_VERSION = 3

# --------------------------------------------------------------------------- #
#  Progression tuning
# --------------------------------------------------------------------------- #
XP_BASE = 400.0
XP_MULTIPLIER = 1.18
LEVEL_UP_CREDITS = 250.0
MAX_LEVEL = 100

SPONSOR_NAMES = ("KIOSK ENERGY", "BYTEWAVE PERIPHERALS", "ORBITAL COURIERS", "HELIX BIOTECH",
                 "VANTA MOTORS", "AURORA NETWORKS", "MONOLITH BANK", "NEXUS CONGLOMERATE")
SPONSOR_COUNT = len(SPONSOR_NAMES)
SPONSOR_BASE_COST = np.array([50.0, 400.0, 3.0e3, 2.5e4, 2.2e5, 2.0e6, 1.9e7, 1.9e8], dtype=np.float64)
SPONSOR_COST_MULT = np.full(SPONSOR_COUNT, 1.15, dtype=np.float64)
SPONSOR_BASE_INCOME = np.array([0.5, 3.0, 18.0, 110.0, 700.0, 4.6e3, 3.1e4, 2.1e5], dtype=np.float64)
SPONSOR_MILESTONES = np.array([10, 25, 50, 100, 200, 400], dtype=np.int64)
SPONSOR_MAX_LEVEL = 1000

PRESTIGE_INCOME_MULT = 1.75
PRESTIGE_COMBAT_BONUS = 0.25
PRESTIGE_BASE_REQUIREMENT = 250000.0
PRESTIGE_REQUIREMENT_MULT = 6.0
PRESTIGE_BASE_LEVEL = 12
PRESTIGE_LEVEL_STEP = 4
MAX_PRESTIGE = 50

OFFLINE_CAP_SECONDS = 8.0 * 3600.0
OFFLINE_EFFICIENCY = 0.5
AUTOSAVE_INTERVAL = 30.0

_SUFFIXES = ("", "K", "M", "B", "T", "Qa", "Qi", "Sx", "Sp", "Oc", "No", "Dc")


# =========================================================================== #
#  Math foundations
# =========================================================================== #
def exp_cost(base, multiplier, level):
    """``Cost = BaseCost * Multiplier ** CurrentLevel`` (scalars or arrays)."""
    return np.asarray(base, dtype=np.float64) * np.power(np.asarray(multiplier, dtype=np.float64),
                                                         np.asarray(level, dtype=np.float64))


def bulk_cost(base, multiplier, level, count):
    """
    Total price of ``count`` consecutive levels starting at ``level``:
    ``base * m**level * (m**count - 1) / (m - 1)`` (geometric series).
    """
    multiplier = np.asarray(multiplier, dtype=np.float64)
    count = np.asarray(count, dtype=np.float64)
    first = exp_cost(base, multiplier, level)
    flat = np.abs(multiplier - 1.0) < 1.0e-12
    safe = np.where(flat, 2.0, multiplier)
    return np.where(flat, first * count, first * (np.power(safe, count) - 1.0) / (safe - 1.0))


def max_affordable(base, multiplier, level, budget):
    """Largest ``n`` with ``bulk_cost(base, multiplier, level, n) <= budget`` (closed form)."""
    multiplier = np.asarray(multiplier, dtype=np.float64)
    budget = np.maximum(np.asarray(budget, dtype=np.float64), 0.0)
    first = exp_cost(base, multiplier, level)
    flat = np.abs(multiplier - 1.0) < 1.0e-12
    safe = np.where(flat, 2.0, multiplier)
    geometric = np.floor(np.log(budget * (safe - 1.0) / first + 1.0) / np.log(safe) + 1.0e-9)
    linear = np.floor(budget / first + 1.0e-9)
    return np.maximum(np.where(flat, linear, geometric), 0.0).astype(np.int64)


def xp_required(level):
    """Experience needed to advance from ``level`` to ``level + 1``."""
    return np.rint(exp_cost(XP_BASE, XP_MULTIPLIER, np.asarray(level, dtype=np.float64) - 1.0))


def sponsor_income(levels, prestige: int = 0) -> float:
    """
    Passive credits per second.  Each contract yields ``base * level`` and
    doubles at every milestone level it has reached; the prestige tier scales
    the whole portfolio.
    """
    levels = np.asarray(levels, dtype=np.int64)
    milestones = np.sum(levels[:, None] >= SPONSOR_MILESTONES[None, :], axis=1)
    per_contract = SPONSOR_BASE_INCOME * levels * np.power(2.0, milestones)
    return float(per_contract.sum() * prestige_income_multiplier(prestige))


def prestige_income_multiplier(tier: int) -> float:
    return float(PRESTIGE_INCOME_MULT ** int(tier))


def prestige_combat_multiplier(tier: int) -> float:
    return 1.0 + PRESTIGE_COMBAT_BONUS * int(tier)


def prestige_requirement(tier: int):
    """``(lifetime_credits, player_level)`` needed to ascend from ``tier``."""
    credits = float(exp_cost(PRESTIGE_BASE_REQUIREMENT, PRESTIGE_REQUIREMENT_MULT, tier))
    return credits, PRESTIGE_BASE_LEVEL + PRESTIGE_LEVEL_STEP * int(tier)


def weapon_upgrade_costs(weapon_index: int, levels) -> np.ndarray:
    """Price of the next level on every track of one weapon (``inf`` when maxed)."""
    levels = np.asarray(levels, dtype=np.int64)
    cost = exp_cost(W.UPGRADE_BASE_COST * W.WEAPONS["tier"][weapon_index], W.UPGRADE_COST_MULT, levels)
    return np.where(levels >= W.UPGRADE_MAX, np.inf, np.rint(cost))


def operator_upgrade_costs(levels) -> np.ndarray:
    levels = np.asarray(levels, dtype=np.int64)
    cost = exp_cost(W.OPERATOR_BASE_COST, W.OPERATOR_COST_MULT, levels)
    return np.where(levels >= W.OPERATOR_MAX, np.inf, np.rint(cost))


def sponsor_costs(levels, count: int = 1) -> np.ndarray:
    """Price of buying ``count`` more levels of every sponsor contract."""
    levels = np.asarray(levels, dtype=np.int64)
    cost = bulk_cost(SPONSOR_BASE_COST, SPONSOR_COST_MULT, levels, count)
    return np.where(levels + count > SPONSOR_MAX_LEVEL, np.inf, np.ceil(cost))


def format_number(value: float, digits: int = 2) -> str:
    """Compact human readable number: 1234567 -> '1.23M'."""
    if value != value or value in (float("inf"), float("-inf")):
        return "MAX"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value < 1000.0:
        return sign + ("%d" % int(value) if value >= 100 or float(value).is_integer() else ("%.1f" % value))
    tier = min(int(math.log10(value) // 3), len(_SUFFIXES) - 1)
    scaled = value / (1000.0 ** tier)
    if scaled >= 999.995:
        scaled = 999.99
    return "%s%.*f%s" % (sign, digits, scaled, _SUFFIXES[tier])


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return "%dh %02dm" % (hours, minutes)
    if minutes:
        return "%dm %02ds" % (minutes, secs)
    return "%ds" % secs


# =========================================================================== #
#  Profile data
# =========================================================================== #
DEFAULT_SETTINGS = {
    "sensitivity": 1.0, "fov": C.BASE_FOV, "volume": 0.8, "invert_y": False, "show_fps": True,
    "bot_count": 7, "bot_difficulty": 1, "map": "datacore", "mode": C.MODE_TDM,
}
STAT_KEYS = ("kills", "deaths", "headshots", "shots", "hits", "damage", "matches", "wins",
             "best_streak", "grenade_kills", "playtime", "credits_from_sponsors", "credits_from_combat")


class SaveTamperError(Exception):
    """The save file failed its integrity verification."""


class Profile:
    """Plain data container; all mutation goes through :class:`ProfileManager`."""

    def __init__(self):
        now = time.time()
        self.version = SAVE_VERSION
        self.uid = uuid.uuid4().hex
        self.name = "OPERATOR"
        self.created_at = now
        self.saved_at = now
        self.credits = 0.0
        self.lifetime_credits = 0.0
        self.xp = 0.0
        self.level = 1
        self.prestige = 0
        self.weapons = {key: {"unlocked": bool(W.WEAPONS["unlock_cost"][i] <= 0.0), "levels": [0] * W.UPGRADE_COUNT}
                        for i, key in enumerate(W.WEAPON_KEYS)}
        self.operator = [0] * W.OPERATOR_COUNT
        self.sponsors = [0] * SPONSOR_COUNT
        self.loadout = [W.WEAPON_KEYS[W.DEFAULT_LOADOUT[0]], W.WEAPON_KEYS[W.DEFAULT_LOADOUT[1]]]
        self.stats = {key: 0.0 for key in STAT_KEYS}
        self.settings = dict(DEFAULT_SETTINGS)
        self.integrity_strikes = 0

    def to_dict(self) -> dict:
        return {
            "version": self.version, "uid": self.uid, "name": self.name, "created_at": self.created_at,
            "saved_at": self.saved_at, "credits": self.credits, "lifetime_credits": self.lifetime_credits,
            "xp": self.xp, "level": self.level, "prestige": self.prestige, "weapons": self.weapons,
            "operator": self.operator, "sponsors": self.sponsors, "loadout": self.loadout, "stats": self.stats,
            "settings": self.settings, "integrity_strikes": self.integrity_strikes,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Profile":
        """Rebuild a profile from decoded JSON; every field is type and range checked."""
        p = cls()
        p.uid = str(raw.get("uid", p.uid))[:64]
        p.name = sanitize_name(raw.get("name", p.name))
        p.created_at = float(raw.get("created_at", p.created_at))
        p.saved_at = float(raw.get("saved_at", p.saved_at))
        p.credits = max(float(raw.get("credits", 0.0)), 0.0)
        p.lifetime_credits = max(float(raw.get("lifetime_credits", 0.0)), p.credits)
        p.level = int(np.clip(int(raw.get("level", 1)), 1, MAX_LEVEL))
        p.xp = max(float(raw.get("xp", 0.0)), 0.0)
        p.prestige = int(np.clip(int(raw.get("prestige", 0)), 0, MAX_PRESTIGE))
        stored = raw.get("weapons", {})
        for index, key in enumerate(W.WEAPON_KEYS):
            entry = stored.get(key, {}) if isinstance(stored, dict) else {}
            levels = list(entry.get("levels", [0] * W.UPGRADE_COUNT))[: W.UPGRADE_COUNT]
            levels += [0] * (W.UPGRADE_COUNT - len(levels))
            free = bool(W.WEAPONS["unlock_cost"][index] <= 0.0)
            p.weapons[key] = {"unlocked": bool(entry.get("unlocked", False)) or free,
                              "levels": [int(v) for v in W.clamp_levels(levels)]}
        operator = list(raw.get("operator", []))[: W.OPERATOR_COUNT]
        operator += [0] * (W.OPERATOR_COUNT - len(operator))
        p.operator = [int(v) for v in W.clamp_operator(operator)]
        sponsors = list(raw.get("sponsors", []))[:SPONSOR_COUNT]
        sponsors += [0] * (SPONSOR_COUNT - len(sponsors))
        p.sponsors = [int(np.clip(int(v), 0, SPONSOR_MAX_LEVEL)) for v in sponsors]
        loadout = [key for key in list(raw.get("loadout", []))[: W.LOADOUT_SLOTS]
                   if key in p.weapons and p.weapons[key]["unlocked"]]
        defaults = [W.WEAPON_KEYS[i] for i in W.DEFAULT_LOADOUT]
        while len(loadout) < W.LOADOUT_SLOTS:
            loadout.append(defaults[len(loadout)])
        p.loadout = loadout
        stats = raw.get("stats", {})
        for key in STAT_KEYS:
            p.stats[key] = max(float(stats.get(key, 0.0)), 0.0) if isinstance(stats, dict) else 0.0
        settings = raw.get("settings", {})
        if isinstance(settings, dict):
            for key, default in DEFAULT_SETTINGS.items():
                value = settings.get(key, default)
                p.settings[key] = type(default)(value) if not isinstance(default, str) else str(value)
        p.settings["sensitivity"] = float(np.clip(p.settings["sensitivity"], 0.05, 10.0))
        p.settings["fov"] = float(np.clip(p.settings["fov"], 60.0, 120.0))
        p.settings["volume"] = float(np.clip(p.settings["volume"], 0.0, 1.0))
        p.settings["bot_count"] = int(np.clip(p.settings["bot_count"], 0, C.MAX_PLAYERS - 1))
        p.settings["bot_difficulty"] = int(np.clip(p.settings["bot_difficulty"], 0, 2))
        p.integrity_strikes = max(int(raw.get("integrity_strikes", 0)), 0)
        return p


def sanitize_name(name) -> str:
    cleaned = "".join(ch for ch in str(name) if ch.isalnum() or ch in "-_ .")[:16].strip()
    return cleaned or "OPERATOR"


# =========================================================================== #
#  Signed, sealed serialisation
# =========================================================================== #
def canonical_json(data: dict) -> str:
    """Deterministic JSON text: sorted keys, no insignificant whitespace."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sign_data_string(data_string: str) -> str:
    """Hex SHA-256 of the internal data string followed by the hard-coded salt."""
    return hashlib.sha256((data_string + SAVE_SALT).encode("utf-8")).hexdigest()


def encode_save(profile: Profile) -> str:
    """Profile -> text of a save file (header, sealed body, signature as the last line)."""
    data_string = canonical_json(profile.to_dict())
    sealed = base64.b64encode(crypto.seal(SAVE_KEY, data_string.encode("utf-8"))).decode("ascii")
    body = "\n".join(sealed[i:i + 76] for i in range(0, len(sealed), 76))
    return "%s\n%s\n%s%s\n" % (SAVE_HEADER, body, SIGNATURE_PREFIX, sign_data_string(data_string))


def decode_save(text: str) -> Profile:
    """Verify and decode the text of a save file; raises :class:`SaveTamperError` on any mismatch."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != SAVE_HEADER or not lines[-1].startswith(SIGNATURE_PREFIX):
        raise SaveTamperError("save file layout is not recognised")
    signature = lines[-1][len(SIGNATURE_PREFIX):]
    try:
        sealed = base64.b64decode("".join(lines[1:-1]), validate=True)
        data_string = crypto.unseal(SAVE_KEY, sealed).decode("utf-8")
    except (ValueError, crypto.AuthenticationError, UnicodeDecodeError) as error:
        raise SaveTamperError("save body failed verification: %s" % error) from error
    if not hmac.compare_digest(signature, sign_data_string(data_string)):
        raise SaveTamperError("salted SHA-256 signature mismatch")
    try:
        raw = json.loads(data_string)
    except json.JSONDecodeError as error:
        raise SaveTamperError("save payload is not valid JSON") from error
    if not isinstance(raw, dict):
        raise SaveTamperError("save payload has the wrong shape")
    return Profile.from_dict(raw)


# =========================================================================== #
#  Profile manager (thread safe facade used by the game and the sponsor thread)
# =========================================================================== #
class ProfileManager:
    """Owns the profile, its lock, the economy operations and the save file."""

    def __init__(self, directory=None, filename: str = "profile.sav"):
        self.directory = Path(directory) if directory is not None else C.SAVE_DIR
        self.path = self.directory / filename
        self.backup_path = self.directory / (filename + ".bak")
        self.lock = threading.RLock()
        self.profile = Profile()
        self.load_status = "new"
        self.offline_seconds = 0.0
        self.offline_credits = 0.0
        self.dirty = False
        self.session_started = time.time()

    # ---- persistence -------------------------------------------------------------
    def load(self) -> str:
        """
        Load the profile.  Returns ``'ok'``, ``'new'``, ``'restored_backup'``
        or ``'tampered_reset'`` and applies offline sponsor earnings.
        """
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            status = "new"
            profile = None
            if self.path.exists():
                try:
                    profile = decode_save(self.path.read_text(encoding="utf-8"))
                    status = "ok"
                except (SaveTamperError, OSError):
                    profile = None
                    status = "tampered_reset"
                    self._quarantine(self.path)
            if profile is None and status != "new" and self.backup_path.exists():
                try:
                    profile = decode_save(self.backup_path.read_text(encoding="utf-8"))
                    profile.integrity_strikes += 1
                    status = "restored_backup"
                except (SaveTamperError, OSError):
                    profile = None
                    self._quarantine(self.backup_path)
            if profile is None:
                profile = Profile()
                if status == "tampered_reset":
                    profile.integrity_strikes = 1
            self.profile = profile
            self.load_status = status
            self.offline_seconds = 0.0
            self.offline_credits = 0.0
            if status in ("ok", "restored_backup"):
                away = min(max(time.time() - profile.saved_at, 0.0), OFFLINE_CAP_SECONDS)
                earned = sponsor_income(profile.sponsors, profile.prestige) * away * OFFLINE_EFFICIENCY
                if earned > 0.0:
                    self.offline_seconds = away
                    self.offline_credits = earned
                    self._credit(earned, "credits_from_sponsors")
            self.dirty = True
            self.session_started = time.time()
            return status

    def _quarantine(self, path: Path) -> None:
        target = path.with_name("%s.tampered-%d" % (path.name, int(time.time())))
        try:
            os.replace(path, target)
        except OSError:
            pass

    def save(self) -> bool:
        """Atomically write the signed save (temp file + rename) and rotate the backup."""
        with self.lock:
            now = time.time()
            self.profile.stats["playtime"] += max(now - self.session_started, 0.0)
            self.session_started = now
            self.profile.saved_at = now
            text = encode_save(self.profile)
            self.dirty = False
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_name(self.path.name + ".tmp")
            with open(temp, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            if self.path.exists():
                os.replace(self.path, self.backup_path)
            os.replace(temp, self.path)
            return True
        except OSError:
            return False

    # ---- low level credit / experience -----------------------------------------------
    def _credit(self, amount: float, stat_key: str | None = None) -> None:
        p = self.profile
        p.credits += amount
        p.lifetime_credits += amount
        if stat_key is not None:
            p.stats[stat_key] += amount
        self.dirty = True

    def add_credits(self, amount: float, combat: bool = False) -> float:
        """Grant credits; combat income is scaled by Bounty Uplink and the prestige tier."""
        with self.lock:
            if amount <= 0.0:
                return 0.0
            if combat:
                amount *= self.combat_multiplier()
            self._credit(float(amount), "credits_from_combat" if combat else None)
            return float(amount)

    def add_xp(self, amount: float) -> int:
        """Grant experience; returns the number of levels gained (each pays a credit bonus)."""
        with self.lock:
            p = self.profile
            gained = 0
            p.xp += max(float(amount), 0.0)
            while p.level < MAX_LEVEL and p.xp >= float(xp_required(p.level)):
                p.xp -= float(xp_required(p.level))
                p.level += 1
                gained += 1
                self._credit(LEVEL_UP_CREDITS * p.level)
            if p.level >= MAX_LEVEL:
                p.xp = min(p.xp, float(xp_required(MAX_LEVEL)))
            self.dirty = True
            return gained

    def accrue(self, dt: float) -> float:
        """Credit ``dt`` seconds of sponsor income (called by the sponsor thread)."""
        with self.lock:
            earned = sponsor_income(self.profile.sponsors, self.profile.prestige) * max(float(dt), 0.0)
            if earned > 0.0:
                self._credit(earned, "credits_from_sponsors")
            return earned

    def spend(self, amount: float) -> bool:
        with self.lock:
            if not math.isfinite(amount) or amount < 0.0 or self.profile.credits + 1.0e-6 < amount:
                return False
            self.profile.credits = max(self.profile.credits - amount, 0.0)
            self.dirty = True
            return True

    # ---- derived values ----------------------------------------------------------------
    def income_per_second(self) -> float:
        with self.lock:
            return sponsor_income(self.profile.sponsors, self.profile.prestige)

    def combat_multiplier(self) -> float:
        with self.lock:
            bounty = 1.0 + W.OPERATOR_EFFECT[W.OP_BOUNTY] * self.profile.operator[W.OP_BOUNTY]
            return float(bounty * prestige_combat_multiplier(self.profile.prestige))

    def xp_progress(self):
        with self.lock:
            need = float(xp_required(self.profile.level))
            return self.profile.xp, need

    def snapshot(self) -> dict:
        """Consistent copy of the headline numbers for the HUD / menus."""
        with self.lock:
            p = self.profile
            return {"credits": p.credits, "lifetime": p.lifetime_credits, "level": p.level, "xp": p.xp,
                    "xp_need": float(xp_required(p.level)), "prestige": p.prestige,
                    "income": sponsor_income(p.sponsors, p.prestige), "name": p.name}

    # ---- weapons --------------------------------------------------------------------------
    def weapon_costs(self, key: str) -> np.ndarray:
        with self.lock:
            return weapon_upgrade_costs(W.WEAPON_INDEX[key], self.profile.weapons[key]["levels"])

    def buy_weapon_upgrade(self, key: str, track: int) -> bool:
        with self.lock:
            entry = self.profile.weapons[key]
            if not entry["unlocked"]:
                return False
            cost = float(self.weapon_costs(key)[track])
            if not self.spend(cost):
                return False
            entry["levels"][track] += 1
            return True

    def unlock_cost(self, key: str) -> float:
        return float(W.WEAPONS["unlock_cost"][W.WEAPON_INDEX[key]])

    def can_unlock(self, key: str) -> bool:
        with self.lock:
            index = W.WEAPON_INDEX[key]
            return (not self.profile.weapons[key]["unlocked"]
                    and self.profile.level >= int(W.WEAPONS["unlock_level"][index])
                    and self.profile.credits >= self.unlock_cost(key))

    def unlock_weapon(self, key: str) -> bool:
        with self.lock:
            if not self.can_unlock(key) or not self.spend(self.unlock_cost(key)):
                return False
            self.profile.weapons[key]["unlocked"] = True
            return True

    def equip(self, slot: int, key: str) -> bool:
        with self.lock:
            if key not in self.profile.weapons or not self.profile.weapons[key]["unlocked"]:
                return False
            other = 1 - slot
            if self.profile.loadout[other] == key:
                self.profile.loadout[other] = self.profile.loadout[slot]
            self.profile.loadout[slot] = key
            self.dirty = True
            return True

    def loadout_arrays(self):
        """``(weapon_ids, levels, operator, prestige)`` ready for the network handshake."""
        with self.lock:
            p = self.profile
            ids = np.array([W.WEAPON_INDEX[key] for key in p.loadout], dtype=np.int64)
            levels = np.array([p.weapons[key]["levels"] for key in p.loadout], dtype=np.int64)
            return ids, levels, np.array(p.operator, dtype=np.int64), int(p.prestige)

    def loadout_blob(self) -> bytes:
        return W.encode_loadout(*self.loadout_arrays())

    # ---- operator upgrades --------------------------------------------------------------------
    def operator_costs(self) -> np.ndarray:
        with self.lock:
            return operator_upgrade_costs(self.profile.operator)

    def buy_operator_upgrade(self, index: int) -> bool:
        with self.lock:
            if not self.spend(float(self.operator_costs()[index])):
                return False
            self.profile.operator[index] += 1
            return True

    # ---- sponsors ---------------------------------------------------------------------------------
    def sponsor_cost(self, index: int, count: int = 1) -> float:
        with self.lock:
            return float(sponsor_costs(self.profile.sponsors, count)[index])

    def sponsor_max_affordable(self, index: int) -> int:
        with self.lock:
            level = self.profile.sponsors[index]
            n = int(max_affordable(SPONSOR_BASE_COST[index], SPONSOR_COST_MULT[index], level, self.profile.credits))
            return max(min(n, SPONSOR_MAX_LEVEL - level), 0)

    def buy_sponsor(self, index: int, count: int = 1) -> bool:
        with self.lock:
            if count <= 0 or not self.spend(self.sponsor_cost(index, count)):
                return False
            self.profile.sponsors[index] += int(count)
            return True

    # ---- prestige -------------------------------------------------------------------------------------
    def can_prestige(self) -> bool:
        with self.lock:
            p = self.profile
            credits, level = prestige_requirement(p.prestige)
            return p.prestige < MAX_PRESTIGE and p.lifetime_credits >= credits and p.level >= level

    def do_prestige(self) -> bool:
        """Ascend one corporate tier: progress resets, permanent multipliers grow."""
        with self.lock:
            if not self.can_prestige():
                return False
            p = self.profile
            p.prestige += 1
            p.credits = 0.0
            p.lifetime_credits = 0.0
            p.xp = 0.0
            p.level = 1
            p.sponsors = [0] * SPONSOR_COUNT
            p.operator = [0] * W.OPERATOR_COUNT
            for index, key in enumerate(W.WEAPON_KEYS):
                p.weapons[key] = {"unlocked": bool(W.WEAPONS["unlock_cost"][index] <= 0.0),
                                  "levels": [0] * W.UPGRADE_COUNT}
            p.loadout = [W.WEAPON_KEYS[i] for i in W.DEFAULT_LOADOUT]
            self.dirty = True
            return True

    # ---- match rewards and statistics ---------------------------------------------------------------------
    def apply_reward(self, credits: float, xp: float) -> tuple:
        """Combat reward from the server: returns ``(credits_granted, levels_gained)``."""
        granted = self.add_credits(credits, combat=True)
        return granted, self.add_xp(xp)

    def record(self, stat: str, amount: float = 1.0, maximum: bool = False) -> None:
        with self.lock:
            if stat in self.profile.stats:
                current = self.profile.stats[stat]
                self.profile.stats[stat] = max(current, amount) if maximum else current + amount
                self.dirty = True

    def set_setting(self, key: str, value) -> None:
        with self.lock:
            if key in self.profile.settings:
                self.profile.settings[key] = value
                self.dirty = True

    def set_name(self, name: str) -> None:
        with self.lock:
            self.profile.name = sanitize_name(name)
            self.dirty = True


# =========================================================================== #
#  Corporate Sponsor Thread
# =========================================================================== #
class CorporateSponsorThread(threading.Thread):
    """
    Background worker that turns elapsed real time into passive income.

    Each wake-up reads ``time.perf_counter_ns`` (monotonic, nanosecond
    resolution), credits ``income * dt`` under the profile lock and writes an
    autosave every ``autosave_interval`` seconds.  It never touches the render
    thread, so the balance grows identically in matches and in menus.
    """

    def __init__(self, manager: ProfileManager, interval: float = 0.25,
                 autosave_interval: float = AUTOSAVE_INTERVAL):
        super().__init__(name="CorporateSponsorThread", daemon=True)
        self.manager = manager
        self.interval = float(interval)
        self.autosave_interval = float(autosave_interval)
        self._halt = threading.Event()
        self.total_accrued = 0.0
        self.wakeups = 0

    def run(self) -> None:
        last = time.perf_counter_ns()
        last_save = last
        while not self._halt.wait(self.interval):
            now = time.perf_counter_ns()
            dt = (now - last) * 1.0e-9
            last = now
            self.total_accrued += self.manager.accrue(dt)
            self.wakeups += 1
            if (now - last_save) * 1.0e-9 >= self.autosave_interval and self.manager.dirty:
                self.manager.save()
                last_save = now
        # credit the partial interval between the last wake-up and the stop request
        self.total_accrued += self.manager.accrue((time.perf_counter_ns() - last) * 1.0e-9)

    def stop(self, save: bool = True) -> None:
        self._halt.set()
        if self.is_alive():
            self.join(timeout=2.0)
        if save:
            self.manager.save()
