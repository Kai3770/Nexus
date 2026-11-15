"""
NEXUS-3D-EXTREME :: engine/report.py
====================================
Medals and the match report.  The client feeds what happens to its own
player during a match; medals are handed out the moment they are earned
(for the HUD popups) and ``summary`` turns the match into the report shown
next to the scoreboard during the intermission.  Pure Python, so it is
tested directly.
"""
from __future__ import annotations

from . import config as C

_COL = {name: i for i, name in enumerate(C.MEDAL_COLUMNS)}
MEDAL_KEYS = tuple(row[_COL["key"]] for row in C.MEDAL_TABLE)
MEDAL_NAMES = {row[_COL["key"]]: row[_COL["name"]] for row in C.MEDAL_TABLE}
MEDAL_DESCRIPTIONS = {row[_COL["key"]]: row[_COL["description"]] for row in C.MEDAL_TABLE}


def _number(value: float) -> str:
    value = float(value)
    if abs(value) >= 1000.0:
        return "%.1fK" % (value / 1000.0)
    return "%d" % round(value)


class MatchReport:
    """Everything the local player did in one match."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.shots = self.hits = self.headshots = 0
        self.kills = self.deaths = self.streak = self.best_streak = 0
        self.equipment_kills = self.reward_kills = self.captures = 0
        self.damage = 0.0
        self.credits = self.xp = 0
        self.medals: list = []                   # keys in the order they were earned
        self.match_kills = 0                     # eliminations anybody scored (FIRST STRIKE)
        self.nemesis = -1                        # who eliminated us last (PAYBACK)
        self.chain = 0
        self.last_kill = -1.0e9

    # ---- feed --------------------------------------------------------------------------------------------
    def _award(self, keys: list) -> list:
        self.medals.extend(keys)
        return keys

    def shot(self) -> None:
        self.shots += 1

    def hit(self, damage: float, headshot: bool = False) -> list:
        self.hits += 1
        self.damage += float(damage)
        if headshot:
            self.headshots += 1
            if self.headshots == 5:
                return self._award(["headhunter"])
        return []

    def kill(self, killer: int, victim: int, me: int, now: float, explosive: bool = False, reward: bool = False,
             suicide: bool = False) -> list:
        """Every elimination of the match (anybody's); returns the medals it earned the local player."""
        hostile = not suicide and killer >= 0 and killer != victim
        first = hostile and self.match_kills == 0
        if hostile:
            self.match_kills += 1
        if victim == me:
            self.deaths += 1
            self.streak = 0
            self.chain = 0
            if hostile:
                self.nemesis = killer
        if not (hostile and killer == me):
            return []
        earned = []
        self.kills += 1
        self.streak += 1
        self.best_streak = max(self.best_streak, self.streak)
        if reward:
            self.reward_kills += 1
            earned.append("starfall")
        elif explosive:
            self.equipment_kills += 1
            if self.equipment_kills == 3:
                earned.append("demolition")
        self.chain = self.chain + 1 if now - self.last_kill <= C.MEDAL_CHAIN_TIME else 1
        self.last_kill = now
        if self.chain == 2:
            earned.append("double")
        elif self.chain == 3:
            earned.append("triple")
        if first:
            earned.append("first_strike")
        if self.streak == 5:
            earned.append("iron_will")
        elif self.streak == 10:
            earned.append("overdrive")
        if victim == self.nemesis:
            earned.append("payback")
            self.nemesis = -1
        return self._award(earned)

    def capture(self) -> list:
        self.captures += 1
        return self._award(["uplinker"]) if self.captures == 3 else []

    def reward(self, credits: int, xp: int) -> None:
        self.credits += int(credits)
        self.xp += int(xp)

    def finish(self, mvp: int, me: int) -> list:
        """Match over: the medals only the whole match can earn."""
        earned = []
        if self.shots >= 30 and self.hits >= 0.5 * self.shots:
            earned.append("sharpshooter")
        if mvp == me:
            earned.append("mvp")
        return self._award(earned)

    # ---- report ------------------------------------------------------------------------------------------
    @property
    def accuracy(self) -> float:
        return self.hits / self.shots if self.shots else 0.0

    def medal_counts(self) -> list:
        """``[(name, count)]`` in the order first earned."""
        counts: dict = {}
        for key in self.medals:
            counts[key] = counts.get(key, 0) + 1
        return [(MEDAL_NAMES[key], n) for key, n in counts.items()]

    def summary(self, mvp_name: str = "", mvp_score: int = 0, uplinks: bool = False) -> str:
        ratio = self.kills / max(self.deaths, 1)
        lines = []
        if mvp_name:
            lines += ["MVP   %s   %s" % (mvp_name, _number(mvp_score)), ""]
        lines.append("ELIMINATIONS %d   DEATHS %d   K/D %.1f" % (self.kills, self.deaths, ratio))
        lines.append("ACCURACY %d%%   (%d of %d)" % (round(self.accuracy * 100.0), self.hits, self.shots) if self.shots
                     else "ACCURACY -")
        lines.append("HEADSHOTS %d   BEST STREAK %d" % (self.headshots, self.best_streak))
        lines.append("DAMAGE %s" % _number(self.damage) + ("   CAPTURES %d" % self.captures if uplinks else ""))
        lines.append("+%s CREDITS   +%s XP" % (_number(self.credits), _number(self.xp)))
        medals = self.medal_counts()
        lines += ["", "MEDALS" if medals else "NO MEDALS THIS TIME"]
        row = []
        for name, count in medals:
            row.append(name + (" x%d" % count if count > 1 else ""))
            if len(row) == 2:
                lines.append("   ".join(row))
                row = []
        if row:
            lines.append("   ".join(row))
        return "\n".join(lines)
