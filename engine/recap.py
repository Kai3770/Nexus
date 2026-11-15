"""
NEXUS-3D-EXTREME :: engine/recap.py
===================================
Death recap.  The client logs the damage it took and dealt since its last
spawn; when it dies the log becomes two lines for the death screen: who
killed you, with what, from how far, and how the exchange went.  Pure Python
(no engine imports), so it is tested directly.
"""
from __future__ import annotations


def _times(count: int) -> str:
    return "%d time%s" % (count, "" if count == 1 else "s")


class CombatLog:
    """Damage taken per attacker and dealt per victim during one life."""

    def __init__(self):
        self.taken: dict = {}      # attacker -> [hits, damage, headshots]
        self.dealt: dict = {}      # victim -> [hits, damage]

    def reset(self) -> None:
        self.taken.clear()
        self.dealt.clear()

    def took(self, attacker: int, damage: float, headshot: bool = False) -> None:
        entry = self.taken.setdefault(int(attacker), [0, 0.0, 0])
        entry[0] += 1
        entry[1] += float(damage)
        entry[2] += int(bool(headshot))

    def gave(self, victim: int, damage: float) -> None:
        entry = self.dealt.setdefault(int(victim), [0, 0.0])
        entry[0] += 1
        entry[1] += float(damage)

    def recap(self, killer: int, killer_name: str, cause: str, distance=None, killer_health=None,
              headshot: bool = False, suicide: bool = False) -> str:
        """
        ``killer`` is a player id (negative for the arena itself), ``cause``
        the weapon / equipment / reward name, ``distance`` in metres and
        ``killer_health`` the killer's vitals as last seen (None if unknown).
        """
        if suicide or killer < 0:
            first = cause or "the arena"
            line = []
        else:
            first = "  -  ".join([cause] + (["%d m" % round(distance)] if distance is not None else [])
                                 + (["HEADSHOT"] if headshot else []))
            hits, damage, _heads = self.taken.get(killer, (0, 0.0, 0))
            dealt_hits, dealt = self.dealt.get(killer, (0, 0.0))
            line = ["%s hit you %s for %d" % (killer_name, _times(hits), round(damage)) if hits
                    else "%s finished you" % killer_name]
            line.append("you hit them %s for %d" % (_times(dealt_hits), round(dealt)) if dealt_hits
                        else "you never hit them")
            if killer_health is not None and killer_health > 0.0:
                line.append("they have %d HP left" % round(killer_health))
        others = sum(entry[1] for attacker, entry in self.taken.items() if attacker != killer)
        if others >= 1.0:
            line.append("%d damage from others" % round(others))
        return first + ("\n" + "  -  ".join(line) if line else "")
