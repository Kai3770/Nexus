"""
NEXUS-3D-EXTREME :: engine/menus.py
===================================
Front-end screens on Ursina's UI layer: main menu, match setup, zero-config
LAN lobby, armory (weapon unlocks / upgrades / loadout / operator upgrades),
sponsor contracts and prestige, settings and the in-match pause overlay.

The screens are thin: every number they show comes from
:class:`engine.secure_save.ProfileManager`, every purchase is one manager call,
and the dynamic labels are refreshed four times per second so the passive
income credited by the Corporate Sponsor Thread is visibly ticking.
"""
from __future__ import annotations

from ursina import Button, Entity, Text, camera, color, window
from ursina.prefabs.input_field import InputField

from . import config as C
from . import maps
from . import secure_save as SS
from . import weapons as W
from .bots import DIFFICULTY_NAMES

COL_TEXT = (0.92, 0.95, 1.00)
COL_DIM = (0.58, 0.64, 0.72)
COL_ACCENT = (0.25, 0.90, 1.00)
COL_GOLD = (1.00, 0.80, 0.25)
COL_GOOD = (0.14, 0.42, 0.26)
COL_BAD = (0.20, 0.21, 0.25)
COL_BUTTON = (0.10, 0.13, 0.18)
COL_DANGER = (0.48, 0.12, 0.14)
REFRESH_INTERVAL = 0.25


def _c(rgb, alpha: float = 1.0):
    return color.rgba(rgb[0], rgb[1], rgb[2], alpha)


def money(value: float) -> str:
    return "CR " + SS.format_number(value)


class Screen:
    """Base class: a root entity plus small factories for labels, buttons and panels."""

    title = ""

    def __init__(self, menu: "MenuManager"):
        self.menu = menu
        self.game = menu.game
        self.pm = menu.game.pm
        self.root = Entity(parent=camera.ui, enabled=False)
        self.half_width = float(window.aspect_ratio) * 0.5
        if self.title:
            self.label(self.title, 0.0, 0.43, 2.0, (0, 0), COL_ACCENT)
        self.build()

    # ---- widget factories ---------------------------------------------------------------
    def label(self, text, x, y, scale=1.0, origin=(-0.5, 0), rgb=COL_TEXT) -> Text:
        return Text(str(text), parent=self.root, position=(x, y), scale=scale, origin=origin, color=_c(rgb))

    def panel(self, x, y, w, h, rgb=(0.04, 0.05, 0.08), alpha=0.82) -> Entity:
        return Entity(parent=self.root, model="quad", color=_c(rgb, alpha), position=(x, y, 0.5), scale=(w, h))

    def button(self, text, x, y, w, h, action, rgb=COL_BUTTON, text_size=0.9) -> Button:
        def clicked():
            self.game.play_ui("ui_click")
            action()
        return Button(text=str(text), parent=self.root, position=(x, y), scale=(w, h), color=_c(rgb, 0.95),
                      text_size=text_size, text_color=_c(COL_TEXT), on_click=clicked, radius=0.12)

    def back_button(self, target: str = "main") -> Button:
        return self.button("BACK", -self.half_width + 0.12, -0.44, 0.16, 0.05, lambda: self.menu.show(target))

    @staticmethod
    def set_text(widget, text: str) -> None:
        if getattr(widget, "_shown", None) != text:
            widget._shown = text
            widget.text = text

    @staticmethod
    def tint(button: Button, rgb) -> None:
        if getattr(button, "_tint", None) != rgb:
            button._tint = rgb
            button.color = _c(rgb, 0.95)
            button.highlight_color = button.color.tint(0.15)
            button.pressed_color = button.color.tint(-0.15)

    # ---- overridables ----------------------------------------------------------------------
    def build(self) -> None:
        """Create the widgets (called once)."""

    def refresh(self) -> None:
        """Update dynamic labels (called on show and four times per second)."""

    def on_show(self) -> None:
        """Hook that runs every time the screen becomes visible."""


# =========================================================================== #
#  Main menu
# =========================================================================== #
class MainScreen(Screen):
    def build(self) -> None:
        self.label(C.GAME_TITLE, 0.0, 0.36, 3.2, (0, 0), COL_ACCENT)
        self.label("LAN ARENA SHOOTER  //  INCREMENTAL CORPORATE WARFARE", 0.0, 0.295, 0.85, (0, 0), COL_DIM)
        entries = (("PLAY SOLO", lambda: self.menu.open_setup(False)), ("HOST LAN GAME", lambda: self.menu.open_setup(True)),
                   ("JOIN LAN GAME", lambda: self.menu.show("lobby")), ("ARMORY", lambda: self.menu.show("armory")),
                   ("SPONSORS", lambda: self.menu.show("sponsors")), ("SETTINGS", lambda: self.menu.show("settings")),
                   ("QUIT", self.game.quit))
        for index, (text, action) in enumerate(entries):
            self.button(text, -0.42, 0.17 - index * 0.075, 0.36, 0.06, action, COL_DANGER if text == "QUIT" else COL_BUTTON, 1.0)
        self.panel(0.30, -0.065, 0.66, 0.50)
        self.name_text = self.label("", 0.0, 0.145, 1.5)
        self.level_text = self.label("", 0.0, 0.095, 0.95, rgb=COL_DIM)
        self.credit_text = self.label("", 0.0, 0.035, 1.6, rgb=COL_GOLD)
        self.income_text = self.label("", 0.0, -0.015, 0.95, rgb=COL_GOLD)
        self.prestige_text = self.label("", 0.0, -0.065, 0.9, rgb=COL_DIM)
        self.record_text = self.label("", 0.0, -0.125, 0.85, rgb=COL_DIM)
        self.notice_text = self.label("", 0.0, -0.215, 0.85, rgb=COL_ACCENT)
        self.label("v%s   //   beacon UDP %d   //   %d Hz rollback netcode" % (C.GAME_VERSION, C.DISCOVERY_PORT, C.TICK_RATE),
                   0.0, -0.47, 0.6, (0, 0), COL_DIM)

    def refresh(self) -> None:
        snap = self.pm.snapshot()
        stats = self.pm.profile.stats
        self.set_text(self.name_text, snap["name"])
        self.set_text(self.level_text, "LEVEL %d     XP %s / %s" % (snap["level"], SS.format_number(snap["xp"]),
                                                                  SS.format_number(snap["xp_need"])))
        self.set_text(self.credit_text, money(snap["credits"]))
        self.set_text(self.income_text, "SPONSOR INCOME  +%s / s" % SS.format_number(snap["income"]))
        self.set_text(self.prestige_text, "CORPORATE PRESTIGE TIER %d    (income x%.2f, combat x%.2f)" % (
            snap["prestige"], SS.prestige_income_multiplier(snap["prestige"]), SS.prestige_combat_multiplier(snap["prestige"])))
        self.set_text(self.record_text, "ELIMINATIONS %d    DEATHS %d    HEADSHOTS %d    MATCHES %d    WINS %d" % (
            stats["kills"], stats["deaths"], stats["headshots"], stats["matches"], stats["wins"]))
        self.set_text(self.notice_text, self.menu.notice)


# =========================================================================== #
#  Match setup (solo / host)
# =========================================================================== #
class SetupScreen(Screen):
    title = "MATCH SETUP"

    def build(self) -> None:
        self.host = False
        settings = self.pm.profile.settings
        self.map_index = maps.map_index(settings["map"]) if settings["map"] in maps.MAP_KEYS else 0
        self.mode = int(settings["mode"])
        self.bots = int(settings["bot_count"])
        self.difficulty = int(settings["bot_difficulty"])
        self.panel(0.0, 0.03, 0.9, 0.58)
        self.rows = {}
        for index, key in enumerate(("ARENA", "MODE", "BOTS", "DIFFICULTY")):
            y = 0.22 - index * 0.09
            self.label(key, -0.38, y, 1.05, rgb=COL_DIM)
            self.button("-", 0.02, y, 0.05, 0.055, lambda k=key: self.step(k, -1))
            self.rows[key] = self.label("", 0.21, y, 1.1, (0, 0))
            self.button("+", 0.40, y, 0.05, 0.055, lambda k=key: self.step(k, 1))
        self.info = self.label("", 0.0, -0.135, 0.8, (0, 0), COL_DIM)
        self.kind = self.label("", 0.0, 0.335, 0.9, (0, 0), COL_DIM)
        self.button("START MATCH", 0.0, -0.215, 0.36, 0.065, self.start, COL_GOOD, 1.1)
        self.back_button()

    def step(self, key: str, delta: int) -> None:
        if key == "ARENA":
            self.map_index = (self.map_index + delta) % maps.MAP_COUNT
        elif key == "MODE":
            self.mode = (self.mode + delta) % len(C.MODE_NAMES)
        elif key == "BOTS":
            self.bots = min(max(self.bots + delta, 0), C.MAX_PLAYERS - 1)
        else:
            self.difficulty = (self.difficulty + delta) % len(DIFFICULTY_NAMES)
        self.refresh()

    def refresh(self) -> None:
        self.set_text(self.rows["ARENA"], maps.MAP_NAMES[self.map_index])
        self.set_text(self.rows["MODE"], C.MODE_NAMES[self.mode])
        self.set_text(self.rows["BOTS"], str(self.bots))
        self.set_text(self.rows["DIFFICULTY"], DIFFICULTY_NAMES[self.difficulty])
        self.set_text(self.info, maps.get_map_data(self.map_index).description)
        self.set_text(self.kind, "LAN HOST: colleagues on this network will see the match in their lobby" if self.host
                      else "SOLO: a private server on this machine, bots only")

    def start(self) -> None:
        for key, value in (("map", maps.MAP_KEYS[self.map_index]), ("mode", self.mode), ("bot_count", self.bots),
                           ("bot_difficulty", self.difficulty)):
            self.pm.set_setting(key, value)
        self.game.start_match(self.host, maps.MAP_KEYS[self.map_index], self.mode, self.bots, self.difficulty)


# =========================================================================== #
#  LAN lobby
# =========================================================================== #
class LobbyScreen(Screen):
    title = "LAN LOBBY"
    ROWS = 7

    def build(self) -> None:
        self.status = self.label("", 0.0, 0.365, 0.8, (0, 0), COL_DIM)
        self.panel(0.0, 0.03, 1.36, 0.60)
        self.label("SERVER", -0.64, 0.285, 0.75, rgb=COL_DIM)
        self.label("ARENA", -0.14, 0.285, 0.75, rgb=COL_DIM)
        self.label("MODE", 0.12, 0.285, 0.75, rgb=COL_DIM)
        self.label("PLAYERS", 0.46, 0.285, 0.75, rgb=COL_DIM)
        self.entries = []
        self.infos: list = []
        for index in range(self.ROWS):
            y = 0.225 - index * 0.072
            button = self.button("", 0.0, y, 1.32, 0.062, lambda i=index: self.join(i))
            texts = [self.label("", x, y, 0.95) for x in (-0.64, -0.14, 0.12, 0.46)]
            for text in texts:
                text.z = -0.1
            self.entries.append((button, texts))
        self.back_button()

    def on_show(self) -> None:
        self.game.ensure_browser()

    def join(self, index: int) -> None:
        if index < len(self.infos):
            info = self.infos[index]
            if info.compatible:
                self.game.join_match(info.address, info.port)
            else:
                self.menu.notify("That server runs a different game version")

    def refresh(self) -> None:
        browser = self.game.browser
        self.infos = browser.servers()[: self.ROWS] if browser is not None else []
        if browser is not None and browser.error:
            self.set_text(self.status, browser.error)
        else:
            self.set_text(self.status, "Listening for beacons on UDP %d ...  %d server(s) found" % (
                C.DISCOVERY_PORT, len(self.infos)))
        for index, (button, texts) in enumerate(self.entries):
            if index < len(self.infos):
                info = self.infos[index]
                values = (info.name, info.map_name, info.mode_name,
                          "%d / %d  (+%d bots)%s" % (info.players, info.max_players, info.bots,
                                                     "" if info.compatible else "  VERSION"))
                button.enabled = True
            else:
                values = ("", "", "", "")
                button.enabled = False
            for text, value in zip(texts, values):
                self.set_text(text, value)


# =========================================================================== #
#  Armory
# =========================================================================== #
class ArmoryScreen(Screen):
    title = "ARMORY"

    def build(self) -> None:
        hw = self.half_width
        self.selected = W.WEAPON_KEYS[0]
        self.wallet = self.label("", hw - 0.04, 0.43, 1.2, (0.5, 0), COL_GOLD)
        self.weapon_buttons = {}
        for index, key in enumerate(W.WEAPON_KEYS):
            self.weapon_buttons[key] = self.button("", -hw + 0.20, 0.33 - index * 0.066, 0.34, 0.056,
                                                   lambda k=key: self.select(k), text_size=0.75)
        self.panel(0.16, 0.0, 1.02, 0.78)
        left = -0.33
        self.name = self.label("", left, 0.355, 1.5)
        self.klass = self.label("", left, 0.315, 0.75, rgb=COL_DIM)
        self.desc = self.label("", left, 0.280, 0.75, rgb=COL_DIM)
        self.stats = self.label("", left, 0.240, 0.78)
        self.action_a = self.button("", 0.44, 0.345, 0.26, 0.05, lambda: self.act(0), COL_GOOD, 0.75)
        self.action_b = self.button("", 0.44, 0.285, 0.26, 0.05, lambda: self.act(1), COL_GOOD, 0.75)
        self.upgrade_rows = []
        for track in range(W.UPGRADE_COUNT):
            y = 0.185 - track * 0.047
            name = self.label(W.UPGRADE_NAMES[track], left, y, 0.8)
            level = self.label("", left + 0.26, y, 0.8, rgb=COL_ACCENT)
            self.label(W.UPGRADE_DESCRIPTIONS[track], left + 0.38, y, 0.62, rgb=COL_DIM)
            buy = self.button("", 0.56, y, 0.17, 0.040, lambda t=track: self.buy_upgrade(t), COL_GOOD, 0.7)
            self.upgrade_rows.append((name, level, buy))
        self.label("OPERATOR UPGRADES  (apply to every loadout)", left, -0.205, 0.8, rgb=COL_ACCENT)
        self.operator_rows = []
        for index in range(W.OPERATOR_COUNT):
            y = -0.250 - index * 0.045
            self.label(W.OPERATOR_NAMES[index], left, y, 0.78)
            level = self.label("", left + 0.30, y, 0.78, rgb=COL_ACCENT)
            self.label(W.OPERATOR_DESCRIPTIONS[index], left + 0.42, y, 0.62, rgb=COL_DIM)
            buy = self.button("", 0.56, y, 0.17, 0.038, lambda i=index: self.buy_operator(i), COL_GOOD, 0.7)
            self.operator_rows.append((level, buy))
        self.back_button()

    def select(self, key: str) -> None:
        self.selected = key
        self.refresh()

    def act(self, which: int) -> None:
        entry = self.pm.profile.weapons[self.selected]
        if not entry["unlocked"]:
            if which == 0:
                self.finish(self.pm.unlock_weapon(self.selected), "Weapon unlocked", "Not enough credits or level")
        else:
            self.finish(self.pm.equip(which, self.selected), "Loadout updated", "Cannot equip")

    def buy_upgrade(self, track: int) -> None:
        self.finish(self.pm.buy_weapon_upgrade(self.selected, track), "", "Cannot afford that upgrade")

    def buy_operator(self, index: int) -> None:
        self.finish(self.pm.buy_operator_upgrade(index), "", "Cannot afford that upgrade")

    def finish(self, ok: bool, good: str, bad: str) -> None:
        self.game.play_ui("ui_buy" if ok else "ui_error")
        if ok:
            self.pm.save()
        if (good if ok else bad):
            self.menu.notify(good if ok else bad)
        self.refresh()

    def refresh(self) -> None:
        profile = self.pm.profile
        credits = profile.credits
        self.set_text(self.wallet, "%s    LEVEL %d" % (money(credits), profile.level))
        for key, button in self.weapon_buttons.items():
            index = W.WEAPON_INDEX[key]
            entry = profile.weapons[key]
            if key in profile.loadout:
                tag = "PRIMARY" if profile.loadout[0] == key else "SECONDARY"
            elif entry["unlocked"]:
                tag = "OWNED"
            else:
                tag = "LOCKED  L%d" % int(W.WEAPONS["unlock_level"][index])
            self.set_text(button, "%s   [%s]" % (W.WEAPON_NAMES[index], tag))
            self.tint(button, (0.10, 0.30, 0.40) if key == self.selected else COL_BUTTON)
        key = self.selected
        index = W.WEAPON_INDEX[key]
        entry = profile.weapons[key]
        stats = W.derive_stats(index, entry["levels"])
        self.set_text(self.name, W.WEAPON_NAMES[index])
        self.set_text(self.klass, W.CLASS_NAMES[int(W.WEAPONS["klass"][index])])
        self.set_text(self.desc, W.WEAPON_DESCRIPTIONS[index])
        rpm = 60.0 * C.TICK_RATE / float(stats["interval"])
        self.set_text(self.stats, "DMG %.1f%s    RPM %.0f    MAG %d    RELOAD %.2fs    RANGE %.0f-%.0fm    MOBILITY %.0f%%" % (
            float(stats["damage"]), (" x%d" % int(stats["pellets"])) if int(stats["pellets"]) > 1 else "", rpm,
            int(stats["mag"]), float(stats["reload_ticks"]) / C.TICK_RATE, float(stats["range_start"]),
            float(stats["range_end"]), float(stats["mobility"]) * 100.0))
        if not entry["unlocked"]:
            cost = self.pm.unlock_cost(key)
            self.action_a.enabled = True
            self.action_b.enabled = False
            self.set_text(self.action_a, "UNLOCK  %s" % money(cost))
            self.tint(self.action_a, COL_GOOD if self.pm.can_unlock(key) else COL_BAD)
        else:
            self.action_a.enabled = True
            self.action_b.enabled = True
            self.set_text(self.action_a, "EQUIP AS PRIMARY")
            self.set_text(self.action_b, "EQUIP AS SECONDARY")
            self.tint(self.action_a, COL_GOOD)
            self.tint(self.action_b, COL_GOOD)
        costs = self.pm.weapon_costs(key)
        for track, (_, level, buy) in enumerate(self.upgrade_rows):
            current = entry["levels"][track]
            self.set_text(level, "LV %d / %d" % (current, int(W.UPGRADE_MAX[track])))
            cost = float(costs[track])
            maxed = cost == float("inf")
            self.set_text(buy, "MAX" if maxed else ("LOCKED" if not entry["unlocked"] else money(cost)))
            self.tint(buy, COL_GOOD if (entry["unlocked"] and not maxed and credits >= cost) else COL_BAD)
        op_costs = self.pm.operator_costs()
        for index, (level, buy) in enumerate(self.operator_rows):
            self.set_text(level, "LV %d / %d" % (profile.operator[index], int(W.OPERATOR_MAX[index])))
            cost = float(op_costs[index])
            maxed = cost == float("inf")
            self.set_text(buy, "MAX" if maxed else money(cost))
            self.tint(buy, COL_GOOD if (not maxed and credits >= cost) else COL_BAD)


# =========================================================================== #
#  Sponsors and prestige
# =========================================================================== #
class SponsorScreen(Screen):
    title = "CORPORATE SPONSORS"

    def build(self) -> None:
        hw = self.half_width
        self.wallet = self.label("", hw - 0.04, 0.43, 1.2, (0.5, 0), COL_GOLD)
        self.income = self.label("", 0.0, 0.365, 1.0, (0, 0), COL_GOLD)
        self.panel(0.0, 0.085, 1.56, 0.50)
        self.rows = []
        for index in range(SS.SPONSOR_COUNT):
            y = 0.295 - index * 0.058
            self.label(SS.SPONSOR_NAMES[index], -0.75, y, 0.85)
            level = self.label("", -0.33, y, 0.85, rgb=COL_ACCENT)
            rate = self.label("", -0.17, y, 0.80, rgb=COL_GOLD)
            one = self.button("", 0.22, y, 0.20, 0.046, lambda i=index: self.buy(i, 1), COL_GOOD, 0.68)
            ten = self.button("", 0.44, y, 0.20, 0.046, lambda i=index: self.buy(i, 10), COL_GOOD, 0.68)
            most = self.button("", 0.66, y, 0.20, 0.046, lambda i=index: self.buy(i, -1), COL_GOOD, 0.68)
            self.rows.append((level, rate, one, ten, most))
        self.panel(0.0, -0.285, 1.56, 0.19, (0.08, 0.05, 0.10))
        self.prestige_title = self.label("", -0.75, -0.225, 1.05, rgb=COL_ACCENT)
        self.prestige_info = self.label("", -0.75, -0.275, 0.75, rgb=COL_DIM)
        self.prestige_need = self.label("", -0.75, -0.325, 0.75, rgb=COL_DIM)
        self.prestige_button = self.button("ASCEND", 0.60, -0.285, 0.26, 0.07, self.ascend, COL_DANGER, 1.0)
        self.back_button()

    def buy(self, index: int, count: int) -> None:
        if count < 0:
            count = self.pm.sponsor_max_affordable(index)
        ok = count > 0 and self.pm.buy_sponsor(index, count)
        self.game.play_ui("ui_buy" if ok else "ui_error")
        if ok:
            self.pm.save()
        self.refresh()

    def ascend(self) -> None:
        ok = self.pm.do_prestige()
        self.game.play_ui("level_up" if ok else "ui_error")
        self.menu.notify("Prestige tier reached: permanent multipliers increased" if ok else "Prestige requirements not met")
        if ok:
            self.pm.save()
        self.refresh()

    def refresh(self) -> None:
        profile = self.pm.profile
        credits = profile.credits
        self.set_text(self.wallet, money(credits))
        self.set_text(self.income, "PASSIVE INCOME  +%s / s   (credited by the Corporate Sponsor Thread, even in menus)" %
                      SS.format_number(self.pm.income_per_second()))
        multiplier = SS.prestige_income_multiplier(profile.prestige)
        for index, (level, rate, one, ten, most) in enumerate(self.rows):
            owned = profile.sponsors[index]
            milestones = int((owned >= SS.SPONSOR_MILESTONES).sum())
            self.set_text(level, "LV %d" % owned)
            self.set_text(rate, "+%s / s" % SS.format_number(SS.SPONSOR_BASE_INCOME[index] * owned * (2 ** milestones) * multiplier))
            for button, count in ((one, 1), (ten, 10)):
                cost = self.pm.sponsor_cost(index, count)
                self.set_text(button, "x%d  %s" % (count, money(cost)))
                self.tint(button, COL_GOOD if credits >= cost else COL_BAD)
            affordable = self.pm.sponsor_max_affordable(index)
            self.set_text(most, "MAX  x%d" % affordable)
            self.tint(most, COL_GOOD if affordable > 0 else COL_BAD)
        need_credits, need_level = SS.prestige_requirement(profile.prestige)
        self.set_text(self.prestige_title, "CORPORATE PRESTIGE  -  TIER %d" % profile.prestige)
        self.set_text(self.prestige_info, "Ascending resets credits, level, weapons and sponsors.  Next tier: income x%.2f, combat credits x%.2f" % (
            SS.prestige_income_multiplier(profile.prestige + 1), SS.prestige_combat_multiplier(profile.prestige + 1)))
        self.set_text(self.prestige_need, "Requires %s lifetime credits (%s earned) and level %d (now %d)" % (
            SS.format_number(need_credits), SS.format_number(profile.lifetime_credits), need_level, profile.level))
        self.tint(self.prestige_button, COL_DANGER if self.pm.can_prestige() else COL_BAD)


# =========================================================================== #
#  Settings
# =========================================================================== #
class SettingsScreen(Screen):
    title = "SETTINGS"
    STEPS = {"sensitivity": 0.1, "fov": 5.0, "volume": 0.1}

    def build(self) -> None:
        self.panel(0.0, 0.02, 0.96, 0.66)
        self.values = {}
        for index, (key, text) in enumerate((("sensitivity", "MOUSE SENSITIVITY"), ("fov", "FIELD OF VIEW"),
                                             ("volume", "VOLUME"), ("invert_y", "INVERT Y AXIS"),
                                             ("show_fps", "SHOW FPS / NETWORK STATS"))):
            y = 0.27 - index * 0.085
            self.label(text, -0.42, y, 0.95, rgb=COL_DIM)
            self.button("-", 0.10, y, 0.05, 0.052, lambda k=key: self.change(k, -1))
            self.values[key] = self.label("", 0.25, y, 1.0, (0, 0))
            self.button("+", 0.40, y, 0.05, 0.052, lambda k=key: self.change(k, 1))
        self.label("CALLSIGN", -0.42, -0.165, 0.95, rgb=COL_DIM)
        self.name_field = InputField(default_value=self.pm.profile.name, parent=self.root, position=(0.13, -0.165),
                                     scale=(0.36, 0.05), character_limit=16)
        self.button("APPLY", 0.40, -0.165, 0.12, 0.05, self.apply_name, COL_GOOD, 0.8)
        self.label("Controls:  WASD move   SHIFT sprint   C / CTRL crouch + slide   SPACE jump   LMB fire   RMB aim",
                   0.0, -0.245, 0.62, (0, 0), COL_DIM)
        self.label("R reload   Q swap weapon   G pulse grenade   TAB scoreboard   ESC pause", 0.0, -0.275, 0.62, (0, 0), COL_DIM)
        self.back_button()

    def change(self, key: str, direction: int) -> None:
        settings = self.pm.profile.settings
        if key in self.STEPS:
            lo, hi = {"sensitivity": (0.1, 5.0), "fov": (60.0, 120.0), "volume": (0.0, 1.0)}[key]
            self.pm.set_setting(key, round(min(max(settings[key] + direction * self.STEPS[key], lo), hi), 2))
        else:
            self.pm.set_setting(key, not settings[key])
        self.game.apply_settings()
        self.refresh()

    def apply_name(self) -> None:
        self.pm.set_name(self.name_field.text)
        self.pm.save()
        self.menu.notify("Callsign set to %s" % self.pm.profile.name)

    def refresh(self) -> None:
        settings = self.pm.profile.settings
        self.set_text(self.values["sensitivity"], "%.1f" % settings["sensitivity"])
        self.set_text(self.values["fov"], "%d" % settings["fov"])
        self.set_text(self.values["volume"], "%d%%" % round(settings["volume"] * 100))
        self.set_text(self.values["invert_y"], "ON" if settings["invert_y"] else "OFF")
        self.set_text(self.values["show_fps"], "ON" if settings["show_fps"] else "OFF")


# =========================================================================== #
#  Connecting / pause overlays
# =========================================================================== #
class ConnectingScreen(Screen):
    def build(self) -> None:
        self.message = self.label("CONNECTING ...", 0.0, 0.05, 1.8, (0, 0), COL_ACCENT)
        self.detail = self.label("", 0.0, -0.02, 0.9, (0, 0), COL_DIM)
        self.button("CANCEL", 0.0, -0.14, 0.24, 0.06, self.game.leave_match, COL_DANGER)

    def refresh(self) -> None:
        self.set_text(self.detail, self.game.connect_detail())


class PauseScreen(Screen):
    def build(self) -> None:
        self.panel(0.0, 0.0, 0.62, 0.52, (0.02, 0.03, 0.05), 0.90)
        self.label("PAUSED", 0.0, 0.19, 1.9, (0, 0), COL_ACCENT)
        self.info = self.label("", 0.0, 0.125, 0.75, (0, 0), COL_DIM)
        self.button("RESUME", 0.0, 0.04, 0.34, 0.065, self.game.resume, COL_GOOD, 1.0)
        self.button("LEAVE MATCH", 0.0, -0.05, 0.34, 0.065, self.game.leave_match, COL_DANGER, 1.0)
        self.label("The match keeps running while this menu is open", 0.0, -0.14, 0.65, (0, 0), COL_DIM)
        self.wallet = self.label("", 0.0, -0.19, 0.85, (0, 0), COL_GOLD)

    def refresh(self) -> None:
        snap = self.pm.snapshot()
        self.set_text(self.info, self.game.match_summary())
        self.set_text(self.wallet, "%s   +%s / s" % (money(snap["credits"]), SS.format_number(snap["income"])))


# =========================================================================== #
#  Manager
# =========================================================================== #
class MenuManager:
    """Creates the screens once, shows one at a time and refreshes it periodically."""

    def __init__(self, game):
        self.game = game
        self.notice = ""
        self.notice_ttl = 0.0
        self.backdrop = Entity(parent=camera.ui, model="quad", color=color.rgba(0.02, 0.03, 0.06, 0.92), scale=(4, 2),
                               z=1.0, enabled=False)
        self.toast = Text("", parent=camera.ui, position=(0, -0.385), scale=0.9, origin=(0, 0), color=_c(COL_ACCENT), z=-3)
        self.screens = {
            "main": MainScreen(self), "setup": SetupScreen(self), "lobby": LobbyScreen(self),
            "armory": ArmoryScreen(self), "sponsors": SponsorScreen(self), "settings": SettingsScreen(self),
            "connecting": ConnectingScreen(self), "pause": PauseScreen(self),
        }
        self.current = None
        self.timer = 0.0

    @property
    def visible(self) -> bool:
        return self.current is not None

    def open_setup(self, host: bool) -> None:
        self.screens["setup"].host = host
        self.show("setup")

    def show(self, name: str) -> None:
        if self.current is not None:
            self.screens[self.current].root.enabled = False
        self.current = name
        screen = self.screens[name]
        self.backdrop.enabled = name not in ("pause",)
        screen.root.enabled = True
        screen.on_show()
        screen.refresh()

    def hide(self) -> None:
        if self.current is not None:
            self.screens[self.current].root.enabled = False
        self.current = None
        self.backdrop.enabled = False

    def notify(self, text: str, seconds: float = 4.0) -> None:
        """Short status line shown at the bottom of whatever screen is open."""
        self.notice = text
        self.notice_ttl = seconds
        self.toast.text = text

    def update(self, dt: float) -> None:
        if self.notice_ttl > 0.0:
            self.notice_ttl -= dt
            if self.notice_ttl <= 0.0:
                self.notice = ""
                self.toast.text = ""
        if self.current is None:
            return
        self.timer += dt
        if self.timer >= REFRESH_INTERVAL:
            self.timer = 0.0
            self.screens[self.current].refresh()
