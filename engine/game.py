"""
NEXUS-3D-EXTREME :: engine/game.py
==================================
Client orchestration: turns input into 60 Hz commands, feeds them to the
predicting network client, and turns everything that comes back -- the
predicted body, interpolated remote entities and server events -- into camera
motion, effects, sound, HUD state and profile progress.

Frame layout while a match is running::

    client.update()            decode snapshots, reconcile prediction, collect events
    scene.begin_frame()        age muzzle / explosion lights
    events -> effects          hit markers, damage numbers, tracers, kill feed, rewards
    mouse look + recoil        view angles are applied every rendered frame
    fixed ticks                sample buttons -> client.send_command() (prediction)
    interpolate                blend the last two predicted positions for rendering
    camera rig / post / view-model / pools -> one frame

Everything runs through the same client path whether the server is a remote
LAN host or the listen server this process started for a solo match.
"""
from __future__ import annotations

import math
import time

import numpy as np
from panda3d.core import ClockObject
from ursina import application, camera, held_keys, mouse

from . import config as C
from . import equipment as EQ
from . import mathx as M
from . import maps
from . import network_lan as N
from . import renderer as R
from . import secure_save as SS
from . import simulation as S
from . import streaks as ST
from . import weapons as W
from .audio import SoundBank
from .controller_3d import CameraRig
from .pacing import QualityGovernor
from .recap import CombatLog
from .report import MEDAL_NAMES, MatchReport
from .menus import MenuManager

STATE_MENU, STATE_CONNECTING, STATE_PLAYING = range(3)
LOOK_DEGREES_PER_UNIT = 42.0
_TRACER_TUNED = ((1.0, 0.85, 0.5, 1.0), (1.0, 0.9, 0.55, 1.0), (1.0, 0.7, 0.35, 0.8), (1.0, 0.95, 0.7, 1.0),
                 (1.0, 0.85, 0.5, 1.0), (1.0, 0.8, 0.45, 1.0), (0.35, 0.9, 1.0, 1.0), (1.0, 0.9, 0.6, 0.9),
                 (1.0, 0.88, 0.55, 1.0), (1.0, 0.82, 0.42, 1.0), (1.0, 0.8, 0.45, 1.0), (1.0, 0.7, 0.35, 0.8))
_TRACER_BY_CLASS = {0: (1.0, 0.84, 0.48, 1.0), 1: (1.0, 0.9, 0.58, 0.9), 2: (1.0, 0.7, 0.35, 0.8),
                    3: (1.0, 0.94, 0.68, 1.0), 4: (1.0, 0.86, 0.5, 1.0), 5: (1.0, 0.8, 0.45, 1.0), 6: (1.0, 0.96, 0.78, 1.0)}
TRACER_COLORS = tuple(_TRACER_TUNED[i] if i < len(_TRACER_TUNED) else _TRACER_BY_CLASS[int(W.WEAPONS["klass"][i])]
                      for i in range(W.WEAPON_COUNT))
KEY_LATCH = {"space": C.BTN_JUMP, "left mouse down": C.BTN_SHOOT, "c": C.BTN_SLIDE, "left control": C.BTN_SLIDE,
             "r": C.BTN_RELOAD, "q": C.BTN_SWAP, "g": C.BTN_ABILITY, "f": C.BTN_TACTICAL, "x": C.BTN_STREAK, "scroll up": C.BTN_SWAP,
             "scroll down": C.BTN_SWAP}
HELD_BUTTONS = (("w", C.BTN_FORWARD), ("s", C.BTN_BACK), ("a", C.BTN_LEFT), ("d", C.BTN_RIGHT), ("space", C.BTN_JUMP),
                ("left shift", C.BTN_SPRINT), ("left control", C.BTN_SLIDE), ("c", C.BTN_SLIDE),
                ("left mouse", C.BTN_SHOOT), ("right mouse", C.BTN_ADS), ("r", C.BTN_RELOAD), ("q", C.BTN_SWAP),
                ("g", C.BTN_ABILITY), ("f", C.BTN_TACTICAL), ("x", C.BTN_STREAK))
CONTROL_HINT = ("WASD move   SHIFT sprint   C slide   SPACE jump   LMB fire   RMB aim   G lethal   F tactical   R reload   "
                "Q swap   X streak   TAB scores")


class AutoPilot:
    """
    Hands-off driver used by ``--autoplay`` and the rendered smoke test: it
    walks, turns away from walls, and engages the nearest visible enemy by
    aiming at the interpolated image exactly like a human would.
    """

    def __init__(self, seed: int = 3):
        self.rng = np.random.default_rng(seed)
        self.turn = 0.0
        self.timer = 0
        self.strafe = 0

    def drive(self, game: "NexusGame"):
        client = game.client
        store = client.predictor.store
        view = client.view
        world = client.loaded.world
        self.timer += 1
        eye = store.eye_position(0)[0]
        yaw, pitch = game.yaw, game.pitch
        buttons = C.BTN_FORWARD | C.BTN_SPRINT
        enemies = view.alive & (view.team != client.team)
        enemies[client.pid] = False
        rows = np.flatnonzero(enemies)
        target = -1
        if rows.size:
            chest = view.pos[rows] + np.array([0.0, 1.2, 0.0])
            seen = world.line_of_sight(np.tile(eye, (rows.size, 1)), chest)
            if seen.any():
                dist = np.sqrt(np.sum((chest - eye) ** 2, axis=1))
                target = int(np.argmin(np.where(seen, dist, np.inf)))
                want_yaw, want_pitch = M.yaw_pitch_from_direction(chest[target] - eye)
                yaw += float(np.clip(M.angle_delta(yaw, float(want_yaw)), -0.16, 0.16))
                pitch += float(np.clip(float(want_pitch) - pitch, -0.10, 0.10))
                aligned = abs(float(M.angle_delta(yaw, float(want_yaw)))) < 0.03
                if self.timer % 50 == 0:
                    self.strafe = int(self.rng.integers(0, 2))
                buttons = (C.BTN_RIGHT if self.strafe else C.BTN_LEFT) | (C.BTN_ADS if dist[target] > 9.0 else 0)
                if aligned:
                    buttons |= C.BTN_SHOOT
                if self.timer % 420 == 0 and 8.0 < dist[target] < 25.0:
                    buttons |= C.BTN_ABILITY
        if target < 0:
            ahead = M.forward_vector(yaw, 0.0)
            hit_t, hit_item, _ = world.raycast(eye - np.array([0.0, 0.9, 0.0]), ahead, 3.0, C.LAYER_PLAYER)
            if hit_item[0] >= 0 or self.timer % 240 == 0:
                self.turn = float(self.rng.choice([-1.0, 1.0])) * float(self.rng.uniform(0.6, 1.6))
            yaw += float(np.clip(self.turn, -0.08, 0.08))
            self.turn -= float(np.clip(self.turn, -0.08, 0.08))
            pitch *= 0.9
            if self.timer % 150 == 60:
                buttons |= C.BTN_JUMP
            if self.timer % 330 == 200:
                buttons |= C.BTN_SLIDE
            if self.timer % 420 == 300:
                buttons |= C.BTN_TACTICAL
            if int(store.weapons.ammo[0, int(store.weapons.slot[0])]) * 2 < int(store.weapons.current("mag", 0)[0]):
                buttons |= C.BTN_RELOAD if self.timer % 30 == 0 else 0
        if int(store.streak_ready[0]) and self.timer % 90 == 45:
            buttons |= C.BTN_STREAK
        return buttons, yaw, pitch


class NexusGame:
    """The running application: menus, sessions and the per-frame match loop."""

    def __init__(self, profile_manager: SS.ProfileManager, autoplay: bool = False):
        self.pm = profile_manager
        self.settings = profile_manager.profile.settings
        self.audio = SoundBank(self.settings["volume"])
        self.scene = R.SceneRenderer(int(self.settings["quality"]), bool(self.settings["dynamic_resolution"]),
                                     float(self.settings["fps_limit"]))
        self.hud = R.HUD(self.scene.world.fx_shader)
        self.hud.set_visible(False)
        self.governor = QualityGovernor()
        self.rig = CameraRig(self.settings["fov"])
        self.server = None
        self.client = None
        self.browser = None
        self.state = STATE_MENU
        self.paused = False
        self.autopilot = AutoPilot() if autoplay else None
        self.grenades = S.GrenadeStore()
        self.zone_fx: list = []                  # burning fire pools and smoke clouds: [kind, position, radius, end]
        self.stun_end = 0.0                      # local clock time when the stun wears off
        self.empty_view = N.EntityView(C.MAX_PLAYERS)
        self.rng = np.random.default_rng(int(time.time()) & 0xFFFF)
        self.yaw = 0.0
        self.pitch = 0.0
        self.latched = 0
        self.accumulator = 0.0
        self.prev_pos = np.zeros(3)
        self.last_frame = time.perf_counter()
        self.clock = 0.0
        self.frame = 0
        self.fps = 60.0
        self.hud_timer = 0.0
        self.hint_timer = 0.0
        self.step_distance = 0.0
        self.empty_click = 0
        self.dead = False
        self.killer = ""
        self.streak = 0
        self.result = ""
        self.banner = ""
        self.banner_sub = ""
        self.banner_rgb = R.HUD_GOLD
        self.banner_ttl = 0.0
        self.death_note = ""
        self.combat = CombatLog()                 # damage taken and dealt this life (death recap)
        self.strikes: list = []                   # marked orbital lances: [position, impact time, owner]
        self.pings = np.full(C.MAX_PLAYERS, -100.0)   # when each player last fired (minimap pings)
        self.alert_end = 0.0
        self.uplink_pos = None                    # UPLINK zone positions of the loaded arena
        self.report = MatchReport()               # this match's stats and medals
        self.report_text = ""
        self.report_due = False                   # match over, waiting for the final roster (MVP)
        self.report_deadline = 0.0                # ... or until this local time at the latest
        self.result_end = 0.0                     # local time the next match starts (intermission countdown)
        self.intro_until = 0.0                    # the ESCALATION intro banner owns the centre until then
        self.mouse_hold = False                   # the click that closed the pause menu is not a shot
        self.orbit = 0.0
        self.last_corrections = 0
        self.backdrop_map = ""
        self.menus = MenuManager(self)
        self.menus.show("main")
        mouse.locked = False
        status = profile_manager.load_status
        if status == "tampered_reset":
            self.menus.notify("PROFILE INTEGRITY CHECK FAILED - the save file was modified and has been reset", 12.0)
        elif status == "restored_backup":
            self.menus.notify("The save file was missing or failed verification - the previous backup was restored", 10.0)
        elif status == "io_error":
            self.menus.notify("THE SAVE FILE COULD NOT BE READ - this session will not be saved; restart the game", 12.0)
        elif profile_manager.offline_credits > 0.0:
            self.menus.notify("While you were away (%s) your sponsors paid %s" % (
                SS.format_duration(profile_manager.offline_seconds), SS.format_number(profile_manager.offline_credits)), 10.0)

    # ================================================================== #
    #  Services used by the menus
    # ================================================================== #
    def play_ui(self, name: str) -> None:
        self.audio.play(name)

    def apply_settings(self) -> None:
        self.audio.set_volume(self.settings["volume"])
        self.rig.base_fov = float(self.settings["fov"])
        self.apply_graphics()

    def apply_graphics(self) -> None:
        """Graphics preset, frame limiter and dynamic resolution from the profile settings."""
        limit = float(self.settings["fps_limit"])
        clock = ClockObject.getGlobalClock()
        if limit > 0.0:
            clock.setMode(ClockObject.MLimited)
            clock.setFrameRate(limit)
        else:
            clock.setMode(ClockObject.MNormal)
        scene = self.scene
        if int(self.settings["quality"]) != scene.quality:
            scene.set_quality(int(self.settings["quality"]))
        if bool(self.settings["dynamic_resolution"]) != scene.resolution.enabled or limit != scene.fps_limit:
            scene.set_pacing(bool(self.settings["dynamic_resolution"]), limit)

    def _set_picking(self, enabled: bool) -> None:
        """Ursina ray-casts the cursor against the UI every frame; that is only needed while a menu is open."""
        mouse.update_step = 1 if enabled else 1.0e9

    def ensure_browser(self) -> None:
        if self.browser is None:
            self.browser = N.LanBrowser()
            self.browser.start()

    def connect_detail(self) -> str:
        if self.client is None or self.client.server is None:
            return ""
        return "%s:%d" % self.client.server

    def match_summary(self) -> str:
        client = self.client
        if client is None or client.loaded is None:
            return ""
        return "%s  -  %s  -  ping %d ms" % (client.loaded.data.name, C.MODE_NAMES[client.mode % len(C.MODE_NAMES)],
                                             round(client.ping_ms))

    def start_match(self, host: bool, map_key: str, mode: int, bots: int, difficulty: int, upgrades=None) -> None:
        """
        Start a listen server in this process (advertised on the LAN when hosting) and join it.
        ``upgrades`` False ignores everybody's upgrade levels: the server cannot verify what a LAN
        player bought, so a host who wants a level playing field turns them off (default: setting).
        """
        self._shutdown_session()
        if upgrades is None:
            upgrades = bool(self.pm.profile.settings.get("upgrades", True))
        config = N.ServerConfig(name="%s ARENA" % self.pm.profile.name, map_key=map_key, mode=mode,
                                bot_fill=bots + 1, bot_difficulty=difficulty, beacon=host,
                                max_players=C.MAX_PLAYERS if host else max(bots + 1, 2), rotate_maps=host,
                                upgrades_enabled=bool(upgrades))
        try:
            self.server = N.host_local_game(config)
        except OSError as error:
            self.menus.notify("Could not start the server: %s" % error)
            self.server = None
            return
        self._connect("127.0.0.1", self.server.port)

    def join_match(self, address: str, port: int) -> None:
        self._shutdown_session()
        self._connect(address, port)

    def _connect(self, address: str, port: int) -> None:
        weapon_ids, levels, operator, prestige = self.pm.loadout_arrays()
        self.client = N.GameClient(self.pm.profile.name, weapon_ids, levels, operator, prestige,
                                   equipment=self.pm.equipment_ids())
        self.client.connect(address, port)
        self.state = STATE_CONNECTING
        self.menus.show("connecting")

    def resume(self) -> None:
        self.paused = False
        self.menus.hide()
        self.hud.set_visible(True)
        self.mouse_hold = True
        self.latched = 0
        mouse.locked = True
        self._set_picking(False)

    def equipment_changed(self) -> None:
        """The player picked a different lethal / tactical: tell the server (it applies at the next respawn)."""
        if self.client is not None:
            self.client.equipment = self.pm.equipment_ids()

    def in_match(self) -> bool:
        return self.state == STATE_PLAYING

    def leave_match(self) -> None:
        self._shutdown_session()
        self.state = STATE_MENU
        self.paused = False
        self.hud.set_visible(False)
        self.scene.viewmodel.set_visible(False)
        mouse.locked = False
        self._set_picking(True)
        self.pm.save()
        self._load_backdrop()
        self.menus.show("main")

    def _shutdown_session(self) -> None:
        if self.client is not None:
            self.client.disconnect()
            self.client = None
        if self.server is not None:
            self.server.stop()
            self.server = None
        self.grenades.clear()
        self.zone_fx.clear()
        self._reset_rewards()
        self.stun_end = 0.0
        self.banner_ttl = 0.0
        self.intro_until = 0.0
        self.result = ""
        self.dead = False
        self.hud.reset()
        self.scene.post.set_stun(0.0)

    def quit(self) -> None:
        self._shutdown_session()
        if self.browser is not None:
            self.browser.stop()
        self.pm.save()
        application.quit()

    # ================================================================== #
    #  Input
    # ================================================================== #
    def input(self, key: str) -> None:
        """Key events from Ursina.  Short taps are latched until the next simulation tick."""
        if self.state == STATE_PLAYING:
            if key == "escape":
                if self.paused:
                    self.resume()
                else:
                    self.paused = True
                    mouse.locked = False
                    self._set_picking(True)
                    self.hud.set_visible(False)                 # menus own the screen while they are open
                    self.menus.show("pause")
            elif not self.paused and key in KEY_LATCH and not (self.mouse_hold and key == "left mouse down"):
                self.latched |= KEY_LATCH[key]
        elif self.state == STATE_MENU and key == "escape" and self.menus.current not in (None, "main"):
            self.menus.show("main")

    def _sample_buttons(self) -> int:
        if self.paused:
            self.latched = 0
            return 0
        buttons = self.latched
        self.latched = 0
        for key, bit in HELD_BUTTONS:
            if held_keys[key]:
                buttons |= bit
        if self.mouse_hold:                       # after RESUME the mouse buttons count once released
            if held_keys["left mouse"] or held_keys["right mouse"]:
                buttons &= ~(C.BTN_SHOOT | C.BTN_ADS)
            else:
                self.mouse_hold = False
        return buttons

    # ================================================================== #
    #  Frame entry point
    # ================================================================== #
    def update(self) -> None:
        now = time.perf_counter()
        elapsed = now - self.last_frame
        dt = min(max(elapsed, 1.0e-4), C.MAX_FRAME_DT)
        self.last_frame = now
        self.clock += dt
        self.frame += 1
        self.fps += (1.0 / dt - self.fps) * 0.05
        self.scene.end_frame(elapsed)                         # dynamic resolution follows the real frame time
        if self.governor.update(self.scene.resolution, self.scene.quality, elapsed):
            self.pm.set_setting("quality", self.scene.quality - 1)
            self.apply_graphics()
            note = "Graphics quality lowered to %s to hold the frame rate" % C.QUALITY_NAMES[self.scene.quality]
            if self.state == STATE_PLAYING:
                self.hud.push_reward(note, R.HUD_DIM)
            else:
                self.menus.notify(note, 6.0)
        self.menus.update(dt)
        if not self.in_match():
            for title, text, credits in self.pm.pop_notices():
                self.menus.notify("%s  -  %s  +%s CR" % (title, text, SS.format_number(credits)), 6.0)
        if self.state == STATE_MENU:
            self._update_backdrop(dt)
            return
        client = self.client
        events = client.update()
        if client.state == N.FAILED:
            reason = client.fail_reason
            self.leave_match()
            self.menus.notify("Disconnected: %s" % reason, 8.0)
            return
        if self.state == STATE_CONNECTING:
            if client.state == N.CONNECTED and client.predictor is not None and client.predictor.initialised:
                self._enter_match()
            else:
                self._update_backdrop(dt)
                return
        self._update_match(dt, events)

    # ---- menu backdrop: slow orbit around the selected arena ------------------------------------
    def _load_backdrop(self) -> None:
        key = self.settings["map"] if self.settings["map"] in maps.MAP_KEYS else maps.MAP_KEYS[0]
        if key != self.backdrop_map:
            self.scene.load_map(maps.get_map_data(key))
            self.backdrop_map = key

    def _update_backdrop(self, dt: float) -> None:
        if self.state == STATE_MENU:
            self._load_backdrop()                 # lazy: a direct --solo / --connect start never bakes the menu arena
        self.orbit += dt * 0.07
        radius, height = 30.0, 15.0
        cam = np.array([math.sin(self.orbit) * radius, height, math.cos(self.orbit) * radius])
        yaw, pitch = M.yaw_pitch_from_direction(np.array([0.0, 1.0, 0.0]) - cam)
        rig = self.rig
        rig.update(dt, cam - np.array([0.0, C.EYE_HEIGHT_STAND, 0.0]), np.zeros(3), 0, 0.0, float(yaw), float(pitch))
        rig.consume_aim_kick()
        R.apply_camera(rig)
        scene = self.scene
        scene.begin_frame(dt)
        scene.post.update(dt, rig, 1.0, True)
        scene.viewmodel.set_visible(False)
        scene.viewmodel.update(dt, rig, 0.0)
        scene.render(dt, self.clock, rig.cam_pos, rig.cam_yaw, rig.cam_pitch, self.empty_view, -1)

    # ---- session start ----------------------------------------------------------------------------
    def _enter_match(self) -> None:
        self.state = STATE_PLAYING
        self.paused = False
        self.menus.hide()
        self.hud.set_visible(True)
        self.hud.set_center("")
        self.hud.set_scoreboard(False)
        self.hint_timer = 12.0
        self.hud.set_hint(CONTROL_HINT)
        self.result = ""
        self.dead = False
        self.streak = 0
        self.accumulator = 0.0
        self.grenades.clear()
        self.zone_fx.clear()
        self._reset_rewards()
        self.stun_end = 0.0
        store = self.client.predictor.store
        self.prev_pos = store.move.pos[0].copy()
        self.last_corrections = self.client.predictor.corrections
        mouse.locked = self.autopilot is None
        self._set_picking(False)
        self.apply_settings()

    def _load_map_visuals(self) -> None:
        client = self.client
        client.map_changed = False
        data = client.loaded.data
        self.scene.load_map(data)
        self.backdrop_map = data.key
        rgba, bounds = maps.minimap_image(data)
        self.hud.set_minimap(rgba, bounds)
        self.uplink_pos, _ = maps.uplink_zones(client.loaded)
        self.grenades.clear()
        self.zone_fx.clear()
        self._reset_rewards()
        self.stun_end = 0.0
        self.result = ""

    # ================================================================== #
    #  Match frame
    # ================================================================== #
    def _update_match(self, dt: float, events: list) -> None:
        client, scene, rig = self.client, self.scene, self.rig
        pred = client.predictor
        store = pred.store
        if client.map_changed:
            self._load_map_visuals()
        scene.begin_frame(dt)
        for event in events:
            self._on_event(event)
        alive = bool(store.alive[0])
        slot = int(store.weapons.slot[0])
        zoom = float(store.weapons.ads_zoom[0, slot])
        look = (0.0, 0.0)
        if not self.paused and self.autopilot is None:
            sensitivity = math.radians(LOOK_DEGREES_PER_UNIT) * self.settings["sensitivity"]
            sensitivity /= 1.0 + (zoom - 1.0) * rig.ads_blend
            if self.clock < self.stun_end:
                sensitivity *= C.STUN_LOOK_SCALE                     # stunned: the aim drags
            look = (float(mouse.velocity[0]), float(mouse.velocity[1]))
            self.yaw += look[0] * sensitivity
            self.pitch += look[1] * sensitivity * (-1.0 if self.settings["invert_y"] else 1.0)
        kick = rig.consume_aim_kick()
        self.pitch = float(np.clip(self.pitch + kick[0], -C.PITCH_LIMIT, C.PITCH_LIMIT))
        self.yaw = float(M.wrap_angle(self.yaw + kick[1]))

        self.accumulator += dt
        steps = 0
        while self.accumulator >= C.TICK_DT and steps < C.MAX_TICKS_PER_FRAME:
            self.accumulator -= C.TICK_DT
            steps += 1
            self._tick()
        if steps == C.MAX_TICKS_PER_FRAME:
            self.accumulator = 0.0
        if pred.corrections != self.last_corrections:
            self.last_corrections = pred.corrections
            self.prev_pos = store.move.pos[0] - store.move.vel[0] * C.TICK_DT
        pred.decay_visual_error(dt)
        alpha = self.accumulator / C.TICK_DT
        feet = self.prev_pos + (store.move.pos[0] - self.prev_pos) * alpha + pred.visual_error
        view = client.sample_entities()
        self._follow_attached(view)
        self._update_zone_fx(dt)
        stunned = max(self.stun_end - self.clock, 0.0)
        if stunned > 0.0:
            rig.trauma = max(rig.trauma, min(0.30 + stunned * 0.15, 0.6))
        scene.post.set_stun(min(stunned / 0.6, 1.0))

        reloading = int(store.weapons.reload_end[0]) > 0
        rig.update(dt, feet, store.move.vel[0], int(store.move.mflags[0]) if alive else 0, float(store.move.crouch_t[0]),
                   self.yaw, self.pitch, look, zoom, reloading, float(store.weapons.recoil_recover[0, slot]))
        R.apply_camera(rig)
        health_fraction = float(store.health[0]) / max(float(store.max_health[0]), 1.0)
        scene.post.update(dt, rig, health_fraction, alive)
        scene.viewmodel.set_weapon(int(store.weapons.weapon[0, slot]))
        scene.viewmodel.set_visible(alive)
        scene.viewmodel.update(dt, rig, scene.lights.flash_level)
        scene.render(dt, self.clock, rig.cam_pos, rig.cam_yaw, rig.cam_pitch, view, client.pid, self.grenades)
        self._update_hud(dt, alive)

    # ---- one 60 Hz simulation tick -----------------------------------------------------------------
    def _tick(self) -> None:
        client, audio, rig = self.client, self.audio, self.rig
        store = client.predictor.store
        buttons = self._sample_buttons()
        if self.autopilot is not None and not self.paused:
            buttons, self.yaw, self.pitch = self.autopilot.drive(self)
        ws = store.weapons
        was_flags = int(store.move.mflags[0])
        was_reloading = int(ws.reload_end[0]) > 0
        was_slot = int(ws.slot[0])
        self.prev_pos = store.move.pos[0].copy()
        fired, ability, tactical = client.send_command(buttons, self.yaw, self.pitch)
        if store.alive[0]:
            flags = int(store.move.mflags[0])
            if flags & C.MF_JUMPED:
                audio.play("jump", 0.45, vary=0.05)
            if flags & C.MF_LANDED:
                impact = float(store.move.fall_speed[0])
                rig.add_landing(impact)
                audio.play("land", min(0.25 + impact * 0.05, 1.0), vary=0.05)
            if flags & C.MF_STEPPED:
                rig.add_step(float(store.move.pos[0, 1] - self.prev_pos[1]))
            if flags & C.MF_SLIDING and not was_flags & C.MF_SLIDING:
                audio.play("slide", 0.7)
            speed = float(np.hypot(store.move.vel[0, 0], store.move.vel[0, 2]))
            if flags & C.MF_ON_GROUND and not flags & C.MF_SLIDING and speed > 2.0:
                self.step_distance += speed * C.TICK_DT
                if self.step_distance > 2.3:
                    self.step_distance = 0.0
                    audio.play("step", 0.22 if flags & C.MF_CROUCHED else 0.38, vary=0.12)
            if fired:
                self._on_local_fire()
            elif buttons & C.BTN_SHOOT and int(ws.ammo[0, int(ws.slot[0])]) == 0 and self.frame - self.empty_click > 20:
                self.empty_click = self.frame
                audio.play("empty", 0.5)
            if ability:
                lethal = int(store.equip[0, 0])
                audio.play("hatchet" if EQ.KIND[lethal] == EQ.KIND_AXE else "throw", 0.7)
            if tactical:
                if EQ.KIND[int(store.equip[0, 1])] == EQ.KIND_STIM:
                    audio.play("stim", 0.8)
                    self.scene.post.add_heal(0.9)
                    self.hud.push_reward("STIM  +%d" % int(EQ.DAMAGE[int(store.equip[0, 1])]), (0.45, 1.0, 0.6))
                else:
                    audio.play("throw", 0.7)
            if int(ws.reload_end[0]) > 0 and not was_reloading:
                audio.play("reload", 0.7)
            if int(ws.slot[0]) != was_slot:
                audio.play("swap", 0.5)
        for slot in S.step_grenades(self.grenades, client.loaded.world).tolist():
            self.grenades.active[slot] = False

    def _follow_attached(self, view) -> None:
        """Charges stuck to players ride along with them."""
        gs = self.grenades
        for slot in np.flatnonzero(gs.active & gs.stuck & (gs.target >= 0)).tolist():
            wearer = int(gs.target[slot])
            if view.alive[wearer]:
                gs.pos[slot] = view.pos[wearer] + gs.offset[slot]

    def _update_zone_fx(self, dt: float) -> None:
        """Fire pools and smoke clouds keep emitting until they burn out."""
        if not self.zone_fx:
            return
        scene = self.scene
        alive = []
        for zone in self.zone_fx:
            kind, position, radius, end, light = zone
            if self.clock >= end:
                continue
            if kind == EQ.KIND_FIRE:
                scene.fx_fire(position, radius, dt)
                zone[4] = light - dt
                if zone[4] <= 0.0:
                    scene.lights.add(position + np.array([0.0, 0.7, 0.0]), (1.0, 0.55, 0.22),
                                     11.0 + 4.0 * math.sin(self.clock * 23.0), 0.25, True)
                    zone[4] = 0.18
            else:
                scene.fx_smoke(position, radius, dt)
            alive.append(zone)
        self.zone_fx = alive

    def _camera_basis(self):
        return (M.forward_vector(self.yaw, self.pitch), M.right_vector(self.yaw), M.up_vector(self.yaw, self.pitch))

    def _on_local_fire(self) -> None:
        """Immediate feedback for a predicted shot: tracers, impacts, flash, sound and recoil."""
        client, scene, rig = self.client, self.scene, self.rig
        store = client.predictor.store
        ws = store.weapons
        slot = int(ws.slot[0])
        weapon = int(ws.weapon[0, slot])
        origins, directions, _, _, _ = S.shot_rays(store, [0], player_ids=[client.pid])
        t, item, normal = client.loaded.world.raycast(origins, directions, C.MAX_BULLET_RANGE)
        ends = origins + directions * t[:, None]
        forward, right, up = self._camera_basis()
        offset = scene.viewmodel.muzzle_offset(rig.ads_blend)
        eye = store.eye_position(0)[0]
        muzzle = eye + right * offset[0] + up * offset[1] + forward * offset[2]
        color = TRACER_COLORS[weapon % len(TRACER_COLORS)]
        for k in range(ends.shape[0]):
            scene.fx_tracer(muzzle, ends[k], color, 0.03 if weapon == 6 else 0.02)
            if item[k] >= 0 and k < 5:
                scene.fx_impact(ends[k], normal[k])
        heavy = float(ws.damage[0, slot]) * int(ws.pellets[0, slot])
        scene.fx_muzzle(muzzle, (0.4, 0.9, 1.0) if weapon == 6 else (1.0, 0.78, 0.45), 7.0 + min(heavy, 120.0) * 0.08)
        scene.viewmodel.fire()
        self.audio.shot(weapon)
        steady = 0.65 if int(store.move.mflags[0]) & C.MF_ADS else 1.0
        rig.add_recoil(float(ws.recoil_pitch[0, slot]) * steady,
                       float(self.rng.uniform(-1.0, 1.0)) * float(ws.recoil_yaw[0, slot]) * steady,
                       1.0 + min(heavy, 120.0) / 60.0)
        self.pm.record("shots")
        self.report.shot()

    # ================================================================== #
    #  Server events -> feedback
    # ================================================================== #
    def _reset_rewards(self) -> None:
        """Forget lance marks, minimap pings, the combat log and alerts (new map, match or session)."""
        self.strikes.clear()
        self.pings[:] = -100.0
        self.combat.reset()
        self.alert_end = 0.0
        self.report.reset()
        self.report_text = ""
        self.report_due = False
        self.hud.set_recap("")
        self.hud.set_alert("")

    def _friendly(self, pid: int) -> bool:
        client = self.client
        return C.team_mode(client.mode) and 0 <= pid < C.MAX_PLAYERS and int(client.view.team[pid]) == int(client.team)

    def _set_alert(self, text: str, rgb, seconds: float) -> None:
        self.hud.set_alert(text, rgb)
        self.alert_end = self.clock + seconds

    def _death_recap(self, killer: int, weapon: int, weapon_name: str, flags: int) -> None:
        """Who, with what, from how far, and how the exchange went (shown while dead)."""
        client = self.client
        if flags & S.KILL_SUICIDE:
            own = killer == client.pid and C.EQUIP_KILL_BASE <= weapon < C.EQUIP_KILL_BASE + EQ.EQUIP_COUNT
            self.hud.set_recap(self.combat.recap(-1, "", ("your own %s" % weapon_name) if own else "the fall",
                                                 suicide=True))
            return
        known = 0 <= killer < C.MAX_PLAYERS and bool(client.view.active[killer])
        distance = float(np.linalg.norm(client.view.pos[killer] - client.predictor.store.move.pos[0])) if known else None
        health = float(client.view.health[killer]) if known and client.view.alive[killer] else None
        self.hud.set_recap(self.combat.recap(killer, self._name(killer), weapon_name, distance, health,
                                             bool(flags & S.KILL_HEADSHOT)))

    def _on_streak(self, event: tuple) -> None:
        """Scorestreak news: rewards ready, used, the lance landing, a shield shot away."""
        _, pid, reward, action, position = event
        client, hud, audio, scene, rig = self.client, self.hud, self.audio, self.scene, self.rig
        me = client.pid
        reward %= ST.STREAK_COUNT
        name = ST.STREAK_NAMES[reward]
        position = np.asarray(position, dtype=np.float64)
        mine = pid == me
        friendly = mine or self._friendly(pid)
        _, right, _ = self._camera_basis()
        if action == S.STREAK_EARNED:
            if mine:
                self._show_banner("%s READY" % name, "press X", R.HUD_GOLD, 2.4)
                audio.play("streak_ready", 0.8)
        elif action == S.STREAK_USED:
            if mine:
                self.pm.record("rewards_used")
            if reward == ST.LANCE:
                self.strikes.append([position, self.clock + float(ST.DURATION[ST.LANCE]), pid])
                hud.push_kill("%s  called in %s" % (self._name(pid), name), R.HUD_GOLD if friendly else R.HUD_CRIMSON)
                audio.play_at("lance_mark", position, rig.cam_pos, right)
            elif reward == ST.SCAN:
                if friendly:
                    hud.push_kill("%s  activated %s" % (self._name(pid), name), R.HUD_GOLD)
                    audio.play("scan", 0.7)
                else:
                    self._set_alert("ENEMY %s  -  you show on their minimap" % name, R.HUD_CRIMSON,
                                    float(ST.DURATION[ST.SCAN]))
                    audio.play("scan", 0.4)
            elif mine:                                     # your own shield: no particles inside the camera
                self._show_banner(name, "+%d overshield for %d seconds" % (C.AEGIS_POINTS, ST.DURATION[ST.AEGIS]),
                                  R.HUD_CYAN, 2.0)
                audio.play("shield_up", 0.8)
            else:
                scene.fx_shield(position)
        elif action == S.STREAK_IMPACT:
            scene.fx_lance_impact(position)
            for _ in range(2):
                rig.add_explosion_shake(rig.cam_pos, position[None, :])
            audio.play_at("lance_impact", position, rig.cam_pos, right)
            self.strikes = [s for s in self.strikes if float(np.linalg.norm(s[0] - position)) > 0.5]
        elif action == S.STREAK_BROKEN:
            if mine:
                scene.post.add_flash(0.25)
                audio.play("shield_break", 0.8)
            else:
                scene.fx_shield(position, broken=True)
                audio.play_at("shield_break", position, rig.cam_pos, right, 0.7)

    def _update_rewards(self, dt: float, store) -> None:
        """Per frame: lance marks, the minimap, the shield bar, the streak lines and timed alerts."""
        client, hud, scene = self.client, self.hud, self.scene
        me, view = client.pid, client.view
        own = client.predictor.render_position()
        marks, live = [], []
        for strike in self.strikes:
            position, end, owner = strike
            if self.clock > end + 0.3:
                continue
            live.append(strike)
            progress = 1.0 - max(end - self.clock, 0.0) / float(ST.DURATION[ST.LANCE])
            scene.fx_lance_mark(position, progress, dt)
            marks.append((position, progress))
            if (self.clock < end and owner != me and not self._friendly(owner)
                    and float(np.linalg.norm(position - own)) < C.LANCE_RADIUS + 3.0):
                self._set_alert("ORBITAL LANCE INCOMING  -  MOVE", R.HUD_CRIMSON, 0.25)
        self.strikes = live
        scanning = int(store.scan_until[0]) > int(store.cmd_tick[0])
        others, colors = [], []
        for pid in np.flatnonzero(view.active & view.alive).tolist():
            if pid == me:
                continue
            if self._friendly(pid):
                rgb, alpha = C.TEAM_COLORS[int(client.team) % 2], 0.95
            else:
                fade = 1.0 - (self.clock - float(self.pings[pid])) / C.SHOT_PING_TIME
                rgb, alpha = (1.0, 0.20, 0.16), 0.95 if scanning else 0.95 * min(max(fade, 0.0), 1.0)
            others.append(view.pos[pid])
            colors.append((float(rgb[0]), float(rgb[1]), float(rgb[2]), alpha))
        hud.set_radar(own, self.yaw, others, colors, marks, 0.5 + 0.5 * math.sin(self.clock * 9.0))
        hud.set_shield(float(store.shield[0]), C.AEGIS_POINTS)
        if client.mode == C.MODE_ESC:
            hud.set_streak("", "")
        else:
            streak = int(store.streak[0])
            earned = sum(1 << i for i in range(ST.STREAK_COUNT) if ST.KILLS[i] <= streak)
            goal = ST.next_goal(streak, earned)
            line = ("STREAK %d  -  %s AT %d" % (streak, ST.STREAK_NAMES[goal], ST.KILLS[goal]) if goal >= 0
                    else "STREAK %d  -  every reward earned" % streak)
            ready = [ST.STREAK_NAMES[i] for i in range(ST.STREAK_COUNT) if (int(store.streak_ready[0]) >> i) & 1]
            hud.set_streak(line, ("X  " + "   +  ".join(ready)) if ready else "")
        if self.alert_end and self.clock >= self.alert_end:
            self.alert_end = 0.0
            hud.set_alert("")

    def _on_uplink(self, event: tuple) -> None:
        """A zone changed hands: feed line and sound from your squad's point of view."""
        _, zone, team, action = event
        client, hud, audio = self.client, self.hud, self.audio
        mine = int(team) == int(client.team) % 2
        name = C.UPLINK_NAMES[zone % len(C.UPLINK_NAMES)]
        squad = C.TEAM_NAMES[team % 2]
        rgb = tuple(float(v) for v in C.TEAM_COLORS[team % 2])
        if action == S.UPLINK_CAPTURED:
            hud.push_kill("%s  captured uplink %s" % (squad, name), rgb)
            audio.play("uplink_captured" if mine else "uplink_lost", 0.8 if mine else 0.5)
        else:
            hud.push_kill("%s  lost uplink %s" % (squad, name), R.HUD_DIM)
            if mine:
                audio.play("uplink_lost", 0.7)

    def _update_uplinks(self, dt: float, alive: bool) -> None:
        """UPLINK: zone strip, capture bar, labels over the zones, minimap discs and the zones' glow."""
        client, hud, scene, rig = self.client, self.hud, self.scene, self.rig
        if client.mode != C.MODE_UPLINK or self.uplink_pos is None:
            hud.set_uplink_strip(None)
            hud.set_uplink_markers(())
            hud.set_uplink_capture("")
            hud.set_radar_zones()
            return
        team = int(client.team) % 2
        own = tuple(float(v) for v in C.TEAM_COLORS[team])
        enemy = tuple(float(v) for v in C.TEAM_COLORS[1 - team])
        flash = 0.5 + 0.5 * math.sin(self.clock * 10.0)
        colors = []
        for zone in range(len(self.uplink_pos)):
            holder = int(client.uplink_owner[zone])
            rgb = np.array(own if holder == team else (enemy if holder == 1 - team else (0.86, 0.89, 0.94)))
            if client.uplink_contested[zone]:
                rgb = rgb * (1.0 - flash) + np.array(R.HUD_GOLD) * flash
            colors.append(tuple(float(v) for v in rgb))
        hud.set_uplink_strip(colors)
        hud.set_radar_zones(self.uplink_pos, [rgb + (0.55,) for rgb in colors])
        forward = M.forward_vector(rig.cam_yaw, rig.cam_pitch)
        right, up = M.right_vector(rig.cam_yaw), M.up_vector(rig.cam_yaw, rig.cam_pitch)
        tan_h = math.tan(math.radians(camera.fov) * 0.5)
        tan_v = math.tan(math.radians(R.vertical_fov(camera.fov)) * 0.5)
        own_pos = client.predictor.render_position()
        markers = []
        for zone, position in enumerate(self.uplink_pos):
            anchor = position + np.array([0.0, 2.4, 0.0]) - rig.cam_pos
            depth = float(anchor @ forward)
            x = float(anchor @ right) / max(depth * tan_h, 1.0e-6) * hud.half_width
            y = float(anchor @ up) / max(depth * tan_v, 1.0e-6) * 0.5
            if depth < 0.5 or abs(x) > hud.half_width * 0.97 or abs(y) > 0.47:
                markers.append(None)
                continue
            metres = float(np.linalg.norm(position - own_pos))
            markers.append((x, y, "%s  %dm" % (C.UPLINK_NAMES[zone], round(metres)), colors[zone]))
        hud.set_uplink_markers(markers)
        flat = own_pos - self.uplink_pos
        inside = ((np.hypot(flat[:, 0], flat[:, 2]) < C.UPLINK_RADIUS) & (flat[:, 1] > -1.0)
                  & (flat[:, 1] < C.UPLINK_HEIGHT))
        if alive and inside.any():
            zone = int(np.argmax(inside))
            name = C.UPLINK_NAMES[zone]
            mine = float(client.uplink_charge[zone]) * (1.0 if team == 1 else -1.0)
            if client.uplink_contested[zone]:
                text = "%s  CONTESTED" % name
            elif int(client.uplink_owner[zone]) == team:
                text = "%s  SECURED" % name
            else:
                text = ("NEUTRALISING  %s" if mine < 0.0 else "CAPTURING  %s") % name
            hud.set_uplink_capture(text, mine, own, enemy)
        else:
            hud.set_uplink_capture("")
        for zone, position in enumerate(self.uplink_pos):
            scene.fx_uplink(position, colors[zone], dt, bool(client.uplink_contested[zone]))

    def _medals(self, keys) -> None:
        """Medal popups: a gold line in the reward feed, a chime, XP and the profile statistic."""
        for key in keys:
            self.hud.push_reward("MEDAL  %s" % MEDAL_NAMES[key])
            self.audio.play("medal", 0.7)
            self.pm.record("medals")
            _, levels = self.pm.apply_reward(0, C.MEDAL_XP)
            self.report.reward(0, C.MEDAL_XP)
            if levels:
                self._show_banner("LEVEL %d" % self.pm.profile.level, "level up bonus credited", R.HUD_GOLD, 3.0)
                self.audio.play("level_up", 0.8)

    def _show_banner(self, text: str, sub: str, rgb, ttl: float) -> None:
        self.banner, self.banner_sub, self.banner_rgb, self.banner_ttl = text, sub, rgb, ttl

    def match_mode(self):
        """Mode of the match being played, or None outside a match."""
        return self.client.mode if self.in_match() and self.client is not None else None

    def _on_ladder(self, event: tuple) -> None:
        """ESCALATION progress: banners for your own steps, feed lines for steals and final weapons."""
        _, pid, step, cause, other = event
        me, hud, audio = self.client.pid, self.hud, self.audio
        steps = W.ESCALATION_STEPS
        gun = W.WEAPON_NAMES[W.escalation_weapon(step)]
        climbed = cause in (S.LADDER_PROMOTED, S.LADDER_STEAL)
        if cause == S.LADDER_STOLEN:
            hud.push_kill("%s  stole a step from  %s" % (self._name(other), self._name(pid)),
                          R.HUD_GOLD if me in (pid, other) else R.HUD_WHITE)
        if pid == me:
            if climbed and step == steps - 1:
                self._show_banner("FINAL WEAPON", "%s  -  one more kill wins" % gun, R.HUD_GOLD, 2.4)
                audio.play("final_weapon", 0.8)
            elif climbed and step < steps:
                sub = gun if cause == S.LADDER_PROMOTED else "%s  -  stolen from %s" % (gun, self._name(other))
                self._show_banner("WEAPON %d / %d" % (step + 1, steps), sub, R.HUD_WHITE, 1.6)
                audio.play("escalate", 0.7)
            elif not climbed:
                self.death_note = ("a step stolen by %s" % self._name(other)) if cause == S.LADDER_STOLEN else "a step lost"
                audio.play("demoted", 0.7)
        elif climbed and step == steps - 1:
            hud.push_kill("%s  reached the FINAL WEAPON" % self._name(pid), R.HUD_CRIMSON)
            audio.play("final_weapon", 0.35)

    def _name(self, pid: int) -> str:
        entry = self.client.roster.get(pid)
        return entry.name if entry is not None else ("PLAYER %d" % pid if pid != S.NO_PLAYER else "THE ARENA")

    def _team_color(self, pid: int):
        entry = self.client.roster.get(pid)
        team = entry.team if entry is not None else 2
        return C.TEAM_COLORS[team] if team < 2 else R.FFA_ENEMY_COLOR

    def _on_event(self, event: tuple) -> None:
        client, scene, hud, audio, rig = self.client, self.scene, self.hud, self.audio, self.rig
        kind = event[0]
        me = client.pid
        if kind == S.EV_HIT:
            _, victim, damage, zone, killed, point, gun = event
            headshot = zone == C.ZONE_HEAD
            hud.hitmarker(headshot, killed)
            scene.fx_damage_number(point, damage, headshot, killed)
            scene.fx_hit(point, headshot)
            scene.characters.hit(victim)
            audio.play("headshot" if headshot else "hit", 0.75)
            if gun:                                    # blasts, fire, hatchets and the lance add damage only
                self.pm.record("hits")
            self.pm.record("damage", damage)
            self.combat.gave(victim, damage)
            self._medals(self.report.hit(damage, headshot, counted=gun))
        elif kind == S.EV_DAMAGE:
            _, attacker, damage, _zone, source = event
            strength = min(damage / 40.0, 1.5)
            scene.post.add_damage(strength)
            offset = np.asarray(source) - rig.cam_pos
            relative = float(M.angle_delta(self.yaw, math.atan2(offset[0], offset[2])))
            rig.add_damage_punch(1.0 if relative > 0.0 else -1.0, min(strength, 1.0) * 0.05)
            if attacker != S.NO_PLAYER and attacker != me:
                hud.damage_from(relative)
            audio.play("hurt", 0.7, vary=0.08)
            self.combat.took(attacker if attacker != S.NO_PLAYER else -1, damage, _zone == C.ZONE_HEAD)
        elif kind == S.EV_KILL:
            _, killer, victim, weapon, flags = event
            equipment = C.EQUIP_KILL_BASE <= weapon < C.EQUIP_KILL_BASE + EQ.EQUIP_COUNT   # grenades, hatchet, fire
            if equipment:
                weapon_name = EQ.EQUIP_NAMES[weapon - C.EQUIP_KILL_BASE]
            elif ST.is_streak_kill(weapon):
                weapon_name = ST.STREAK_NAMES[weapon - C.STREAK_KILL_BASE]
            elif weapon == S.WEAPON_GRENADE:
                weapon_name = "KNOCKED OFF"
            else:
                weapon_name = W.WEAPON_NAMES[weapon % W.WEAPON_COUNT]
            if flags & S.KILL_SUICIDE:
                line = "%s  was eliminated" % self._name(victim)
            else:
                tag = "HEADSHOT  " if flags & S.KILL_HEADSHOT else ""
                line = "%s  [%s%s]  %s" % (self._name(killer), tag, weapon_name, self._name(victim))
            mine = me in (killer, victim)
            self._medals(self.report.kill(killer, victim, me, self.clock, equipment, ST.is_streak_kill(weapon),
                                          bool(flags & S.KILL_SUICIDE), bool(flags & S.KILL_FIRST)))
            hud.push_kill(line, R.HUD_CRIMSON if flags & S.KILL_HEADSHOT else (R.HUD_GOLD if mine else R.HUD_WHITE))
            if 0 <= victim < C.MAX_PLAYERS and victim != me and client.view.active[victim]:
                scene.fx_death(client.view.pos[victim], self._team_color(victim))
            if victim == me:
                self.dead = True
                self.killer = "" if flags & S.KILL_SUICIDE else self._name(killer)
                self.streak = 0
                self.pm.record("deaths")
                self._death_recap(killer, weapon, weapon_name, flags)
            elif killer == me:
                self.streak += 1
                self.pm.record("kills")
                self.pm.record("best_streak", self.streak, maximum=True)
                if flags & S.KILL_HEADSHOT:
                    self.pm.record("headshots")
                if equipment:
                    self.pm.record("grenade_kills")
                audio.play("kill", 0.8)
        elif kind == S.EV_SPAWN:
            _, pid, position, yaw = event
            if pid == me:
                self.yaw, self.pitch = float(yaw), 0.0
                self.dead = False
                self.prev_pos = np.asarray(position, dtype=np.float64)
                rig.recoil_target[:] = 0.0
                rig.recoil_current[:] = 0.0
                rig.trauma = 0.0
                audio.play("respawn", 0.5)
                self.death_note = ""
                self.combat.reset()
                hud.set_recap("")
                if client.mode == C.MODE_ESC and client.predictor is not None and self.clock >= self.intro_until:
                    ws = client.predictor.store.weapons
                    gun = int(ws.weapon[0, int(ws.slot[0])])
                    self._show_banner("WEAPON %d / %d" % (W.ESCALATION_STEP_OF.get(gun, 0) + 1, W.ESCALATION_STEPS),
                                      W.WEAPON_NAMES[gun], R.HUD_WHITE, 1.4)
        elif kind == S.EV_SHOT:
            _, shooter, weapon, origin, end = event
            if 0 <= shooter < C.MAX_PLAYERS:
                self.pings[shooter] = self.clock
            origin = np.asarray(origin) - np.array([0.0, 0.12, 0.0])
            direction = M.normalize(np.asarray(end) - origin)
            start = origin + direction * 0.7
            scene.fx_tracer(start, end, TRACER_COLORS[weapon % len(TRACER_COLORS)])
            scene.fx_muzzle(start, intensity=5.0)
            _, right, _ = self._camera_basis()
            audio.shot_at(weapon, origin, rig.cam_pos, right)
        elif kind == S.EV_EXPLOSION:
            _, position, _owner, gid, equip = event
            position = np.asarray(position, dtype=np.float64)
            effect = int(EQ.KIND[equip % EQ.EQUIP_COUNT])
            distance = float(np.sqrt(np.sum((position - rig.cam_pos) ** 2)))
            _, right, _ = self._camera_basis()
            if effect in (EQ.KIND_FRAG, EQ.KIND_STICKY):
                scene.fx_explosion(position)
                rig.add_explosion_shake(rig.cam_pos, position[None, :])
                scene.post.add_flash(min(0.5, 6.0 / max(distance * distance, 1.0)))
                audio.play_at("explosion", position, rig.cam_pos, right)
            elif effect == EQ.KIND_STUN:
                scene.fx_stun(position)
                scene.post.add_flash(min(0.7, 12.0 / max(distance * distance, 1.0)))
                audio.play_at("shock", position, rig.cam_pos, right)
            else:
                radius = float(EQ.RADIUS[equip % EQ.EQUIP_COUNT])
                end = self.clock + float(EQ.DURATION[equip % EQ.EQUIP_COUNT])
                self.zone_fx.append([effect, position, radius, end, 0.0])
                if effect == EQ.KIND_FIRE:
                    scene.fx_fire(position, radius, 0.0, burst=True)
                    audio.play_at("fire", position, rig.cam_pos, right)
                else:
                    scene.fx_smoke(position, radius, 0.0, burst=True)
                    audio.play_at("smoke", position, rig.cam_pos, right)
            self.grenades.remove_gid(gid)
        elif kind == S.EV_STICK:
            _, gid, target, position = event
            slot = self.grenades.find(gid)
            _, right, _ = self._camera_basis()
            if slot >= 0:
                axe = EQ.KIND[int(self.grenades.equip[slot])] == EQ.KIND_AXE
                if axe and target != S.NO_PLAYER:
                    self.grenades.active[slot] = False                   # buried in somebody
                    scene.fx_hit(position)
                else:
                    offset = None
                    if target != S.NO_PLAYER and 0 <= target < len(client.view.pos):
                        offset = np.asarray(position) - client.view.pos[target]
                    self.grenades.stick(slot, position, None, target if offset is not None else -1, offset)
                audio.play_at("hatchet_hit" if axe else "stick", np.asarray(position), rig.cam_pos, right)
        elif kind == S.EV_STATUS:
            _, status, ticks = event
            if status == C.STATUS_STUN:
                self.stun_end = max(self.stun_end, self.clock + ticks * C.TICK_DT)
                scene.post.add_flash(0.45)
                audio.play("stun_ring", 0.7)
        elif kind == S.EV_GRENADE:
            _, gid, owner, position, velocity, fuse, spawn_tick, equip = event
            slot = self.grenades.spawn(position, velocity, fuse, owner, gid, equip)
            behind = int(np.clip(client.render_ms / C.TICK_MS - spawn_tick, 0, 30))
            for _ in range(behind):
                expired = S.step_grenades(self.grenades, client.loaded.world)
                if slot in expired.tolist():
                    self.grenades.active[slot] = False
                    break
        elif kind == S.EV_LADDER:
            self._on_ladder(event)
        elif kind == S.EV_STREAK:
            self._on_streak(event)
        elif kind == S.EV_UPLINK:
            self._on_uplink(event)
        elif kind == S.EV_REWARD:
            _, credits, xp, reason = event
            granted, levels = self.pm.apply_reward(credits, xp)
            self.report.reward(granted, xp)
            if reason == S.REWARD_CAPTURE:
                self.pm.record("captures")
                self._medals(self.report.capture())
            hud.push_reward("+%s  %s" % (SS.format_number(granted), S.REWARD_NAMES[reason % len(S.REWARD_NAMES)]))
            if levels:
                self._show_banner("LEVEL %d" % self.pm.profile.level, "level up bonus credited", R.HUD_GOLD, 3.0)
                audio.play("level_up", 0.8)
        elif kind == S.EV_MATCH:
            _, phase, winner, _map_index, mode = event
            if phase == C.PHASE_ENDED:
                if winner == S.NO_PLAYER:
                    self.result, won = "DRAW", False
                else:
                    won = (winner == client.team) if C.team_mode(mode) else (winner == me)
                    self.result = "VICTORY" if won else "DEFEAT"
                self.result_end = self.clock + C.INTERMISSION_TIME
                self.pm.record("matches")
                if won:
                    self.pm.record("wins")
                audio.play("victory" if won else "defeat", 0.8)
                self.report_due = True                     # the MVP needs the roster taken after the end
                self.report_deadline = self.clock + 2.0
                self.report_text = self._report_summary()  # provisional until then
                self._finish_report()
            else:
                self.result = ""
                self.death_note = ""
                self.report.reset()
                self.report_text = ""
                self.report_due = False
                if mode == C.MODE_ESC:
                    self._show_banner(C.MODE_NAMES[mode], "climb %d guns  -  first up: %s" % (
                        W.ESCALATION_STEPS, W.WEAPON_NAMES[W.escalation_weapon(0)]), R.HUD_GOLD, 3.0)
                    self.intro_until = self.clock + 3.0
                self.streak = 0

    def _best(self):
        """Roster entry of the match's best player (score, then eliminations)."""
        return max(self.client.roster.values(), key=lambda e: (e.score, e.kills), default=None)

    def _report_summary(self) -> str:
        best = self._best()
        return self.report.summary(best.name if best is not None else "", best.score if best is not None else 0,
                                   self.client.mode == C.MODE_UPLINK)

    def _finish_report(self, force: bool = False) -> None:
        """End-of-match medals and the final report, from the first roster the server took after the end."""
        client = self.client
        if not self.report_due or not (force or client.roster_phase == C.PHASE_ENDED):
            return
        self.report_due = False
        best = self._best()
        self._medals(self.report.finish(best.pid if best is not None and best.score > 0 else -1, client.pid))
        self.report_text = self._report_summary()
        self.pm.save()

    # ================================================================== #
    #  HUD
    # ================================================================== #
    def _update_hud(self, dt: float, alive: bool) -> None:
        client, hud, rig = self.client, self.hud, self.rig
        store = client.predictor.store
        ws = store.weapons
        slot = int(ws.slot[0])
        weapon = int(ws.weapon[0, slot])
        speed = float(np.hypot(store.move.vel[0, 0], store.move.vel[0, 2]))
        spread = float(W.spread_angle(ws.klass[0, slot], ws.spread_hip[0, slot], ws.spread_ads[0, slot], ws.bloom[0],
                                      store.move.mflags[0], speed))
        scoped = float(ws.ads_zoom[0, slot]) >= 2.0
        hud.set_crosshair(spread, R.vertical_fov(camera.fov), 0.0 if not alive else 1.0 - rig.ads_blend * (1.0 if scoped else 0.6), dt)
        hud.set_health(float(store.health[0]) if alive else 0.0, float(store.max_health[0]))
        hud.set_weapon(W.WEAPON_NAMES[weapon], int(ws.ammo[0, slot]), int(ws.mag[0, slot]), int(ws.reload_end[0]) > 0)
        lethal_left = float(store.ability_ready[0] - store.cmd_tick[0]) / max(float(store.ability_cooldown[0]), 1.0)
        tactical_left = float(store.tactical_ready[0] - store.cmd_tick[0]) / max(float(store.tactical_cooldown[0]), 1.0)
        hud.set_equipment(EQ.EQUIP_NAMES[int(store.equip[0, 0])], 1.0 - min(max(lethal_left, 0.0), 1.0),
                          EQ.EQUIP_NAMES[int(store.equip[0, 1])], 1.0 - min(max(tactical_left, 0.0), 1.0))
        self._update_rewards(dt, store)
        self._update_uplinks(dt, alive)
        for title, text, credits in self.pm.pop_notices():
            hud.push_reward("%s  +%s CR" % (title, SS.format_number(credits)))
            self.audio.play("contract", 0.8)
            self._show_banner(title, text, R.HUD_GOLD, 2.6)
        own = client.roster.get(client.pid)
        if client.mode == C.MODE_ESC:
            leader = max(client.roster.values(), key=lambda e: (e.score, e.kills), default=None)
            detail = ("LEADER  %s  ON WEAPON %d" % (leader.name, min(leader.score + 1, client.score_limit))
                      if leader is not None and leader.score > 0 else "a kill with the last gun wins")
            hud.set_match(client.mode, client.scores, client.time_left, client.team, own.kills if own is not None else 0,
                          client.score_limit, W.ESCALATION_STEP_OF.get(weapon, 0), detail)
        else:
            hud.set_match(client.mode, client.scores, client.time_left, client.team, own.kills if own is not None else 0,
                          client.score_limit)
        ended = client.phase == C.PHASE_ENDED
        if self.report_due:
            self._finish_report(self.clock >= self.report_deadline)
        show_board = ended or bool(held_keys["tab"])
        between = ended and bool(self.result)
        hud.set_report(self.report_text if ended else None, self.result if between else "MATCH REPORT",
                       R.HUD_GOLD if self.result in ("VICTORY", "") else R.HUD_WHITE)
        if show_board:
            entries = sorted(client.roster.values(), key=lambda e: (e.team if C.team_mode(client.mode) else 0, -e.score, -e.kills))
            rows = []
            for entry in entries:
                tag = "* " if entry.pid == client.pid else "  "
                side = ("%s  " % C.TEAM_NAMES[entry.team % 2]) if C.team_mode(client.mode) and entry.team < 2 else ""
                score = min(entry.score + 1, client.score_limit) if client.mode == C.MODE_ESC else entry.score
                rows.append(("%s%s%s" % (side, entry.name, "  (bot)" if entry.bot else ""), entry.kills, entry.deaths,
                             score, entry.ping, tag))
            hud.set_scoreboard(True, rows, "%s  -  %s" % (C.MODE_NAMES[client.mode % len(C.MODE_NAMES)], client.loaded.data.name),
                               "WEAPON" if client.mode == C.MODE_ESC else "SCORE")
        else:
            hud.set_scoreboard(False)
        self.banner_ttl = max(self.banner_ttl - dt, 0.0)
        if between:                    # the boards fill the middle: news and the countdown go under them
            if self.banner_ttl > 0.0:
                status = "%s  -  %s" % (self.banner, self.banner_sub) if self.banner_sub else self.banner
            else:
                status = "next match starts in %d seconds" % max(int(math.ceil(self.result_end - self.clock)), 0)
            hud.set_intermission(status)
            hud.set_center("")
            hud.set_hint("")
        elif self.dead:
            parts = (["by %s" % self.killer] if self.killer else []) + ([self.death_note] if self.death_note else [])
            hud.set_center("ELIMINATED", "  -  ".join(parts + ["respawning"]), R.HUD_CRIMSON)
        elif self.banner_ttl > 0.0:
            hud.set_center(self.banner, self.banner_sub, self.banner_rgb)
        else:
            hud.set_center("")
        if not between:
            hud.set_intermission(None)
        if self.hint_timer > 0.0:
            self.hint_timer -= dt
            if self.hint_timer <= 0.0:
                hud.set_hint("")
        self.hud_timer -= dt
        if self.hud_timer <= 0.0:
            self.hud_timer = 0.25
            snap = self.pm.snapshot()
            hud.set_economy("%s    +%s / s" % ("CR " + SS.format_number(snap["credits"]), SS.format_number(snap["income"])),
                            "LEVEL %d    XP %s / %s    PRESTIGE %d" % (snap["level"], SS.format_number(snap["xp"]),
                                                                     SS.format_number(snap["xp_need"]), snap["prestige"]))
            if self.settings["show_fps"]:
                pred = client.predictor
                hud.set_net("FPS %d  |  %s %d%%  |  PING %d ms  |  CORRECTIONS %d  |  PREDICTION ERROR %.3f" % (
                    round(self.fps), C.QUALITY_NAMES[self.scene.quality], round(self.scene.post.render_scale * 100.0),
                    round(client.ping_ms), pred.corrections, pred.last_error))
            else:
                hud.set_net("")
        hud.update(dt)
