"""
NEXUS-3D-EXTREME :: engine/bots.py
==================================
Vectorised bot brains.

Bots are ordinary players as far as the simulation is concerned: every tick
:class:`BotController` produces one input command per bot (button mask, yaw,
pitch) and the server feeds those commands through exactly the same
``simulate_commands`` path that human input takes.  There is no cheating
shortcut -- bots walk with the kinematic controller, obey weapon cadence and
spread, and their rounds are resolved by the same ray casts.

All per-bot state lives in arrays indexed by player slot, and one ``think``
call evaluates perception (batched line-of-sight rays through the octree),
target selection, path following on the navigation graph, aiming and trigger
discipline for every bot at once.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import mathx as M
from . import simulation as S
from .controller_3d import eye_height

DIFFICULTY_NAMES = ("RECRUIT", "REGULAR", "VETERAN")
# reaction (s), aim noise (rad), turn rate (rad/s), head preference, burst (s), pause (s), grenade chance / tick
_DIFFICULTY = np.array([
    [0.55, 0.070, 3.4, 0.05, 0.35, 0.75, 0.004],
    [0.32, 0.036, 5.8, 0.20, 0.50, 0.50, 0.008],
    [0.18, 0.016, 9.5, 0.45, 0.70, 0.30, 0.012],
], dtype=np.float64)

PERCEPTION_INTERVAL = 6          # ticks between line-of-sight refreshes
VIEW_RANGE = 75.0
FIELD_OF_VIEW_COS = float(np.cos(np.radians(105.0)))
MEMORY_TICKS = 4 * C.TICK_RATE   # how long a lost target is hunted
ARRIVE_RADIUS = 1.1
PREFERRED_RANGE = 11.0

# button combination for each of the 8 movement sectors relative to the view yaw
_SECTOR_BUTTONS = np.array([
    C.BTN_FORWARD,
    C.BTN_FORWARD | C.BTN_RIGHT,
    C.BTN_RIGHT,
    C.BTN_BACK | C.BTN_RIGHT,
    C.BTN_BACK,
    C.BTN_BACK | C.BTN_LEFT,
    C.BTN_LEFT,
    C.BTN_FORWARD | C.BTN_LEFT,
], dtype=np.uint16)


class BotController:
    """Per-slot AI state plus the batched decision step."""

    def __init__(self, game: S.GameWorld, difficulty: int = 1, seed: int = 4242):
        self.game = game
        self.difficulty = int(np.clip(difficulty, 0, len(DIFFICULTY_NAMES) - 1))
        self.rng = np.random.default_rng(seed)
        n = game.store.capacity
        self.target = np.full(n, -1, dtype=np.int64)
        self.react_until = np.zeros(n, dtype=np.int64)
        self.last_seen = np.zeros((n, 3), dtype=np.float64)
        self.memory_until = np.zeros(n, dtype=np.int64)
        self.visible = np.zeros((n, n), dtype=bool)
        self.alerted_by = np.full(n, -1, dtype=np.int64)
        self.alert_until = np.zeros(n, dtype=np.int64)
        self.goal_node = np.zeros(n, dtype=np.int64)
        self.goal_until = np.zeros(n, dtype=np.int64)
        self.path_to = np.full(n, -1, dtype=np.int64)
        self.path_deadline = np.zeros(n, dtype=np.int64)
        self.aim_yaw = np.zeros(n, dtype=np.float64)
        self.aim_pitch = np.zeros(n, dtype=np.float64)
        self.noise = np.zeros((n, 2), dtype=np.float64)
        self.burst_until = np.zeros(n, dtype=np.int64)
        self.pause_until = np.zeros(n, dtype=np.int64)
        self.strafe_dir = np.ones(n, dtype=np.float64)
        self.strafe_until = np.zeros(n, dtype=np.int64)
        self.stuck_ref = np.zeros((n, 3), dtype=np.float64)
        self.stuck_count = np.zeros(n, dtype=np.int64)
        self.jump_until = np.zeros(n, dtype=np.int64)
        self.spawn_seq = np.full(n, -1, dtype=np.int64)
        self.phase = self.rng.integers(0, 64, n)
        self.map_index = -1

    # ---- events from the simulation -----------------------------------------------
    def observe(self, events) -> None:
        """Let bots react to being shot: the attacker becomes known regardless of view cone."""
        tick = self.game.tick
        store = self.game.store
        for event in events:
            if event[0] == S.EV_DAMAGE:
                victim, attacker = event[1], event[2]
                if attacker != S.NO_PLAYER and store.is_bot[victim] and attacker != victim:
                    self.alerted_by[victim] = attacker
                    self.alert_until[victim] = tick + 2 * C.TICK_RATE

    def _reset(self, rows: np.ndarray) -> None:
        store = self.game.store
        self.target[rows] = -1
        self.memory_until[rows] = 0
        self.alert_until[rows] = 0
        self.path_to[rows] = -1
        self.goal_until[rows] = 0
        self.aim_yaw[rows] = store.yaw[rows]
        self.aim_pitch[rows] = 0.0
        self.noise[rows] = 0.0
        self.stuck_ref[rows] = store.move.pos[rows]
        self.stuck_count[rows] = 0
        self.spawn_seq[rows] = store.spawn_seq[rows]

    # ---- perception ---------------------------------------------------------------------
    def _perceive(self, rows: np.ndarray) -> None:
        """Refresh the bot x player visibility matrix with one batched octree query."""
        store = self.game.store
        tick = self.game.tick
        n = store.capacity
        alive = store.active & store.alive
        enemy = alive[None, :] & (store.team[None, :] != store.team[rows][:, None])
        eyes = store.eye_position(rows)
        centers = store.hitboxes.bound_center
        delta = centers[None, :, :] - eyes[:, None, :]
        dist = np.sqrt(np.sum(delta * delta, axis=2))
        facing = M.forward_vector(self.aim_yaw[rows], self.aim_pitch[rows])
        cos_angle = np.sum(delta * facing[:, None, :], axis=2) / np.maximum(dist, 1.0e-6)
        alerted = (self.alerted_by[rows][:, None] == np.arange(n)[None, :]) & (self.alert_until[rows] > tick)[:, None]
        consider = enemy & (dist < VIEW_RANGE) & ((cos_angle > FIELD_OF_VIEW_COS) | alerted | (dist < 4.0))
        seen = np.zeros((rows.size, n), dtype=bool)
        bi, ti = np.nonzero(consider)
        if bi.size:
            seen[bi, ti] = self.game.world.line_of_sight(eyes[bi], centers[ti])
        self.visible[rows] = seen

    # ---- decision step ----------------------------------------------------------------------
    def think(self, rows):
        """
        Commands for the bots in ``rows`` for this tick.
        Returns ``(buttons, yaw, pitch)`` aligned with ``rows``.
        """
        rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))
        k = rows.size
        buttons = np.zeros(k, dtype=np.uint16)
        if k == 0:
            return buttons, np.zeros(0), np.zeros(0)
        game, store = self.game, self.game.store
        nav = game.loaded.nav
        tick = game.tick
        rng = self.rng
        params = _DIFFICULTY[self.difficulty]
        reaction, aim_noise, turn_rate, head_pref, burst_len, pause_len, grenade_chance = params

        if self.map_index != game.map_index:
            self.map_index = game.map_index
            self.path_to[:] = -1
            self.goal_until[:] = 0
        fresh = self.spawn_seq[rows] != store.spawn_seq[rows]
        if fresh.any():
            self._reset(rows[fresh])
        alive = store.alive[rows] & (game.phase == C.PHASE_LIVE)
        if tick % PERCEPTION_INTERVAL == 0 or fresh.any():
            self._perceive(rows)

        pos = store.move.pos[rows]
        eyes = pos.copy()
        eyes[:, 1] += eye_height(store.move.crouch_t[rows])

        # ---- target selection: nearest visible enemy ---------------------------------------
        vis = self.visible[rows] & (store.active & store.alive)[None, :]
        centers = store.hitboxes.bound_center
        dist_all = np.sqrt(np.sum((centers[None, :, :] - eyes[:, None, :]) ** 2, axis=2))
        masked = np.where(vis, dist_all, np.inf)
        best = masked.argmin(axis=1)
        has_target = np.isfinite(masked[np.arange(k), best])
        new_target = np.where(has_target, best, -1)
        changed = has_target & (new_target != self.target[rows])
        jitter = rng.uniform(0.8, 1.35, k)
        self.react_until[rows] = np.where(changed, tick + (reaction * jitter * C.TICK_RATE).astype(np.int64),
                                          self.react_until[rows])
        self.target[rows] = new_target
        tgt = np.where(has_target, new_target, 0)
        self.last_seen[rows] = np.where(has_target[:, None], store.move.pos[tgt], self.last_seen[rows])
        self.memory_until[rows] = np.where(has_target, tick + MEMORY_TICKS, self.memory_until[rows])
        hunting = ~has_target & (self.memory_until[rows] > tick)
        target_dist = np.where(has_target, masked[np.arange(k), best], 1.0e9)

        # ---- goal node -------------------------------------------------------------------------
        if nav.count:
            chase = has_target | hunting
            chase_node = nav.nearest(self.last_seen[rows])
            expired = (self.goal_until[rows] <= tick) & ~chase
            wander = rng.integers(0, nav.count, k)
            goal = np.where(chase, chase_node, np.where(expired, wander, self.goal_node[rows]))
            self.goal_until[rows] = np.where(expired, tick + rng.integers(6, 14, k) * C.TICK_RATE, self.goal_until[rows])
            self.goal_node[rows] = goal

            # ---- path following (commit to one edge at a time) -----------------------------------
            path_to = self.path_to[rows]
            lost = (path_to < 0) | (self.path_deadline[rows] <= tick)
            if lost.any():
                path_to = np.where(lost, nav.nearest(pos), path_to)
                self.path_deadline[rows] = np.where(lost, tick + 3 * C.TICK_RATE, self.path_deadline[rows])
            node_pos = nav.nodes[path_to]
            flat = node_pos - pos
            flat[:, 1] = 0.0
            arrived = (np.sqrt(np.sum(flat * flat, axis=1)) < ARRIVE_RADIUS) & (np.abs(node_pos[:, 1] - pos[:, 1]) < 1.6)
            if arrived.any():
                at_goal = arrived & (path_to == goal)
                self.goal_until[rows] = np.where(at_goal & ~chase, 0, self.goal_until[rows])
                path_to = np.where(arrived, nav.step_towards(path_to, goal), path_to)
                self.path_deadline[rows] = np.where(arrived, tick + 4 * C.TICK_RATE, self.path_deadline[rows])
                node_pos = nav.nodes[path_to]
            self.path_to[rows] = path_to
            move = node_pos - pos
        else:
            move = self.last_seen[rows] - pos
        move[:, 1] = 0.0
        move_len = np.sqrt(np.sum(move * move, axis=1))
        move_dir = move / np.maximum(move_len, 1.0e-6)[:, None]
        wants_move = move_len > 0.25

        # ---- combat footwork: strafe, hold range, back off -------------------------------------------
        flip = self.strafe_until[rows] <= tick
        self.strafe_dir[rows] = np.where(flip, np.where(rng.random(k) < 0.5, -1.0, 1.0), self.strafe_dir[rows])
        self.strafe_until[rows] = np.where(flip, tick + rng.integers(30, 90, k), self.strafe_until[rows])
        to_target = store.move.pos[tgt] - pos
        to_target[:, 1] = 0.0
        to_target = to_target / np.maximum(np.sqrt(np.sum(to_target * to_target, axis=1)), 1.0e-6)[:, None]
        side = np.stack([to_target[:, 2], np.zeros(k), -to_target[:, 0]], axis=1) * self.strafe_dir[rows][:, None]
        engaged = has_target & (self.react_until[rows] <= tick)
        advance = np.where(target_dist > PREFERRED_RANGE, 0.9, np.where(target_dist < 4.5, -0.8, 0.0))
        combat_dir = side + to_target * advance[:, None]
        combat_dir = combat_dir / np.maximum(np.sqrt(np.sum(combat_dir * combat_dir, axis=1)), 1.0e-6)[:, None]
        close_fight = engaged & (target_dist < PREFERRED_RANGE * 1.6)
        move_dir = np.where(close_fight[:, None], combat_dir, move_dir)
        wants_move = wants_move | close_fight

        # ---- aiming ------------------------------------------------------------------------------------
        head = rng.random(k) < head_pref
        aim_point = np.where(head[:, None], store.hitboxes.world_center[tgt, 0], store.hitboxes.world_center[tgt, 1])
        want_yaw, want_pitch = M.yaw_pitch_from_direction(aim_point - eyes)
        travel_yaw = np.arctan2(move_dir[:, 0], move_dir[:, 2])
        look_yaw, look_pitch = M.yaw_pitch_from_direction(self.last_seen[rows] + np.array([0.0, 1.2, 0.0]) - eyes)
        desired_yaw = np.where(has_target, want_yaw, np.where(hunting & (move_len < 3.0), look_yaw, travel_yaw))
        desired_pitch = np.where(has_target, want_pitch, 0.0)
        self.noise[rows] = self.noise[rows] * 0.90 + rng.normal(0.0, aim_noise * 0.44, (k, 2))
        range_scale = np.clip(target_dist / 14.0, 0.45, 1.6)
        goal_yaw = desired_yaw + np.where(has_target, self.noise[rows, 0] * range_scale, 0.0)
        goal_pitch = desired_pitch + np.where(has_target, self.noise[rows, 1] * range_scale * 0.6, 0.0)
        max_turn = turn_rate * C.TICK_DT
        yaw_error = M.angle_delta(self.aim_yaw[rows], goal_yaw)
        speed_up = 1.0 + 2.2 * np.clip(np.abs(yaw_error) / 1.2, 0.0, 1.0)
        aim_yaw = M.wrap_angle(self.aim_yaw[rows] + np.clip(yaw_error, -max_turn * speed_up, max_turn * speed_up))
        aim_pitch = np.clip(self.aim_pitch[rows] + np.clip(goal_pitch - self.aim_pitch[rows], -max_turn, max_turn),
                            -C.PITCH_LIMIT, C.PITCH_LIMIT)
        self.aim_yaw[rows] = aim_yaw
        self.aim_pitch[rows] = aim_pitch

        # ---- trigger discipline -----------------------------------------------------------------------------
        on_target = (np.abs(M.angle_delta(aim_yaw, want_yaw)) < np.arctan2(0.55, np.maximum(target_dist, 0.5)) + 0.02) \
            & (np.abs(aim_pitch - want_pitch) < np.arctan2(0.9, np.maximum(target_dist, 0.5)) + 0.02)
        ws = store.weapons
        slot = ws.slot[rows]
        ammo = ws.ammo[rows, slot]
        reloading = ws.reload_end[rows] > 0
        can_shoot = engaged & on_target & ~reloading & (ammo > 0)
        start_burst = can_shoot & (self.burst_until[rows] <= tick) & (self.pause_until[rows] <= tick)
        burst_ticks = (burst_len * rng.uniform(0.7, 1.4, k) * C.TICK_RATE).astype(np.int64)
        pause_ticks = (pause_len * rng.uniform(0.6, 1.3, k) * C.TICK_RATE).astype(np.int64)
        self.burst_until[rows] = np.where(start_burst, tick + burst_ticks, self.burst_until[rows])
        self.pause_until[rows] = np.where(start_burst, tick + burst_ticks + pause_ticks, self.pause_until[rows])
        in_burst = can_shoot & (self.burst_until[rows] > tick)
        period = np.maximum(np.ceil(ws.interval[rows, slot]).astype(np.int64) + 2, 4)
        pulse = ((tick + self.phase[rows]) % period) == 0
        shoot = in_burst & (ws.auto[rows, slot] | pulse)
        ads = has_target & (target_dist > 9.0)

        # ---- movement buttons -----------------------------------------------------------------------------------
        relative = M.wrap_angle(np.arctan2(move_dir[:, 0], move_dir[:, 2]) - aim_yaw)
        sector = np.rint(relative / (np.pi / 4.0)).astype(np.int64) % 8
        move_buttons = np.where(wants_move, _SECTOR_BUTTONS[sector], 0).astype(np.uint16)
        sprint = wants_move & ~has_target & (np.abs(relative) < 0.6)
        hspeed = np.hypot(store.move.vel[rows, 0], store.move.vel[rows, 2])
        slide = sprint & (hspeed > 7.6) & (rng.random(k) < 0.004)

        # ---- stuck recovery ----------------------------------------------------------------------------------------
        check = (tick + self.phase[rows]) % 45 == 0
        if check.any():
            moved = np.sqrt(np.sum((pos - self.stuck_ref[rows]) ** 2, axis=1))
            stuck = check & wants_move & (moved < 0.6) & ~close_fight
            self.stuck_count[rows] = np.where(check, np.where(stuck, self.stuck_count[rows] + 1, 0), self.stuck_count[rows])
            self.jump_until[rows] = np.where(stuck, tick + 12, self.jump_until[rows])
            repath = stuck & (self.stuck_count[rows] >= 2)
            self.path_to[rows] = np.where(repath, -1, self.path_to[rows])
            self.goal_until[rows] = np.where(repath, 0, self.goal_until[rows])
            self.memory_until[rows] = np.where(repath, 0, self.memory_until[rows])
            self.stuck_ref[rows] = np.where(check[:, None], pos, self.stuck_ref[rows])
        on_ground = (store.move.mflags[rows] & C.MF_ON_GROUND) != 0
        jump = (self.jump_until[rows] > tick) & on_ground & ((tick + self.phase[rows]) % 2 == 0)

        # ---- utility ---------------------------------------------------------------------------------------------------
        low_ammo = ammo * 2 < ws.mag[rows, slot]
        reload = ~has_target & low_ammo & ~reloading & ((tick + self.phase[rows]) % 20 == 0)
        throw = (engaged & (target_dist > 8.0) & (target_dist < 26.0) & (store.cmd_tick[rows] >= store.ability_ready[rows])
                 & (rng.random(k) < grenade_chance))

        buttons |= move_buttons
        buttons |= np.where(sprint, C.BTN_SPRINT, 0).astype(np.uint16)
        buttons |= np.where(slide, C.BTN_SLIDE, 0).astype(np.uint16)
        buttons |= np.where(jump, C.BTN_JUMP, 0).astype(np.uint16)
        buttons |= np.where(shoot, C.BTN_SHOOT, 0).astype(np.uint16)
        buttons |= np.where(ads, C.BTN_ADS, 0).astype(np.uint16)
        buttons |= np.where(reload, C.BTN_RELOAD, 0).astype(np.uint16)
        buttons |= np.where(throw, C.BTN_ABILITY, 0).astype(np.uint16)
        buttons = np.where(alive, buttons, 0).astype(np.uint16)
        return buttons, aim_yaw, aim_pitch


def run_bot_match(map_key="crucible", bots: int = 8, ticks: int = 3600, difficulty: int = 1,
                  mode: int = C.MODE_TDM, seed: int = 7) -> dict:
    """
    Headless bots-only match used by the soak tests and for balancing.
    Returns a summary dictionary (kills, shots, stuck statistics, timings).
    """
    import time

    game = S.GameWorld(map_key, mode=mode, seed=seed)
    for index in range(bots):
        game.add_player("BOT-%02d" % index, is_bot=True)
    controller = BotController(game, difficulty, seed=seed + 1)
    game.drain_events()
    shots = hits = kills = explosions = 0
    worst = 0.0
    started = time.perf_counter()
    for _ in range(ticks):
        t0 = time.perf_counter()
        rows = game.bots()
        buttons, yaw, pitch = controller.think(rows)
        fired = game.step_commands(rows, buttons, yaw, pitch, game.tick + 1)
        game.resolve_shots(rows[fired])
        game.end_tick()
        events = game.drain_events()
        controller.observe(events)
        for event in events:
            kind = event[0]
            shots += kind == S.EV_SHOT
            hits += kind == S.EV_HIT
            kills += kind == S.EV_KILL
            explosions += kind == S.EV_EXPLOSION
        worst = max(worst, time.perf_counter() - t0)
    elapsed = time.perf_counter() - started
    store = game.store
    return {
        "ticks": ticks, "shots": int(shots), "hits": int(hits), "kills": int(kills), "explosions": int(explosions),
        "team_scores": game.team_scores.tolist(), "matches": game.matches_played,
        "mean_tick_ms": elapsed / ticks * 1000.0, "worst_tick_ms": worst * 1000.0,
        "min_y": float(store.move.pos[store.active, 1].min()), "finite": bool(np.isfinite(store.move.pos).all()),
        "kills_per_bot": store.kills[store.active].tolist(), "deaths_per_bot": store.deaths[store.active].tolist(),
    }
