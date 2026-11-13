"""
NEXUS-3D-EXTREME :: engine/simulation.py
========================================
The authoritative game state and the deterministic per-command simulation
shared by server and client.

Layout
------
``PlayerStore``        every per-player component as one contiguous array
                       (structure of arrays): movement, weapons, hitboxes,
                       vitals, scores.
``simulate_commands``  the function both sides run for one input command per
                       row: weapon state machine + kinematics.  Client side
                       prediction calls it on a one-row store, the server on
                       all rows at once.
``GrenadeStore``       pooled thrown equipment (grenades, charges, flasks,
                       hatchets), also stepped identically on client and server.
``ZoneStore``          fire pools and smoke clouds left behind by equipment.
``GameWorld``          server side orchestration: joining / leaving, hit
                       resolution, damage, explosions, respawns, match rules
                       and the event stream consumed by the network layer.

Nothing in this module touches sockets or the renderer.
"""
from __future__ import annotations

import struct

import numpy as np

from . import config as C
from . import equipment as EQ
from . import mathx as M
from . import maps
from . import weapons as W
from .controller_3d import (CollisionWorld, HitboxSet, KinematicState, HITBOX_MULTIPLIER, HITBOX_ZONE,
                            eye_height, step_kinematics)

# --------------------------------------------------------------------------- #
#  Event identifiers (tuples appended to ``GameWorld.events``)
# --------------------------------------------------------------------------- #
EV_HIT = 1          # (EV_HIT, shooter, victim, damage, zone, killed, (x, y, z))
EV_DAMAGE = 2       # (EV_DAMAGE, victim, attacker, damage, zone, (ax, ay, az))
EV_KILL = 3         # (EV_KILL, killer, victim, weapon, flags)
EV_SPAWN = 4        # (EV_SPAWN, player, (x, y, z), yaw)
EV_SHOT = 5         # (EV_SHOT, shooter, weapon, (ox, oy, oz), (ex, ey, ez))
EV_EXPLOSION = 6    # (EV_EXPLOSION, (x, y, z), owner, grenade_id, equipment)  blasts, stuns, fire and smoke
EV_GRENADE = 7      # (EV_GRENADE, grenade_id, owner, (x, y, z), (vx, vy, vz), fuse_ticks, spawn_tick, equipment)
EV_REWARD = 8       # (EV_REWARD, player, credits, xp, reason)
EV_MATCH = 9        # (EV_MATCH, phase, winner, map_index, mode)
EV_STICK = 10       # (EV_STICK, grenade_id, target player or NO_PLAYER for a wall, (x, y, z))
EV_STATUS = 11      # (EV_STATUS, player, status, ticks)   status: C.STATUS_STUN or C.STATUS_STIM

KILL_HEADSHOT = 1
KILL_EXPLOSIVE = 2
KILL_SUICIDE = 4

REWARD_COMBAT = 0
REWARD_KILL = 1
REWARD_HEADSHOT = 2
REWARD_WIN = 3
REWARD_MATCH = 4
REWARD_NAMES = ("COMBAT", "ELIMINATION", "HEADSHOT", "VICTORY", "MATCH COMPLETE")

WEAPON_GRENADE = 255                 # pseudo weapon id of kills by falling after a knock-back
NO_PLAYER = 255

_TICKS = C.TICK_RATE


# =========================================================================== #
#  Player store
# =========================================================================== #
class PlayerStore:
    """All per-player state for ``capacity`` slots, one array per component."""

    OWNER_STRUCT = struct.Struct("<3f3fHHH5f2fBHHIdfIIfBIHBBBII")
    OWNER_SIZE = OWNER_STRUCT.size

    def __init__(self, capacity: int = C.MAX_PLAYERS):
        n = int(capacity)
        self.capacity = n
        self.move = KinematicState(n)
        self.weapons = W.WeaponState(n)
        self.hitboxes = HitboxSet(n)
        self.active = np.zeros(n, dtype=bool)
        self.is_bot = np.zeros(n, dtype=bool)
        self.alive = np.zeros(n, dtype=bool)
        self.team = np.zeros(n, dtype=np.int64)
        self.yaw = np.zeros(n, dtype=np.float64)
        self.pitch = np.zeros(n, dtype=np.float64)
        self.health = np.zeros(n, dtype=np.float64)
        self.max_health = np.full(n, C.BASE_HEALTH, dtype=np.float64)
        self.cmd_tick = np.zeros(n, dtype=np.int64)
        self.respawn_tick = np.zeros(n, dtype=np.int64)
        self.protect_until = np.zeros(n, dtype=np.int64)
        self.last_damage_tick = np.full(n, -100000, dtype=np.int64)
        self.last_attacker = np.full(n, -1, dtype=np.int64)
        self.ability_ready = np.zeros(n, dtype=np.int64)                 # lethal (G) usable from this command tick
        self.ability_cooldown = np.full(n, EQ.cooldown_ticks(EQ.DEFAULT_EQUIPMENT[0]), dtype=np.int64)
        self.tactical_ready = np.zeros(n, dtype=np.int64)                # tactical (F) usable from this command tick
        self.tactical_cooldown = np.full(n, EQ.cooldown_ticks(EQ.DEFAULT_EQUIPMENT[1]), dtype=np.int64)
        self.equip = np.tile(np.array(EQ.DEFAULT_EQUIPMENT, dtype=np.int64), (n, 1))   # (lethal, tactical) ids
        self.pending_equip = self.equip.copy()                           # takes effect at the next spawn
        self.stun_until = np.zeros(n, dtype=np.int64)                    # command tick until which movement is slowed
        self.heal_left = np.zeros(n, dtype=np.float64)                   # stim healing still to be delivered
        self.heal_rate = np.zeros(n, dtype=np.float64)                   # health per second while it lasts
        self.regen_delay = np.full(n, int(round(C.HEALTH_REGEN_DELAY * _TICKS)), dtype=np.int64)
        self.kills = np.zeros(n, dtype=np.int64)
        self.deaths = np.zeros(n, dtype=np.int64)
        self.score = np.zeros(n, dtype=np.int64)
        self.streak = np.zeros(n, dtype=np.int64)
        self.spawn_seq = np.zeros(n, dtype=np.int64)
        self.operator = np.zeros((n, W.OPERATOR_COUNT), dtype=np.int64)
        self.prestige = np.zeros(n, dtype=np.int64)
        self.pending_credits = np.zeros(n, dtype=np.float64)
        self.blast_resist = np.ones(n, dtype=np.float64)           # explosive damage multiplier (Blast Plating)
        self.names = [""] * n

    # ---- convenience -------------------------------------------------------------
    def active_rows(self) -> np.ndarray:
        return np.flatnonzero(self.active)

    def eye_position(self, rows) -> np.ndarray:
        rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))
        eye = self.move.pos[rows].copy()
        eye[:, 1] += eye_height(self.move.crouch_t[rows])
        return eye

    def refresh_hitboxes(self, rows=None) -> None:
        """Re-derive the anatomical boxes of ``rows`` from their root state."""
        rows = self.active_rows() if rows is None else np.atleast_1d(np.asarray(rows, dtype=np.intp))
        if rows.size:
            self.hitboxes.update(rows, self.move.pos[rows], self.yaw[rows], self.move.crouch_t[rows],
                                 self.alive[rows] & self.active[rows])

    def apply_operator(self, row: int, operator, prestige: int) -> None:
        """Install operator upgrade levels and derive the stats they control."""
        op = W.clamp_operator(operator)
        self.operator[row] = op
        self.prestige[row] = int(prestige)
        eff = W.OPERATOR_EFFECT
        self.max_health[row] = C.BASE_HEALTH                       # fixed vital ceiling for everybody
        self.blast_resist[row] = 1.0 - eff[W.OP_PLATING] * op[W.OP_PLATING]
        self.refresh_cooldowns(row)
        self.regen_delay[row] = max(int(round(C.HEALTH_REGEN_DELAY * (1.0 - eff[W.OP_NANITES] * op[W.OP_NANITES]) * _TICKS)), _TICKS // 2)

    def refresh_cooldowns(self, row: int) -> None:
        """Cooldowns of the equipped lethal and tactical after the operator's capacitor upgrade."""
        reduction = float(W.OPERATOR_EFFECT[W.OP_CAPACITOR] * self.operator[row, W.OP_CAPACITOR])
        self.ability_cooldown[row] = EQ.cooldown_ticks(int(self.equip[row, 0]), reduction)
        self.tactical_cooldown[row] = EQ.cooldown_ticks(int(self.equip[row, 1]), reduction)

    def apply_equipment(self, row: int, equipment) -> None:
        """Equip ``(lethal, tactical)`` immediately (invalid ids fall back to the defaults)."""
        lethal, tactical = EQ.sanitize(equipment)
        self.equip[row] = (lethal, tactical)
        self.pending_equip[row] = (lethal, tactical)
        self.refresh_cooldowns(row)

    # ---- owner state (everything the predicting client must agree on) ---------------
    def pack_owner(self, row: int) -> bytes:
        mv, ws = self.move, self.weapons
        p, v = mv.pos[row], mv.vel[row]
        return self.OWNER_STRUCT.pack(
            p[0], p[1], p[2], v[0], v[1], v[2],
            int(mv.mflags[row]), int(mv.prev_buttons[row]), int(ws.prev_buttons[row]),
            mv.slide_t[row], mv.slide_cd[row], mv.crouch_t[row], mv.air_t[row], mv.jump_buf[row],
            mv.slide_dir[row, 0], mv.slide_dir[row, 1],
            int(ws.slot[row]), int(ws.ammo[row, 0]) & 0xFFFF, int(ws.ammo[row, 1]) & 0xFFFF,
            int(ws.reload_end[row]) & 0xFFFFFFFF, float(ws.next_fire[row]), ws.bloom[row],
            int(ws.shots_fired[row]) & 0xFFFFFFFF, int(self.ability_ready[row]) & 0xFFFFFFFF,
            self.health[row], int(self.alive[row]), int(self.cmd_tick[row]) & 0xFFFFFFFF,
            int(round(self.max_health[row])), int(self.spawn_seq[row]) & 0xFF,
            int(self.equip[row, 0]), int(self.equip[row, 1]), int(self.tactical_ready[row]) & 0xFFFFFFFF,
            int(self.stun_until[row]) & 0xFFFFFFFF)

    def unpack_owner(self, row: int, payload: bytes, offset: int = 0) -> None:
        (px, py, pz, vx, vy, vz, mflags, mprev, wprev, slide_t, slide_cd, crouch_t, air_t, jump_buf, sdx, sdz,
         slot, ammo0, ammo1, reload_end, next_fire, bloom, shots, ability_ready, health, alive, cmd_tick,
         max_health, spawn_seq, lethal, tactical, tactical_ready, stun_until) = self.OWNER_STRUCT.unpack_from(payload, offset)
        mv, ws = self.move, self.weapons
        mv.pos[row] = (px, py, pz)
        mv.vel[row] = (vx, vy, vz)
        mv.mflags[row] = mflags
        mv.prev_buttons[row] = mprev
        ws.prev_buttons[row] = wprev
        mv.slide_t[row] = slide_t
        mv.slide_cd[row] = slide_cd
        mv.crouch_t[row] = crouch_t
        mv.air_t[row] = air_t
        mv.jump_buf[row] = jump_buf
        mv.slide_dir[row] = (sdx, sdz)
        ws.slot[row] = slot
        ws.ammo[row] = (ammo0, ammo1)
        ws.reload_end[row] = reload_end
        ws.next_fire[row] = next_fire
        ws.bloom[row] = bloom
        ws.shots_fired[row] = shots
        self.ability_ready[row] = ability_ready
        self.health[row] = health
        self.alive[row] = bool(alive)
        self.cmd_tick[row] = cmd_tick
        self.max_health[row] = max_health
        self.spawn_seq[row] = spawn_seq
        if (lethal, tactical) != (int(self.equip[row, 0]), int(self.equip[row, 1])):
            self.apply_equipment(row, (lethal, tactical))
        self.tactical_ready[row] = tactical_ready
        self.stun_until[row] = stun_until


def simulate_commands(store: PlayerStore, world: CollisionWorld, rows, buttons, yaw, pitch, cmd_tick):
    """
    Run one input command for each of ``rows`` (the deterministic core shared
    by prediction and authority).  Returns ``(fired, lethal, tactical)``
    boolean masks aligned with ``rows``: a round left the barrel, the lethal
    (G) or the tactical (F) was used.  Dead rows only record the command tick.
    """
    rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))
    n = rows.size
    fired = np.zeros(n, dtype=bool)
    ability = np.zeros(n, dtype=bool)
    tactical = np.zeros(n, dtype=bool)
    if n == 0:
        return fired, ability, tactical
    buttons = np.broadcast_to(np.asarray(buttons, dtype=np.uint16), (n,))
    yaw = np.broadcast_to(np.asarray(yaw, dtype=np.float64), (n,))
    pitch = np.broadcast_to(np.asarray(pitch, dtype=np.float64), (n,))
    ticks = np.broadcast_to(np.asarray(cmd_tick, dtype=np.int64), (n,))
    store.cmd_tick[rows] = ticks
    alive = store.alive[rows]
    if not alive.any():
        return fired, ability, tactical
    live = rows[alive]
    b = buttons[alive]
    t = ticks[alive]
    store.yaw[live] = M.wrap_angle(yaw[alive])
    store.pitch[live] = np.clip(pitch[alive], -C.PITCH_LIMIT, C.PITCH_LIMIT)
    previous = store.weapons.prev_buttons[live]
    pressed = ((b & C.BTN_ABILITY) != 0) & ((previous & C.BTN_ABILITY) == 0)
    used = pressed & (t >= store.ability_ready[live])
    store.ability_ready[live] = np.where(used, t + store.ability_cooldown[live], store.ability_ready[live])
    pressed_tactical = ((b & C.BTN_TACTICAL) != 0) & ((previous & C.BTN_TACTICAL) == 0)
    used_tactical = pressed_tactical & (t >= store.tactical_ready[live])
    store.tactical_ready[live] = np.where(used_tactical, t + store.tactical_cooldown[live], store.tactical_ready[live])
    shot = W.step_weapons(store.weapons, live, b, t, True)
    stunned = t < store.stun_until[live]
    step_kinematics(store.move, live, b, store.yaw[live], world,
                    speed_mul=store.weapons.current("mobility", live) * np.where(stunned, EQ.STUN_FACTOR, 1.0))
    fired[alive] = shot
    ability[alive] = used
    tactical[alive] = used_tactical
    return fired, ability, tactical


def shot_rays(store: PlayerStore, rows, player_ids=None):
    """
    Rays of the rounds that ``rows`` have just fired, for any number of
    shooters at once.  Returns ``(origins, directions, starts, counts, cones)``
    where pellet ``k`` of shooter ``i`` sits at index ``starts[i] + k``.

    Deterministic: every pellet's deflection is hashed from the shooter's
    player id and lifetime shot counter, so the predicting client (which
    passes its real id through ``player_ids``) draws exactly the tracers the
    server evaluates.
    """
    rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))
    ids = rows if player_ids is None else np.atleast_1d(np.asarray(player_ids, dtype=np.int64))
    ws, mv = store.weapons, store.move
    slot = ws.slot[rows]
    origin = mv.pos[rows].copy()
    origin[:, 1] += eye_height(mv.crouch_t[rows])
    forward = M.forward_vector(store.yaw[rows], store.pitch[rows])
    counts = ws.pellets[rows, slot]
    speed = np.hypot(mv.vel[rows, 0], mv.vel[rows, 2])
    bloom = np.maximum(ws.bloom[rows] - ws.bloom_shot[rows, slot], 0.0)
    cones = W.spread_angle(ws.klass[rows, slot], ws.spread_hip[rows, slot], ws.spread_ads[rows, slot],
                           bloom, mv.mflags[rows], speed)
    starts = np.cumsum(counts) - counts
    own = np.repeat(np.arange(rows.size), counts)
    pellet = np.arange(own.size) - starts[own]
    keys = ((ids[own] * 7919 + ws.shots_fired[rows][own] * 31 + 17) * 16 + pellet) & 0xFFFFFFFF
    directions = M.apply_spread(forward[own], M.spread_disc_keys(keys), cones[own])
    return origin[own], directions, starts, counts, cones


# =========================================================================== #
#  Grenades
# =========================================================================== #
class GrenadeStore:
    """Fixed capacity pool of thrown equipment (grenades, charges, flasks, hatchets)."""

    def __init__(self, capacity: int = C.GRENADE_CAPACITY):
        n = int(capacity)
        self.capacity = n
        self.active = np.zeros(n, dtype=bool)
        self.pos = np.zeros((n, 3), dtype=np.float64)
        self.vel = np.zeros((n, 3), dtype=np.float64)
        self.fuse = np.zeros(n, dtype=np.int64)
        self.owner = np.full(n, -1, dtype=np.int64)
        self.gid = np.zeros(n, dtype=np.int64)
        self.equip = np.zeros(n, dtype=np.int64)
        self.stuck = np.zeros(n, dtype=bool)
        self.target = np.full(n, -1, dtype=np.int64)              # player a charge is stuck to
        self.offset = np.zeros((n, 3), dtype=np.float64)          # its position relative to that player
        self.normal = np.zeros((n, 3), dtype=np.float64)          # surface a stuck item sits in
        self.heading = np.zeros((n, 3), dtype=np.float64)         # direction of flight (orientation of hatchets)
        self.just_hit = np.zeros(n, dtype=bool)                   # stopped in a wall during the last step
        self.next_gid = 1

    def spawn(self, position, velocity, fuse_ticks: int, owner: int, gid: int | None = None, equip: int = 0) -> int:
        free = np.flatnonzero(~self.active)
        slot = int(free[0]) if free.size else int(np.argmin(self.fuse))
        if gid is None:
            gid = self.next_gid
            self.next_gid = self.next_gid % 250 + 1
        velocity = np.asarray(velocity, dtype=np.float64)
        speed = float(np.sqrt(np.dot(velocity, velocity)))
        self.active[slot] = True
        self.pos[slot] = position
        self.vel[slot] = velocity
        self.fuse[slot] = int(fuse_ticks)
        self.owner[slot] = int(owner)
        self.gid[slot] = int(gid)
        self.equip[slot] = int(equip)
        self.stuck[slot] = False
        self.target[slot] = -1
        self.offset[slot] = 0.0
        self.normal[slot] = (0.0, 1.0, 0.0)
        self.heading[slot] = velocity / speed if speed > 1.0e-6 else (0.0, 0.0, 1.0)
        self.just_hit[slot] = False
        return slot

    def stick(self, slot: int, position, normal=None, target: int = -1, offset=None) -> None:
        """Stop an item where it is (a sticky charge starts its short fuse)."""
        self.pos[slot] = position
        self.vel[slot] = 0.0
        self.stuck[slot] = True
        self.target[slot] = int(target)
        if offset is not None:
            self.offset[slot] = offset
        if normal is not None:
            self.normal[slot] = normal
        equip = int(self.equip[slot])
        if EQ.KIND[equip] == EQ.KIND_STICKY:
            self.fuse[slot] = min(int(self.fuse[slot]), EQ.stick_ticks(equip))

    def find(self, gid: int) -> int:
        slots = np.flatnonzero(self.active & (self.gid == gid))
        return int(slots[0]) if slots.size else -1

    def remove_gid(self, gid: int) -> None:
        self.active[self.active & (self.gid == gid)] = False

    def clear(self) -> None:
        self.active[:] = False


def step_grenades(gs: GrenadeStore, world: CollisionWorld, dt: float = C.TICK_DT) -> np.ndarray:
    """
    Advance every live item one tick and return the slots whose fuse ran out.

    Flying items fall with their own gravity scale and are swept against the
    bullet layer.  Bouncing kinds (grenades, canisters) reflect with
    restitution; sticky charges, hatchets and fire flasks stop dead in the
    first wall they touch and are flagged in ``just_hit`` for the caller.
    """
    gs.just_hit[:] = False
    live = np.flatnonzero(gs.active)
    if live.size == 0:
        return live
    idx = live[~gs.stuck[live]]
    if idx.size:
        equip = gs.equip[idx]
        vel = gs.vel[idx]
        vel[:, 1] -= C.GRAVITY * EQ.GRAVITY[equip] * dt
        delta = vel * dt
        dist = M.length(delta)
        direction = delta / np.maximum(dist, 1.0e-9)[:, None]
        t, item, normal = world.raycast(gs.pos[idx], direction, dist, C.LAYER_BULLET)
        hit = item >= 0
        travel = np.where(hit, np.maximum(t - 0.02, 0.0), dist)
        pos = gs.pos[idx] + direction * travel[:, None]
        bounces = EQ.BOUNCES[equip]
        into = M.dot(vel, normal)
        bounce = vel - (1.0 + C.GRENADE_RESTITUTION) * into[:, None] * normal
        bounce *= 0.86
        resting = hit & (normal[:, 1] > 0.7) & (np.abs(into) < 1.4)
        bounce[resting, 1] = 0.0
        bounce[resting] *= 0.80
        vel = np.where((hit & bounces)[:, None], bounce, vel)
        gs.pos[idx] = pos
        gs.vel[idx] = vel
        moving = dist > 1.0e-6
        gs.heading[idx[moving]] = direction[moving]
        for k in np.flatnonzero(hit & ~bounces).tolist():
            slot = int(idx[k])
            gs.stick(slot, pos[k], normal[k])
            gs.just_hit[slot] = True
    gs.fuse[live] -= 1
    return live[gs.fuse[live] <= 0]


class ZoneStore:
    """Fire pools and smoke clouds: where they are, how large and the tick they end."""

    def __init__(self, capacity: int = C.ZONE_CAPACITY):
        n = int(capacity)
        self.capacity = n
        self.active = np.zeros(n, dtype=bool)
        self.pos = np.zeros((n, 3), dtype=np.float64)
        self.radius = np.zeros(n, dtype=np.float64)
        self.born = np.zeros(n, dtype=np.int64)
        self.until = np.zeros(n, dtype=np.int64)
        self.owner = np.full(n, -1, dtype=np.int64)
        self.equip = np.zeros(n, dtype=np.int64)

    def spawn(self, position, radius: float, born: int, until: int, owner: int, equip: int) -> int:
        free = np.flatnonzero(~self.active)
        slot = int(free[0]) if free.size else int(np.argmin(self.until))
        self.active[slot] = True
        self.pos[slot] = position
        self.radius[slot] = float(radius)
        self.born[slot] = int(born)
        self.until[slot] = int(until)
        self.owner[slot] = int(owner)
        self.equip[slot] = int(equip)
        return slot

    def of_kind(self, kind: int) -> np.ndarray:
        return np.flatnonzero(self.active & (EQ.KIND[self.equip] == kind))

    def clear(self) -> None:
        self.active[:] = False


# =========================================================================== #
#  Game world (authoritative)
# =========================================================================== #
class GameWorld:
    """Authoritative match state: players, projectiles, rules and events."""

    def __init__(self, map_key="datacore", mode: int = C.MODE_TDM, score_limit: int | None = None,
                 time_limit: float = C.MATCH_TIME_LIMIT, upgrades_enabled: bool = True, seed: int = 1234,
                 capacity: int = C.MAX_PLAYERS):
        self.mode = int(mode)
        self.upgrades_enabled = bool(upgrades_enabled)
        self.score_limit = int(score_limit) if score_limit else (
            C.TDM_SCORE_LIMIT if self.mode == C.MODE_TDM else C.FFA_SCORE_LIMIT)
        self.time_limit = float(time_limit)
        self.rng = np.random.default_rng(seed)
        self.store = PlayerStore(capacity)
        self.grenades = GrenadeStore()
        self.zones = ZoneStore()
        self.events: list = []
        self.tick = 0
        self.phase = C.PHASE_LIVE
        self.time_left = self.time_limit
        self.team_scores = np.zeros(2, dtype=np.int64)
        self.winner = NO_PLAYER
        self.intermission_end = 0
        self.matches_played = 0
        self.next_map = None                 # arena to load when the intermission ends (map rotation)
        self.loaded = maps.load_map(map_key)
        self.map_index = self.loaded.data.index
        self.world = self.loaded.world

    # ---- map handling ----------------------------------------------------------------
    def change_map(self, map_key) -> None:
        self.loaded = maps.load_map(map_key)
        self.map_index = self.loaded.data.index
        self.world = self.loaded.world
        self.grenades.clear()
        self.zones.clear()
        self.store.move._cache_version = -1

    # ---- membership --------------------------------------------------------------------
    def add_player(self, name: str, is_bot: bool = False, weapon_ids=None, levels=None, operator=None,
                   prestige: int = 0, team: int | None = None, equipment=None) -> int:
        """Claim a free slot; returns the player id or -1 when the server is full."""
        s = self.store
        free = np.flatnonzero(~s.active)
        if free.size == 0:
            return -1
        row = int(free[0])
        if weapon_ids is None:
            weapon_ids = W.DEFAULT_LOADOUT
        if levels is None or not self.upgrades_enabled:
            levels = np.zeros((W.LOADOUT_SLOTS, W.UPGRADE_COUNT), dtype=np.int64)
        if operator is None or not self.upgrades_enabled:
            operator = np.zeros(W.OPERATOR_COUNT, dtype=np.int64)
        s.active[row] = True
        s.is_bot[row] = bool(is_bot)
        s.names[row] = str(name)[:16] or ("BOT-%02d" % row if is_bot else "PLAYER-%02d" % row)
        s.kills[row] = s.deaths[row] = s.score[row] = s.streak[row] = 0
        s.pending_credits[row] = 0.0
        s.cmd_tick[row] = self.tick
        s.ability_ready[row] = 0
        s.weapons.shots_fired[row] = 0
        s.weapons.set_loadout(row, weapon_ids, levels)
        s.apply_operator(row, operator, prestige)
        s.apply_equipment(row, EQ.DEFAULT_EQUIPMENT if equipment is None else equipment)
        s.tactical_ready[row] = 0
        if self.mode == C.MODE_TDM:
            if team is None or team not in (0, 1):
                counts = np.bincount(s.team[s.active & (np.arange(s.capacity) != row)], minlength=2)[:2]
                team = int(np.argmin(counts))
            s.team[row] = team
        else:
            s.team[row] = 2 + row
        self.spawn_player(row)
        return row

    def request_equipment(self, row: int, equipment) -> bool:
        """Queue a different lethal / tactical; it is handed out at the player's next spawn."""
        chosen = EQ.sanitize(equipment)
        changed = (int(self.store.pending_equip[row, 0]), int(self.store.pending_equip[row, 1])) != chosen
        self.store.pending_equip[row] = chosen
        return changed

    def remove_player(self, row: int) -> None:
        s = self.store
        s.active[row] = False
        s.alive[row] = False
        s.is_bot[row] = False
        s.hitboxes.valid[row] = False
        s.names[row] = ""

    def humans(self) -> np.ndarray:
        s = self.store
        return np.flatnonzero(s.active & ~s.is_bot)

    def bots(self) -> np.ndarray:
        s = self.store
        return np.flatnonzero(s.active & s.is_bot)

    # ---- spawning ------------------------------------------------------------------------
    def choose_spawn(self, row: int):
        """Spawn point of the player's side that is farthest from every living enemy."""
        s, data = self.store, self.loaded.data
        if self.mode == C.MODE_TDM:
            options = np.flatnonzero(data.spawn_team == s.team[row])
        else:
            options = np.arange(data.spawn_pos.shape[0])
        points = data.spawn_pos[options]
        others = s.active & s.alive
        others[row] = False
        score = self.rng.uniform(0.0, 3.0, options.size)
        if others.any():
            delta = points[:, None, :] - s.move.pos[others][None, :, :]
            dist = np.sqrt(np.sum(delta * delta, axis=2))
            enemy = (s.team[others] != s.team[row])[None, :]
            nearest_enemy = np.where(enemy, dist, np.inf).min(axis=1)
            nearest_any = dist.min(axis=1)
            score += np.minimum(nearest_enemy, 60.0)
            score -= np.where(nearest_any < 1.6, 1000.0, 0.0)
        pick = int(options[int(np.argmax(score))])
        return data.spawn_pos[pick].copy(), float(data.spawn_yaw[pick])

    def spawn_player(self, row: int) -> None:
        s = self.store
        position, yaw = self.choose_spawn(row)
        s.move.reset_rows(np.array([row]), position)
        s.weapons.reset_rows(np.array([row]))
        s.yaw[row] = yaw
        s.pitch[row] = 0.0
        s.health[row] = s.max_health[row]
        s.alive[row] = True
        s.protect_until[row] = self.tick + int(C.SPAWN_PROTECTION * _TICKS)
        s.last_damage_tick[row] = -100000
        s.last_attacker[row] = -1
        if (int(s.pending_equip[row, 0]), int(s.pending_equip[row, 1])) != (int(s.equip[row, 0]), int(s.equip[row, 1])):
            s.apply_equipment(row, s.pending_equip[row])            # a new class: both items ready at once
            s.ability_ready[row] = s.cmd_tick[row]
            s.tactical_ready[row] = s.cmd_tick[row]
        else:
            s.ability_ready[row] = min(int(s.ability_ready[row]), int(s.cmd_tick[row]) + int(s.ability_cooldown[row]))
            s.tactical_ready[row] = min(int(s.tactical_ready[row]), int(s.cmd_tick[row]) + int(s.tactical_cooldown[row]))
        s.stun_until[row] = 0
        s.heal_left[row] = 0.0
        s.spawn_seq[row] = (int(s.spawn_seq[row]) + 1) & 0xFF
        s.refresh_hitboxes(np.array([row]))
        self.events.append((EV_SPAWN, row, tuple(position.tolist()), yaw))

    # ---- command execution -------------------------------------------------------------------
    def step_commands(self, rows, buttons, yaw, pitch, cmd_tick):
        """Simulate one command per row and use any equipment that was triggered."""
        fired, lethal, tactical = simulate_commands(self.store, self.world, rows, buttons, yaw, pitch, cmd_tick)
        rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))
        for row in rows[lethal].tolist():
            self.throw_equipment(int(row), int(self.store.equip[row, 0]))
        for row in rows[tactical].tolist():
            self.use_tactical(int(row))
        return fired

    # ---- hitscan ---------------------------------------------------------------------------------
    def resolve_shots(self, rows, rewind=None) -> None:
        """
        Evaluate the rounds ``rows`` have just fired.  All pellets of all
        shooters share one batched world ray cast; each shooter is then tested
        against the hitboxes inside its own ``rewind(row)`` context (supplied
        by the lag compensator, which swaps the historical hitbox state the
        shooter was looking at into ``store.hitboxes`` for that moment).
        """
        rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))
        if rows.size == 0:
            return
        origins, directions, starts, counts, _ = shot_rays(self.store, rows)
        t_world, _, _ = self.world.raycast(origins, directions, C.MAX_BULLET_RANGE)
        for k, row in enumerate(rows.tolist()):
            span = slice(int(starts[k]), int(starts[k] + counts[k]))
            if rewind is None:
                self._score_shot(row, origins[span], directions[span], t_world[span])
            else:
                with rewind(row):
                    self._score_shot(row, origins[span], directions[span], t_world[span])

    def resolve_shot(self, row: int) -> None:
        """Single shooter convenience wrapper around :meth:`resolve_shots`."""
        self.resolve_shots(np.array([row]))

    def _score_shot(self, row: int, origins, directions, t_world) -> None:
        s = self.store
        ws = s.weapons
        slot = int(ws.slot[row])
        s.protect_until[row] = 0
        targets = s.active & s.alive & (s.team != s.team[row]) & (s.protect_until <= self.tick)
        targets[row] = False
        if self.phase != C.PHASE_LIVE:
            targets[:] = False
        t_hit, entity, box = s.hitboxes.raycast(origins, directions, t_world, targets)
        ends = origins + directions * t_hit[:, None]
        self.events.append((EV_SHOT, row, int(ws.weapon[row, slot]), tuple(origins[0].tolist()), tuple(ends[0].tolist())))
        struck = entity >= 0
        if not struck.any():
            return
        falloff = W.damage_falloff(t_hit[struck], ws.range_start[row, slot], ws.range_end[row, slot],
                                   ws.range_min[row, slot])
        damage = ws.damage[row, slot] * falloff * HITBOX_MULTIPLIER[box[struck]]
        victims = entity[struck]
        zones = HITBOX_ZONE[box[struck]]
        points = ends[struck]
        for victim in np.unique(victims).tolist():
            mine = victims == victim
            total = float(damage[mine].sum())
            zone = int(zones[mine].min())
            point = points[mine][int(np.argmin(zones[mine]))]
            self.apply_damage(int(victim), row, total, zone, int(ws.weapon[row, slot]), point)

    # ---- damage ---------------------------------------------------------------------------------------
    def apply_damage(self, victim: int, attacker: int, amount: float, zone: int, weapon: int, point,
                     explosive: bool = False) -> float:
        """Apply damage, emit hit / damage / kill events and credit the attacker."""
        s = self.store
        if not s.alive[victim] or amount <= 0.0 or self.phase != C.PHASE_LIVE:
            return 0.0
        dealt = float(min(amount, s.health[victim]))
        s.health[victim] -= dealt
        s.last_damage_tick[victim] = self.tick
        killed = s.health[victim] <= 1.0e-6
        hostile = 0 <= attacker != victim
        if hostile:
            s.last_attacker[victim] = attacker
            s.pending_credits[attacker] += dealt
            source = tuple(s.move.pos[attacker].tolist())
            self.events.append((EV_HIT, attacker, victim, dealt, zone, bool(killed), tuple(np.asarray(point).tolist())))
        else:
            source = tuple(np.asarray(point).tolist())
        self.events.append((EV_DAMAGE, victim, attacker if attacker >= 0 else NO_PLAYER, dealt, zone, source))
        if killed:
            flags = (KILL_HEADSHOT if zone == C.ZONE_HEAD and not explosive else 0) | (KILL_EXPLOSIVE if explosive else 0)
            self.kill_player(victim, attacker, weapon, flags)
        return dealt

    def kill_player(self, victim: int, attacker: int, weapon: int, flags: int) -> None:
        s = self.store
        if not s.alive[victim]:
            return
        s.alive[victim] = False
        s.health[victim] = 0.0
        s.deaths[victim] += 1
        s.streak[victim] = 0
        s.respawn_tick[victim] = self.tick + int(C.RESPAWN_TIME * _TICKS)
        s.hitboxes.valid[victim] = False
        hostile = 0 <= attacker != victim
        if hostile:
            s.kills[attacker] += 1
            s.streak[attacker] += 1
            bonus = 25 * int(min(s.streak[attacker] - 1, 8))
            head = 50 if flags & KILL_HEADSHOT else 0
            s.score[attacker] += 100 + head
            if self.mode == C.MODE_TDM and s.team[attacker] < 2:
                self.team_scores[s.team[attacker]] += 1
            self.events.append((EV_REWARD, attacker, 100 + bonus, 100, REWARD_KILL))
            if head:
                self.events.append((EV_REWARD, attacker, head, 50, REWARD_HEADSHOT))
            killer = attacker
        else:
            flags |= KILL_SUICIDE
            s.score[victim] = max(int(s.score[victim]) - 50, 0)
            killer = victim if attacker == victim else NO_PLAYER
        self.events.append((EV_KILL, killer, victim, weapon, flags))

    # ---- equipment -------------------------------------------------------------------------------------------
    def throw_equipment(self, row: int, equip: int) -> int:
        """Throw a grenade, charge, flask or hatchet from the player's eyes; returns the projectile slot."""
        s = self.store
        eye = s.eye_position(row)[0]
        lift = 0.03 if EQ.KIND[equip] == EQ.KIND_AXE else 0.10
        direction = M.forward_vector(s.yaw[row], min(float(s.pitch[row]) + lift, float(C.PITCH_LIMIT)))
        t, item, _ = self.world.raycast(eye, direction, 0.7, C.LAYER_BULLET)
        start = eye + direction * (max(float(t[0]) - 0.15, 0.0) if item[0] >= 0 else 0.6)
        velocity = direction * float(EQ.SPEED[equip]) + s.move.vel[row] * 0.5
        fuse = EQ.flight_ticks(equip)
        slot = self.grenades.spawn(start, velocity, fuse, row, equip=equip)
        gid = int(self.grenades.gid[slot])
        self.events.append((EV_GRENADE, gid, row, tuple(start.tolist()), tuple(velocity.tolist()), fuse, self.tick,
                            int(equip)))
        return slot

    def throw_grenade(self, row: int) -> None:
        """Throw the player's lethal (the G key)."""
        self.throw_equipment(row, int(self.store.equip[row, 0]))

    def use_tactical(self, row: int) -> None:
        """The F key: inject the stim, or throw the shock grenade / smoke canister."""
        s = self.store
        equip = int(s.equip[row, 1])
        if EQ.KIND[equip] == EQ.KIND_STIM:
            s.heal_left[row] = float(EQ.DAMAGE[equip])
            s.heal_rate[row] = float(EQ.DAMAGE[equip]) / max(float(EQ.DURATION[equip]), 1.0e-6)
            s.last_damage_tick[row] = -100000                   # regeneration restarts at once
            self.events.append((EV_STATUS, row, C.STATUS_STIM, int(round(float(EQ.DURATION[equip]) * _TICKS))))
        else:
            self.throw_equipment(row, equip)

    def _affected(self, position, radius: float, owner: int, always: int = -1):
        """Players a blast at ``position`` reaches: in range, in sight, hostile (or the owner), not protected."""
        s = self.store
        rows = np.flatnonzero(s.active & s.alive)
        if rows.size == 0:
            return rows, np.zeros((0, 3)), np.zeros(0)
        centers = s.hitboxes.bound_center[rows]
        delta = centers - position
        dist = M.length(delta)
        near = (dist < radius) | (rows == always)
        if owner >= 0 and s.active[owner]:
            near &= (s.team[rows] != s.team[owner]) | (rows == owner)
        near &= (s.protect_until[rows] <= self.tick) | (rows == owner)
        rows, delta, dist = rows[near], delta[near], dist[near]
        if rows.size == 0:
            return rows, delta, dist
        lifted = position + np.array([0.0, 0.12, 0.0])
        visible = self.world.line_of_sight(np.tile(lifted, (rows.size, 1)), s.hitboxes.bound_center[rows])
        visible |= rows == always
        return rows[visible], delta[visible], dist[visible]

    def explode(self, position, owner: int, gid: int, equip: int = 0, stuck_to: int = -1) -> None:
        """
        Radial damage with line of sight and knock-back, vectorised over all
        players (grenades and sticky charges).  The player a charge is stuck
        to takes ``STICK_DAMAGE_MULT`` times the centre damage.
        """
        s = self.store
        position = np.asarray(position, dtype=np.float64)
        radius = float(EQ.RADIUS[equip])
        self.events.append((EV_EXPLOSION, tuple(position.tolist()), owner if owner >= 0 else NO_PLAYER, gid, int(equip)))
        rows, delta, dist = self._affected(position, radius, owner, stuck_to)
        if rows.size == 0:
            return
        damage = EQ.splash_damage(equip, dist)
        if stuck_to >= 0:
            damage = np.where(rows == stuck_to, np.maximum(damage, float(EQ.DAMAGE[equip]) * C.STICK_DAMAGE_MULT), damage)
        damage = damage * np.where(rows == owner, 0.5, 1.0) * s.blast_resist[rows]
        reach = np.clip(1.0 - dist / radius, 0.0, 1.0)
        push = delta / np.maximum(dist, 1.0e-6)[:, None]
        push[:, 1] = np.maximum(push[:, 1], 0.35)
        s.move.vel[rows] += push * (C.EXPLOSION_KNOCKBACK * reach)[:, None]
        s.move.mflags[rows] &= np.uint16(0xFFFF ^ (C.MF_ON_GROUND | C.MF_SLIDING))
        weapon = C.EQUIP_KILL_BASE + int(equip)
        for row, amount in zip(rows.tolist(), damage.tolist()):
            self.apply_damage(int(row), owner, float(amount), C.ZONE_TORSO, weapon, position, explosive=True)

    def stun_burst(self, position, owner: int, gid: int, equip: int) -> None:
        """Shock grenade: slows and disorients every player it reaches (shorter towards the edge)."""
        s = self.store
        position = np.asarray(position, dtype=np.float64)
        self.events.append((EV_EXPLOSION, tuple(position.tolist()), owner if owner >= 0 else NO_PLAYER, gid, int(equip)))
        rows, _, dist = self._affected(position, float(EQ.RADIUS[equip]), owner)
        if rows.size == 0:
            return
        ticks = EQ.stun_ticks(equip, dist)
        ticks = np.where(rows == owner, np.round(ticks * C.STUN_OWNER_SCALE).astype(np.int64), ticks)
        for row, length in zip(rows.tolist(), ticks.tolist()):
            s.stun_until[row] = max(int(s.stun_until[row]), int(s.cmd_tick[row]) + int(length))
            self.events.append((EV_STATUS, int(row), C.STATUS_STUN, int(length)))

    def burst_zone(self, position, owner: int, gid: int, equip: int) -> None:
        """Fire pool (settles on the ground under the point of impact) or smoke cloud."""
        position = np.array(position, dtype=np.float64)
        if EQ.KIND[equip] == EQ.KIND_FIRE:
            t, item, _ = self.world.raycast((position + np.array([0.0, 0.3, 0.0]))[None, :], np.array([[0.0, -1.0, 0.0]]),
                                            6.0, C.LAYER_PLAYER)
            if item[0] >= 0:
                position[1] += 0.3 - float(t[0])
        until = self.tick + int(round(float(EQ.DURATION[equip]) * _TICKS))
        self.zones.spawn(position, float(EQ.RADIUS[equip]), self.tick, until, owner, equip)
        self.events.append((EV_EXPLOSION, tuple(position.tolist()), owner if owner >= 0 else NO_PLAYER, gid, int(equip)))

    def detonate(self, slot: int) -> None:
        """A fuse ran out: blow up, stun, burn or smoke according to the item's kind."""
        gs = self.grenades
        equip = int(gs.equip[slot])
        kind = int(EQ.KIND[equip])
        position, owner, gid = gs.pos[slot].copy(), int(gs.owner[slot]), int(gs.gid[slot])
        gs.active[slot] = False
        if kind in (EQ.KIND_FRAG, EQ.KIND_STICKY):
            self.explode(position, owner, gid, equip, int(gs.target[slot]) if kind == EQ.KIND_STICKY else -1)
        elif kind == EQ.KIND_STUN:
            self.stun_burst(position, owner, gid, equip)
        elif kind in (EQ.KIND_FIRE, EQ.KIND_SMOKE):
            self.burst_zone(position, owner, gid, equip)
        # a hatchet that hit nothing simply falls out of the game

    def _projectile_contacts(self) -> None:
        """Sticky charges, hatchets and fire flasks in flight against the players' hitboxes."""
        gs, s = self.grenades, self.store
        flying = np.flatnonzero(gs.active & ~gs.stuck & EQ.CONTACT[gs.equip])
        for slot in flying.tolist():
            equip = int(gs.equip[slot])
            owner = int(gs.owner[slot])
            velocity = gs.vel[slot].copy()
            velocity[1] -= C.GRAVITY * float(EQ.GRAVITY[equip]) * C.TICK_DT
            delta = velocity * C.TICK_DT
            dist = float(np.sqrt(np.dot(delta, delta)))
            if dist < 1.0e-6:
                continue
            origin = gs.pos[slot][None, :]
            direction = (delta / dist)[None, :]
            t_world, _, _ = self.world.raycast(origin, direction, dist, C.LAYER_BULLET)
            targets = s.active & s.alive & (s.protect_until <= self.tick)
            if 0 <= owner < s.capacity:
                targets[owner] = False
                if s.active[owner]:
                    targets &= s.team != s.team[owner]
            if self.phase != C.PHASE_LIVE or not targets.any():
                continue
            t_hit, entity, box = s.hitboxes.raycast(origin, direction, t_world, targets)
            victim = int(entity[0])
            if victim < 0:
                continue
            point = origin[0] + direction[0] * float(t_hit[0])
            kind = int(EQ.KIND[equip])
            gid = int(gs.gid[slot])
            if kind == EQ.KIND_AXE:
                gs.active[slot] = False
                self.events.append((EV_STICK, gid, victim, tuple(point.tolist())))
                self.apply_damage(victim, owner, float(EQ.DAMAGE[equip]), int(HITBOX_ZONE[box[0]]),
                                  C.EQUIP_KILL_BASE + equip, point)
            elif kind == EQ.KIND_STICKY:
                gs.stick(slot, point, -direction[0], victim, point - s.move.pos[victim])
                self.events.append((EV_STICK, gid, victim, tuple(point.tolist())))
            else:
                gs.active[slot] = False
                self.burst_zone(point, owner, gid, equip)

    def _step_equipment(self) -> None:
        gs, s = self.grenades, self.store
        attached = np.flatnonzero(gs.active & gs.stuck & (gs.target >= 0))
        for slot in attached.tolist():
            wearer = int(gs.target[slot])
            if s.active[wearer] and s.alive[wearer]:
                gs.pos[slot] = s.move.pos[wearer] + gs.offset[slot]
            else:
                gs.target[slot] = -1                             # the wearer died first: it stays where it is
        self._projectile_contacts()
        expired = step_grenades(gs, self.world)
        for slot in np.flatnonzero(gs.just_hit & gs.active).tolist():
            equip = int(gs.equip[slot])
            kind = int(EQ.KIND[equip])
            if kind == EQ.KIND_FIRE:
                gs.active[slot] = False
                self.burst_zone(gs.pos[slot].copy(), int(gs.owner[slot]), int(gs.gid[slot]), equip)
                continue
            self.events.append((EV_STICK, int(gs.gid[slot]), NO_PLAYER, tuple(gs.pos[slot].tolist())))
            if kind == EQ.KIND_AXE:
                gs.active[slot] = False                         # stuck in the wall: nothing left to simulate
        for slot in expired.tolist():
            if gs.active[slot]:
                self.detonate(int(slot))
        self._tick_zones()

    FIRE_PULSE = 10                                             # fire damage is dealt six times per second

    def _tick_zones(self) -> None:
        z, s = self.zones, self.store
        z.active &= z.until > self.tick
        if self.tick % self.FIRE_PULSE or self.phase != C.PHASE_LIVE:
            return
        fires = z.of_kind(EQ.KIND_FIRE)
        if fires.size == 0:
            return
        rows = np.flatnonzero(s.active & s.alive)
        if rows.size == 0:
            return
        feet = s.move.pos[rows]
        for zone in fires.tolist():
            equip, owner, centre = int(z.equip[zone]), int(z.owner[zone]), z.pos[zone]
            flat = np.hypot(feet[:, 0] - centre[0], feet[:, 2] - centre[2])
            height = feet[:, 1] - centre[1]
            inside = (flat < z.radius[zone]) & (height > -0.6) & (height < C.FIRE_HEIGHT)
            inside &= (s.protect_until[rows] <= self.tick) | (rows == owner)
            if 0 <= owner < s.capacity and s.active[owner]:
                inside &= (s.team[rows] != s.team[owner]) | (rows == owner)
            amount = float(EQ.DAMAGE[equip]) * self.FIRE_PULSE / _TICKS
            for row in rows[inside].tolist():
                self.apply_damage(int(row), owner, amount * (0.5 if row == owner else 1.0), C.ZONE_TORSO,
                                  C.EQUIP_KILL_BASE + equip, centre)

    def smoke_blocks(self, starts, ends) -> np.ndarray:
        """True where the segment ``start -> end`` passes through a smoke cloud (bots cannot see through)."""
        starts = np.atleast_2d(np.asarray(starts, dtype=np.float64))
        ends = np.atleast_2d(np.asarray(ends, dtype=np.float64))
        blocked = np.zeros(starts.shape[0], dtype=bool)
        z = self.zones
        clouds = z.of_kind(EQ.KIND_SMOKE)
        if clouds.size == 0 or starts.shape[0] == 0:
            return blocked
        delta = ends - starts
        length2 = np.maximum(np.sum(delta * delta, axis=1), 1.0e-9)
        for zone in clouds.tolist():
            grown = min(1.0, (self.tick - int(z.born[zone])) / (0.6 * _TICKS))
            radius = float(z.radius[zone]) * (0.35 + 0.65 * grown)
            centre = z.pos[zone] + np.array([0.0, 1.0, 0.0])
            t = np.clip(np.sum((centre - starts) * delta, axis=1) / length2, 0.0, 1.0)
            closest = starts + delta * t[:, None]
            blocked |= np.sum((closest - centre) ** 2, axis=1) < radius * radius
        return blocked

    # ---- end of tick -----------------------------------------------------------------------------------------
    def end_tick(self) -> None:
        """Projectiles, rules and hitbox refresh.  Call once per server tick after all commands."""
        s = self.store
        self.tick += 1
        self._step_equipment()
        live = s.active & s.alive
        fell = np.flatnonzero(live & (s.move.pos[:, 1] < self.loaded.data.kill_y))
        for row in fell.tolist():
            attacker = int(s.last_attacker[row]) if self.tick - s.last_damage_tick[row] < 5 * _TICKS else -1
            self.kill_player(int(row), attacker, WEAPON_GRENADE if attacker >= 0 else 0, 0)
        live = s.active & s.alive
        regen = live & (self.tick - s.last_damage_tick > s.regen_delay) & (s.health < s.max_health)
        s.health[regen] = np.minimum(s.health[regen] + C.HEALTH_REGEN_RATE * C.TICK_DT, s.max_health[regen])
        healing = np.flatnonzero(live & (s.heal_left > 0.0))
        if healing.size:
            amount = np.minimum(s.heal_rate[healing] * C.TICK_DT, s.heal_left[healing])
            s.heal_left[healing] -= amount
            s.health[healing] = np.minimum(s.health[healing] + amount, s.max_health[healing])
        if self.tick % 30 == 0:
            owed = np.flatnonzero(s.active & (s.pending_credits >= 1.0))
            for row in owed.tolist():
                amount = int(s.pending_credits[row])
                s.pending_credits[row] -= amount
                self.events.append((EV_REWARD, int(row), amount, amount // 2, REWARD_COMBAT))
        if self.phase == C.PHASE_LIVE:
            due = np.flatnonzero(s.active & ~s.alive & (self.tick >= s.respawn_tick))
            for row in due.tolist():
                self.spawn_player(int(row))
            self.time_left = max(self.time_left - C.TICK_DT, 0.0)
            self._check_match_end()
        elif self.phase == C.PHASE_ENDED and self.tick >= self.intermission_end:
            self.restart_match(self.next_map)
            self.next_map = None
        s.refresh_hitboxes()

    # ---- match flow ----------------------------------------------------------------------------------------------
    def leading_side(self) -> int:
        """Winning team (TDM) or player id (FFA); ``NO_PLAYER`` on a draw / empty server."""
        s = self.store
        if self.mode == C.MODE_TDM:
            if self.team_scores[0] == self.team_scores[1]:
                return NO_PLAYER
            return int(np.argmax(self.team_scores))
        rows = s.active_rows()
        if rows.size == 0:
            return NO_PLAYER
        return int(rows[int(np.argmax(s.kills[rows]))])

    def _check_match_end(self) -> None:
        s = self.store
        if self.mode == C.MODE_TDM:
            reached = int(self.team_scores.max()) >= self.score_limit
        else:
            rows = s.active_rows()
            reached = rows.size > 0 and int(s.kills[rows].max()) >= self.score_limit
        if not reached and self.time_left > 0.0:
            return
        self.phase = C.PHASE_ENDED
        self.winner = self.leading_side()
        self.intermission_end = self.tick + int(C.INTERMISSION_TIME * _TICKS)
        self.matches_played += 1
        for row in self.humans().tolist():
            won = (s.team[row] == self.winner) if self.mode == C.MODE_TDM else (row == self.winner)
            self.events.append((EV_REWARD, int(row), 250, 150, REWARD_MATCH))
            if won:
                self.events.append((EV_REWARD, int(row), 500, 300, REWARD_WIN))
        self.events.append((EV_MATCH, self.phase, self.winner, self.map_index, self.mode))

    def restart_match(self, map_key=None) -> None:
        """Reset scores and respawn everybody (optionally on another arena)."""
        s = self.store
        if map_key is not None:
            self.change_map(map_key)
        self.phase = C.PHASE_LIVE
        self.time_left = self.time_limit
        self.team_scores[:] = 0
        self.winner = NO_PLAYER
        self.grenades.clear()
        self.zones.clear()
        s.kills[:] = 0
        s.deaths[:] = 0
        s.score[:] = 0
        s.streak[:] = 0
        s.pending_credits[:] = 0.0
        self.events.append((EV_MATCH, self.phase, NO_PLAYER, self.map_index, self.mode))
        for row in s.active_rows().tolist():
            self.spawn_player(int(row))

    def drain_events(self) -> list:
        events, self.events = self.events, []
        return events
