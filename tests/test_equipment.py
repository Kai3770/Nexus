"""Lethal and tactical equipment: balance rules and every item's behaviour in the real simulation."""
import numpy as np

import helpers  # noqa: F401

from engine import bots
from engine import config as C
from engine import equipment as EQ
from engine import mathx as M
from engine import secure_save as SS
from engine import simulation as S

IDLE = np.uint16(0)


def duel(distance: float, lethal: str = "pulse", tactical: str = "stim", mode: int = C.MODE_FFA):
    """Two players on open ground in CRUCIBLE, ``distance`` metres apart, protection off."""
    game = S.GameWorld("crucible", mode=mode, seed=4)
    gear = (EQ.EQUIP_INDEX[lethal], EQ.EQUIP_INDEX[tactical])
    a = game.add_player("alpha", equipment=gear)
    b = game.add_player("bravo")
    s = game.store
    s.move.pos[a] = (-10.0, 0.0, 14.0)
    s.move.pos[b] = (-10.0 + distance, 0.0, 14.0)
    s.yaw[a], s.yaw[b] = np.pi / 2, -np.pi / 2
    s.protect_until[:] = 0
    s.refresh_hitboxes()
    game.drain_events()
    return game, a, b


def aim_at(game, shooter: int, point) -> tuple:
    s = game.store
    yaw, pitch = M.yaw_pitch_from_direction(np.asarray(point, dtype=np.float64) - s.eye_position(shooter)[0])
    return float(yaw), float(pitch)


def run(game, ticks: int, buttons=None, aims=None, until=None) -> list:
    """Step every player ``ticks`` times (``buttons`` / ``aims`` per player for the first tick); returns events."""
    s = game.store
    rows = s.active_rows()
    events = []
    for step in range(ticks):
        for row in rows.tolist():
            pressed = buttons.get(row, IDLE) if (buttons and step == 0) else IDLE
            yaw, pitch = aims[row] if aims and row in aims else (float(s.yaw[row]), float(s.pitch[row]))
            game.step_commands(np.array([row]), np.uint16(pressed), yaw, pitch, int(s.cmd_tick[row]) + 1)
        game.end_tick()
        events += game.drain_events()
        if until is not None and until(events):
            break
    return events


def kinds(events, kind) -> list:
    return [event for event in events if event[0] == kind]


def test_equipment_table_obeys_the_balance_rules():
    assert EQ.validate_equipment() == []
    assert set(C.EQUIPMENT_KINDS) == {C.EQUIPMENT_KINDS[k] for k in EQ.KIND}            # every kind is in use
    assert len(EQ.LETHALS) >= 4 and len(EQ.TACTICALS) >= 3
    report = {row["key"]: row for row in EQ.balance_report()}
    assert report["pulse"]["kill_radius"] < 1.0 and report["pulse"]["kill_radius_plated"] == 0.0
    assert report["cling"]["stuck_damage_plated"] >= C.BASE_HEALTH                       # wearing it is fatal
    assert report["thermite"]["fire_ttk"] >= 2.0 and report["thermite"]["fire_total"] >= C.BASE_HEALTH
    assert report["stim"]["heal_rate"] <= 0.5 * C.BASE_HEALTH / (C.TTK_MAX_MS / 1000.0)
    axe = EQ.EQUIP_INDEX["hatchet"]
    assert all(EQ.COOLDOWN[axe] < EQ.COOLDOWN[e] for e in EQ.LETHALS if e != axe)       # direct hit only: fastest
    assert EQ.sanitize((EQ.EQUIP_INDEX["stim"], EQ.EQUIP_INDEX["pulse"])) == EQ.DEFAULT_EQUIPMENT   # wrong slots
    assert EQ.sanitize((EQ.EQUIP_INDEX["cling"], EQ.EQUIP_INDEX["smoke"])) == (EQ.EQUIP_INDEX["cling"],
                                                                              EQ.EQUIP_INDEX["smoke"])
    assert EQ.splash_damage(0, [0.0, EQ.RADIUS[0], 99.0]).tolist() == [EQ.DAMAGE[0], 0.0, 0.0]


def test_lethal_and_tactical_have_their_own_cooldowns():
    game, a, b = duel(6.0, "pulse", "shock")
    s = game.store
    events = run(game, 1, {a: C.BTN_ABILITY | C.BTN_TACTICAL})
    thrown = kinds(events, S.EV_GRENADE)
    assert sorted(event[7] for event in thrown) == sorted([EQ.EQUIP_INDEX["pulse"], EQ.EQUIP_INDEX["shock"]])
    assert s.ability_ready[a] - s.cmd_tick[a] == EQ.cooldown_ticks(EQ.EQUIP_INDEX["pulse"])
    assert s.tactical_ready[a] - s.cmd_tick[a] == EQ.cooldown_ticks(EQ.EQUIP_INDEX["shock"])
    events = run(game, 1, {a: C.BTN_ABILITY | C.BTN_TACTICAL})                                 # still cooling down
    assert not kinds(events, S.EV_GRENADE)
    capacitor = np.zeros(4, int)
    capacitor[1] = 10                                                                           # -50 % cooldowns
    s.apply_operator(a, capacitor, 0)
    assert s.ability_cooldown[a] == EQ.cooldown_ticks(EQ.EQUIP_INDEX["pulse"], 0.5)
    assert s.tactical_cooldown[a] == EQ.cooldown_ticks(EQ.EQUIP_INDEX["shock"], 0.5)


def test_stim_heals_fifty_over_a_second_and_never_overheals():
    game, a, _ = duel(6.0, "pulse", "stim")
    s = game.store
    s.health[a] = 30.0
    s.last_damage_tick[a] = game.tick
    events = run(game, 30, {a: C.BTN_TACTICAL})
    status = kinds(events, S.EV_STATUS)
    assert status and status[0][1:3] == (a, C.STATUS_STIM)
    healed = float(s.health[a]) - 30.0
    regen = C.HEALTH_REGEN_RATE * 30 * C.TICK_DT                                            # restarted at once
    assert 25.0 - 1e-6 <= healed <= 25.0 + regen + 1e-6
    run(game, 90)
    assert s.health[a] == s.max_health[a] and s.heal_left[a] == 0.0


def test_hatchet_kills_on_a_direct_hit_and_sticks_in_walls_otherwise():
    game, a, b = duel(7.0, "hatchet")
    s = game.store
    target = s.hitboxes.world_center[b, 1]
    events = run(game, 40, {a: C.BTN_ABILITY}, {a: aim_at(game, a, target)},
                 until=lambda ev: bool(kinds(ev, S.EV_KILL)))
    kill = kinds(events, S.EV_KILL)
    assert kill and kill[0][1:4] == (a, b, C.EQUIP_KILL_BASE + EQ.EQUIP_INDEX["hatchet"])
    assert kinds(events, S.EV_STICK)[0][2] == b and not kinds(events, S.EV_EXPLOSION)      # no splash
    game, a, b = duel(7.0, "hatchet")
    s = game.store
    away = s.eye_position(a)[0] + np.array([0.0, -0.5, -4.0])                               # at the floor, not at b
    events = run(game, 60, {a: C.BTN_ABILITY}, {a: aim_at(game, a, away)})
    stuck = kinds(events, S.EV_STICK)
    assert stuck and stuck[0][2] == S.NO_PLAYER and s.health[b] == 100.0
    assert not game.grenades.active.any()


def test_cling_charge_sticks_to_a_player_and_kills_through_blast_plating():
    game, a, b = duel(5.0, "cling")
    s = game.store
    plating = np.zeros(4, int)
    plating[0] = 5                                                                              # -30 % explosive damage
    s.apply_operator(b, plating, 0)
    target = s.hitboxes.world_center[b, 0] + np.array([0.0, 0.25, 0.0])
    events = run(game, 140, {a: C.BTN_ABILITY}, {a: aim_at(game, a, target)},
                 until=lambda ev: bool(kinds(ev, S.EV_KILL)))
    stuck = kinds(events, S.EV_STICK)
    assert stuck and stuck[0][2] == b                                                           # tagged
    kill = kinds(events, S.EV_KILL)
    assert kill and kill[0][1:4] == (a, b, C.EQUIP_KILL_BASE + EQ.EQUIP_INDEX["cling"])
    stick_tick = next(i for i, e in enumerate(events) if e[0] == S.EV_STICK)
    assert kinds(events[stick_tick:], S.EV_EXPLOSION)                                           # blew up after sticking


def test_cling_charge_on_a_wall_detonates_after_its_short_fuse():
    game, a, b = duel(9.0, "cling")
    s = game.store
    floor = s.eye_position(a)[0] + np.array([1.5, -1.62, 0.0])
    events = run(game, 200, {a: C.BTN_ABILITY}, {a: aim_at(game, a, floor)},
                 until=lambda ev: bool(kinds(ev, S.EV_EXPLOSION)))
    stick_at = next(i for i, e in enumerate(events) if e[0] == S.EV_STICK)
    assert events[stick_at][2] == S.NO_PLAYER
    assert kinds(events, S.EV_EXPLOSION) and s.health[b] == 100.0                             # 9 m away: safe


def test_thermite_burns_over_time_and_never_kills_in_under_two_seconds():
    game, a, b = duel(4.0, "thermite")
    s = game.store
    thermite = EQ.EQUIP_INDEX["thermite"]
    game.burst_zone(s.move.pos[b].copy(), a, 77, thermite)
    assert game.zones.of_kind(EQ.KIND_FIRE).size == 1
    s.regen_delay[b] = 10 ** 6                                                                  # isolate the fire
    run(game, int(1.9 * C.TICK_RATE))
    assert s.alive[b] and 15.0 < s.health[b] < 45.0                                            # about 38 / s so far
    events = run(game, int(1.5 * C.TICK_RATE), until=lambda ev: bool(kinds(ev, S.EV_KILL)))
    kill = kinds(events, S.EV_KILL)
    assert kill and kill[0][3] == C.EQUIP_KILL_BASE + thermite
    run(game, int(EQ.DURATION[thermite] * C.TICK_RATE))
    assert game.zones.of_kind(EQ.KIND_FIRE).size == 0                                           # burnt out


def test_thermite_flask_bursts_on_impact_into_a_fire_pool():
    game, a, b = duel(9.0, "thermite")
    s = game.store
    floor = s.eye_position(a)[0] + np.array([4.0, -1.62, 0.0])
    events = run(game, 120, {a: C.BTN_ABILITY}, {a: aim_at(game, a, floor)},
                 until=lambda ev: bool(kinds(ev, S.EV_EXPLOSION)))
    burst = kinds(events, S.EV_EXPLOSION)
    assert burst and burst[0][4] == EQ.EQUIP_INDEX["thermite"]
    zone = game.zones.of_kind(EQ.KIND_FIRE)
    assert zone.size == 1 and abs(float(game.zones.pos[zone[0], 1])) < 0.05                    # settled on the floor


def test_teammates_are_safe_from_fire_and_shock():
    game, a, b = duel(4.0, "thermite", "shock", mode=C.MODE_TDM)
    s = game.store
    s.team[b] = s.team[a]
    game.burst_zone(s.move.pos[b].copy(), a, 1, EQ.EQUIP_INDEX["thermite"])
    game.stun_burst(s.move.pos[b] + np.array([0.0, 1.0, 0.0]), a, 2, EQ.EQUIP_INDEX["shock"])
    run(game, 60)
    assert s.health[b] == 100.0 and s.stun_until[b] <= s.cmd_tick[b]


def test_shock_grenade_slows_movement_in_the_shared_step():
    game, a, b = duel(4.0, "pulse", "shock")
    s = game.store
    shock = EQ.EQUIP_INDEX["shock"]
    game.stun_burst(s.move.pos[b] + np.array([0.0, 1.0, 0.0]), a, 3, shock)
    status = {e[1]: e for e in kinds(game.drain_events(), S.EV_STATUS)}
    assert set(status) == {a, b} and all(e[2] == C.STATUS_STUN for e in status.values())
    length = status[b][3]
    assert status[a][3] < length * C.STUN_OWNER_SCALE + 1                    # the thrower, further out, at half length
    assert EQ.DURATION[shock] * C.STUN_EDGE_SCALE * C.TICK_RATE <= length <= EQ.DURATION[shock] * C.TICK_RATE
    start = s.move.pos[b].copy()
    run_ticks = 40
    for _ in range(run_ticks):
        game.step_commands(np.array([b]), np.uint16(C.BTN_FORWARD), float(s.yaw[b]), 0.0, int(s.cmd_tick[b]) + 1)
        game.end_tick()
    slowed = float(np.linalg.norm((s.move.pos[b] - start)[[0, 2]]))
    fresh, _, d = duel(4.0)
    t = fresh.store
    begin = t.move.pos[d].copy()
    for _ in range(run_ticks):
        fresh.step_commands(np.array([d]), np.uint16(C.BTN_FORWARD), float(t.yaw[d]), 0.0, int(t.cmd_tick[d]) + 1)
        fresh.end_tick()
    normal = float(np.linalg.norm((t.move.pos[d] - begin)[[0, 2]]))
    assert slowed < normal * 0.75 and abs(slowed / normal - EQ.STUN_FACTOR) < 0.12


def test_smoke_blocks_sight_for_bots():
    game, a, b = duel(10.0, "pulse", "smoke")
    s = game.store
    eye, target = s.eye_position(a)[0], s.hitboxes.bound_center[b]
    assert not game.smoke_blocks(eye, target)[0]
    controller = bots.BotController(game, 1, seed=3)
    controller.aim_yaw[a], controller.aim_pitch[a] = M.yaw_pitch_from_direction(target - eye)
    controller._perceive(np.array([a]))
    assert controller.visible[a, b]
    middle = (s.move.pos[a] + s.move.pos[b]) * 0.5
    game.burst_zone(middle, b, 5, EQ.EQUIP_INDEX["smoke"])
    run(game, int(0.8 * C.TICK_RATE))                                                          # the cloud spreads
    assert game.smoke_blocks(eye, target)[0]
    controller._perceive(np.array([a]))
    assert not controller.visible[a, b]
    run(game, int(EQ.DURATION[EQ.EQUIP_INDEX["smoke"]] * C.TICK_RATE))
    assert not game.smoke_blocks(eye, target)[0]                                               # cleared


def test_equipment_changes_wait_for_the_next_respawn():
    game, a, b = duel(6.0)
    s = game.store
    wanted = (EQ.EQUIP_INDEX["hatchet"], EQ.EQUIP_INDEX["smoke"])
    assert game.request_equipment(a, wanted)
    assert tuple(s.equip[a]) == EQ.DEFAULT_EQUIPMENT and tuple(s.pending_equip[a]) == wanted
    run(game, 1, {a: C.BTN_ABILITY})
    assert s.ability_ready[a] > s.cmd_tick[a]                                                  # pulse on cooldown
    game.kill_player(a, b, 0, 0)
    run(game, int((C.RESPAWN_TIME + 0.2) * C.TICK_RATE))
    assert s.alive[a] and tuple(s.equip[a]) == wanted
    assert s.ability_ready[a] <= s.cmd_tick[a] and s.tactical_ready[a] <= s.cmd_tick[a]        # a new class: ready
    assert s.ability_cooldown[a] == EQ.cooldown_ticks(wanted[0])
    assert not game.request_equipment(a, wanted)                                                # nothing new


def test_owner_state_carries_equipment_and_stun():
    store = S.PlayerStore(2)
    store.apply_equipment(0, (EQ.EQUIP_INDEX["thermite"], EQ.EQUIP_INDEX["shock"]))
    store.tactical_ready[0] = 1234
    store.stun_until[0] = 777
    copy = S.PlayerStore(2)
    copy.unpack_owner(1, store.pack_owner(0))
    assert tuple(copy.equip[1]) == tuple(store.equip[0]) and copy.tactical_ready[1] == 1234 and copy.stun_until[1] == 777
    assert copy.ability_cooldown[1] == EQ.cooldown_ticks(EQ.EQUIP_INDEX["thermite"])


def test_profile_unlocks_and_equips_gear():
    profile = SS.Profile()
    assert profile.equipment == [EQ.EQUIP_KEYS[i] for i in EQ.DEFAULT_EQUIPMENT]
    assert all(profile.gear[EQ.EQUIP_KEYS[i]]["unlocked"] for i in EQ.DEFAULT_EQUIPMENT)
    old = SS.Profile.from_dict({"level": 9, "credits": 10000.0})                                # a pre-equipment save
    assert old.equipment == profile.equipment and not old.gear["thermite"]["unlocked"]
    forged = SS.Profile.from_dict({"equipment": ["thermite", "pulse"], "gear": {"thermite": {"unlocked": False}}})
    assert forged.equipment == profile.equipment                                                # locked / wrong slot
    import tempfile
    pm = SS.ProfileManager(tempfile.mkdtemp())
    pm.load()
    assert not pm.unlock_gear("thermite")                                                       # level and credits
    pm.profile.level = int(EQ.UNLOCK_LEVEL[EQ.EQUIP_INDEX["thermite"]])
    pm.profile.credits = EQ.UNLOCK_COST[EQ.EQUIP_INDEX["thermite"]] + 10.0
    assert pm.unlock_gear("thermite") and pm.profile.credits == 10.0
    assert pm.equip_gear("thermite") and pm.equip_gear("smoke") is False                        # smoke still locked
    assert pm.equipment_ids() == (EQ.EQUIP_INDEX["thermite"], EQ.DEFAULT_EQUIPMENT[1])
    restored = SS.Profile.from_dict(pm.profile.to_dict())
    assert restored.equipment == ["thermite", "stim"] and restored.gear["thermite"]["unlocked"]


def test_bots_use_their_equipment():
    game = S.GameWorld("crucible", mode=C.MODE_FFA, seed=21)
    choices = [(EQ.LETHALS[i % len(EQ.LETHALS)], EQ.TACTICALS[i % len(EQ.TACTICALS)]) for i in range(8)]
    for index, gear in enumerate(choices):
        game.add_player("BOT-%02d" % index, is_bot=True, equipment=gear)
    controller = bots.BotController(game, 2, seed=22)
    game.drain_events()
    used = set()
    for _ in range(2400):
        rows = game.bots()
        buttons, yaw, pitch = controller.think(rows)
        fired = game.step_commands(rows, buttons, yaw, pitch, game.tick + 1)
        game.resolve_shots(rows[fired])
        game.end_tick()
        events = game.drain_events()
        controller.observe(events)
        for event in events:
            if event[0] == S.EV_GRENADE:
                used.add(int(event[7]))
            elif event[0] == S.EV_STATUS and event[2] == C.STATUS_STIM:
                used.add(EQ.EQUIP_INDEX["stim"])
    assert np.isfinite(game.store.move.pos).all()
    assert len(used) >= 5, sorted(EQ.EQUIP_KEYS[e] for e in used)
