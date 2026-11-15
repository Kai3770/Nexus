"""ESCALATION: the weapon ladder mode -- rules, wire format, prediction and bots."""
import time

import numpy as np

import helpers  # noqa: F401

from engine import bots
from engine import config as C
from engine import equipment as EQ
from engine import mathx as M
from engine import network_lan as N
from engine import simulation as S
from engine import weapons as W

HATCHET = EQ.EQUIP_INDEX[C.ESCALATION_LETHAL]
LADDER = W.ESCALATION_LADDER


def esc_game(players=2, seed=3):
    game = S.GameWorld("crucible", mode=C.MODE_ESC, seed=seed)
    rows = [game.add_player("p%d" % index) for index in range(players)]
    game.store.protect_until[:] = 0
    game.drain_events()
    return game, rows


def kill(game, victim, attacker, weapon):
    """Kill ``victim`` (respawning it first if needed); returns the kill and ladder events."""
    if not game.store.alive[victim]:
        game.spawn_player(victim)
    game.kill_player(victim, attacker, weapon, 0)
    return [event for event in game.drain_events() if event[0] in (S.EV_KILL, S.EV_LADDER)]


def gun(game, row):
    return int(game.store.weapons.weapon[row, 0])


def test_the_ladder_is_valid_and_climbs_through_every_weapon_once():
    assert W.validate_escalation() == []
    assert W.ESCALATION_STEPS == len(C.ESCALATION_LADDER) == W.WEAPON_COUNT
    assert sorted(LADDER) == list(range(W.WEAPON_COUNT))
    assert W.escalation_weapon(-3) == LADDER[0] and W.escalation_weapon(99) == LADDER[-1]
    assert all(W.ESCALATION_STEP_OF[weapon] == step for step, weapon in enumerate(LADDER))
    assert C.MODE_NAMES[C.MODE_ESC] == "ESCALATION" and len(C.MODE_DESCRIPTIONS) == len(C.MODE_NAMES)


def test_joining_hands_out_the_first_gun_in_both_slots_and_the_hatchet():
    game, (a, b) = esc_game()
    s = game.store
    assert game.score_limit == W.ESCALATION_STEPS
    for row in (a, b):
        assert tuple(s.weapons.weapon[row]) == (LADDER[0], LADDER[0]) and not s.weapons.levels[row].any()
        assert int(s.equip[row, 0]) == HATCHET and game.ladder[row] == 0 and s.score[row] == 0
    # the player's own loadout and its upgrades never reach the ladder
    late = game.add_player("late", weapon_ids=(3, 4), levels=np.full((2, W.UPGRADE_COUNT), 3),
                           equipment=(EQ.EQUIP_INDEX["cling"], EQ.EQUIP_INDEX["smoke"]))
    assert tuple(s.weapons.weapon[late]) == (LADDER[0], LADDER[0]) and not s.weapons.levels[late].any()
    assert (int(s.equip[late, 0]), int(s.equip[late, 1])) == (HATCHET, EQ.EQUIP_INDEX["smoke"])
    # a lethal queued in the pause menu is ignored, the tactical still changes at the respawn
    game.request_equipment(late, (EQ.EQUIP_INDEX["thermite"], EQ.EQUIP_INDEX["stim"]))
    game.spawn_player(late)
    assert (int(s.equip[late, 0]), int(s.equip[late, 1])) == (HATCHET, EQ.EQUIP_INDEX["stim"])


def test_a_gun_kill_climbs_one_step_with_a_full_magazine():
    game, (a, b) = esc_game()
    s = game.store
    s.weapons.ammo[a] = 1
    events = kill(game, b, a, gun(game, a))
    assert events == [(S.EV_KILL, a, b, LADDER[0], 0), (S.EV_LADDER, a, 1, S.LADDER_PROMOTED, b)]
    assert tuple(s.weapons.weapon[a]) == (LADDER[1], LADDER[1])
    assert (s.weapons.ammo[a] == s.weapons.mag[a]).all()
    assert game.ladder[a] == 1 and s.score[a] == 1 and s.kills[a] == 1 and game.ladder[b] == 0


def test_a_hatchet_kill_steals_a_step_and_a_suicide_costs_one():
    game, (a, b) = esc_game()
    for _ in range(3):
        kill(game, b, a, gun(game, a))
    assert game.ladder[a] == 3
    events = kill(game, a, b, C.EQUIP_KILL_BASE + HATCHET)
    assert (S.EV_LADDER, b, 1, S.LADDER_STEAL, a) in events
    assert (S.EV_LADDER, a, 2, S.LADDER_STOLEN, b) in events
    assert gun(game, a) == LADDER[2] and gun(game, b) == LADDER[1]
    events = kill(game, a, a, gun(game, a))                              # own fault: one step down
    assert (S.EV_LADDER, a, 1, S.LADDER_SUICIDE, S.NO_PLAYER) in events and game.ladder[a] == 1
    # nobody falls below the first gun ...
    game.set_ladder_step(b, 0)
    events = kill(game, b, a, C.EQUIP_KILL_BASE + HATCHET)
    assert [e for e in events if e[0] == S.EV_LADDER] == [(S.EV_LADDER, a, 2, S.LADDER_STEAL, b)]
    kill(game, b, b, 0)
    assert game.ladder[b] == 0 and game.store.score[b] == 0
    # ... and kills by anything but a gun or the hatchet do not climb
    kill(game, b, a, C.EQUIP_KILL_BASE + EQ.EQUIP_INDEX["pulse"])
    kill(game, b, a, S.WEAPON_GRENADE)
    assert game.ladder[a] == 2


def test_a_kill_with_the_last_gun_wins_and_the_next_match_starts_at_the_bottom():
    game, (a, b) = esc_game()
    game.set_ladder_step(a, W.ESCALATION_STEPS - 1)
    game.set_ladder_step(b, 6)
    assert gun(game, a) == LADDER[-1]
    kill(game, b, a, gun(game, a))
    game.end_tick()
    events = game.drain_events()
    assert game.phase == C.PHASE_ENDED and game.winner == a
    assert (S.EV_MATCH, C.PHASE_ENDED, a, game.map_index, C.MODE_ESC) in events
    game.restart_match()
    assert game.ladder[[a, b]].tolist() == [0, 0] and game.store.score[[a, b]].tolist() == [0, 0]
    assert gun(game, a) == gun(game, b) == LADDER[0]


def test_at_the_time_limit_the_player_furthest_up_wins():
    game, (a, b, c) = esc_game(3)
    game.set_ladder_step(b, 9)
    game.set_ladder_step(c, 9)
    game.store.kills[c] = 4                                              # kills break the tie
    game.time_left = C.TICK_DT / 2
    game.end_tick()
    assert game.phase == C.PHASE_ENDED and game.winner == c


def test_one_volley_that_kills_twice_credits_the_gun_that_fired():
    game, (a, b, c) = esc_game(3)
    s = game.store
    breacher = W.WEAPON_INDEX["breacher"]
    step = W.ESCALATION_STEP_OF[breacher]
    game.set_ladder_step(a, step)
    s.move.pos[a] = (-8.0, 0.0, 14.0)
    s.move.pos[b] = (-6.0, 0.0, 13.8)
    s.move.pos[c] = (-6.0, 0.0, 14.2)
    s.yaw[b] = s.yaw[c] = -np.pi / 2
    s.health[[b, c]] = 1.0
    s.refresh_hitboxes()
    target = (s.hitboxes.world_center[b, 1] + s.hitboxes.world_center[c, 1]) / 2
    s.yaw[a], s.pitch[a] = M.yaw_pitch_from_direction(target - s.eye_position(a)[0])
    s.weapons.shots_fired[a] += 1
    game.resolve_shot(a)
    kills = [event for event in game.drain_events() if event[0] == S.EV_KILL]
    assert sorted(event[2] for event in kills) == sorted((b, c))
    assert all(event[3] == breacher for event in kills)                    # not the gun the first kill handed out
    assert game.ladder[a] == step + 2


def test_identical_slots_make_the_swap_key_a_no_op():
    game, (a, _b) = esc_game()
    ws = game.store.weapons
    rows = np.array([a])
    W.step_weapons(ws, rows, np.array([C.BTN_SWAP], dtype=np.uint16), np.array([50]), np.array([True]))
    assert int(ws.slot[a]) == 0 and float(ws.next_fire[a]) == 0.0         # no swap delay either


def test_the_owner_state_hands_the_ladder_gun_to_the_predictor():
    game, (a, _b) = esc_game()
    s = game.store
    game.set_ladder_step(a, 4)
    s.weapons.ammo[a, 0] = 7
    levels = np.zeros((W.LOADOUT_SLOTS, W.UPGRADE_COUNT), dtype=np.int64)
    predictor = N.Predictor(game.loaded.world, a, W.DEFAULT_LOADOUT, levels, np.zeros(W.OPERATOR_COUNT, dtype=np.int64))
    assert tuple(predictor.store.weapons.weapon[0]) == tuple(W.DEFAULT_LOADOUT)
    predictor.predict(1, 0, 0.0, 0.0)
    assert predictor.reconcile(1, s.pack_owner(a))
    ws = predictor.store.weapons
    assert tuple(ws.weapon[0]) == (LADDER[4], LADDER[4]) and not ws.levels[0].any()
    assert int(ws.ammo[0, 0]) == 7 and int(ws.mag[0, 0]) == int(s.weapons.mag[a, 0])
    assert int(predictor.store.equip[0, 0]) == HATCHET
    # upgrade levels travel too (a normal match keeps the player's own upgraded loadout)
    s.weapons.set_loadout(a, (0, 4), np.full((2, W.UPGRADE_COUNT), 2))
    other = S.PlayerStore(1)
    other.unpack_owner(0, s.pack_owner(a))
    assert tuple(other.weapons.weapon[0]) == (0, 4) and (other.weapons.levels[0] == 2).all()
    assert np.isclose(other.weapons.damage[0, 0], s.weapons.damage[a, 0])


def test_the_ladder_event_round_trips_through_the_wire_codec():
    event = (S.EV_LADDER, 3, 17, S.LADDER_STOLEN, 5)
    blob = N.encode_event(event)
    decoded, offset = N.decode_event(blob, 0)
    assert decoded == event and offset == len(blob)


def test_live_session_the_client_predicts_with_the_gun_the_server_hands_out():
    config = N.ServerConfig(name="LADDER", map_key="crucible", mode=C.MODE_ESC, bot_fill=0, beacon=False,
                            port=28190, seed=4)
    server = N.host_local_game(config)
    zeros = np.zeros((W.LOADOUT_SLOTS, W.UPGRADE_COUNT), dtype=np.int64)
    client = N.GameClient("climber", W.DEFAULT_LOADOUT, zeros, np.zeros(W.OPERATOR_COUNT, dtype=np.int64))
    events = []

    def play(seconds, buttons):
        end = next_tick = time.perf_counter()
        end += seconds
        while time.perf_counter() < end:
            events.extend(client.update())
            now = time.perf_counter()
            if now >= next_tick and client.state == N.CONNECTED:
                next_tick += C.TICK_DT
                client.sample_entities()
                client.send_command(buttons(now), 0.3 * np.sin(now), 0.0)
            time.sleep(0.0008)

    try:
        client.connect("127.0.0.1", server.port)
        deadline = time.time() + 4.0
        while time.time() < deadline and not (client.state == N.CONNECTED and client.predictor.initialised):
            events.extend(client.update())
            time.sleep(0.01)
        assert client.state == N.CONNECTED and client.mode == C.MODE_ESC
        assert client.score_limit == W.ESCALATION_STEPS
        play(1.0, lambda now: C.BTN_FORWARD)
        store = client.predictor.store
        assert tuple(store.weapons.weapon[0]) == (LADDER[0], LADDER[0])     # not the loadout the client sent
        assert int(store.equip[0, 0]) == HATCHET
        with server.lock:
            server.world.set_ladder_step(client.pid, 6, S.LADDER_PROMOTED, S.NO_PLAYER)
        before = client.predictor.corrections
        play(1.5, lambda now: C.BTN_FORWARD | (C.BTN_SHOOT if int(now * 4) % 2 else 0))
        assert tuple(store.weapons.weapon[0]) == (LADDER[6], LADDER[6])
        assert any(e[0] == S.EV_LADDER and e[1] == client.pid and e[2] == 6 for e in events)
        assert client.predictor.corrections - before <= 2                 # the hand-over, then exact again
        with server.lock:
            server_ammo = int(server.world.store.weapons.ammo[client.pid, 0])
        assert server_ammo < int(store.weapons.mag[0, 0])                  # it fired the new gun
    finally:
        client.disconnect()
        server.stop()


def test_bots_climb_the_ladder_and_always_hold_the_gun_of_their_step():
    summary = bots.run_bot_match("skybridge", bots=8, ticks=60 * 90, difficulty=2, mode=C.MODE_ESC, seed=5)
    assert summary["finite"] and summary["ladder_ok"]
    assert summary["kills"] > 60 and summary["ladder_events"] > 60
    assert max(summary["ladder"]) >= 6 or summary["matches"] >= 1
