"""Sprint 2: scorestreak rewards, the minimap picture and the death recap."""
import time

import numpy as np

import helpers  # noqa: F401

from engine import bots
from engine import config as C
from engine import maps
from engine import mathx as M
from engine import network_lan as N
from engine import simulation as S
from engine import streaks as ST
from engine import weapons as W
from engine.recap import CombatLog

LANCE_KILL = C.STREAK_KILL_BASE + ST.LANCE


def game_with(players=2, mode=C.MODE_FFA, teams=None):
    game = S.GameWorld("crucible", mode=mode, seed=3)
    rows = [game.add_player("p%d" % i, team=None if teams is None else teams[i]) for i in range(players)]
    game.store.protect_until[:] = 0
    game.drain_events()
    return game, rows


def kill(game, victim, attacker, weapon=0):
    if not game.store.alive[victim]:
        game.spawn_player(victim)
    game.kill_player(victim, attacker, weapon, 0)
    return [event for event in game.drain_events() if event[0] == S.EV_STREAK]


def press(game, row, held=True):
    s = game.store
    game.step_commands(np.array([row]), np.array([C.BTN_STREAK if held else 0], dtype=np.uint16), s.yaw[row],
                       s.pitch[row], int(s.cmd_tick[row]) + 1)
    return [event for event in game.drain_events() if event[0] == S.EV_STREAK]


def test_the_reward_table_and_its_helpers():
    assert ST.validate_streaks() == []
    assert ST.STREAK_NAMES == ("PULSE SCAN", "AEGIS SHIELD", "ORBITAL LANCE") and ST.KILLS.tolist() == [3, 5, 7]
    assert ST.earned_by(5, 0b001) == [1] and ST.earned_by(2, 0) == []
    assert ST.next_ready(0b110) == 1 and ST.next_ready(0) == -1
    assert ST.next_goal(4, 0b001) == 1 and ST.next_goal(9, 0b111) == -1
    assert ST.is_streak_kill(LANCE_KILL) and not ST.is_streak_kill(C.EQUIP_KILL_BASE) and not ST.is_streak_kill(0)
    assert np.allclose(ST.lance_damage([0.0, C.LANCE_RADIUS / 2, C.LANCE_RADIUS, 20.0]),
                       [C.LANCE_DAMAGE, C.LANCE_DAMAGE / 2, 0.0, 0.0])


def test_rewards_are_earned_once_per_life_and_wait_through_death():
    game, (a, b) = game_with()
    s = game.store
    events = []
    for _ in range(3):
        events += kill(game, b, a)
    assert events == [(S.EV_STREAK, a, ST.SCAN, S.STREAK_EARNED, events[0][4])] and s.streak_ready[a] == 0b001
    for _ in range(2):
        kill(game, b, a)
    assert s.streak_ready[a] == 0b011 and s.streak_earned[a] == 0b011
    kill(game, a, b)                                                    # death: the streak resets, the rewards wait
    assert s.streak[a] == 0 and s.streak_earned[a] == 0 and s.streak_ready[a] == 0b011
    again = []
    for _ in range(3):
        again += kill(game, b, a)
    assert again == [] and s.streak_ready[a] == 0b011                  # a waiting reward is not announced twice
    game.restart_match()
    assert s.streak_ready[a] == 0 and s.streak[a] == 0


def test_x_uses_the_smallest_ready_reward_once_per_press():
    game, (a, _b) = game_with()
    s = game.store
    s.streak_ready[a] = 0b011
    events = press(game, a)
    assert [(e[2], e[3]) for e in events] == [(ST.SCAN, S.STREAK_USED)] and s.streak_ready[a] == 0b010
    assert press(game, a) == []                                         # held: nothing more
    events = press(game, a, held=False) + press(game, a)
    assert [(e[2], e[3]) for e in events] == [(ST.AEGIS, S.STREAK_USED)] and s.streak_ready[a] == 0
    assert s.shield[a] == C.AEGIS_POINTS


def test_the_dead_and_escalation_get_nothing():
    game, (a, b) = game_with()
    s = game.store
    s.streak_ready[a] = 0b001
    kill(game, a, b)
    assert not game.use_streak(a) and s.streak_ready[a] == 0b001
    esc = S.GameWorld("crucible", mode=C.MODE_ESC, seed=3)
    x, y = esc.add_player("x"), esc.add_player("y")
    for _ in range(8):
        if not esc.store.alive[y]:
            esc.spawn_player(y)
        esc.kill_player(y, x, int(esc.store.weapons.weapon[x, 0]), 0)
    assert esc.store.streak_ready[x] == 0


def test_the_aegis_shield_soaks_damage_first_and_fades():
    game, (a, b) = game_with()
    s = game.store
    s.streak_ready[a] = 1 << ST.AEGIS
    assert game.use_streak(a)
    game.drain_events()
    game.apply_damage(a, b, 30.0, C.ZONE_TORSO, 0, s.move.pos[a])
    assert s.shield[a] == 20.0 and s.health[a] == 100.0
    game.apply_damage(a, b, 40.0, C.ZONE_TORSO, 0, s.move.pos[a])
    events = game.drain_events()
    assert s.shield[a] == 0.0 and s.health[a] == 80.0
    assert [e[3] for e in events if e[0] == S.EV_HIT] == [30.0, 40.0]          # absorbed damage still counts as dealt
    assert any(e[0] == S.EV_STREAK and e[3] == S.STREAK_BROKEN for e in events)
    s.streak_ready[a] = 1 << ST.AEGIS
    game.use_streak(a)
    s.cmd_tick[a] = s.shield_until[a]
    game.end_tick()
    assert s.shield[a] == 0.0                                                    # faded after 20 seconds


def test_the_pulse_scan_covers_the_squad_in_tdm_and_only_you_in_ffa():
    game, (a, b, c, d) = game_with(4, C.MODE_TDM, teams=(0, 1, 0, 1))
    s = game.store
    s.streak_ready[a] = 1 << ST.SCAN
    game.use_streak(a)
    ticks = ST.duration_ticks(ST.SCAN)
    assert (s.scan_until[[a, c]] - s.cmd_tick[[a, c]] == ticks).all() and (s.scan_until[[b, d]] == 0).all()
    game, (a, b, c) = game_with(3)
    s = game.store
    s.streak_ready[a] = 1 << ST.SCAN
    game.use_streak(a)
    assert s.scan_until[a] > 0 and (s.scan_until[[b, c]] == 0).all()


def test_the_orbital_lance_marks_the_crosshair_and_lands_after_the_warning():
    game, (a, b, c) = game_with(3)
    s = game.store
    s.move.pos[a] = (2.0, 0.0, 10.0)
    s.move.pos[b] = (2.0, 0.0, 14.0)
    s.move.pos[c] = (14.0, 0.0, 14.0)                                    # far outside the blast
    s.refresh_hitboxes()
    target = np.array([2.0, 0.0, 14.5])
    s.yaw[a], s.pitch[a] = M.yaw_pitch_from_direction(target - s.eye_position(a)[0])
    s.streak[a] = 7
    s.streak_ready[a] = 1 << ST.LANCE
    events = press(game, a)
    assert [(e[2], e[3]) for e in events] == [(ST.LANCE, S.STREAK_USED)]
    marked = np.asarray(events[0][4])
    assert np.linalg.norm(marked - target) < 0.4
    for _ in range(ST.duration_ticks(ST.LANCE) - 1):
        game.end_tick()
    assert s.alive[b] and s.health[b] == 100.0                           # the warning time is honoured
    game.end_tick()
    events = game.drain_events()
    assert any(e[0] == S.EV_STREAK and e[3] == S.STREAK_IMPACT for e in events)
    assert (S.EV_KILL, a, b, LANCE_KILL, S.KILL_EXPLOSIVE) in events
    assert s.health[a] == 100.0 and s.health[c] == 100.0                 # the caller is spared, c is out of reach
    assert s.streak[a] == 7                                              # reward kills never extend the streak


def test_the_lance_spares_the_squad():
    game, (a, b, c) = game_with(3, C.MODE_TDM, teams=(0, 0, 1))
    s = game.store
    for row, x in ((a, 2.0), (b, 3.0), (c, 1.0)):
        s.move.pos[row] = (x, 0.0, 14.0)
    s.refresh_hitboxes()
    game.lance_impact(np.array([2.0, 0.05, 14.0]), a)
    assert s.health[a] == s.health[b] == 100.0 and not s.alive[c]


def test_wire_formats_carry_the_rewards():
    event = (S.EV_STREAK, 4, ST.LANCE, S.STREAK_IMPACT, (12.5, 0.0, -3.25))
    decoded, offset = N.decode_event(N.encode_event(event), 0)
    assert decoded[:4] == event[:4] and np.allclose(decoded[4], event[4], atol=0.05)
    game, (a, b) = game_with()
    s = game.store
    s.shield[a], s.shield_until[a], s.scan_until[a], s.streak[a], s.streak_ready[a] = 37.5, 900, 640, 6, 0b101
    other = S.PlayerStore(1)
    other.unpack_owner(0, s.pack_owner(a))
    assert other.reward_state(0) == (37.5, 900, 640, 6, 0b101)
    frame = N.quantize_world(s, game.tick)
    decoded_frame, _ = N.decode_entities(N.encode_entities(frame, None), 0, None)
    assert decoded_frame.S["shield"][a] == 38 and decoded_frame.S["shield"][b] == 0


def test_live_session_x_raises_the_shield_and_the_client_shows_it():
    config = N.ServerConfig(name="AEGIS", map_key="crucible", mode=C.MODE_FFA, bot_fill=0, beacon=False,
                            port=28196, seed=4)
    server = N.host_local_game(config)
    zeros = np.zeros((W.LOADOUT_SLOTS, W.UPGRADE_COUNT), dtype=np.int64)
    client = N.GameClient("shield", W.DEFAULT_LOADOUT, zeros, np.zeros(W.OPERATOR_COUNT, dtype=np.int64))
    events = []
    try:
        client.connect("127.0.0.1", server.port)
        deadline = time.time() + 4.0
        while time.time() < deadline and not (client.state == N.CONNECTED and client.predictor.initialised):
            events.extend(client.update())
            time.sleep(0.01)
        assert client.state == N.CONNECTED
        with server.lock:
            server.world.store.streak_ready[client.pid] = 1 << ST.AEGIS
        end = next_tick = time.perf_counter()
        end += 1.0
        while time.perf_counter() < end:
            events.extend(client.update())
            now = time.perf_counter()
            if now >= next_tick:
                next_tick += C.TICK_DT
                client.send_command(C.BTN_STREAK if now < end - 0.6 else 0, 0.0, 0.0)
            time.sleep(0.0008)
        with server.lock:
            assert server.world.store.shield[client.pid] == C.AEGIS_POINTS
        assert client.predictor.store.shield[0] == C.AEGIS_POINTS and client.predictor.store.streak_ready[0] == 0
        assert any(e[0] == S.EV_STREAK and e[1] == client.pid and e[3] == S.STREAK_USED for e in events)
    finally:
        client.disconnect()
        server.stop()


def test_bots_earn_and_use_their_rewards():
    summary = bots.run_bot_match("datacore", bots=8, ticks=60 * 90, difficulty=2, mode=C.MODE_TDM, seed=9)
    assert summary["finite"] and summary["streaks_used"] >= 3


def test_every_arena_has_a_readable_minimap():
    for key in maps.MAP_KEYS:
        data = maps.get_map_data(key)
        rgba, (x0, z0, span) = maps.minimap_image(data, size=128)
        assert rgba.shape == (128, 128, 4) and rgba.dtype == np.uint8
        assert (rgba[0, :, 3] > 200).all() and (rgba[:, -1, 3] > 200).all()          # framed
        assert np.mean(rgba[..., 3] > 200) > 0.3                                     # mostly arena
        col = ((data.spawn_pos[:, 0] - x0) / span * 128).astype(int)
        row = ((z0 + span - data.spawn_pos[:, 2]) / span * 128).astype(int)
        assert ((col > 0) & (col < 127) & (row > 0) & (row < 127)).all()
        assert (rgba[row, col, 3] > 200).all()                                       # spawns stand on geometry


def test_the_death_recap_tells_the_story():
    log = CombatLog()
    log.took(3, 40.0)
    log.took(3, 35.0, headshot=True)
    log.took(5, 20.0)
    log.gave(3, 27.0)
    text = log.recap(3, "BOT VEXA", "KR-11 KESTREL", 23.4, 61.0, headshot=True)
    first, second = text.split("\n")
    assert first == "KR-11 KESTREL  -  23 m  -  HEADSHOT"
    assert second == ("BOT VEXA hit you 2 times for 75  -  you hit them 1 time for 27  -  they have 61 HP left"
                      "  -  20 damage from others")
    assert log.recap(-1, "", "the fall", suicide=True) == "the fall\n95 damage from others"
    log.reset()
    assert log.recap(7, "BOT ECHO", "ORBITAL LANCE", 41.0, None) == (
        "ORBITAL LANCE  -  41 m\nBOT ECHO finished you  -  you never hit them")
