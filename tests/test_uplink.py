"""Sprint 3: the UPLINK zone-control mode."""
import time

import numpy as np

import helpers  # noqa: F401

from engine import bots
from engine import config as C
from engine import maps
from engine import network_lan as N
from engine import simulation as S
from engine import weapons as W


def uplink_game(squads=(0, 0, 1), key="crucible"):
    game = S.GameWorld(key, mode=C.MODE_UPLINK, seed=3)
    rows = [game.add_player("p%d" % i, team=team) for i, team in enumerate(squads)]
    game.store.protect_until[:] = 0
    game.drain_events()
    return game, rows


def stand(game, row, zone):
    game.store.move.pos[row] = game.uplink_pos[zone] if zone >= 0 else np.array([0.0, -500.0, 0.0])
    game.store.move.vel[row] = 0.0


def run(game, seconds, keep=()):
    """Tick the match; players in ``keep`` are pinned where they stand.  Returns the uplink / reward events."""
    pinned = {row: game.store.move.pos[row].copy() for row in keep}
    events = []
    for _ in range(int(round(seconds * C.TICK_RATE))):
        game.end_tick()
        for row, spot in pinned.items():
            game.store.move.pos[row] = spot
            game.store.move.vel[row] = 0.0
        events += [e for e in game.drain_events() if e[0] in (S.EV_UPLINK, S.EV_REWARD)]
    return events


def test_every_arena_gets_three_fair_uplinks():
    for key in maps.MAP_KEYS:
        loaded = maps.load_map(key)
        pos, nodes = maps.uplink_zones(loaded)
        assert pos.shape == (3, 3) and np.all(nodes >= 0) and np.allclose(pos, loaded.nav.nodes[nodes])
        data = loaded.data
        homes = [data.spawn_pos[data.spawn_team == team].mean(axis=0) for team in (0, 1)]
        assert abs(np.linalg.norm(pos[0] - homes[0]) - np.linalg.norm(pos[2] - homes[1])) < 0.5, key
        ab = np.linalg.norm(pos[0, [0, 2]] - pos[1, [0, 2]])
        bc = np.linalg.norm(pos[1, [0, 2]] - pos[2, [0, 2]])
        assert abs(ab - bc) < 1.0 and min(ab, bc) > 2.0 * C.UPLINK_RADIUS, key


def test_squads_spawn_like_team_deathmatch():
    game, (a, b, c) = uplink_game()
    s = game.store
    assert C.team_mode(C.MODE_UPLINK) and game.score_limit == C.UPLINK_SCORE_LIMIT
    assert s.team[[a, b, c]].tolist() == [0, 0, 1]
    spawn_team = game.loaded.data.spawn_team
    for row in (a, b, c):
        nearest = np.argmin(np.linalg.norm(game.loaded.data.spawn_pos - s.move.pos[row], axis=1))
        assert spawn_team[nearest] == s.team[row]


def test_one_player_captures_a_neutral_uplink_in_the_capture_time():
    game, (a, b, c) = uplink_game()
    stand(game, a, 0)
    stand(game, b, -1)
    stand(game, c, -1)
    early = run(game, C.UPLINK_CAPTURE_TIME - 0.2, keep=(a,))
    assert game.uplink_owner[0] == -1 and not any(e[0] == S.EV_UPLINK for e in early)
    late = run(game, 0.3, keep=(a,))
    assert game.uplink_owner[0] == 0
    assert (S.EV_UPLINK, 0, 0, S.UPLINK_CAPTURED) in late
    assert (S.EV_REWARD, a, C.UPLINK_REWARD[0], C.UPLINK_REWARD[1], S.REWARD_CAPTURE) in late


def test_more_squad_members_capture_faster_and_enemies_contest():
    game, (a, b, c) = uplink_game()
    stand(game, a, 1)
    stand(game, b, 1)
    stand(game, c, -1)
    run(game, 4.05, keep=(a, b))
    assert game.uplink_owner[1] == 0                                         # two players: four seconds
    game, (a, b, c) = uplink_game()
    stand(game, a, 1)
    stand(game, c, 1)
    stand(game, b, -1)
    run(game, 10.0, keep=(a, c))
    assert game.uplink_contested[1] and game.uplink_charge[1] == 0.0 and game.uplink_owner[1] == -1


def test_an_enemy_uplink_is_neutralised_before_it_is_captured():
    game, (a, b, c) = uplink_game()
    stand(game, a, 0)
    stand(game, b, -1)
    stand(game, c, -1)
    run(game, C.UPLINK_CAPTURE_TIME + 0.1, keep=(a,))
    stand(game, a, -1)
    stand(game, c, 0)
    events = run(game, 2.0 * C.UPLINK_CAPTURE_TIME + 0.2, keep=(c,))
    changes = [(e[2], e[3]) for e in events if e[0] == S.EV_UPLINK]
    assert changes == [(0, S.UPLINK_NEUTRALISED), (1, S.UPLINK_CAPTURED)] and game.uplink_owner[0] == 1


def test_idle_uplinks_settle_and_held_uplinks_score_until_the_limit():
    game, (a, b, c) = uplink_game()
    game.uplink_owner[:] = (0, -1, 1)
    game.uplink_charge[:] = (-0.4, 0.5, 1.0)
    for row in (a, b, c):
        stand(game, row, -1)
    run(game, 3.0)
    assert np.isclose(game.uplink_charge[0], -0.9, atol=0.02)                # an owned uplink recharges
    assert np.isclose(game.uplink_charge[1], 0.25, atol=0.02)                # a neutral one drains at half speed
    game.uplink_owner[:] = (0, 0, 1)
    game.uplink_charge[:] = (-1.0, -1.0, 1.0)
    before = game.team_scores.copy()
    run(game, 10.0)
    gained = game.team_scores - before
    assert gained.tolist() == [10, 5]                                        # one point per held uplink every 2 s
    game.kill_player(c, a, 0, 0)
    assert (game.team_scores - before).tolist() == [10, 5]                   # kills score nothing in UPLINK
    game.team_scores[0] = C.UPLINK_SCORE_LIMIT - 1
    run(game, C.UPLINK_POINT_INTERVAL + 0.1)
    assert game.phase == C.PHASE_ENDED and game.winner == 0


def test_bots_spread_over_the_uplinks_in_mirror_image():
    spread = bots._spread(np.array([0, 1, 0, 1, 0, 1, 0, 1]), 3)
    picks = np.argmax(spread, axis=1).tolist()
    assert picks[0::2] == [0, 1, 2, 0] and picks[1::2] == [2, 1, 0, 2]


def test_bots_fight_over_the_uplinks():
    game = S.GameWorld("crucible", mode=C.MODE_UPLINK, seed=2)
    for index in range(8):
        game.add_player("BOT-%02d" % index, is_bot=True)
    controller = bots.BotController(game, 1, seed=3)
    game.drain_events()
    captured = set()
    for _ in range(60 * 120):
        rows = game.bots()
        buttons, yaw, pitch = controller.think(rows)
        fired = game.step_commands(rows, buttons, yaw, pitch, game.tick + 1)
        game.resolve_shots(rows[fired])
        game.end_tick()
        events = game.drain_events()
        controller.observe(events)
        captured |= {(e[1], e[2]) for e in events if e[0] == S.EV_UPLINK and e[3] == S.UPLINK_CAPTURED}
    assert len(captured) >= 3 and game.team_scores.sum() > 20


def test_zone_state_travels_in_every_snapshot():
    game, _ = uplink_game()
    game.uplink_owner[:] = (0, -1, 1)
    game.uplink_charge[:] = (-1.0, 0.25, 0.6)
    game.uplink_contested[:] = (False, True, False)
    owner, contested, charge = N.unpack_uplinks(N.pack_uplinks(game))
    assert owner.tolist() == [0, -1, 1] and contested.tolist() == [False, True, False]
    assert np.allclose(charge, [-1.0, 0.25, 0.6], atol=0.01)
    event = (S.EV_UPLINK, 2, 1, S.UPLINK_CAPTURED)
    assert N.decode_event(N.encode_event(event), 0)[0] == event


def test_live_session_the_client_sees_the_uplinks():
    config = N.ServerConfig(name="UPLINK", map_key="crucible", mode=C.MODE_UPLINK, bot_fill=0, beacon=False,
                            port=28198, seed=4)
    server = N.host_local_game(config)
    zeros = np.zeros((W.LOADOUT_SLOTS, W.UPGRADE_COUNT), dtype=np.int64)
    client = N.GameClient("zone", W.DEFAULT_LOADOUT, zeros, np.zeros(W.OPERATOR_COUNT, dtype=np.int64))
    try:
        client.connect("127.0.0.1", server.port)
        deadline = time.time() + 4.0
        while time.time() < deadline and client.state != N.CONNECTED:
            client.update()
            time.sleep(0.01)
        with server.lock:
            server.world.uplink_owner[:] = (1, -1, 0)
            server.world.uplink_charge[:] = (1.0, -0.5, -1.0)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            client.update()
            client.send_command(0, 0.0, 0.0)
            time.sleep(C.TICK_DT)
        assert client.mode == C.MODE_UPLINK and client.uplink_owner.tolist() == [1, -1, 0]
        assert np.allclose(client.uplink_charge[[0, 2]], [1.0, -1.0], atol=0.01)
    finally:
        client.disconnect()
        server.stop()
