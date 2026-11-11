"""Module A: wire codecs, LAN discovery, rollback, prediction and interpolation."""
import time

import numpy as np

import helpers  # noqa: F401

from engine import config as C
from engine import mathx as M
from engine import network_lan as N
from engine import simulation as S
from engine import weapons as W

ZERO_LEVELS = np.zeros((2, 8), int)
ZERO_OPERATOR = np.zeros(4, int)


def make_client(name, **net):
    return N.GameClient(name, W.DEFAULT_LOADOUT, ZERO_LEVELS, ZERO_OPERATOR, **net)


def test_command_delta_codec():
    f32 = lambda v: float(np.float32(v))
    commands = [(3, f32(0.5), f32(0.1), (10, 0, -3)), (3, f32(0.5), f32(0.1), (10, 0, -3)),
                (67, f32(0.51), f32(0.1), (11, 0, -3)), (0, f32(-1.0), f32(0.2), (0, 0, 0))]
    encoded = N.encode_commands(commands)
    decoded, offset = N.decode_commands(encoded, 0, len(commands))
    assert decoded == commands and offset == len(encoded)
    assert len(encoded) == 17 + 1 + 13 + 17                          # full, unchanged (1 byte), partial, full
    assert len(N.encode_commands(commands[:1] * 4)) == 17 + 3        # three identical repeats cost one byte each


def test_snapshot_entity_delta_codec():
    game = S.GameWorld("crucible", seed=1)
    for i in range(10):
        game.add_player("p%d" % i, is_bot=True)
    first = N.quantize_world(game.store, 1)
    full = N.encode_entities(first, None)
    decoded_first, used = N.decode_entities(full, 0, None)
    assert used == len(full)
    for name in ("T", "O", "S", "active"):
        assert np.array_equal(getattr(decoded_first, name), getattr(first, name))
    game.store.move.pos[3] += 1.0
    game.store.yaw[5] += 0.3
    game.store.health[7] = 40.0
    game.remove_player(9)
    game.add_player("late", is_bot=True)
    second = N.quantize_world(game.store, 2)
    delta = N.encode_entities(second, first)
    decoded_second, _ = N.decode_entities(delta, 0, decoded_first)
    for name in ("T", "O", "S", "active"):
        assert np.array_equal(getattr(decoded_second, name), getattr(second, name))
    assert len(delta) < len(full) / 3
    assert len(N.encode_entities(second, second)) == N.ENT_MASKS.size      # nothing changed: masks only
    assert abs(first.T["px"][0] / C.POS_QUANT - game.store.move.pos[0, 0]) <= 0.5 / C.POS_QUANT + 1e-9


def test_event_codec_round_trip():
    events = [(S.EV_HIT, 1, 2, 26.4, 0, True, (1.5, 2.25, -3.0)), (S.EV_DAMAGE, 2, 1, 26.4, 1, (4.0, 0.0, 1.0)),
              (S.EV_KILL, 1, 2, 3, 1), (S.EV_SPAWN, 4, (1.0, 0.0, 2.0), 1.5), (S.EV_SHOT, 1, 0, (0, 1.5, 0), (10, 1.5, 30)),
              (S.EV_EXPLOSION, (1, 2, 3), 255, 7), (S.EV_GRENADE, 7, 1, (0, 1, 2), (5, 6, -7), 96, 12345),
              (S.EV_REWARD, 1, 125, 100, 1), (S.EV_MATCH, 2, 0, 1, 0)]
    for event in events:
        blob = N.encode_event(event)
        decoded, offset = N.decode_event(blob, 0)
        assert offset == len(blob) and decoded[0] == event[0]
    assert N.decode_event(N.encode_event(events[0]), 0)[0] == (S.EV_HIT, 2, 26.4, 0, True, (1.5, 2.25, -3.0))
    assert N.decode_event(N.encode_event(events[6]), 0)[0] == (S.EV_GRENADE, 7, 1, (0.0, 1.0, 2.0), (5.0, 6.0, -7.0), 96, 12345)
    spawn = N.decode_event(N.encode_event(events[3]), 0)[0]
    assert abs(spawn[3] - 1.5) < 1e-3


def test_beacon_is_sealed_and_validated():
    datagram = N.encode_beacon("FLOOR 3 ARENA", 28016, 2, 12, 6, 1, 0, 0xDEADBEEF)
    info = N.decode_beacon(datagram, "10.0.0.5")
    assert (info.name, info.port, info.players, info.max_players, info.bots, info.uid) == ("FLOOR 3 ARENA", 28016, 2, 12, 6, 0xDEADBEEF)
    assert info.compatible and info.map_name == "SKYBRIDGE" and info.mode_name == C.MODE_NAMES[0]
    assert b"FLOOR" not in datagram                                           # encrypted on the wire
    forged = bytearray(datagram)
    forged[-1] ^= 1
    assert N.decode_beacon(bytes(forged), "x") is None and N.decode_beacon(b"hello", "x") is None
    assert (C.BROADCAST_ADDRESS, C.DISCOVERY_PORT) in N.broadcast_targets()
    assert N.seq_newer(1, 65535) and not N.seq_newer(65535, 1) and not N.seq_newer(5, 5)


def test_history_ring_holds_exactly_500_ms():
    assert C.HISTORY_TICKS == 30 and abs(C.HISTORY_MS - 500.0) < 1e-9
    game = S.GameWorld("crucible", seed=2)
    game.add_player("a")
    lag = N.LagCompensator(game.store.capacity)
    assert lag.root.shape == (30, C.MAX_PLAYERS, 4, 4)
    for tick in range(1, 100):
        game.store.move.pos[0, 0] = tick
        game.store.refresh_hitboxes()
        lag.record(tick, game.store)
    assert sorted(lag.ticks.tolist()) == list(range(70, 100))                # only the newest 30 ticks survive
    root, _, _, _ = lag.sample(85.0 * C.TICK_MS)
    assert root[0, 0, 3] == 85.0                                             # exact matrix of that tick
    root, _, _, _ = lag.sample(85.25 * C.TICK_MS)
    assert np.isclose(root[0, 0, 3], 85.25)                                  # interpolated between two ticks
    lag.sample(-1.0e9)
    assert lag.last_rewind_ticks == 29 and lag.clamped >= 1


def test_rollback_registers_a_hit_the_present_would_miss():
    game = S.GameWorld("crucible", seed=2)
    shooter, runner = game.add_player("shooter"), game.add_player("runner")
    s = game.store
    s.protect_until[:] = 0
    s.weapons.spread_hip[:] = 0.0
    lag = N.LagCompensator(s.capacity)
    s.move.pos[shooter] = (-8.0, 0.0, 14.0)
    s.move.pos[runner] = (2.0, 0.0, 12.0)
    seen = {}
    for tick in range(1, 41):                                                 # runner strafes at 9 units / s
        s.move.pos[runner, 2] += 9.0 * C.TICK_DT
        game.tick = tick
        s.refresh_hitboxes()
        lag.record(tick, s)
        seen[tick] = s.hitboxes.world_center[runner, 1].copy()
    view_tick = 32.6                                                          # about 120 ms in the past
    old_chest = M.lerp(seen[32], seen[33], 0.6)
    assert np.linalg.norm(seen[40] - old_chest) > 1.0
    s.yaw[shooter], s.pitch[shooter] = M.yaw_pitch_from_direction(old_chest - s.eye_position(shooter)[0])
    s.move.mflags[shooter] = C.MF_ON_GROUND
    game.drain_events()
    game.resolve_shots(np.array([shooter]))
    assert not [e for e in game.drain_events() if e[0] == S.EV_HIT]           # present-time hitboxes: miss
    present = s.hitboxes.root.copy()
    game.resolve_shots(np.array([shooter]), rewind=lambda row: lag.rewind(s, view_tick * C.TICK_MS, row))
    hits = [e for e in game.drain_events() if e[0] == S.EV_HIT]
    assert len(hits) == 1 and hits[0][2] == runner                            # rewound hitboxes: hit
    assert np.array_equal(present, s.hitboxes.root)                           # world snapped back to the present
    try:
        with lag.rewind(s, view_tick * C.TICK_MS, shooter):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert np.array_equal(present, s.hitboxes.root)                           # restored even after an exception


def test_rollback_never_hits_a_respawned_player_at_the_old_spot():
    game = S.GameWorld("crucible", seed=2)
    shooter, runner = game.add_player("shooter"), game.add_player("runner")
    s = game.store
    lag = N.LagCompensator(s.capacity)
    for tick in range(1, 20):
        game.tick = tick
        s.refresh_hitboxes()
        lag.record(tick, s)
    s.spawn_seq[runner] += 1                                                  # died and respawned since
    with lag.rewind(s, 10 * C.TICK_MS, shooter):
        assert not s.hitboxes.valid[runner] and not s.hitboxes.valid[shooter]


def test_interpolation_buffer_lerp_slerp_and_teleport_snap():
    buffer = N.InterpolationBuffer()
    game = S.GameWorld("crucible", seed=3)
    pid = game.add_player("mover")
    s = game.store
    for tick, x, yaw in ((10, 0.0, 3.0), (11, 1.0, -3.0), (12, 2.0, -2.8)):
        s.move.pos[pid] = (x, 0.0, 0.0)
        s.yaw[pid] = yaw
        buffer.push(N.quantize_world(s, tick), tick * C.TICK_MS)
    view = buffer.sample(10.5 * C.TICK_MS)
    assert abs(view.pos[pid, 0] - 0.5) < 0.01 and view.alive[pid]
    assert abs(abs(view.yaw[pid]) - np.pi) < 0.01                             # short way round through pi
    assert abs(buffer.sample(11.25 * C.TICK_MS).pos[pid, 0] - 1.25) < 0.01
    assert abs(buffer.sample(5.0 * C.TICK_MS).pos[pid, 0]) < 0.01             # before the buffer: clamp to oldest
    s.move.pos[pid] = (40.0, 0.0, 0.0)
    s.spawn_seq[pid] += 1
    buffer.push(N.quantize_world(s, 13), 13 * C.TICK_MS)
    early, late = buffer.sample(12.3 * C.TICK_MS), None
    assert abs(early.pos[pid, 0] - 2.0) < 0.01                                # respawn: no sweep across the map
    late = buffer.sample(12.7 * C.TICK_MS)
    assert abs(late.pos[pid, 0] - 40.0) < 0.01
    assert C.INTERP_DELAY_MS == 100.0


def test_predictor_replays_unacknowledged_commands_after_a_correction():
    from engine import maps
    world = maps.load_map("crucible").world
    server = S.PlayerStore(1)
    server.active[0] = server.alive[0] = True
    server.move.reset_rows([0], [-15.0, 0.0, -5.0])
    predictor = N.Predictor(world, 0, W.DEFAULT_LOADOUT, ZERO_LEVELS, ZERO_OPERATOR)
    predictor.reconcile(0, server.pack_owner(0))
    packets = {}
    for tick in range(1, 31):
        buttons, yaw = C.BTN_FORWARD | C.BTN_SPRINT, float(np.float32(0.4))
        predictor.predict(tick, buttons, yaw, 0.0)
        S.simulate_commands(server, world, [0], buttons, yaw, 0.0, tick)
        packets[tick] = server.pack_owner(0)
    assert not predictor.reconcile(20, packets[20]) and predictor.corrections == 0   # agreement: nothing to do
    assert predictor.last_error < 1e-4
    nudged = S.PlayerStore(1)
    nudged.active[0] = True
    nudged.unpack_owner(0, packets[20])
    nudged.move.pos[0, 0] += 0.04                                                 # below the 0.05 threshold
    assert not predictor.reconcile(20, nudged.pack_owner(0))
    nudged.move.pos[0, 0] += 0.40                                                 # server knows better
    truth = S.PlayerStore(1)
    truth.active[0] = True
    truth.weapons.set_loadout(0, W.DEFAULT_LOADOUT, ZERO_LEVELS)
    truth.unpack_owner(0, nudged.pack_owner(0))
    for tick in range(21, 31):
        S.simulate_commands(truth, world, [0], C.BTN_FORWARD | C.BTN_SPRINT, float(np.float32(0.4)), 0.0, tick)
    before = predictor.store.move.pos[0].copy()
    assert predictor.reconcile(20, nudged.pack_owner(0))
    assert predictor.corrections == 1 and predictor.replayed == 10                # 10 pending commands re-run
    assert np.allclose(predictor.store.move.pos[0], truth.move.pos[0], atol=1e-6)
    assert np.allclose(predictor.render_position(), before, atol=1e-6)            # visual error hides the snap
    predictor.decay_visual_error(1.0)
    assert np.allclose(predictor.render_position(), predictor.store.move.pos[0], atol=1e-4)


def run_session(seconds, drive, port, lag=0.0, jitter=0.0, loss=0.0, clients=1, bots=0):
    config = N.ServerConfig(name="TEST", map_key="crucible", bot_fill=bots, beacon=False, port=port, seed=9,
                            fake_lag_ms=lag, fake_jitter_ms=jitter, fake_loss=loss)
    server = N.host_local_game(config)
    players = [make_client("p%d" % i, fake_lag_ms=lag, fake_jitter_ms=jitter, fake_loss=loss, seed=i + 1)
               for i in range(clients)]
    for index, client in enumerate(players):
        client.connect("127.0.0.1", server.port, team=index % 2)
    events = [[] for _ in players]
    started = next_tick = time.perf_counter()
    try:
        while time.perf_counter() - started < seconds:
            for index, client in enumerate(players):
                events[index] += client.update()
            now = time.perf_counter()
            if now >= next_tick and all(c.state == N.CONNECTED for c in players):
                next_tick += C.TICK_DT
                for client in players:
                    client.sample_entities()
                drive(now - started, server, players)
            time.sleep(0.0008)
        return server.describe(), players, events
    finally:
        for client in players:
            client.disconnect()
        server.stop()


def test_live_session_prediction_matches_the_server_exactly():
    def drive(t, server, players):
        buttons = C.BTN_FORWARD | (C.BTN_SPRINT if t > 0.8 else 0) | (C.BTN_JUMP if int(t * 2) % 3 == 0 else 0)
        buttons |= (C.BTN_RIGHT if t > 1.6 else 0) | (C.BTN_SLIDE if 1.2 < t < 1.5 else 0)
        players[0].send_command(buttons, 0.4 * np.sin(t), 0.0)

    info, players, _ = run_session(3.0, drive, 28110)
    client = players[0]
    assert client.stats["snapshots"] > 120 and info["full_snapshots"] <= 3 and info["malformed"] == 0
    assert client.predictor.corrections == 0 and client.predictor.max_error < 1e-3
    assert client.predictor.acked_tick > client.cmd_tick - 12
    assert client.pid in client.roster and client.ping_ms < 60.0


def test_live_session_rollback_under_latency_jitter_and_loss():
    state = {"placed": False, "shots": 0}

    def drive(t, server, players):
        alpha, bravo = players
        if not state["placed"] and t > 1.0:
            with server.lock:
                store = server.world.store
                store.move.reset_rows([alpha.pid], np.array([-8.0, 0.0, 15.0]))
                store.move.reset_rows([bravo.pid], np.array([6.0, 0.0, 12.5]))
                store.protect_until[:] = 0
                store.weapons.spread_hip[alpha.pid] = 0.0
                store.weapons.bloom_shot[alpha.pid] = 0.0
            alpha.predictor.store.weapons.spread_hip[0] = 0.0
            alpha.predictor.store.weapons.bloom_shot[0] = 0.0
            state["placed"] = True
        with server.lock:
            server.world.store.health[bravo.pid] = 100.0                        # keep the target alive
        yaw = pitch = 0.0
        buttons = 0
        view = alpha.view
        if state["placed"] and t > 2.0 and view.alive[bravo.pid]:
            eye = alpha.predictor.store.eye_position(0)[0]
            yaw, pitch = M.yaw_pitch_from_direction(view.pos[bravo.pid] + np.array([0.0, 1.22, 0.0]) - eye)
            buttons = C.BTN_SHOOT                                               # aim at the *interpolated* image
        fired, _ = alpha.send_command(buttons, float(yaw), float(pitch))
        state["shots"] += fired
        bravo.send_command((C.BTN_LEFT if int(t / 0.7) % 2 else C.BTN_RIGHT) if t > 2.0 else 0, np.pi, 0.0)

    info, players, events = run_session(6.5, drive, 28120, lag=45.0, jitter=8.0, loss=0.03, clients=2)
    hits = [e for e in events[0] if e[0] == S.EV_HIT]
    assert state["shots"] >= 20
    assert len(hits) >= state["shots"] * 0.9, (len(hits), state["shots"])       # hit registration despite 90 ms RTT
    assert info["rewinds"] >= state["shots"] - 2 and info["fire_missing"] <= 2
    assert 8.0 < info["last_rewind_ticks"] < 20.0                               # 100 ms interp + latency, inside 500 ms
    assert 60.0 < players[0].ping_ms < 140.0
    assert players[1].predictor.corrections <= 6                                # strafing stays predicted


def test_lan_discovery_finds_a_hosted_match():
    browser = N.LanBrowser()
    browser.start()
    server = N.host_local_game(N.ServerConfig(name="DISCOVERY TEST", map_key="skybridge", bot_fill=3, port=28130))
    try:
        deadline = time.time() + 4.0
        found = []
        while time.time() < deadline and not found:
            found = [s for s in browser.servers() if s.name == "DISCOVERY TEST"]
            time.sleep(0.05)
        assert found, browser.error or "no beacon received"
        info = found[0]
        assert info.port == server.port and info.map_name == "SKYBRIDGE" and info.max_players == 12 and info.compatible
    finally:
        server.stop()
        browser.stop()


def test_server_refuses_wrong_protocol_and_full_house():
    config = N.ServerConfig(name="TINY", map_key="crucible", bot_fill=0, beacon=False, port=28140, max_players=1)
    server = N.host_local_game(config)
    first, second = make_client("one"), make_client("two")
    try:
        first.connect("127.0.0.1", server.port)
        deadline = time.time() + 3.0
        while time.time() < deadline and first.state != N.CONNECTED:
            first.update()
            time.sleep(0.01)
        assert first.state == N.CONNECTED
        second.connect("127.0.0.1", server.port)
        deadline = time.time() + 3.0
        while time.time() < deadline and second.state == N.CONNECTING:
            first.update()
            second.update()
            time.sleep(0.01)
        assert second.state == N.FAILED and "full" in second.fail_reason
    finally:
        first.disconnect()
        second.close()
        server.stop()
