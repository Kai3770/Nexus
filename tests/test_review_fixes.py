"""Regression tests for defects found in the full code review (fuzzing, strict-numerics soak, reviewers)."""
import numpy as np

import helpers  # noqa: F401

from engine import config as C
from engine import simulation as S


def test_an_idle_neutral_uplink_drains_exactly_to_zero():
    # The drain used to multiply after clipping, so a neutral zone's charge halved every tick
    # and never reached zero (it ended in floating point underflow).
    game = S.GameWorld("crucible", mode=C.MODE_UPLINK, seed=3)
    for team in (0, 1):
        row = game.add_player("p%d" % team, team=team)
        game.store.move.pos[row] = (0.0, -500.0, 0.0)
    game.uplink_owner[:] = -1
    game.uplink_charge[:] = (0.004, -0.004, 0.3)
    with np.errstate(all="raise"):
        for _ in range(C.TICK_RATE):
            game.end_tick()
            game.store.move.pos[:2] = (0.0, -500.0, 0.0)
            game.store.move.vel[:2] = 0.0
    assert game.uplink_charge[0] == 0.0 and game.uplink_charge[1] == 0.0
    assert np.isclose(game.uplink_charge[2], 0.3 - 0.5 / C.UPLINK_CAPTURE_TIME, atol=1.0e-9)


# ---------------------------------------------------------------------------------------------------------
#  Hostile or broken network input (security review)
# ---------------------------------------------------------------------------------------------------------
import tempfile          # noqa: E402
import threading         # noqa: E402
import time              # noqa: E402

from engine import network_lan as N     # noqa: E402
from engine import secure_save as SS    # noqa: E402
from engine import weapons as W         # noqa: E402


def _server(port):
    """A server without its threads; packets are handed to it directly."""
    server = N.GameServer(N.ServerConfig(name="R", map_key="crucible", bot_fill=0, beacon=False, port=port))
    client = N.GameClient("r", W.DEFAULT_LOADOUT, np.zeros((2, W.UPGRADE_COUNT), int), np.zeros(W.OPERATOR_COUNT, int))
    client.connect("127.0.0.1", port)
    addr = ("10.0.0.5", 41000)
    server._handle_packet(client._request, addr, 100.0)
    client.close()                                  # only its request is needed; no receive thread left behind
    return server, server.conns[addr], addr, client._request


def _frame(conn, newest, commands, fires=()):
    payload = N.FRAME_HEADER.pack(0, 0, 0, newest, 0, len(commands), len(fires), 0, 4) + N.encode_commands(commands)
    for seq, tick in fires:
        payload += N.FIRE_RECORD.pack(seq, tick, 0, 0)
    return N.make_packet(N.PKT_CLIENT_FRAME, conn.token, payload)


def test_a_non_finite_view_angle_neither_crashes_nor_poisons_the_server():
    server, conn, addr, _ = _server(29801)
    base = conn.newest_cmd
    commands = [(C.BTN_SHOOT, float("inf"), float("nan"), (0, 0, 0)), (C.BTN_FORWARD, float("nan"), 0.2, (0, 0, 0))]
    server._handle_packet(_frame(conn, base + 2, commands), addr, 100.1)
    for _ in range(10):
        server.tick_once()                                                   # used to raise struct.error
    store = server.world.store
    assert np.isfinite(store.move.pos).all() and np.isfinite(store.yaw).all() and np.isfinite(store.pitch).all()
    assert N.quantize_position((float("nan"), float("inf"), -float("inf"))) == (0, 32767, -32767)


def test_command_ticks_cannot_run_ahead_of_real_time():
    server, conn, addr, _ = _server(29802)
    start = conn.newest_cmd
    # a cheat that jumps the command clock to skip cadence and cooldowns gets nothing
    for i in range(1, 50):
        server._handle_packet(_frame(conn, start + 1000 * i, [(C.BTN_SHOOT, 0.0, 0.0, (0, 0, 0))]), addr, 100.0)
    assert conn.newest_cmd == start
    # an honest client, one command per tick in real time, is never held back
    now = 100.0
    for i in range(1, 600):
        now += 1.0 / C.TICK_RATE
        server._handle_packet(_frame(conn, start + i, [(0, 0.0, 0.0, (0, 0, 0))]), addr, now)
    assert conn.newest_cmd == start + 599
    # running twice as fast as real time only gets through until the burst allowance is spent
    tick = conn.newest_cmd
    for _ in range(600):
        now += 1.0 / C.TICK_RATE
        tick += 2
        server._handle_packet(_frame(conn, tick, [(0, 0.0, 0.0, (0, 0, 0))]), addr, now)
    assert conn.newest_cmd - (start + 599) <= 600 + C.CMD_TICK_BURST


def test_fire_records_stay_bounded():
    server, conn, addr, _ = _server(29803)
    base = conn.newest_cmd
    for i in range(100):
        fires = [(i * 255 + k, 10_000_000 + i * 255 + k) for k in range(255)]      # far in the future
        server._handle_packet(_frame(conn, base + 1 + i, [(0, 0.0, 0.0, (0, 0, 0))], fires), addr, 100.0 + i / 60.0)
    assert len(conn.fire) == 0
    fires = [(5000 + k, base + 1 + k) for k in range(255)]                          # plausible ticks
    server._handle_packet(_frame(conn, base + 300, [(0, 0.0, 0.0, (0, 0, 0))], fires), addr, 110.0)
    assert 0 < len(conn.fire) <= C.FIRE_RECORD_LIMIT


def test_an_unauthenticated_connect_cannot_replace_a_live_session():
    server, conn, addr, request = _server(29804)
    spoof = N.GameClient("x", W.DEFAULT_LOADOUT, np.zeros((2, W.UPGRADE_COUNT), int), np.zeros(W.OPERATOR_COUNT, int))
    spoof.connect("127.0.0.1", 29804)
    spoof.close()
    token = conn.token
    server._handle_packet(spoof._request, addr, conn.last_recv + 0.5)
    assert server.conns[addr].token == token                                         # ignored while it is alive
    server._handle_packet(spoof._request, addr, conn.last_recv + C.RECONNECT_GRACE + 1.0)
    assert server.conns[addr].token != token                                         # a silent one may be replaced
    assert server.inbox.maxlen == C.INBOX_LIMIT
    other = ("10.0.0.6", 41001)
    server._handle_packet(request, other, 300.0)
    count = len(server.conns)
    server._handle_packet(spoof._request, other, 300.0 + C.CONNECT_INTERVAL / 2)    # rate limited per address
    assert len(server.conns) == count


def test_a_flood_waits_instead_of_stalling_one_tick():
    server, conn, addr, request = _server(29805)
    for i in range(C.PACKETS_PER_TICK * 3):
        server.inbox.append((b"\x00" * 12, ("10.9.%d.%d" % (i // 250, i % 250), 5000), 100.0))
    server._drain_inbox()
    assert len(server.inbox) == C.PACKETS_PER_TICK * 2


def test_names_from_the_network_lose_markup():
    assert N.clean_name(b"<scale:x>BO<B>") == "scale:xBOB" and N.clean_name("<>", "PLAYER 3") == "PLAYER 3"
    server = N.GameServer(N.ServerConfig(name="R", map_key="crucible", bot_fill=0, beacon=False, port=29806))
    client = N.GameClient("<scale:x>", W.DEFAULT_LOADOUT, np.zeros((2, W.UPGRADE_COUNT), int),
                          np.zeros(W.OPERATOR_COUNT, int))
    client.connect("127.0.0.1", 29806)
    client.close()
    server._handle_packet(client._request, ("10.0.0.7", 4000), 100.0)
    assert "<" not in server.world.store.names[0] and ">" not in server.world.store.names[0]
    beacon = N.encode_beacon("<scale:400>ARENA", 28016, 1, 8, 0, 0, 0, uid=7, map_crc=0)    # a forged beacon
    info = N.decode_beacon(beacon, "10.0.0.1")
    assert info is not None and info.name == "scale:400ARENA"


def test_zero_and_tiny_time_limits_are_raised():
    assert S.GameWorld("crucible", mode=C.MODE_FFA, time_limit=0.0).time_limit == C.MATCH_TIME_LIMIT
    assert S.GameWorld("crucible", mode=C.MODE_FFA, time_limit=-5.0).time_limit == C.MATCH_TIME_LIMIT
    assert S.GameWorld("crucible", mode=C.MODE_FFA, time_limit=5.0).time_limit == C.MIN_TIME_LIMIT


# ---------------------------------------------------------------------------------------------------------
#  Saves (security review)
# ---------------------------------------------------------------------------------------------------------
def _manager():
    pm = SS.ProfileManager(tempfile.mkdtemp())
    assert pm.load() == "new"
    pm.add_credits(1234.0)
    assert pm.save()
    return pm


def test_concurrent_saves_never_lose_the_main_file():
    pm = _manager()
    missing, failed = [], []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            if not pm.path.exists():
                missing.append(1)

    def writer():
        for _ in range(120):
            if not pm.save():
                failed.append(1)

    watcher = threading.Thread(target=watch)
    watcher.start()
    writers = [threading.Thread(target=writer) for _ in range(2)]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop.set()
    watcher.join()
    assert not missing and not failed
    again = SS.ProfileManager(pm.directory)
    assert again.load() == "ok" and again.profile.credits >= 1234.0


def test_a_missing_main_save_is_restored_from_the_backup():
    pm = _manager()
    assert pm.save()                                            # the first save's copy is now the backup
    pm.path.unlink()
    again = SS.ProfileManager(pm.directory)
    assert again.load() == "restored_backup" and again.profile.credits >= 1234.0
    assert again.profile.integrity_strikes == 0                 # a missing file is not tampering


def test_an_unreadable_save_is_kept_and_never_overwritten():
    pm = _manager()
    before = pm.path.read_bytes()
    again = SS.ProfileManager(pm.directory)
    again._read = lambda path, attempts=3: None                 # every read fails (permissions, disk errors)
    assert again.load() == "io_error"
    assert pm.path.exists() and pm.path.read_bytes() == before and not list(pm.directory.glob("*.tampered-*"))
    assert not again.save() and pm.path.read_bytes() == before


def test_non_finite_and_malformed_numbers_in_a_save_are_sanitised():
    raw = SS.Profile().to_dict()
    raw.update(credits=float("nan"), lifetime_credits=float("inf"), xp=float("nan"), level=float("nan"))
    raw["stats"]["kills"] = float("inf")
    raw["settings"]["sensitivity"] = float("nan")
    raw["weapons"]["vanguard"]["levels"] = [float("nan")] * W.UPGRADE_COUNT
    clean = SS.Profile.from_dict(raw)
    assert clean.credits == 0.0 and clean.lifetime_credits == 0.0 and clean.xp == 0.0 and clean.level == 1
    assert clean.stats["kills"] == 0.0 and np.isfinite(clean.settings["sensitivity"])
    assert clean.weapons["vanguard"]["levels"] == [0] * W.UPGRADE_COUNT

    class Broken(SS.Profile):
        def to_dict(self):
            data = SS.Profile().to_dict()
            data["loadout"] = 5                                 # not iterable: from_dict raises TypeError
            return data

    raised = False
    try:
        SS.decode_save(SS.encode_save(Broken()))
    except SS.SaveTamperError:
        raised = True
    assert raised                                               # a damaged save, not a crash at start-up


# ---------------------------------------------------------------------------------------------------------
#  Match rules (rules review)
# ---------------------------------------------------------------------------------------------------------
from engine import equipment as EQ      # noqa: E402
from engine import maps                 # noqa: E402
from engine import streaks as ST        # noqa: E402
from engine.report import MatchReport   # noqa: E402

FRAG, FIRE, SMOKE, STIM = (EQ.EQUIP_INDEX[key] for key in ("pulse", "thermite", "smoke", "stim"))
HATCHET = EQ.EQUIP_INDEX[C.ESCALATION_LETHAL]


def _world(count=2, mode=C.MODE_FFA, key="crucible", **extra):
    game = S.GameWorld(key, mode=mode, seed=5, **extra)
    rows = [game.add_player("p%d" % i) for i in range(count)]
    game.store.protect_until[:] = 0
    game.drain_events()
    return game, rows


def test_a_player_who_leaves_takes_what_is_in_play_with_them():
    game, (a, b) = _world()
    s = game.store
    near_b = s.move.pos[b] + np.array([0.0, 0.5, 0.0])
    slot = game.grenades.spawn(near_b, np.zeros(3), 5, a, equip=FRAG)                # about to blow up next to b
    fire = game.zones.spawn(s.move.pos[b], 3.0, game.tick, game.tick + 600, a, FIRE)
    smoke = game.zones.spawn(s.move.pos[b], 3.0, game.tick, game.tick + 600, a, SMOKE)
    game.strikes.append([game.tick + 30, s.move.pos[b].copy(), a])
    s.last_attacker[b] = a
    game.remove_player(a)
    assert not game.grenades.active[slot] and not game.zones.active[fire] and game.zones.active[smoke]
    assert game.strikes == [] and s.last_attacker[b] == -1
    c = game.add_player("newcomer")                                                 # takes the free slot ...
    assert c == a
    s.protect_until[:] = 0
    events = []
    for _ in range(2 * C.TICK_RATE):
        game.end_tick()
        events += game.drain_events()
    assert s.health[b] == s.max_health[b] and s.kills[c] == 0                       # ... and inherits nothing
    assert not any(e[0] in (S.EV_HIT, S.EV_KILL, S.EV_REWARD) for e in events)


def test_an_arena_without_spawns_for_one_squad_still_spawns_everybody():
    yard = {"name": "ONE SIDED", "mirror": False,
            "center": [(0.0, -1.0, 0.0, 28.0, 1.0, 20.0, "floor", -1), (0.0, 0.0, 10.5, 30.0, 4.0, 1.0, "brick", -1),
                       (0.0, 0.0, -10.5, 30.0, 4.0, 1.0, "brick", -1), (-14.5, 0.0, 0.0, 1.0, 4.0, 22.0, "concrete", -1),
                       (14.5, 0.0, 0.0, 1.0, 4.0, 22.0, "concrete", -1)],
            "half": [(-12.0, 0.0, -4.0, 1.0, 1.0, 1.0, "spawn", 0), (-12.0, 0.0, 4.0, 1.0, 1.0, 1.0, "spawn", 0)]}
    maps.register_map("one_sided", yard)
    try:
        game = S.GameWorld("one_sided", mode=C.MODE_TDM, seed=2)
        rows = [game.add_player("p%d" % team, team=team) for team in (0, 1)]       # used to raise in np.argmax
        assert game.store.alive[rows].all()
    finally:
        maps.unregister_map("one_sided")


def _end_match(game):
    game.time_left = C.TICK_DT * 0.5
    game.end_tick()
    assert game.phase == C.PHASE_ENDED
    game.drain_events()


def test_nothing_scores_between_matches():
    game, (a, b) = _world()
    s = game.store
    s.last_attacker[b] = a
    s.last_hostile_tick[b] = game.tick
    _end_match(game)
    kills, deaths = s.kills.copy(), s.deaths.copy()
    s.move.pos[b] = (0.0, game.loaded.data.kill_y - 2.0, 0.0)                     # pushed off the arena
    game.end_tick()
    events = game.drain_events()
    assert not any(e[0] == S.EV_KILL for e in events) and any(e[0] == S.EV_SPAWN and e[1] == b for e in events)
    assert s.alive[b] and np.array_equal(s.kills, kills) and np.array_equal(s.deaths, deaths)
    velocity = s.move.vel[b].copy()
    game.explode(s.hitboxes.bound_center[b] + np.array([1.0, 0.0, 0.0]), a, 7, FRAG)
    assert np.array_equal(s.move.vel[b], velocity)                                  # no knock-back either


def test_a_fall_is_credited_to_hostile_damage_only():
    game, (a, b) = _world()
    s = game.store
    game.apply_damage(b, a, 10.0, C.ZONE_TORSO, 0, s.move.pos[b])
    for _ in range(6 * C.TICK_RATE):                                                # long ago ...
        game.end_tick()
    game.apply_damage(b, b, 5.0, C.ZONE_TORSO, C.EQUIP_KILL_BASE + FRAG, s.move.pos[b], explosive=True)
    game.drain_events()
    s.move.pos[b] = (0.0, game.loaded.data.kill_y - 2.0, 0.0)                     # ... then b blows himself off
    game.end_tick()
    kill = [e for e in game.drain_events() if e[0] == S.EV_KILL]
    assert kill and kill[0][1] == S.NO_PLAYER and kill[0][4] & S.KILL_SUICIDE and s.kills[a] == 0


def test_posthumous_kills_neither_extend_the_streak_nor_earn_rewards():
    game, rows = _world(4)
    a, others = rows[0], rows[1:]
    s = game.store
    game.kill_player(a, others[0], 0, 0)                                            # a dies, a fire keeps burning
    game.drain_events()
    for victim in others:
        game.apply_damage(victim, a, 500.0, C.ZONE_TORSO, C.EQUIP_KILL_BASE + FIRE, s.move.pos[victim])
    events = game.drain_events()
    assert s.kills[a] == 3 and s.streak[a] == 0 and s.streak_ready[a] == 0          # the kills count, nothing else
    assert not any(e[0] == S.EV_STREAK for e in events)
    rewards = [e[2] for e in events if e[0] == S.EV_REWARD and e[4] == S.REWARD_KILL]
    assert rewards == [100, 100, 100]


def test_the_kill_bonus_never_goes_negative():
    game, (a, b) = _world()
    game.kill_player(b, a, C.STREAK_KILL_BASE + ST.LANCE, S.KILL_EXPLOSIVE)          # a reward kill on streak 0
    rewards = [e for e in game.drain_events() if e[0] == S.EV_REWARD and e[4] == S.REWARD_KILL]
    assert rewards and rewards[0][2] == 100                                         # used to pay 75


def test_ties_are_draws_in_free_for_all_and_escalation():
    game, (a, b) = _world()
    game.store.kills[[a, b]] = 3
    assert game.leading_side() == S.NO_PLAYER                                       # used to pick the lower slot
    game.store.kills[b] = 4
    assert game.leading_side() == b
    game, (a, b) = _world(mode=C.MODE_ESC)
    game.ladder[[a, b]] = 7
    game.store.kills[[a, b]] = 9
    assert game.leading_side() == S.NO_PLAYER
    game.store.kills[a] = 10
    assert game.leading_side() == a


def test_escalation_is_stock_for_everybody():
    assert not S.mode_upgrades(C.MODE_ESC, True) and S.mode_upgrades(C.MODE_TDM, True)
    game = S.GameWorld("crucible", mode=C.MODE_ESC, seed=5)
    row = game.add_player("veteran", operator=np.full(W.OPERATOR_COUNT, 5))
    s = game.store
    assert not s.operator[row].any() and s.blast_resist[row] == 1.0
    assert s.regen_delay[row] == int(round(C.HEALTH_REGEN_DELAY * C.TICK_RATE))


def test_a_hatchet_kill_on_the_last_step_does_not_win():
    game, (a, b) = _world(mode=C.MODE_ESC)
    last = W.ESCALATION_STEPS - 1
    game.set_ladder_step(a, last)
    game.set_ladder_step(b, 4)
    game.kill_player(b, a, C.EQUIP_KILL_BASE + HATCHET, 0)
    events = game.drain_events()
    game.end_tick()
    assert game.ladder[a] == last and game.ladder[b] == 4 - C.ESCALATION_STEAL and game.phase == C.PHASE_LIVE
    assert not any(e[0] == S.EV_LADDER and e[1] == a for e in events)               # no second FINAL WEAPON banner
    game.spawn_player(b)
    game.kill_player(b, a, int(game.store.weapons.weapon[a, 0]), 0)                 # the last gun still wins
    game.end_tick()
    assert game.phase == C.PHASE_ENDED and game.winner == a


def test_equipment_and_rewards_end_spawn_protection():
    for act in ("lethal", "stim", "reward"):
        game, (a, b) = _world()
        s = game.store
        s.equip[a] = (FRAG, STIM)
        s.protect_until[a] = game.tick + 600
        if act == "lethal":
            game.throw_equipment(a, FRAG)
        elif act == "stim":
            game.use_tactical(a)
        else:
            s.streak_ready[a] = 1 << ST.AEGIS
            assert game.use_streak(a)
        assert s.protect_until[a] == 0, act


def test_there_is_room_for_every_players_fire_and_smoke():
    assert C.ZONE_CAPACITY >= 2 * C.MAX_PLAYERS and S.ZoneStore().active.size == C.ZONE_CAPACITY


# ---------------------------------------------------------------------------------------------------------
#  What the client is told (client review)
# ---------------------------------------------------------------------------------------------------------
def test_hits_say_whether_a_bullet_landed():
    game, (a, b) = _world()
    s = game.store
    game.apply_damage(b, a, 10.0, C.ZONE_TORSO, int(s.weapons.weapon[a, 0]), s.move.pos[b])
    game.apply_damage(b, a, 10.0, C.ZONE_TORSO, C.EQUIP_KILL_BASE + FRAG, s.move.pos[b], explosive=True)
    hits = [e for e in game.drain_events() if e[0] == S.EV_HIT]
    assert [hit[7] for hit in hits] == [True, False]
    report = MatchReport()
    report.shot()
    report.hit(10.0, counted=True)
    for _ in range(4):
        report.hit(25.0, headshot=True, counted=False)                              # splash, fire, the lance ...
    assert report.hits == 1 and report.headshots == 0 and report.damage == 110.0 and report.accuracy == 1.0


def test_first_strike_is_the_servers_call():
    game, (a, b, c) = _world(3)
    game.kill_player(b, b, 0, 0)                                                   # a fall is no elimination
    game.kill_player(c, a, 0, 0)
    game.kill_player(b, a, 0, 0) if game.store.alive[b] else None
    flags = [e[4] for e in game.drain_events() if e[0] == S.EV_KILL]
    assert flags[0] & S.KILL_SUICIDE and flags[1] & S.KILL_FIRST and not any(f & S.KILL_FIRST for f in flags[2:])
    game.restart_match()
    game.drain_events()
    game.kill_player(a, b, 0, 0)
    assert [e[4] for e in game.drain_events() if e[0] == S.EV_KILL] == [S.KILL_FIRST]
    late = MatchReport()                                                            # joined after the first kill
    assert late.kill(0, 1, 0, 5.0, first=False) == [] and late.kill(0, 2, 0, 60.0, first=False) == []


def test_the_roster_goes_out_at_once_when_the_match_ends():
    server, conn, addr, _ = _server(29807)
    sent = []
    server._send_raw = lambda datagram, to: sent.append(datagram)
    conn.last_recv = time.perf_counter()                                             # not a silent client
    server.tick_once()
    sent.clear()
    with server.lock:
        server.world.time_left = C.TICK_DT * 0.5
    conn.last_recv = time.perf_counter()
    server.tick_once()                                                               # long before ROSTER_INTERVAL
    rosters = [d for d in sent if N.HEADER.unpack_from(d, 0)[1] == N.PKT_ROSTER]
    assert len(rosters) == 1 and server.world.phase == C.PHASE_ENDED
    client = N.GameClient("r", W.DEFAULT_LOADOUT, np.zeros((2, W.UPGRADE_COUNT), int), np.zeros(W.OPERATOR_COUNT, int))
    client._on_roster(rosters[0], N.HEADER.size)
    assert client.roster_phase == C.PHASE_ENDED and conn.pid in client.roster


def test_drop_in_arena_text_cannot_carry_markup_and_broken_files_never_crash():
    clean = maps._normalise_definition("x", {"name": "<scale:99>EVIL", "description": "a <b>bold</b> claim" * 20,
                                             "center": [(0, -1, 0, 4, 1, 4, "floor", -1)]})
    assert clean["name"] == "scale:99EVIL" and "<" not in clean["description"] and len(clean["description"]) <= 120
    original = C.MAPS_DIR
    with tempfile.TemporaryDirectory() as folder:
        C.MAPS_DIR = __import__("pathlib").Path(folder)
        try:
            (C.MAPS_DIR / "numbers.json").write_text('{"center": [5]}')               # TypeError inside
            (C.MAPS_DIR / "empty.json").write_text('{"center": [[0, -1, 0, 4, 1, 4, "floor", -1]]}')   # no spawn
            keys = maps.reload_registry()
            assert "numbers" not in keys and "empty" not in keys and len(maps.REGISTRY_ERRORS) == 2
        finally:
            C.MAPS_DIR = original
            maps.reload_registry()


# ---------------------------------------------------------------------------------------------------------
#  Netcode and determinism (netcode review)
# ---------------------------------------------------------------------------------------------------------
from contextlib import nullcontext       # noqa: E402

from engine import mathx as M           # noqa: E402


def _client():
    return N.GameClient("r", W.DEFAULT_LOADOUT, np.zeros((2, W.UPGRADE_COUNT), int), np.zeros(W.OPERATOR_COUNT, int))


def test_a_client_that_stalled_for_minutes_skips_the_events_it_lost():
    server, conn, _, _ = _server(29808)
    blob = N.encode_event((S.EV_UPLINK, 0, 0, S.UPLINK_CAPTURED))
    for _ in range(600):
        server._queue_reliable(conn, blob)
    assert len(conn.events) == 512 and server.stats["events_dropped"] >= 88      # memory stays bounded ...
    client = _client()
    client.event_seq = 1                                                           # ... it still expects event 2
    first = conn.events[0][0]
    data = N._U32.pack(first) + blob * 3
    assert client._take_reliable(data, 0, 3) == len(data)
    assert client.event_seq == first + 2 and len(client.events) == 3 and client.stats["event_gaps"] == 1
    client._take_reliable(N._U32.pack(first + 1) + blob * 2, 0, 2)                # an ordinary resend changes nothing
    assert client.event_seq == first + 2 and len(client.events) == 3


def test_the_snapshot_names_the_arena_so_a_missed_map_change_cannot_strand_a_client():
    config = N.ServerConfig(name="MAP", map_key="crucible", bot_fill=0, beacon=False, port=28199, seed=4)
    server = N.host_local_game(config)
    client = _client()
    try:
        client.connect("127.0.0.1", server.port)
        deadline = time.time() + 4.0
        while time.time() < deadline and client.state != N.CONNECTED:
            client.update()
            time.sleep(0.01)
        with server.lock:
            server.world.change_map((server.world.map_index + 1) % maps.MAP_COUNT)    # no EV_MATCH announces it
            target = server.world.map_index
        deadline = time.time() + 2.0
        while time.time() < deadline and client.map_index != target:
            client.update()
            client.send_command(0, 0.0, 0.0)
            time.sleep(C.TICK_DT)
        assert client.state == N.CONNECTED and client.map_index == target and client.map_changed
        assert client.loaded.data.checksum == server.world.loaded.data.checksum
    finally:
        client.disconnect()
        server.stop()


def test_the_lance_locks_on_to_what_the_player_saw():
    game = S.GameWorld("datacore", mode=C.MODE_TDM, seed=3)
    s = game.store
    a, b = game.add_player("aim", team=0), game.add_player("strafe", team=1)
    data = game.loaded.data
    best = None                                                     # a long open sight line from a squad spawn
    for spot in data.spawn_pos[data.spawn_team == 0]:
        for yaw in np.linspace(-np.pi, np.pi, 72, endpoint=False):
            eye = spot + np.array([0.0, C.EYE_HEIGHT_STAND, 0.0])
            ahead = M.forward_vector(yaw, 0.0)
            t, _, _ = game.world.raycast(eye, ahead, 200.0, C.LAYER_BULLET)
            ground = spot + ahead * 12.0
            drop, floor, _ = game.world.raycast(ground + np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]), 3.0,
                                                C.LAYER_PLAYER)
            if t[0] > 26.0 and floor[0] >= 0 and (best is None or t[0] > best[0]):
                best = (float(t[0]), spot.copy(), float(yaw), ground + np.array([0.0, 1.0 - float(drop[0]), 0.0]))
    _, spot, yaw, ground = best
    side = M.forward_vector(yaw + np.pi / 2.0, 0.0)
    lag = N.LagCompensator()
    for k in range(30):                                             # b strafes at 6.5 m/s
        s.move.pos[a], s.yaw[a] = spot, yaw
        s.move.pos[b] = ground + side * 6.5 * (k - 29) * C.TICK_DT
        game.tick += 1
        s.refresh_hitboxes()
        lag.record(game.tick, s)
    view_tick = game.tick - 8                                       # interpolation delay plus latency
    seen = lag.pos[view_tick % lag.history, b]
    aim_yaw, aim_pitch = (float(v) for v in M.yaw_pitch_from_direction(seen + np.array([0.0, 1.2, 0.0])
                                                                         - s.eye_position(a)[0]))
    s.protect_until[:] = 0
    s.streak_ready[a] = 1 << ST.LANCE
    game.drain_events()
    game.step_commands(np.array([a]), C.BTN_STREAK, aim_yaw, aim_pitch, s.cmd_tick[a] + 1,
                       rewind=lambda row: lag.rewind(s, view_tick * C.TICK_MS, row))
    used = [e for e in game.drain_events() if e[0] == S.EV_STREAK and e[3] == S.STREAK_USED]
    assert used and np.linalg.norm(np.asarray(used[0][4]) - s.move.pos[b]) < 1.0      # under b's feet, not the wall


def test_the_owner_state_travels_at_full_precision():
    game, (a, _) = _world()
    s = game.store
    s.move.pos[a] = (1.0 / 3.0, 2.0 / 7.0, -5.0 / 9.0)
    s.move.vel[a] = (0.1, -7.000000001, 1.0 / 3.0)
    s.move.slide_t[a], s.move.air_t[a], s.weapons.bloom[a] = 0.123456789123, 1.0 / 7.0, 1.0 / 3.0
    copy = S.PlayerStore(1)
    copy.active[0] = True
    copy.unpack_owner(0, s.pack_owner(a))
    assert np.array_equal(copy.move.pos[0], s.move.pos[a]) and np.array_equal(copy.move.vel[0], s.move.vel[a])
    assert copy.move.slide_t[0] == s.move.slide_t[a] and copy.move.air_t[0] == s.move.air_t[a]
    assert copy.weapons.bloom[0] == s.weapons.bloom[a]                  # replays after a correction are bit-exact


def test_a_full_server_snapshot_still_fits_one_datagram():
    server, conn, _, _ = _server(29809)
    while server.world.add_player("filler", is_bot=True) >= 0:
        pass
    sent = []
    server._send_raw = lambda datagram, to: sent.append(datagram)
    blob = N.encode_event((S.EV_HIT, 1, conn.pid, 10.0, 0, False, (0.0, 0.0, 0.0), True))
    for _ in range(60):
        server._queue_reliable(conn, blob)
    server._send_snapshots(time.perf_counter())
    assert sent and all(len(datagram) <= C.MAX_DATAGRAM for datagram in sent)


def test_a_joiner_learns_about_fire_smoke_and_lances_already_in_play():
    server = N.GameServer(N.ServerConfig(name="R", map_key="crucible", bot_fill=0, beacon=False, port=29810))
    world = server.world
    owner = world.add_player("thrower")
    spot = world.store.move.pos[owner].copy()
    world.zones.spawn(spot, 3.0, world.tick, world.tick + 300, owner, FIRE)
    world.zones.spawn(spot + 4.0, 5.0, world.tick, world.tick + 480, owner, SMOKE)
    world.strikes.append([world.tick + 60, spot + np.array([2.0, 0.0, 0.0]), owner])
    client = _client()
    client.connect("127.0.0.1", 29810)
    client.close()
    server._handle_packet(client._request, ("10.0.0.8", 4100), 100.0)
    conn = server.conns[("10.0.0.8", 4100)]
    replay = [N.decode_event(blob, 0)[0] for _, blob in conn.events]
    zones = sorted((e[2], e[3]) for e in replay if e[0] == S.EV_ZONE)
    assert zones == sorted([(FIRE, 300), (SMOKE, 480)])
    assert any(e[0] == S.EV_STREAK and e[2] == ST.LANCE and e[3] == S.STREAK_USED for e in replay)


def test_the_view_time_of_a_shot_unwraps_after_49_days():
    server, conn, _, _ = _server(29811)
    store = server.world.store
    seen = []
    server.lag.rewind = lambda st, view_ms, row: (seen.append(view_ms), nullcontext())[1]
    tick = (1 << 32) // C.TICK_MS + 12345                              # more than 2^32 ms of uptime
    server.lag.newest = tick
    true_view = int(tick * C.TICK_MS) - 140
    store.cmd_tick[conn.pid] = tick
    conn.fire[tick] = (1, true_view & 0xFFFFFFFF)                        # the client sends 32 bits
    server._rewind_for(conn.pid)
    assert seen == [float(true_view)]


# ---------------------------------------------------------------------------------------------------------
#  Stability, tests and docs (performance review)
# ---------------------------------------------------------------------------------------------------------
def test_command_line_switches_last_one_run():
    pm = _manager()
    stored = dict(pm.profile.settings)
    pm.override_setting("quality", 0)
    pm.override_setting("fps_limit", 30)
    assert pm.profile.settings["quality"] == 0 and pm.profile.settings["fps_limit"] == 30 and pm.save()
    again = SS.ProfileManager(pm.directory)
    assert again.load() == "ok"
    assert again.profile.settings["quality"] == stored["quality"] and again.profile.settings["fps_limit"] == stored["fps_limit"]
    pm.set_setting("quality", 1)                                                  # chosen in the menu: kept
    assert pm.save()
    again = SS.ProfileManager(pm.directory)
    again.load()
    assert again.profile.settings["quality"] == 1 and again.profile.settings["fps_limit"] == stored["fps_limit"]


def test_closing_a_client_stops_its_latency_simulator():
    client = N.GameClient("lag", W.DEFAULT_LOADOUT, np.zeros((2, W.UPGRADE_COUNT), int), np.zeros(W.OPERATOR_COUNT, int),
                          fake_lag_ms=30.0)
    client.connect("127.0.0.1", 29812)
    old = client.conditioner
    assert old.is_alive()
    client.close()
    old.join(timeout=2.0)
    assert not old.is_alive() and not client.conditioner.is_alive() and client.conditioner.latency == old.latency


def test_shots_queued_during_a_stall_are_not_replayed():
    config = N.ServerConfig(name="STALL", map_key="crucible", mode=C.MODE_FFA, bot_fill=6, max_players=8, beacon=False,
                            port=28196, seed=4)
    server = N.host_local_game(config)
    client = _client()
    try:
        client.connect("127.0.0.1", server.port)
        deadline = time.time() + 4.0
        while time.time() < deadline and client.state != N.CONNECTED:
            client.update()
            time.sleep(0.01)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            client.update()
            client.send_command(0, 0.0, 0.0)
            time.sleep(C.TICK_DT)
        time.sleep(4.0)                                                           # the game stalls; bots keep firing
        fresh = 0                                                                 # shots that may still play
        cutoff = time.perf_counter() - C.STALE_EVENT_AGE
        for data, _, received in list(client.inbox):
            if N.HEADER.unpack_from(data, 0)[1] == N.PKT_SNAPSHOT and received >= cutoff:
                fresh += N.SNAP_HEADER.unpack_from(data, N.HEADER.size)[-1]
        events = client.update()
        shots = sum(1 for e in events if e[0] == S.EV_SHOT)
        assert client.state == N.CONNECTED and client.stats["stale_transients"] > 0
        assert shots <= fresh                                                     # no burst of old tracers and sounds
    finally:
        client.disconnect()
        server.stop()
