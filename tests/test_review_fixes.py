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
