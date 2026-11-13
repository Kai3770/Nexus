#!/usr/bin/env python3
"""
NEXUS-3D-EXTREME :: dedicated_server.py
=======================================
Headless authoritative server.  Needs only Python and NumPy (no window, no
Ursina), so it runs on any machine of the LAN:

    python dedicated_server.py --map datacore --bots 8 --name "FLOOR 3 ARENA"
    python dedicated_server.py --list-maps

Clients find it automatically through the LAN lobby (beacon on UDP 28015).
Arenas come from ``content/maps_registry.py`` and ``content/maps/*.json``;
clients need the same arena files (the handshake compares checksums).
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

MODES = {"tdm": 0, "ffa": 1}
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args(argv=None) -> argparse.Namespace:
    from engine import maps

    parser = argparse.ArgumentParser(description="NEXUS-3D-EXTREME dedicated server")
    parser.add_argument("--name", default="NEXUS DEDICATED")
    parser.add_argument("--port", type=int, default=0, help="gameplay UDP port (default 28016)")
    parser.add_argument("--map", default=maps.MAP_KEYS[0], choices=maps.MAP_KEYS,
                        help="arena key from content/maps_registry.py or content/maps/*.json")
    parser.add_argument("--mode", default="tdm", choices=sorted(MODES))
    parser.add_argument("--max-players", type=int, default=12)
    parser.add_argument("--bots", type=int, default=8, help="the server tops the match up to this many players with bots")
    parser.add_argument("--difficulty", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--score-limit", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--no-upgrades", action="store_true", help="ignore player upgrade levels (even playing field)")
    parser.add_argument("--no-rotate", action="store_true", help="stay on one arena")
    parser.add_argument("--no-beacon", action="store_true", help="do not advertise on the LAN")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (0 = run until Ctrl+C)")
    parser.add_argument("--status-every", type=float, default=10.0)
    parser.add_argument("--list-maps", action="store_true", help="print the arena registry and exit")
    return parser.parse_args(argv)


def list_maps() -> int:
    """Print every registered arena with its checksum and validation result."""
    from engine import maps

    for error in maps.REGISTRY_ERRORS:
        print("skipped %s" % error)
    for key in maps.MAP_KEYS:
        loaded = maps.load_map(key)
        data, problems = loaded.data, maps.validate_map(loaded)
        print("%-12s %-12s crc %08x  %3d boxes  %2d spawns  %3d nav nodes  %s" % (
            key, data.name, data.checksum, int(data.box_visible.sum()), data.spawn_pos.shape[0], loaded.nav.count,
            "ok" if not problems else "; ".join(problems)))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_maps:
        return list_maps()
    from engine import config as C
    from engine import network_lan as N

    config = N.ServerConfig(name=args.name, port=args.port or C.GAME_PORT, map_key=args.map, mode=MODES[args.mode],
                            max_players=min(max(args.max_players, 1), C.MAX_PLAYERS),
                            bot_fill=min(max(args.bots, 0), C.MAX_PLAYERS), bot_difficulty=args.difficulty,
                            score_limit=args.score_limit, time_limit=args.time_limit,
                            upgrades_enabled=not args.no_upgrades, rotate_maps=not args.no_rotate,
                            beacon=not args.no_beacon)
    server = N.GameServer(config)
    port = server.open()
    print("%s dedicated server '%s' listening on UDP %d (%d Hz), beacon %s" % (
        C.GAME_TITLE, config.name, port, C.TICK_RATE, "on UDP %d" % C.DISCOVERY_PORT if config.beacon else "off"))

    def halt(*_):
        server.running = False

    signal.signal(signal.SIGINT, halt)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, halt)

    def report() -> None:
        started = time.time()
        while server.running:
            time.sleep(0.25)
            elapsed = time.time() - started
            if args.duration and elapsed >= args.duration:
                server.running = False
                break
            if args.status_every > 0 and int(elapsed / args.status_every) != int((elapsed - 0.25) / args.status_every):
                info = server.describe()
                print("[%6.0fs] %s %s | humans %d bots %d | score %s | %d:%02d left | tick %.2f ms (max %.1f) | rewinds %d" % (
                    elapsed, info["map"], info["mode"], info["humans"], info["bots"], info["scores"],
                    int(info["time_left"]) // 60, int(info["time_left"]) % 60, info["mean_tick_ms"], info["max_tick_ms"],
                    info["rewinds"]))

    threading.Thread(target=report, name="ServerStatus", daemon=True).start()
    try:
        server.serve_forever()
    finally:
        server.stop()
        info = server.describe()
        print("server stopped after %d ticks; mean tick %.2f ms" % (info["ticks"], info["mean_tick_ms"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
