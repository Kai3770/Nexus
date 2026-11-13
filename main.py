#!/usr/bin/env python3
"""
NEXUS-3D-EXTREME :: main.py
===========================
Game client entry point.

    python main.py                       open the main menu
    python main.py --solo crucible       jump straight into a solo match against bots
    python main.py --host datacore       host a LAN match (advertised by beacon) and join it
    python main.py --connect 10.0.0.7    join a server directly (the LAN lobby needs no address)
    python main.py --quality 1 --fps-limit 60    override the saved graphics settings for this run

Arenas are read from ``content/maps_registry.py`` and ``content/maps/*.json``
(``python dedicated_server.py --list-maps`` prints them).

Developer / test switches: ``--autoplay`` lets a scripted pilot drive the
player, ``--frames N`` quits after N frames (and prints frame statistics) and
``--screenshots DIR`` stores a PNG every ``--shot-every`` frames.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

MODES = {"tdm": 0, "ffa": 1}
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args(argv=None) -> argparse.Namespace:
    early = argparse.ArgumentParser(add_help=False)       # the save directory must be known before the engine loads
    early.add_argument("--save-dir")
    known, _ = early.parse_known_args(argv)
    if known.save_dir:
        os.environ["NEXUS_SAVE_DIR"] = os.path.abspath(known.save_dir)

    from engine import maps

    parser = argparse.ArgumentParser(description="NEXUS-3D-EXTREME client")
    parser.add_argument("--solo", metavar="MAP", choices=maps.MAP_KEYS,
                        help="start a solo match on MAP (%s)" % ", ".join(maps.MAP_KEYS))
    parser.add_argument("--host", metavar="MAP", choices=maps.MAP_KEYS, help="host a LAN match on MAP and join it")
    parser.add_argument("--connect", metavar="HOST[:PORT]", help="join a server by address")
    parser.add_argument("--mode", choices=sorted(MODES), default="tdm")
    parser.add_argument("--bots", type=int, default=7)
    parser.add_argument("--difficulty", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--size", default="1600x900", help="window size, e.g. 1920x1080")
    parser.add_argument("--quality", type=int, choices=(0, 1, 2, 3), help="graphics preset: 0 low ... 3 ultra")
    parser.add_argument("--fps-limit", type=int, help="frame limiter in frames per second (0 = uncapped)")
    parser.add_argument("--vsync", action="store_true", help="synchronise with the display (default: off, see README)")
    parser.add_argument("--no-vsync", action="store_true", help="force vsync off even if the profile enables it")
    parser.add_argument("--no-dynamic-resolution", action="store_true", help="keep the 3D render scale fixed")
    parser.add_argument("--save-dir", help="directory of the profile save (default: ./saves)")
    parser.add_argument("--autoplay", action="store_true", help="let the built-in pilot play (demo / testing)")
    parser.add_argument("--frames", type=int, default=0, help="quit after this many frames (0 = never)")
    parser.add_argument("--screenshots", metavar="DIR", help="save screenshots into DIR")
    parser.add_argument("--shot-every", type=int, default=120)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from engine import config as C
    from engine import secure_save as SS

    C.ensure_directories()
    sys.setswitchinterval(C.THREAD_SWITCH_INTERVAL)       # the listen server thread shares the interpreter with rendering
    profiles = SS.ProfileManager()
    status = profiles.load()
    print("profile: %s (%s)  credits %s  sponsor income %s/s" % (
        profiles.profile.name, status, SS.format_number(profiles.profile.credits),
        SS.format_number(profiles.income_per_second())))

    try:
        width, height = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        width, height = 1600, 900
    settings = profiles.profile.settings                     # command line switches override the profile for this run
    if args.quality is not None:
        settings["quality"] = args.quality
    if args.fps_limit is not None:
        settings["fps_limit"] = max(args.fps_limit, 0)
    if args.no_dynamic_resolution:
        settings["dynamic_resolution"] = False
    vsync = (bool(settings["vsync"]) or args.vsync) and not args.no_vsync

    from ursina import Entity, Ursina, application, window

    app = Ursina(title=C.GAME_TITLE, development_mode=False, fullscreen=args.fullscreen, borderless=args.fullscreen,
                 size=(width, height), vsync=vsync)
    window.fps_counter.enabled = False
    window.exit_button.enabled = False

    from engine.game import NexusGame

    game = NexusGame(profiles, autoplay=args.autoplay)
    game.apply_graphics()
    sponsor = SS.CorporateSponsorThread(profiles)
    sponsor.start()
    counter = {"frames": 0, "logic": 0.0, "started": time.perf_counter(), "scale": 0.0}
    if args.screenshots:
        os.makedirs(args.screenshots, exist_ok=True)

    def frame() -> None:
        started = time.perf_counter()
        game.update()
        counter["logic"] += time.perf_counter() - started
        counter["frames"] += 1
        counter["scale"] += game.scene.post.render_scale
        if args.screenshots and counter["frames"] % max(args.shot_every, 1) == 0:
            name = os.path.join(args.screenshots, "frame_%05d.png" % counter["frames"])
            application.base.screenshot(namePrefix=name, defaultFilename=False)
        if args.frames and counter["frames"] >= args.frames:
            frames = counter["frames"]
            wall = time.perf_counter() - counter["started"]
            print("frames %d   %.1f fps (%.2f ms per frame)   client logic %.2f ms   mean render scale %.0f%%   quality %s" % (
                frames, frames / wall, wall / frames * 1000.0, counter["logic"] / frames * 1000.0,
                counter["scale"] / frames * 100.0, C.QUALITY_NAMES[game.scene.quality]))
            game.quit()

    driver = Entity(name="nexus_driver", ignore_paused=True)
    driver.update = frame
    driver.input = game.input

    if args.solo or args.host:
        game.start_match(bool(args.host), args.host or args.solo, MODES[args.mode], args.bots, args.difficulty)
    elif args.connect:
        host, _, port = args.connect.partition(":")
        game.join_match(host, int(port) if port else C.GAME_PORT)

    try:
        app.run(info=False)
    finally:
        sponsor.stop(save=True)
        game._shutdown_session()
        if game.browser is not None:
            game.browser.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
