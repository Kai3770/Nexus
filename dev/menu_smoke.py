"""Opens every menu screen once and saves a screenshot of each (run under xvfb-run when headless)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.environ["NEXUS_SAVE_DIR"] = tempfile.mkdtemp()

from engine import secure_save as SS                                         # noqa: E402

profiles = SS.ProfileManager()
profiles.load()
profiles.add_credits(48000.0)
profiles.add_xp(9000.0)
profiles.buy_sponsor(0, 12)
profiles.buy_sponsor(1, 4)
profiles.buy_weapon_upgrade("vanguard", 0)

from ursina import Entity, Ursina, application, window                       # noqa: E402

app = Ursina(title="NEXUS", development_mode=False, fullscreen=False, borderless=False, size=(1280, 720), vsync=False)
window.fps_counter.enabled = False
window.exit_button.enabled = False

from engine.game import NexusGame                                            # noqa: E402

game = NexusGame(profiles)
sponsor = SS.CorporateSponsorThread(profiles, interval=0.05)
sponsor.start()
order = ["main", "setup", "lobby", "armory", "equipment", "sponsors", "settings", "pause"]
state = {"frame": 0}


def frame():
    game.update()
    state["frame"] += 1
    index, phase = divmod(state["frame"], 12)
    if index >= len(order):
        sponsor.stop(save=False)
        game.quit()
        return
    if phase == 1:
        game.menus.show(order[index])
        if order[index] == "armory":
            game.menus.screens["armory"].select("breacher")
        if order[index] == "equipment":
            game.menus.screens["equipment"].select("cling")
    if phase == 10:
        application.base.screenshot(namePrefix=os.path.join(OUT, "menu_%s.png" % order[index]), defaultFilename=False)


driver = Entity(ignore_paused=True)
driver.update = frame
app.run(info=False)
