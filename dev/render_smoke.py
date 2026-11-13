"""
Renders a scripted scene with every renderer feature and saves screenshots.

    xvfb-run -a -s "-screen 0 1280x720x24" python dev/render_smoke.py OUT_DIR [MAP] [QUALITY 0-3]

Writes ``<map>_<frame>.png`` (first-person shots with effects), ``<map>_board.png``
(scoreboard) and ``<map>_overview.png`` (bird's-eye view for shadows and the skyline).
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
MAP = sys.argv[2] if len(sys.argv) > 2 else "datacore"
QUALITY = int(sys.argv[3]) if len(sys.argv) > 3 else 2

from ursina import Ursina, application, camera, window                      # noqa: E402

app = Ursina(development_mode=False, fullscreen=False, borderless=False, size=(1280, 720), vsync=False, title="NEXUS")
window.fps_counter.enabled = False
window.exit_button.enabled = False

from engine import config as C, maps, mathx as M, network_lan as N, renderer as R, weapons as W   # noqa: E402
from engine.controller_3d import CameraRig                                                        # noqa: E402

started = time.perf_counter()
scene_renderer = R.SceneRenderer(QUALITY, dynamic_resolution=False, fps_limit=0)
hud = R.HUD(scene_renderer.world.fx_shader)
loaded = maps.load_map(MAP)
scene_renderer.load_map(loaded.data)
print("load %.2fs" % (time.perf_counter() - started), scene_renderer.describe(), scene_renderer.world.stats)

view = N.EntityView(C.MAX_PLAYERS)
spots = {"datacore": ((-21.0, 0.0, 1.5), 90.0), "skybridge": ((-24.0, 0.0, 2.0), 90.0), "crucible": ((-15.0, 0.0, -6.0), 60.0),
         "terminal": ((-24.0, 0.0, 3.0), 80.0), "atrium": ((-19.0, 0.0, 2.0), 85.0)}
feet, yaw_deg = spots.get(MAP, (tuple(loaded.data.spawn_pos[0]), math.degrees(loaded.data.spawn_yaw[0])))
feet = np.array(feet)
yaw = math.radians(yaw_deg)
fwd, right = M.forward_vector(yaw), M.right_vector(yaw)
for i, (ahead, side, team, crouch) in enumerate(((7, -2.0, 1, 0.0), (9, 1.2, 1, 1.0), (12, 3.5, 0, 0.0), (15, -3.5, 1, 0.0))):
    view.pos[i] = feet + fwd * ahead + right * side
    view.vel[i] = right * (3.0 if i % 2 else -4.0)
    view.yaw[i] = yaw + math.pi + 0.3 * i
    view.pitch[i] = 0.15 * (i - 1)
    view.crouch[i] = crouch
    view.flags[i] = 1 | 2
    view.team[i] = team
    view.active[i] = view.alive[i] = True
rig = CameraRig(90.0)
state = {"frame": 0, "time": 0.0}
hud.set_health(64, 100)
hud.set_weapon(W.WEAPON_NAMES[0], 23, 30, False)
hud.set_ability(0.6)
hud.set_match(C.MODE_TDM, (12, 9), 431, 0, 4, 75)
hud.set_economy("CR 12.4K   +38.5 / s", "LEVEL 7   XP 1,240 / 2,150   PRESTIGE 0")
hud.set_net("FPS 60  |  PING 3 ms  |  CORRECTIONS 0")
hud.push_kill("OPERATOR  [VK-9 VANGUARD]  BOT KODO")
hud.push_kill("BOT NOVA  [HEADSHOT]  BOT RIVET", R.HUD_CRIMSON)
hud.push_reward("+100  ELIMINATION")
hud.set_hint("WASD move   SHIFT sprint   C slide   SPACE jump   G grenade   R reload   Q swap   TAB scores")


def shot(name: str) -> None:
    application.base.screenshot(namePrefix=os.path.join(OUT, "%s_%s.png" % (MAP, name)), defaultFilename=False)


def update():
    dt = 1.0 / 60.0
    state["frame"] += 1
    state["time"] += dt
    frame = state["frame"]
    scene_renderer.begin_frame(dt)
    eye = feet + np.array([0.0, C.EYE_HEIGHT_STAND, 0.0])
    muzzle = eye + fwd * 0.9 + right * 0.2 - np.array([0.0, 0.16, 0.0])
    if frame in (8, 14, 20, 28):
        target = view.pos[0] + np.array([0.0, 1.2, 0.0])
        scene_renderer.fx_muzzle(muzzle)
        scene_renderer.fx_tracer(muzzle, target)
        scene_renderer.viewmodel.fire()
        rig.add_recoil(0.012, 0.004)
        scene_renderer.fx_hit(target, headshot=frame == 20)
        scene_renderer.characters.hit(0)
        scene_renderer.fx_damage_number(target, 26 if frame != 20 else 78, headshot=frame == 20)
        hud.hitmarker(headshot=frame == 20)
        wall = eye + fwd * 6.0 + right * 3.0
        scene_renderer.fx_impact(wall - np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    if frame == 24:
        blast = feet + fwd * 10.0 + right * 1.0 + np.array([0.0, 0.2, 0.0])
        scene_renderer.fx_explosion(blast)
        rig.add_explosion_shake(eye, blast[None, :])
        scene_renderer.post.add_damage(0.8)
        hud.damage_from(0.9)
        scene_renderer.fx_damage_number(view.pos[1] + np.array([0.0, 1.0, 0.0]), 112, killed=True)
    rig.update(dt, feet, right * 2.0, C.MF_ON_GROUND, 0.0, yaw, -0.04, (0.0, 0.0))
    rig.consume_aim_kick()
    R.apply_camera(rig)
    scene_renderer.post.update(dt, rig, 0.64, True)
    scene_renderer.viewmodel.update(dt, rig, scene_renderer.lights.flash_level)
    cam_pos, cam_yaw, cam_pitch = rig.cam_pos, rig.cam_yaw, rig.cam_pitch
    if frame >= 47:                                                     # bird's-eye view of the whole arena
        data = loaded.data
        extent = float(np.max(data.bounds_max - data.bounds_min))
        cam_pos = np.array([data.bounds_min[0] * 0.55, data.bounds_max[1] + extent * 0.30, data.bounds_min[2] * 1.25])
        look = M.normalize(np.array([0.0, 1.0, 0.0]) - cam_pos)
        cam_yaw, cam_pitch = math.atan2(look[0], look[2]), math.asin(look[1])
        camera.position = tuple(float(v) for v in cam_pos)
        camera.rotation = (-math.degrees(cam_pitch), math.degrees(cam_yaw), 0.0)
        camera.fov = 90.0
        scene_renderer.viewmodel.set_visible(False)
        hud.set_visible(False)
    scene_renderer.render(dt, state["time"], cam_pos, cam_yaw, cam_pitch, view, -1,
                          np.array([feet + fwd * 5.0 + right * -1.0 + np.array([0.0, 0.6, 0.0])]))
    hud.set_crosshair(math.radians(2.2) + 0.01 * (frame % 10 < 5), R.vertical_fov(camera.fov), 1.0, dt)
    hud.update(dt)
    if frame in (9, 21, 27, 40):
        shot("%02d" % frame)
    if frame == 43:
        hud.set_scoreboard(True, [("OPERATOR", 4, 1, 450, 3, "* "), ("BOT NOVA", 3, 2, 300, 0, "  ")],
                           "TEAM DEATHMATCH - %s" % loaded.data.name)
    if frame == 46:
        shot("board")
    if frame == 50:
        shot("overview")
        print("final", scene_renderer.describe())
        application.quit()


app.run()
