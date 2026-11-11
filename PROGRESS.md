# NEXUS-3D-EXTREME: Checkpoint / Handoff

> Kurzfassung (DE): Diese Datei ist der gespeicherte Arbeitsstand. Sie enthält, was fertig ist,
> was geprüft wurde und den vollständigen Bauplan für alle noch fehlenden Module. Beim
> Weitermachen wird ab Abschnitt 4 ("Next step") fortgesetzt.

Project root: `/home/user/nexus3d`. Original game (own names, maps, weapons, look); it borrows
genre mechanics only, it is not a recreation of any existing title. Reply to the user in German.

## 1. Environment facts (all verified in this container)

- Python 3.13.18, numpy 2.5.3. Installed with `pip install --break-system-packages ursina`:
  ursina 8.3.0, panda3d 1.10.16.
- Headless rendering works: `xvfb-run -a -s "-screen 0 1280x720x24" python3 script.py`
  (Mesa llvmpipe, GL 4.5 compatibility, GLSL 4.50). Screenshot:
  `base.screenshot(namePrefix='x.png', defaultFilename=False)`. ALSA / "AL lib" / audio errors
  are harmless noise (no sound device); filter them with grep.
- `import ursina` initialises audio and needs Panda3D, so headless modules (server, tests) must
  never import ursina. Window-bound modules: renderer, audio, menus, game.
- Ursina 8.3 sets `coordinate-system y-up-left`: Panda coordinates == Ursina coordinates, no
  y/z swap. World up is +Y. (The spec's "global Z axis" for rising damage numbers is therefore
  implemented as world-up = +Y; say so in the final summary.)
- Camera mapping verified: `camera.rotation_y = degrees(yaw)`, `camera.rotation_x = -degrees(pitch)`
  gives forward = (sin yaw cos pitch, sin pitch, cos yaw cos pitch), right = (cos yaw, 0, -sin yaw).
- Front-face winding for raw Panda Geoms in this setup (verified by screenshot): a triangle is
  visible when `dot(cross(p1-p0, p2-p0), outward_normal) < 0`.
- Raw geometry recipe (verified): one GeomVertexArrayFormat with float32 columns vertex(3),
  normal(3), color(4, C_color), texcoord(2) -> stride 48, numpy dtype
  `[('pos','<f4',3),('nrm','<f4',3),('col','<f4',4),('uv','<f4',2)]`; upload with
  `vdata.uncleanSetNumRows(n); memoryview(vdata.modifyArray(0)).cast('B')[:] = arr.tobytes()`;
  indices: `GeomTriangles`, `setIndexType(NTUint32)`, `modifyVertices()`, same memoryview write.
  Dynamic meshes: `node.setBounds(OmniBoundingVolume()); node.setFinal(True)`, rewrite the
  memoryview each frame. Attach NodePaths under `scene`.
- Shaders: GLSL `#version 140`; inputs `p3d_Vertex, p3d_Normal, p3d_Color, p3d_MultiTexCoord0`;
  uniforms `p3d_ModelViewProjectionMatrix, p3d_ModelMatrix`; `out vec4 fragColor`. Uniform vec4
  arrays via `PTA_LVecBase4f` (pushBack N entries, assign by index, `setShaderInput(name, pta)`).
- Post-processing (verified): `fm = FilterManager(base.win, base.cam)`;
  `quad = fm.renderSceneInto(colortex=ctex, depthtex=dtex)`; `quad.setShader(...)`;
  `quad.setShaderInput('tex', ctex)`. View-model layer inside the same offscreen buffer:
  `buf = fm.buffers[-1]`; own NodePath root + PandaCamera + PerspectiveLens;
  `dr = buf.makeDisplayRegion(); dr.setSort(10); dr.setClearDepthActive(True);
  dr.setClearColorActive(False); dr.setCamera(vm_cam)`. The HUD on `camera.ui` stays on top and
  is not post-processed. Reference script: `dev/probe_render_pipeline.py`.
- Ursina API notes: `Ursina(development_mode=False, fullscreen=False, borderless=False,
  size=(w,h), vsync=False, title=...)`; `mouse.locked = True` then `mouse.velocity` is the
  per-frame delta (x in aspect units, y in [-0.5, 0.5]); `held_keys['left mouse']`,
  `'left shift'`, `'left control'`, `'space'`; `app.input(key)` simulates input; UI space under
  `camera.ui` is y in [-0.5, 0.5], x in +-aspect/2; rotation_z is roll (sign still to be checked
  by screenshot); disable `window.fps_counter`, `window.exit_button`, `window.cog_button`.

## 2. Files finished so far (checkpoint 2: the whole headless core is done)

| File | State |
| --- | --- |
| `engine/config.py`, `mathx.py`, `crypto.py` | done, checked |
| `engine/controller_3d.py` (MODULE B) | done, checked: octree == brute force, movement, hitboxes, camera rig |
| `engine/weapons.py` | done, checked: tables, upgrades, deterministic `step_weapons`, spread matrix |
| `engine/maps.py` | done, checked: DATACORE, SKYBRIDGE, CRUCIBLE all pass `validate_map`, nav graphs strongly connected |
| `engine/simulation.py` | done, checked: `PlayerStore`, `simulate_commands`, `shot_rays`, `GameWorld` (batched `resolve_shots(rows, rewind)`, damage, grenades, respawn, match flow, `next_map`) |
| `engine/secure_save.py` (MODULE D) | done, checked: cost math, sponsors, `CorporateSponsorThread`, sealed + salted-SHA-256 save, tamper -> backup -> quarantine |
| `engine/bots.py` | done, checked: `BotController.think`, `run_bot_match` (1 min matches on all maps: 40 to 67 kills, no NaN) |
| `engine/network_lan.py` (MODULE A) | done, checked (see below) |

Network checks that passed over loopback:
- Codecs round-trip (delta commands, entity delta 52 bytes vs 228 full, all events, sealed beacon).
- `LanBrowser` finds a server through the beacon; connect, 60 snapshots/s, roster, events.
- Prediction: 244 ticks of running / jumping / strafing with max position error 0.00000.
- `LagCompensator` unit case: target moved 1.11 units since the client saw it; miss without
  rollback, hit with rollback, hitboxes identical after restore, >500 ms clamped to 29 ticks.
- Live with 45 ms latency each way, 8 ms jitter, 3 percent loss: 50 of 50 rounds fired at the
  interpolated image of a strafing target confirmed; rewind 10 to 13 ticks; no missing FIRE records.

API the client layer must use (already implemented):
- `N.host_local_game(N.ServerConfig(...))` -> `GameServer` (`.port`, `.stop()`, `.describe()`, `.lock`, `.world`).
- `N.LanBrowser().start()`, `.servers()` -> list of `ServerInfo` (name, address, port, players, bots, map_name, mode_name).
- `cl = N.GameClient(name, weapon_ids, levels, operator, prestige)`; `cl.connect(host, port)`;
  per frame `events = cl.update()`; per 60 Hz tick `fired, ability = cl.send_command(buttons, yaw, pitch)`;
  per frame `view = cl.sample_entities()` (EntityView: pos, vel, yaw, pitch, crouch, flags, health, weapon, team, active, alive);
  `cl.state` in `N.DISCONNECTED / CONNECTING / CONNECTED / FAILED`, `cl.fail_reason`, `cl.pid`, `cl.team`,
  `cl.map_index`, `cl.map_changed`, `cl.loaded` (LoadedMap), `cl.mode`, `cl.scores`, `cl.time_left`, `cl.phase`,
  `cl.winner`, `cl.roster` {pid: RosterEntry(name, team, bot, alive, kills, deaths, score, ping)}, `cl.ping_ms`.
- `cl.predictor`: `.store` (PlayerStore with 1 row: `.move`, `.weapons`, `.health`, `.max_health`, `.alive`,
  `.ability_ready`, `.ability_cooldown`, `.cmd_tick`), `.render_position()`, `.decay_visual_error(dt)`,
  `.corrections`, `.last_error`.
- Decoded client events: `(EV_HIT, victim, damage, zone, killed, pos)`, `(EV_DAMAGE, attacker, damage, zone, source_pos)`,
  `(EV_KILL, killer, victim, weapon, flags)`, `(EV_SPAWN, pid, pos, yaw)`, `(EV_SHOT, shooter, weapon, origin, end)`,
  `(EV_EXPLOSION, pos, owner, gid)`, `(EV_GRENADE, gid, owner, pos, vel, fuse, spawn_tick)`,
  `(EV_REWARD, credits, xp, reason)`, `(EV_MATCH, phase, winner, map_index, mode)`.
- Own tracers: `S.shot_rays(cl.predictor.store, [0], player_ids=[cl.pid])` right after a predicted shot.
- Grenades on the client: `S.GrenadeStore` + `S.step_grenades(gs, world)` from EV_GRENADE; remove on EV_EXPLOSION.
- Profile: `SS.ProfileManager().load()`, `.loadout_arrays()`, `.apply_reward(credits, xp)`, `.snapshot()`,
  `SS.CorporateSponsorThread(pm).start()`.

## 3. Decisions already taken (do not re-derive)

- Sim state float64; wire float32 / quantised ints; client quantises view angles to float32 before predicting.
- Discovery beacon on UDP 28015; gameplay socket on 28016 (next free port if busy).
- The server derives each shot from the SHOOT button in the deterministic weapon step; the FIRE record
  (shot_seq, cmd_tick, view_time_ms, slot) carries the client-side execution timestamp and selects the rollback
  time; fallback estimate if it is missing.
- Server simulates a player only when its commands arrive; token bucket plus 12 percent backlog drain.
- Snapshots at 60 Hz. FFA: unique team id per player. Players do not collide with each other.
- Recoil is client-side on the view angles (`CameraRig.add_recoil`, `consume_aim_kick`).
- Remote players are drawn from the hitbox box layout (one dynamic mesh for everyone).
- Honest notes for the final summary: no `numpy.lerp` (own `lerp`); keys / salt compiled in (deters casual
  tampering only); world-up is +Y in Ursina 8 (spec said global Z); beacon port 28015 vs game port 28016;
  original game, not a recreation of an existing title.

## 4. Next step and remaining work (in this order)

DONE since checkpoint 2: `engine/renderer.py` (MODULE C, 1510 lines) written and verified by screenshots on all
three arenas with `xvfb-run -a -s "-screen 0 1280x720x24" python3 dev/render_smoke.py <out_dir> <map>`:
world mesh + sky + glow strips, `LightPool` (muzzle flash 0.03 s, explosion), `ParticlePool`, `TracerPool`,
`DecalPool`, `CharacterBatch`, `DamageTextPool` (seven-segment digits), `PostFX`, `ViewModel`, `HUD`,
`SceneRenderer` facade, `apply_camera(rig)`, `vertical_fov(hfov)`.

Renderer API for the game layer:
- `sr = R.SceneRenderer()`; `sr.load_map(map_data)`; each frame `sr.begin_frame(dt)` first, then effect calls
  (`fx_muzzle(pos)`, `fx_tracer(start, end)`, `fx_impact(pos, normal)`, `fx_hit(pos, headshot)`,
  `fx_death(pos, rgb)`, `fx_explosion(pos)`, `fx_damage_number(pos, amount, headshot, killed)`,
  `sr.characters.hit(pid)`), then `rig.update(...)`, `R.apply_camera(rig)`, `sr.post.update(dt, rig, health_frac, alive)`,
  `sr.post.add_damage(strength)`, `sr.viewmodel.set_weapon(i) / fire() / set_visible(b) / update(dt, rig, sr.lights.flash_level)`,
  `sr.viewmodel.muzzle_offset(ads_blend)` (camera space right/up/forward), and finally
  `sr.render(dt, time_now, cam_pos, yaw, pitch, entity_view, own_pid, grenade_positions)`.
- `hud = R.HUD()`: `set_crosshair(spread, vfov, alpha, dt)`, `hitmarker(headshot, kill)`, `damage_from(rel_angle)`,
  `push_kill(text, rgb)`, `push_reward(text)`, `set_health`, `set_weapon(name, ammo, mag, reloading)`,
  `set_ability(frac)`, `set_match(mode, scores, seconds, own_team, own_kills, limit)`, `set_center(text, sub, rgb)`,
  `set_hint`, `set_economy(credits_text, level_text)`, `set_net(text)`, `set_scoreboard(visible, rows, title)`,
  `set_visible(b)`, `update(dt)`.
- Camera roll: Ursina `rotation_z = -degrees(rig.cam_roll)` (already inside `apply_camera`).

DONE since checkpoint 3: `engine/audio.py` (27 synthesised effects, `SoundBank`), `engine/menus.py` (8 screens,
verified by `dev/menu_smoke.py` screenshots), `engine/game.py` (`NexusGame`, `AutoPilot`), `main.py`,
`dedicated_server.py`. Full client verified headless:
`xvfb-run -a -s "-screen 0 1280x720x24" python3 main.py --solo crucible --mode ffa --bots 5 --autoplay --frames 420
--screenshots <dir> --shot-every 70 --size 1280x720 --no-vsync --save-dir <dir>` ran 420 frames (about 32 FPS on
software rendering), autopilot got eliminations, prediction error 0.000, profile saved.
Ursina gotcha: never put the characters `<` or `>` in Text / Button strings (tag parser crashes).

DONE since checkpoint 4: `tests/` (69 tests, all passing with `python -m pytest tests -q` and with the
dependency-free `python tests/run_all.py`), `README.md`, `requirements.txt`, frame-logic timing in `main.py`.

STATUS: the project is complete. Every module of the brief exists, runs and is tested. Nothing is left on the
required list. Optional follow-ups if the user asks: bot balance pass, measuring frame rate on a real GPU.

## 5. Checkpoint verification log

```
final checkpoint written: 2026-11-11T12:04:00Z

source lines:
    14 engine/__init__.py
   264 engine/audio.py
   360 engine/bots.py
   247 engine/config.py
  1413 engine/controller_3d.py
    98 engine/crypto.py
   685 engine/game.py
   664 engine/maps.py
   501 engine/mathx.py
   589 engine/menus.py
  1840 engine/network_lan.py
  1510 engine/renderer.py
   701 engine/secure_save.py
   694 engine/simulation.py
   369 engine/weapons.py
    95 dedicated_server.py
   118 main.py
     8 tests/conftest.py
    18 tests/helpers.py
    47 tests/run_all.py
    69 tests/test_audio_render.py
    91 tests/test_maps_bots.py
    99 tests/test_math_crypto.py
   337 tests/test_network.py
   260 tests/test_octree_movement.py
   184 tests/test_secure_save.py
   237 tests/test_weapons_simulation.py
    53 dev/menu_smoke.py
    28 dev/probe_headless_window.py
   132 dev/probe_render_pipeline.py
   101 dev/render_smoke.py
 11826 total
```
