# NEXUS-3D-EXTREME

A first-person LAN arena shooter with an incremental economy, written in pure Python on
Ursina / Panda3D with a NumPy data-oriented core. It is an original game: its own arenas,
weapons, names and look, built around fast slide-and-gun movement, hitscan combat with
server-side lag compensation, and a "corporate sponsor" idle economy that keeps paying while
you sit in the menus.

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

| Command | What it does |
| --- | --- |
| `python main.py --solo terminal --mode ffa --bots 5` | Jump straight into a solo match |
| `python main.py --solo crucible --mode esc` | ESCALATION: climb the ladder of all 29 guns |
| `python main.py --solo datacore --mode uplink` | UPLINK: two squads fight over three zones |
| `python main.py --host datacore` | Host a LAN match (advertised automatically) and join it |
| `python main.py --connect 10.0.0.7` | Join by address (normally not needed, see the LAN lobby) |
| `python main.py --quality 1 --fps-limit 60` | Override the saved graphics settings for one run |
| `python dedicated_server.py --map atrium --bots 8` | Headless server, needs only Python + NumPy |
| `python dedicated_server.py --list-maps` | Print the arena registry with checksums and validation |
| `python main.py --solo datacore --autoplay --frames 600` | Self-playing demo that prints frame statistics |

Controls: `WASD` move, `SHIFT` sprint, `C` / `CTRL` crouch and slide (press while sprinting),
`SPACE` jump, left mouse fire, right mouse aim, `R` reload, `Q` swap weapon, `G` lethal equipment,
`F` tactical equipment, `X` scorestreak reward, `TAB` scoreboard, `ESC` pause.

## Playing on a LAN

Nobody types an IP address. The host picks **HOST LAN GAME**; the server broadcasts an encrypted
beacon to `255.255.255.255:28015` once per second. Everybody else opens **JOIN LAN GAME** and
clicks the match. Gameplay traffic uses UDP 28016 (the next free port if that one is taken), so
allow UDP 28015 and 28016 through the local firewall.

Trust on a LAN: there is no account server, so the upgrade levels a player brings come from their own
local profile and the server cannot verify them. A host who wants a level playing field sets
**UPGRADES OFF** in MATCH SETUP (or starts with `--no-upgrades`; `dedicated_server.py --no-upgrades`).

Modes: Team Deathmatch (VANTA vs HELIX), Free For All, ESCALATION and UPLINK on five arenas (DATACORE,
SKYBRIDGE, CRUCIBLE, TERMINAL, ATRIUM). The server tops the match up with bots (three difficulty levels).

## UPLINK

Two squads fight over three zones, A on VANTA's side, B in the centre and C on HELIX's side. They are
derived from every arena's data -- the midpoint of the two squads' spawn areas and the points halfway
towards each squad, snapped to the navigation graph, with B chosen so it is as far from A as from C -- so
every arena, JSON drop-ins included, supports the mode without extra data.

Standing in a zone with no enemy present swings it towards your squad: one player needs 6 seconds, two
4, three or more 3. An enemy zone is neutralised first and then captured; with both squads inside it is
contested and does not move. Empty zones settle back (held ones recharge, neutral ones drain). Every held
zone scores a point every 2 seconds, first to 150 wins; kills score nothing, captures pay 150 credits and
75 XP to every capturer. The HUD shows the zones under the score in their holders' colours, labels with
the distance over each zone, a capture bar while you stand in one and the zones on the minimap. Bots play
the objective: they head for zones their squad does not hold, fight from inside a zone instead of chasing,
and split over the zones in mirror image for the two squads. Tunables are the `UPLINK_*` constants in
`engine/config.py`.

## ESCALATION

A weapon ladder. Everybody starts with the same stock gun in both slots (swapping does nothing),
and every kill with a gun hands the killer the next one, with a full magazine. A kill with the last
gun wins the match; at the time limit the player furthest up wins (kills break ties). The lethal
slot is always the **RAZOR HATCHET**: a hatchet kill also climbs, and drops the victim one step
("stole a step"). Killing yourself (falling, your own blast) costs a step. Nobody drops below the
first gun, and your own loadout and upgrades stay out of it, so the mode is fair for new profiles.

The ladder runs through all 29 weapons: automatics first, then LMGs and shotguns, marksman rifles
and snipers, and the pistols at the top. It is data, `ESCALATION_LADDER` in `engine/config.py`;
reorder or shorten it freely, `weapons.validate_escalation()` reports unknown or doubled keys.
In bot matches the first player finishes after roughly two and a half to four minutes.

The HUD shows `WEAPON n / 29` and the leader, a banner announces every new gun (and the final one),
the kill feed reports stolen steps, and the scoreboard lists the weapon number instead of the score.
Planned work is in `ROADMAP.md` (German).

## Arenas are data

The engine contains no level geometry. `content/maps_registry.py` holds every arena as a nested
dictionary of tuples, and `engine/maps.py` turns that data into render arrays, a collision octree
and a bot navigation graph:

```python
(x, y, z, scale_x, scale_y, scale_z, texture_type, spawn_team)
```

`(x, z)` is the centre of the footprint, `y` the bottom, the scales are full sizes in metres.
`texture_type` is a surface (`concrete floor metal rack crate hazard grate roof brick paint glass
container tower armor`), a prefab (`server_rack pillar barrier railing container_unit lamp_warm
lamp_ice`), an OBJ prop (`barrel pylon`, loaded from `content/models`), a ramp (`ramp_+x ramp_-x
ramp_+z ramp_-z`, `scale_y` is the rise), decoration (`glow_* paint_team`), a light (`light_*`,
`scale_x` is the radius) or `spawn`. With `"mirror": True` only one team's half is written; the
other half is generated by a 180 degree rotation with the teams swapped.

To add an arena, either add an entry to `MAPS`, or drop a JSON file with the same structure into
`content/maps/` (lists instead of tuples, optional `"key"`), or call `maps.register_map(key,
definition)` at run time. Nothing else is needed: waypoints for the bots are sampled
automatically on every storey and over ramps, and lighting is baked on first load. Both sides of
a connection must have the same arena: beacon, handshake and map rotation carry a CRC of the
definition, and a client without it is refused with a clear message instead of desynchronising.

## Graphics

* **Materials.** 15 tileable surface materials (albedo + gloss, normal + cavity) are synthesised
  at load and sampled from two texture arrays; normal mapping needs no tangents. Dropping
  `content/textures/<material>.png` (and `<material>_n.png`) replaces a material with your own
  image, `content/sky/<arena>.png` replaces the sky panorama, and any `.obj` in `content/models`
  can be registered as a prop in `maps.MODEL_PROPS`.
* **Lighting.** Hemisphere ambient, sun with PCF shadow maps (level rendered once, characters
  every frame), arena lamps and ambient occlusion baked per vertex with real occlusion, eight
  per-pixel dynamic lights (`I0 / d^2`) for muzzle flashes and explosions, bloom.
* **Sky.** Equirectangular panorama per arena (procedural clouds, sun halo), analytic sun disc
  and stars, and a ring of distant towers as real geometry.
* **Batching.** One static buffer and one draw call for the arena; one buffer per effect pool,
  for all players, and for all HUD vectors.
* **Frame pacing.** Vsync is off by default and a frame limiter (144 fps) is used instead,
  because double-buffered vsync halves 59 fps to 30. The 3D scene renders into an off-screen
  buffer whose resolution follows the measured frame time (50 to 100 %, target 60 fps); if the
  lowest resolution is still too slow the graphics preset drops one step. Presets (LOW, MEDIUM,
  HIGH, ULTRA), dynamic resolution, limiter and vsync are in **SETTINGS**.

The simulation always advances in fixed 60 Hz ticks (prediction and rollback need identical
steps on client and server); rendering interpolates between ticks with the real frame delta, so
motion is frame-rate independent at any refresh rate.

## Equipment

Every player carries one **lethal** (`G`) and one **tactical** (`F`), each with its own cooldown. Pick them on
the **EQUIPMENT** screen (main menu, or the pause menu during a match: a change takes effect at your next
respawn, like a class change). Bots carry random equipment and use it.

| Item | Slot | What it does | Cooldown | Unlock |
| --- | --- | --- | --- | --- |
| PULSE GRENADE | Lethal | bouncing grenade, 1.6 s fuse, 120 damage at the centre, 6.5 m radius | 12 s | free |
| RAZOR HATCHET | Lethal | thrown axe: lethal on a direct hit, no splash; sticks in walls | 9 s | L3, CR 2000 |
| CLING CHARGE | Lethal | sticks to walls and players, detonates 1.2 s later: 140 (210 to the player wearing it), 4.5 m | 16 s | L5, CR 3000 |
| THERMITE FLASK | Lethal | bursts on impact into a 3.2 m fire pool: 38 damage per second for 4 s | 15 s | L8, CR 4500 |
| NANO STIM | Tactical | heals 50 over one second and restarts regeneration at once | 16 s | free |
| SHOCK GRENADE | Tactical | stuns up to 2.5 s: movement x0.55, shaky aim, washed-out view (thrower: half) | 14 s | L2, CR 1800 |
| SMOKE CANISTER | Tactical | 5 m smoke sphere for 8 s; nobody, bots included, sees through it | 18 s | L4, CR 2600 |

`equipment.validate_equipment()` checks the balance rules, and the test-suite plays every item in the real
simulation: no blast kills a full-health player more than 1 m away (none at all through full Blast Plating),
a player wearing a cling charge always dies, fire needs more than two seconds to kill, the hatchet (direct
hits only) recharges fastest, and a stim heals too slowly to win a gunfight with. The table is
`EQUIPMENT_TABLE` in `engine/config.py`.

## Scorestreaks, minimap and death recap

Kills in a row within one life earn rewards (not in ESCALATION). A reward is earned once per life and
then waits, also through deaths, until `X` uses it (the smallest ready one first) or the match ends.
Kills made by a reward never extend the streak.

| Reward | Kills | What it does |
| --- | --- | --- |
| PULSE SCAN | 3 | Every enemy shows on the minimap for 8 s -- yours and your squad's in Team Deathmatch. The enemy gets a warning |
| AEGIS SHIELD | 5 | A 50 point overshield takes damage before your vitals for 20 s; others see you shimmer cyan |
| ORBITAL LANCE | 7 | Marks the spot under your crosshair; everyone sees the red ring, the beam and the danger zone on the minimap, and players inside get an alert. 2 s later it strikes: 240 at the centre, falling off to 0 at 7 m, blocked by cover; you and your squad are spared |

The **minimap** (top left) is rasterised from the arena geometry when a map loads (`maps.minimap_image`):
walls and cover outlined, higher floors brighter. It shows you, your squad, enemies for 1.6 s after they
fire, everybody during a PULSE SCAN, and marked lances. While you are dead the **death recap** says who
killed you, with what, from how far, how the exchange went and how much health they kept
(`engine/recap.py`). Bots earn and use rewards too. The table is `STREAK_TABLE` in `engine/config.py`.

## Weapon balance

29 weapons in seven classes; maximum health is 100 for everybody. Damage per trigger pull scales inversely
with rate of fire; recoil is proportional to damage; reload time grows with damage. Time to kill on body hits:

| Weapon | Class | Damage | RPM | Shots | TTK | TTK maxed | Reload | Unlock |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MX-2 MIDGE | SMG (auto) | 14.3 | 1125 | 7 | 333 ms | 300 ms | 1.25 s | L2 |
| VX-3 VIPER | Pistol (auto) | 14.4 | 1091 | 7 | 333 ms | 300 ms | 1.3 s | L4 |
| TK-7 SHRIKE | SMG (auto) | 16.8 | 923 | 6 | 333 ms | 300 ms | 1.4 s | L3 |
| HX-4 HORNET | SMG (auto) | 17 | 900 | 6 | 333 ms | 300 ms | 1.5 s | L1 |
| WD-17 WARDEN | Pistol (semi) | 17.3 | 818 | 6 | 367 ms | 333 ms | 1.55 s | L2 |
| P9 SIDEKICK | Pistol (semi) | 20 | 720 | 5 | 333 ms | 300 ms | 1.6 s | L1 |
| AX-5 AXIOM | Assault rifle (auto) | 20.2 | 692 | 5 | 350 ms | 317 ms | 1.65 s | L4 |
| WR-5 WRAITH PDW | SMG (auto) | 20.5 | 655 | 5 | 367 ms | 333 ms | 1.75 s | L5 |
| BP-4 SPARROW | Assault rifle (auto) | 20.8 | 621 | 5 | 400 ms | 350 ms | 1.8 s | L5 |
| VK-9 VANGUARD | Assault rifle (auto) | 21 | 600 | 5 | 400 ms | 367 ms | 1.9 s | L1 |
| TR-6 TEMPEST | Assault rifle (auto) | 21.4 | 554 | 5 | 433 ms | 400 ms | 2 s | L7 |
| LM-3 HERON | LMG (auto) | 21.7 | 529 | 5 | 467 ms | 417 ms | 2.1 s | L9 |
| BR-14 BASALT | Assault rifle (auto) | 25 | 522 | 4 | 350 ms | 317 ms | 2.15 s | L11 |
| KR-11 KESTREL | Assault rifle (auto) | 25.5 | 514 | 4 | 350 ms | 317 ms | 2.2 s | L8 |
| DM-7 SENTRY | Marksman (semi) | 26.5 | 480 | 4 | 383 ms | 350 ms | 2.35 s | L10 |
| MG-6 MAMMOTH | LMG (auto) | 27.5 | 462 | 4 | 400 ms | 367 ms | 2.5 s | L14 |
| BULWARK LMG | LMG (auto) | 28 | 450 | 4 | 400 ms | 367 ms | 2.6 s | L10 |
| HP-50 ANVIL | Pistol (semi) | 28.5 | 400 | 4 | 450 ms | 417 ms | 2.62 s | L6 |
| VERDICT REVOLVER | Pistol (semi) | 34 | 360 | 3 | 333 ms | 300 ms | 2.65 s | L7 |
| LV-36 RANGER | Marksman (semi) | 36 | 327 | 3 | 367 ms | 333 ms | 2.67 s | L12 |
| LONGBOW DMR | Marksman (semi) | 40 | 300 | 3 | 400 ms | 367 ms | 2.7 s | L6 |
| SC-43 OSPREY | Sniper (semi) | 43 | 267 | 3 | 450 ms | 417 ms | 2.75 s | L16 |
| SL-1 BADGER | Shotgun (semi) | 52 | 185 | 2 | 333 ms | 300 ms | 2.78 s | L8 |
| CYCLONE AUTO-8 | Shotgun (auto) | 8 x 7 | 180 | 2 | 333 ms | 300 ms | 2.8 s | L9 |
| PX-10 GAVEL | Shotgun (semi) | 10 x 6 | 164 | 2 | 367 ms | 333 ms | 2.85 s | L13 |
| BREACHER-12 | Shotgun (semi) | 8 x 8.5 | 138 | 2 | 433 ms | 400 ms | 2.9 s | L3 |
| DB-2 TWINFANG | Shotgun (semi) | 12 x 6 | 133 | 2 | 450 ms | 417 ms | 3 s | L18 |
| AURORA RAIL | Sniper (semi) | 80 | 129 | 2 | 467 ms | 433 ms | 3.1 s | L15 |
| HB-84 HALBERD | Sniper (semi) | 84 | 124 | 2 | 483 ms | 450 ms | 3.4 s | L20 |

`weapons.validate_balance()` checks the 300 to 500 ms band for stock weapons and for every
combination of damage and fire-rate upgrades, and the test-suite measures the same numbers in
the real simulation. To add a weapon, add a row to `WEAPON_TABLE` in `engine/config.py` and run
`python -c "from engine import weapons; print(weapons.validate_balance())"`: an empty list means it fits the
model. Its shot sound, tracer colour, first-person model (from `WEAPON_LOOKS` in `engine/renderer.py`, or a
class default), armory page and bot use follow automatically. Headshots keep the 3.0x multiplier of the
original brief and are therefore faster than the band (set the head entry of `ZONE_MULTIPLIER` in
`engine/config.py` to about 1.3 if they should stay inside it).

## Audio

Every sound goes through one gain stage: `master (clamped 0..1) x category ceiling x volume x
distance falloff`. Ceilings: weapons 0.25, impacts 0.20, explosions 0.30. Positional sounds decay
with `1 / distance^2` beyond a per-category reference distance, and stacked identical sounds are
ducked.

## Progression

* Kills, damage and match results pay credits and experience.
* **Armory**: unlock weapons, buy upgrade levels (damage, fire rate, magazine, reload) and
  attachments, plus operator upgrades. Every price is `BaseCost * Multiplier ** Level`.
* **Sponsors**: eight contracts that produce passive income. The Corporate Sponsor Thread credits
  `income * dt` a few times per second from a nanosecond clock, in matches and in menus; up to
  eight hours accrue (at half rate) while the game is closed.
* **Prestige**: reset everything for permanent income and combat-credit multipliers.

A host can run a level playing field with `dedicated_server.py --no-upgrades`.

### Medals, match report and contracts

Twelve medals are earned during a match and pop up in the reward feed (+50 XP each): FIRST STRIKE, DOUBLE
DOWN and TRIPLE THREAT (eliminations within 4 s of each other), IRON WILL and OVERDRIVE (5 and 10 without
dying), PAYBACK, HEADHUNTER (5 headshots), DEMOLITION (3 equipment eliminations), STARFALL (an ORBITAL LANCE
elimination), UPLINKER (3 captures), SHARPSHOOTER (half of at least 30 shots hit) and MVP. When the match
ends a **match report** appears next to the scoreboard: the MVP, eliminations, deaths, K/D, accuracy,
headshots, best streak, damage, captures, the credits and XP earned and the medals (`engine/report.py`).

**Contracts** (main menu): three daily contracts and one weekly contract, picked from ten kinds by the UTC
date so everybody gets the same ones on the same day -- eliminations, headshots, wins, matches, damage,
equipment eliminations, uplink captures, medals, scorestreak rewards used, hits. Progress counts what the
profile records; a finished contract pays 2,500 (daily) or 15,000 credits (weekly) at once. They live in the
signed save (`CONTRACT_TABLE` in `engine/config.py`).

## Layout

| File | Contents |
| --- | --- |
| `content/maps_registry.py` | All arenas as data; `content/models`, `content/textures`, `content/sky`, `content/maps` are drop-in folders |
| `engine/config.py` | Every tuning constant: weapon table, upgrade tables, audio ceilings, graphics presets, pacing |
| `engine/maps.py` | Map injector: registry, prefabs, ramps, OBJ loader, automatic navigation graph, checksums |
| `engine/renderer.py` | **Module C.** Shaders, texture loaders, shadow maps, sky, batched effect pools, post process, HUD |
| `engine/textures.py`, `engine/worldmesh.py`, `engine/pacing.py` | Procedural materials and sky; mesh tessellation, UVs, light and occlusion baking; dynamic resolution |
| `engine/controller_3d.py` | **Module B.** NumPy loose octree, kinematic controller (sprint, crouch, slide, coyote jump, stair stepping), hitboxes, camera rig |
| `engine/network_lan.py` | **Module A.** Beacon discovery, `struct` protocol with delta compression, 30-tick rollback, prediction and reconciliation, interpolation |
| `engine/secure_save.py` | **Module D.** Cost math, sponsors, Corporate Sponsor Thread, sealed and signed profile file |
| `engine/simulation.py`, `engine/weapons.py`, `engine/bots.py` | Shared deterministic simulation, weapon state machine and balance model, bot AI |
| `engine/equipment.py` | Lethal / tactical equipment table, per-kind helpers and balance rules |
| `engine/streaks.py`, `engine/recap.py` | Scorestreak reward table and helpers; the death recap's combat log |
| `engine/report.py` | Medals and the match report |
| `engine/game.py`, `engine/menus.py`, `engine/audio.py` | Client loop, front end, synthesised sound and the mixer |
| `engine/mathx.py`, `engine/crypto.py` | Vector / quaternion math, stream cipher |

## How the netcode works

* **60 Hz authoritative server.** Clients send their input (button bit mask, yaw, pitch,
  velocity) for every tick, each datagram repeating the previous three commands and delta
  compressed against them. The server simulates a player once per received command.
* **Prediction.** The client runs the very same `simulate_commands` function immediately and
  keeps every unacknowledged command. When the server's state for an acknowledged command
  differs from the prediction by more than 0.05 units (or in velocity, flags or weapon state),
  the client adopts the server state and replays all pending commands inside that frame.
  That owner state also carries the complete loadout (both weapon ids and every upgrade level),
  so a gun the server hands out -- the ESCALATION ladder -- reaches the prediction with the next
  snapshot, without a separate message that could arrive out of order. Scorestreak state (shield,
  scan timer, streak, ready rewards) rides along for the HUD; it is shown, never predicted.
* **Interpolation.** Other players are drawn 100 ms in the past, positions interpolated linearly
  and orientations with quaternion SLERP between the two snapshots around that time.
* **Lag compensation.** Every shot carries a FIRE record with the client's view timestamp. The
  server keeps the last 30 ticks (500 ms) of every player's root transform, rewinds all hitboxes
  to that timestamp, ray casts, applies the zone multiplier and restores the present.
* **Snapshots** are delta compressed against the last one the client acknowledged.
* **Hardening.** Every inbound packet path was fuzzed in the code review. The server replaces non-finite
  view angles, lets a client's command clock run no faster than real time (a two second burst
  covers packet loss, so jumping ticks cannot skip fire rates or cooldowns), keeps at most 48 FIRE
  records per client, bounds its inbox and the packets it handles per tick, rate-limits connect
  attempts per address, never lets an unauthenticated connect request replace a live session,
  strips `<` / `>` (Ursina markup) from every name it receives and survives a failing tick.

## Save file integrity

The profile is canonical JSON, sealed with a SHA-256 counter-mode cipher and followed by a
final line `#SHA256:<hex>` holding `SHA-256(json + salt)`. A modified file fails verification;
the previous good save is restored (or the profile is reset) and a strike is recorded. The salt
and cipher key are constants in `engine/secure_save.py`: that stops hand-editing a save file, it
does not stop someone who reads the source and re-signs a file. The same applies to the beacon key.

Saves are serialised (the game and the Corporate Sponsor Thread both save) and the previous file is
copied to `profile.sav.bak` before the new one replaces it atomically, so the main file never goes
missing; if it does anyway, the backup is loaded. A file that cannot be *read* (permissions, disk
errors) is never treated as tampering: it is left alone and that session is not saved over it. Every
number in a save is checked to be finite and in range.

## Tests

```bash
python -m pytest tests -q
```

```bash
python tests/run_all.py
```

The second command needs no pytest. 170 tests cover the math, the octree against a brute-force
reference, movement, weapons and the balance model, the simulation, all arenas and the injector
(run-time registration, JSON drop-ins, malformed definitions, checksums), navigation on every
storey, bot-only matches, the save system, wire codecs, rollback, live client / server sessions
over loopback with latency, jitter and packet loss, a client that lacks the server's arena, the
mixer, every equipment item (hatchet hits, charges sticking to players, fire, stuns, smoke against bots,
stims, respawn-time class changes), the ESCALATION ladder (climbing, stolen steps, a shotgun volley that
kills twice, the winner, a live session in which the client predicts with the gun the server hands out,
bot matches), every scorestreak reward (earning per life, the X key, the shield soaking damage, the scan's
reach, the lance's warning time, falloff and squad immunity, a live session raising the shield, bots using
them), the minimap of every arena, the death recap, UPLINK (fair zones on every arena, capture times,
contesting, neutralising, scoring, bots spreading over the zones, the zone state in snapshots and a live
session), every medal and the match report, contracts (calendar rotation, payout, the signed save),
procedural materials, light baking, dynamic resolution,
camera springs at 10 fps, regression tests for every defect the code review found (hostile packets,
command clocks running ahead, unbounded queues, session hijacking, markup in names, concurrent and
unreadable saves, non-finite numbers in saves), a client
that stalls while baking, and two rendered end-to-end runs
(under `xvfb-run` when there is no display; skipped if neither exists).

`dev/render_smoke.py OUT_DIR MAP QUALITY` and `dev/menu_smoke.py OUT_DIR` write screenshots.

## Measurements and known limits

* Everything here was measured in a container **without a GPU** (Mesa llvmpipe software
  rasteriser). At 1280x720 with 8 players and the listen server in the same process the previous
  flat-shaded build ran at 25 fps; this build runs at 36 to 38 fps on LOW (dynamic resolution at
  50 %) and 28 fps on HIGH before the preset fallback acts. Frame rates on real graphics hardware
  have not been measured; software rasterisation says little about them.
* CPU side per frame (measured at 320x180, where rasterisation is negligible): about 4.5 to 5 ms
  of client logic and about 1.8 ms per server tick. The listen server shares the Python
  interpreter with the render loop; moving it to its own process is the next step for hosts.
* Baked lamps and ambient occlusion are per vertex (0.8 to 3 m grid depending on the preset),
  not lightmaps. Characters and the view-model are box built; real meshes can be added as OBJ
  props, but there is no skeletal animation.
* Players do not collide with each other, and ramps are walked as fine staircases (bullets and
  grenades hit the true ramp surface; the camera smooths the steps).
* Recoil is applied on the client to the view angles; the server trusts the angles it receives.
* Changing vsync takes effect at the next start. Audio is synthesised at first start into
  `cache/sfx`; baked navigation, meshes, materials and skies are cached in `cache/bake`.
