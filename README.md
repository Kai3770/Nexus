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

Useful shortcuts:

| Command | What it does |
| --- | --- |
| `python main.py --solo crucible --mode ffa --bots 5` | Jump straight into a solo match |
| `python main.py --host datacore` | Host a LAN match (advertised automatically) and join it |
| `python main.py --connect 10.0.0.7` | Join by address (normally not needed, see the LAN lobby) |
| `python dedicated_server.py --map datacore --bots 8` | Headless server, needs only Python + NumPy |
| `python main.py --solo datacore --autoplay --frames 600` | Self-playing demo that prints the frame-logic time |

Controls: `WASD` move, `SHIFT` sprint, `C` / `CTRL` crouch and slide (press while sprinting),
`SPACE` jump, left mouse fire, right mouse aim, `R` reload, `Q` swap weapon, `G` pulse grenade,
`TAB` scoreboard, `ESC` pause.

## Playing on a LAN

Nobody types an IP address. The host picks **HOST LAN GAME**; the server broadcasts an encrypted
beacon to `255.255.255.255:28015` once per second. Everybody else opens **JOIN LAN GAME** and
clicks the match. Gameplay traffic uses UDP 28016 (the next free port if that one is taken), so
allow UDP 28015 and 28016 through the local firewall.

Modes: Team Deathmatch (VANTA vs HELIX) and Free For All, on three arenas (DATACORE, SKYBRIDGE,
CRUCIBLE). The server tops the match up with bots (three difficulty levels).

## Progression

* Kills, damage and match results pay credits and experience.
* **Armory**: unlock weapons, buy upgrade levels (damage, fire rate, magazine, reload) and
  attachments, plus operator upgrades. Every price is `BaseCost * Multiplier ** Level`.
* **Sponsors**: eight contracts that produce passive income. The Corporate Sponsor Thread credits
  `income * dt` a few times per second from a nanosecond clock, in matches and in menus; up to
  eight hours accrue (at half rate) while the game is closed.
* **Prestige**: reset everything for permanent income and combat-credit multipliers.

A host can run a level playing field with `dedicated_server.py --no-upgrades`.

## Layout

| File | Contents |
| --- | --- |
| `engine/controller_3d.py` | **Module B.** NumPy loose octree (batched ray / box queries), vectorised kinematic controller (sprint, crouch, exponential-friction slide, coyote jump, stair stepping, swept AABB), anatomical hitboxes (head 3.0x, torso 1.0x, limbs 0.75x), camera rig (sway, banking, recoil, shake) |
| `engine/network_lan.py` | **Module A.** Beacon discovery, `struct` protocol with delta compression, 30-tick history rollback, client prediction and reconciliation, LERP / SLERP entity interpolation, server and client |
| `engine/renderer.py` | **Module C.** GLSL shaders (world lighting with `I0 / d^2` point lights, post process), static level mesh, particle / tracer / decal pools, character batch, seven-segment damage numbers, view-model, HUD |
| `engine/secure_save.py` | **Module D.** Cost math, sponsors, Corporate Sponsor Thread, sealed and signed profile file |
| `engine/simulation.py` | Structure-of-arrays player store, shared deterministic command step, shots, damage, grenades, match rules |
| `engine/weapons.py` | Weapon and upgrade tables, bullet spread matrix, deterministic weapon state machine |
| `engine/maps.py` | Level builder, three arenas, navigation graph (vectorised Floyd-Warshall) |
| `engine/bots.py` | Vectorised bot AI that plays through the same command path as humans |
| `engine/game.py`, `engine/menus.py`, `engine/audio.py` | Client loop, front end, procedurally synthesised sound |
| `engine/mathx.py`, `engine/crypto.py`, `engine/config.py` | Vector / quaternion math, stream cipher, all tuning constants |

## How the netcode works

* **60 Hz authoritative server.** Clients send their input (button bit mask, yaw, pitch,
  velocity) for every tick, each datagram repeating the previous three commands and delta
  compressed against them. The server simulates a player once per received command.
* **Prediction.** The client runs the very same `simulate_commands` function immediately and
  keeps every unacknowledged command. When the server's state for an acknowledged command
  differs from the prediction by more than 0.05 units (or in velocity, flags or weapon state),
  the client adopts the server state and replays all pending commands inside that frame.
* **Interpolation.** Other players are drawn 100 ms in the past, positions interpolated linearly
  and orientations with quaternion SLERP between the two snapshots around that time.
* **Lag compensation.** Every shot carries a FIRE record with the client's view timestamp. The
  server keeps the last 30 ticks (500 ms) of every player's root transform, rewinds all hitboxes
  to that timestamp, ray casts, applies the zone multiplier and restores the present.
* **Snapshots** are delta compressed against the last one the client acknowledged: three 16-bit
  change masks plus packed record arrays (a typical delta is about 50 bytes for 10 players).

## Save file integrity

The profile is canonical JSON, sealed with a SHA-256 counter-mode cipher and followed by a
final line `#SHA256:<hex>` holding `SHA-256(json + salt)`. A modified file fails verification;
the previous good save is restored (or the profile is reset) and a strike is recorded.

Be aware of the limit of this design: the salt and cipher key are constants in
`engine/secure_save.py`, as the brief asked for. That stops hand-editing a save file. It does
not stop someone who reads the source and re-signs a file. The same applies to the beacon key.

## Tests

```bash
python -m pytest tests -q
```

```bash
python tests/run_all.py
```

The second command needs no pytest. The suite covers the math, the octree against a brute-force
reference, movement (including bit-identical batch versus single stepping, which prediction
relies on), weapons, the simulation, all three arenas, bot-only matches, the save system
(including tampering), wire codecs, rollback, a live client / server session over loopback with
45 ms latency, jitter and 3 % packet loss, LAN discovery, audio synthesis and one rendered
end-to-end match (run under `xvfb-run` when there is no display; skipped if neither exists).

`dev/render_smoke.py` and `dev/menu_smoke.py` write screenshots of the renderer and of every
menu screen.

## Design notes and known limits

* World up is +Y (Ursina 8 runs Panda3D in a Y-up coordinate system). Damage numbers therefore
  rise along +Y, the axis stock Panda3D documentation calls Z.
* NumPy has no `numpy.lerp`; `engine/mathx.py` provides `lerp` and `quat_slerp`.
* Discovery uses port 28015; gameplay uses 28016 so lobby listeners and the game socket never
  compete for one port.
* Players do not collide with each other, and ramps are walked as fine staircases (bullets and
  grenades hit the true ramp surface).
* Recoil is applied on the client to the view angles; the server trusts the angles it receives.
  On a LAN among colleagues that is a reasonable trade, but it is not cheat-proof.
* Measured in a headless container with software rendering: about 35 to 40 frames per second at
  1280x720, with about 6 ms of client logic per frame in an eight-player match that also hosts
  the server. The frame rate on real GPUs has not been measured.
* Audio is synthesised at first start into `cache/sfx`; without a sound device the game runs
  silently.
