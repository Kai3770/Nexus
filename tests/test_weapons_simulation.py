"""Weapon tables / state machine and the authoritative simulation."""
import numpy as np

import helpers  # noqa: F401

from engine import config as C
from engine import mathx as M
from engine import simulation as S
from engine import weapons as W


def test_weapon_table_is_consistent():
    assert W.WEAPONS.shape[0] == W.WEAPON_COUNT == len(W.WEAPON_NAMES) == 7
    assert np.all(W.WEAPONS["damage"] > 0) and np.all(W.WEAPONS["interval"] >= 1) and np.all(W.WEAPONS["mag"] > 0)
    assert np.all(W.WEAPONS["range_end"] > W.WEAPONS["range_start"])
    assert W.SPREAD_MATRIX.shape == (7, 6)
    assert np.all(W.SPREAD_MATRIX[:, W.MOVE_AIR] > W.SPREAD_MATRIX[:, W.MOVE_WALK])     # jumping is least accurate
    assert np.all(W.SPREAD_MATRIX[:, W.MOVE_CROUCH] < 1.0)
    assert np.allclose(C.ZONE_MULTIPLIER, [3.0, 1.0, 0.75])


def test_upgrades_change_derived_stats():
    base = W.derive_stats(0, np.zeros(8, int))
    maxed = W.derive_stats(0, W.UPGRADE_MAX)
    assert np.isclose(maxed["damage"], base["damage"] * 2.0)                            # 20 levels x 5 %
    assert maxed["interval"] < base["interval"] and maxed["mag"] > base["mag"]
    assert maxed["reload_ticks"] < base["reload_ticks"] and maxed["recoil_pitch"] < base["recoil_pitch"]
    assert maxed["spread_hip"] < base["spread_hip"] and maxed["range_end"] > base["range_end"]
    assert maxed["mobility"] > base["mobility"]
    clipped = W.derive_stats(0, np.full(8, 99))
    assert np.isclose(clipped["damage"], maxed["damage"])                               # levels are clamped


def test_loadout_blob_round_trip():
    levels = np.arange(16).reshape(2, 8) % 5
    blob = W.encode_loadout([3, 4], levels, [1, 2, 3, 4], 7)
    assert len(blob) == W.LOADOUT_BLOB_SIZE
    ids, lv, op, prestige = W.decode_loadout(blob)
    assert ids.tolist() == [3, 4] and np.array_equal(lv, W.clamp_levels(levels)) and op.tolist() == [1, 2, 3, 4]
    assert prestige == 7


def test_spread_matrix_drives_the_accuracy_cone():
    hip, ads = W.WEAPONS["spread_hip"][0], W.WEAPONS["spread_ads"][0]
    def cone(flags, speed, bloom=0.0):
        return float(W.spread_angle(W.CLASS_AR, hip, ads, bloom, np.uint16(flags), speed))
    stand = cone(C.MF_ON_GROUND, 0.0)
    assert np.isclose(stand, hip)
    assert cone(C.MF_ON_GROUND, 5.0) > stand                                            # walking
    assert cone(C.MF_ON_GROUND | C.MF_SLIDING, 9.0) > cone(C.MF_ON_GROUND, 5.0)         # sliding
    assert cone(0, 5.0) > cone(C.MF_ON_GROUND | C.MF_SLIDING, 9.0)                      # airborne
    assert cone(C.MF_ON_GROUND | C.MF_CROUCHED, 0.0) < stand
    assert cone(C.MF_ON_GROUND | C.MF_ADS, 0.0) < stand * 0.2
    assert cone(C.MF_ON_GROUND, 0.0, bloom=0.01) > stand


def test_automatic_fire_cadence_reload_and_semi_auto():
    ws = W.WeaponState(3)
    ws.set_loadout(1, [W.WEAPON_INDEX["sidekick"], W.WEAPON_INDEX["sidekick"]], np.zeros((2, 8), int))
    ws.set_loadout(2, [W.WEAPON_INDEX["vanguard"], W.WEAPON_INDEX["sidekick"]], np.full((2, 8), 99))
    rows, shots, shot_ticks = np.arange(3), np.zeros(3, int), [[], [], []]
    for tick in range(1, 361):
        fired = W.step_weapons(ws, rows, np.full(3, C.BTN_SHOOT, np.uint16), tick, True)
        for row in np.flatnonzero(fired):
            shot_ticks[row].append(tick)
        shots += fired
    assert np.all(np.diff(shot_ticks[0][:30]) == 6)                                     # 600 rpm at 60 Hz
    assert shots[0] == 30 + (360 - (175 + 126)) // 6 + 1                                # full mag, reload, resume
    assert shots[1] == 1                                                                # semi-auto: held trigger = 1 round
    gaps = np.diff(shot_ticks[2][:20])
    assert set(gaps.tolist()) <= {4, 5} and abs(gaps.mean() - 6.0 * 0.7) < 0.15         # fractional cadence carried
    ws.reset_rows([1])
    taps = sum(bool(W.step_weapons(ws, [1], np.uint16(C.BTN_SHOOT if t % 2 else 0), 1000 + t, True)[0]) for t in range(40))
    assert taps == 40 // 2 // 4 + 1 or 4 <= taps <= 6                                   # tapping is limited by the cadence


def test_swap_cancels_reload_and_dead_players_cannot_fire():
    ws = W.WeaponState(2)
    ws.ammo[0, 0] = 0
    W.step_weapons(ws, [0], np.uint16(0), 1, True)
    assert ws.reload_end[0] > 0                                                         # empty magazine auto reloads
    W.step_weapons(ws, [0], np.uint16(C.BTN_SWAP), 2, True)
    assert ws.slot[0] == 1 and ws.reload_end[0] == 0 and ws.next_fire[0] > 2
    assert not W.step_weapons(ws, [1], np.uint16(C.BTN_SHOOT), 5, False)[0]


def make_duel(seed=2):
    game = S.GameWorld("crucible", mode=C.MODE_TDM, seed=seed)
    a, b = game.add_player("alpha"), game.add_player("bravo")
    s = game.store
    s.move.pos[a] = (-8.0, 0.0, 14.0)
    s.move.pos[b] = (2.0, 0.0, 14.0)
    s.yaw[b] = -np.pi / 2
    s.protect_until[:] = 0
    s.weapons.spread_hip[:] = 0.0
    s.weapons.bloom_shot[:] = 0.0
    s.refresh_hitboxes()
    game.drain_events()
    return game, a, b


def aim(game, shooter, target, box):
    s = game.store
    s.yaw[shooter], s.pitch[shooter] = M.yaw_pitch_from_direction(s.hitboxes.world_center[target, box] - s.eye_position(shooter)[0])


def test_zone_multipliers_and_falloff_are_applied():
    game, a, b = make_duel()
    s = game.store
    base = float(s.weapons.damage[a, 0])
    for box, zone, multiplier in ((0, C.ZONE_HEAD, 3.0), (1, C.ZONE_TORSO, 1.0), (5, C.ZONE_LIMB, 0.75)):
        s.health[b] = 500.0
        aim(game, a, b, box)
        s.weapons.shots_fired[a] += 1
        game.resolve_shot(a)
        hits = [e for e in game.drain_events() if e[0] == S.EV_HIT]
        assert len(hits) == 1 and hits[0][4] == zone and np.isclose(hits[0][3], base * multiplier)
    s.move.pos[a] = (-16.0, 0.0, 14.0)
    s.move.pos[b] = (16.5, 0.0, 14.0)                                                   # 32.5 m: inside the falloff band
    s.refresh_hitboxes()
    aim(game, a, b, 1)
    game.resolve_shot(a)
    far = [e for e in game.drain_events() if e[0] == S.EV_HIT][0][3]
    assert base * 0.6 < far < base


def test_world_geometry_blocks_shots_and_teammates_are_safe():
    game, a, b = make_duel()
    s = game.store
    s.move.pos[b] = (-2.0, 0.0, 0.0)                                                    # behind the centre platform edge
    s.move.pos[a] = (9.0, 0.0, 0.0)
    s.refresh_hitboxes()
    aim(game, a, b, 5)
    game.resolve_shot(a)
    assert not [e for e in game.drain_events() if e[0] == S.EV_HIT]
    game2, a2, b2 = make_duel()
    game2.store.team[b2] = game2.store.team[a2]
    aim(game2, a2, b2, 1)
    game2.resolve_shot(a2)
    assert not [e for e in game2.drain_events() if e[0] == S.EV_HIT]


def test_kill_respawn_rewards_and_spawn_protection():
    game, a, b = make_duel()
    s = game.store
    aim(game, a, b, 0)
    kills = []
    for _ in range(40):
        fired = game.step_commands(np.array([a]), np.uint16(C.BTN_SHOOT), s.yaw[a], s.pitch[a], game.tick + 1)
        game.resolve_shots(np.array([a])[fired])
        game.end_tick()
        kills += [e for e in game.drain_events() if e[0] in (S.EV_KILL, S.EV_REWARD)]
        if not s.alive[b]:
            break
    assert not s.alive[b] and s.kills[a] == 1 and s.deaths[b] == 1 and game.team_scores[s.team[a]] == 1
    kill = [e for e in kills if e[0] == S.EV_KILL][0]
    assert kill[1] == a and kill[2] == b and kill[4] & S.KILL_HEADSHOT
    assert any(e[0] == S.EV_REWARD and e[4] == S.REWARD_HEADSHOT for e in kills)
    seq = int(s.spawn_seq[b])
    for _ in range(int(C.RESPAWN_TIME * C.TICK_RATE) + 2):
        game.end_tick()
    assert s.alive[b] and s.health[b] == s.max_health[b] and s.spawn_seq[b] == seq + 1
    assert s.protect_until[b] > game.tick
    s.move.pos[b] = (2.0, 0.0, 14.0)
    s.refresh_hitboxes()
    aim(game, a, b, 1)
    game.drain_events()
    game.resolve_shot(a)
    assert not [e for e in game.drain_events() if e[0] == S.EV_HIT]                     # protected after respawn


def test_grenade_bounces_explodes_and_respects_cover():
    game, a, b = make_duel()
    s = game.store
    s.move.pos[b] = (-2.5, 0.0, 14.0)
    s.refresh_hitboxes()
    s.yaw[a], s.pitch[a] = np.pi / 2, -0.25
    game.throw_grenade(a)
    spawned = [e for e in game.drain_events() if e[0] == S.EV_GRENADE]
    assert len(spawned) == 1 and spawned[0][5] == int(C.GRENADE_FUSE * C.TICK_RATE)
    blast, hits = None, []
    for _ in range(140):
        game.end_tick()
        for event in game.drain_events():
            if event[0] == S.EV_EXPLOSION:
                blast = np.array(event[1])
            if event[0] == S.EV_HIT:
                hits.append(event)
    assert blast is not None and blast[1] < 0.6 and blast[0] > -8.0                     # came to rest on the floor
    distance = np.linalg.norm(s.hitboxes.bound_center[b] - blast)
    assert (distance < C.GRENADE_RADIUS) == bool(hits)
    game.explode(np.array([0.0, 0.3, 8.6]), a, 99)                                      # pillar at (0, 10.5) shields z > 11.5
    s.health[:] = 100.0
    s.alive[:2] = True
    s.move.pos[b] = (0.0, 0.0, 12.6)
    s.refresh_hitboxes()
    game.drain_events()
    game.explode(np.array([0.0, 0.3, 8.6]), a, 100)
    assert not [e for e in game.drain_events() if e[0] == S.EV_HIT and e[2] == b]


def test_match_ends_restarts_and_rotates_map():
    game = S.GameWorld("crucible", mode=C.MODE_TDM, seed=4)
    a = game.add_player("alpha")
    game.drain_events()
    game.team_scores[game.store.team[a]] = game.score_limit
    game.end_tick()
    events = game.drain_events()
    assert game.phase == C.PHASE_ENDED and game.winner == game.store.team[a]
    assert {e[4] for e in events if e[0] == S.EV_REWARD} == {S.REWARD_MATCH, S.REWARD_WIN}
    game.next_map = "datacore"
    for _ in range(int(C.INTERMISSION_TIME * C.TICK_RATE) + 2):
        game.end_tick()
    assert game.phase == C.PHASE_LIVE and game.team_scores.sum() == 0 and game.loaded.data.key == "datacore"
    assert game.store.alive[a]
    ffa = S.GameWorld("crucible", mode=C.MODE_FFA, seed=4)
    ids = [ffa.add_player("p%d" % i) for i in range(3)]
    assert len(set(ffa.store.team[ids].tolist())) == 3                                  # everybody is an enemy


def test_owner_state_round_trip_and_client_spread_agreement():
    game, a, b = make_duel()
    s = game.store
    for tick in range(1, 40):
        game.step_commands(np.array([a]), np.uint16(C.BTN_FORWARD | C.BTN_SHOOT | C.BTN_SPRINT), 0.3, 0.05, tick)
    mirror = S.PlayerStore(1)
    mirror.weapons.set_loadout(0, s.weapons.weapon[a], s.weapons.levels[a])
    mirror.weapons.spread_hip[:] = s.weapons.spread_hip[a]
    mirror.weapons.bloom_shot[:] = s.weapons.bloom_shot[a]
    mirror.unpack_owner(0, s.pack_owner(a))
    mirror.yaw[0], mirror.pitch[0] = s.yaw[a], s.pitch[a]
    assert len(s.pack_owner(a)) == S.PlayerStore.OWNER_SIZE
    assert np.allclose(mirror.move.pos[0], s.move.pos[a], atol=1e-4) and mirror.weapons.ammo[0, 0] == s.weapons.ammo[a, 0]
    assert mirror.weapons.next_fire[0] == s.weapons.next_fire[a] and mirror.move.mflags[0] == s.move.mflags[a]
    server_rays = S.shot_rays(s, [a])
    client_rays = S.shot_rays(mirror, [0], player_ids=[a])
    assert np.allclose(server_rays[1], client_rays[1], atol=1e-6)                       # identical pellet directions
