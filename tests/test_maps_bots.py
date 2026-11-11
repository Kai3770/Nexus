"""Arena data, navigation graphs and bot-only soak matches."""
import numpy as np

import helpers  # noqa: F401

from engine import bots
from engine import config as C
from engine import maps
from engine.controller_3d import KinematicState, step_kinematics


def test_all_arenas_validate():
    assert maps.MAP_COUNT == 3
    for key in maps.MAP_KEYS:
        loaded = maps.load_map(key)
        assert maps.validate_map(loaded) == [], key
        data = loaded.data
        assert data.spawn_pos.shape[0] >= 12 and (data.spawn_team == 0).sum() == (data.spawn_team == 1).sum()
        assert data.lights_pos.shape[0] <= C.MAX_DYNAMIC_LIGHTS
        assert np.all(np.abs(data.bounds_min) < C.WORLD_LIMIT) and np.all(np.abs(data.bounds_max) < C.WORLD_LIMIT)


def test_arenas_are_point_symmetric():
    for key in maps.MAP_KEYS:
        data = maps.load_map(key).data
        team0 = data.spawn_pos[data.spawn_team == 0]
        team1 = data.spawn_pos[data.spawn_team == 1]
        mirrored = team0 * np.array([-1.0, 1.0, -1.0])
        gaps = np.linalg.norm(mirrored[:, None, :] - team1[None, :, :], axis=2).min(axis=1)
        assert gaps.max() < 1e-9, key                                           # both sides are identical


def test_navigation_graph_is_strongly_connected_and_consistent():
    for key in maps.MAP_KEYS:
        nav = maps.load_map(key).nav
        assert nav.is_connected(), key
        assert np.allclose(np.diag(nav.dist), 0.0)
        src, dst = np.triu_indices(nav.count, 1)
        hop = nav.step_towards(src, dst)
        assert np.all(nav.adjacency[src, hop])                                  # the next hop is always a real edge
        assert np.all(nav.dist[hop, dst] < nav.dist[src, dst])                  # and always gets closer
        assert np.all(nav.nearest(nav.nodes) == np.arange(nav.count))


def test_ramps_are_walkable_both_ways():
    loaded = maps.load_map("datacore")
    ks = KinematicState(1)
    ks.reset_rows([0], [-11.0, 0.0, 0.0])
    for _ in range(140):
        step_kinematics(ks, [0], np.uint16(C.BTN_FORWARD), np.pi / 2, loaded.world)
    assert abs(ks.pos[0, 1] - 1.2) < 1e-6 and ks.pos[0, 0] > -4.0                # up the ramp onto the console
    off_ground = 0
    for _ in range(140):
        step_kinematics(ks, [0], np.uint16(C.BTN_FORWARD), -np.pi / 2, loaded.world)
        off_ground += not ks.mflags[0] & C.MF_ON_GROUND
    assert abs(ks.pos[0, 1]) < 1e-6 and off_ground == 0                          # and smoothly back down


def test_bullets_hit_the_ramp_surface_not_the_hidden_stairs():
    world = maps.load_map("datacore").world
    t, item, normal = world.raycast(np.array([[-7.0, 5.0, 0.0]]), np.array([[0.0, -1.0, 0.0]]), 20.0, C.LAYER_BULLET)
    assert item[0] >= 0 and abs((5.0 - t[0]) - 0.6) < 1e-6                       # halfway up a 1.2 high ramp
    assert normal[0, 1] > 0.9 and normal[0, 0] < -0.2                            # sloped normal


def soak(key, mode, ticks=1500):
    result = bots.run_bot_match(key, bots=8, ticks=ticks, difficulty=1, mode=mode, seed=11)
    assert result["finite"] and result["shots"] > 50 and result["hits"] > 10 and result["kills"] >= 3, result
    assert result["mean_tick_ms"] < 12.0, result
    return result


def test_bot_match_crucible_free_for_all():
    result = soak("crucible", C.MODE_FFA)
    assert sum(result["kills_per_bot"]) == result["kills"] or result["kills"] >= sum(result["kills_per_bot"])


def test_bot_match_datacore_team_deathmatch():
    result = soak("datacore", C.MODE_TDM)
    assert sum(result["team_scores"]) >= 3


def test_bot_match_skybridge_nobody_gets_stuck_in_the_void():
    result = soak("skybridge", C.MODE_TDM)
    assert result["min_y"] > maps.load_map("skybridge").data.kill_y


def test_bot_difficulty_changes_lethality():
    easy = bots.run_bot_match("crucible", bots=6, ticks=1800, difficulty=0, mode=C.MODE_FFA, seed=5)
    hard = bots.run_bot_match("crucible", bots=6, ticks=1800, difficulty=2, mode=C.MODE_FFA, seed=5)
    assert hard["hits"] / max(hard["shots"], 1) > easy["hits"] / max(easy["shots"], 1)
