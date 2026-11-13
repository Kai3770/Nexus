"""Arena registry, map injection, navigation graphs and bot-only soak matches."""
import json
import tempfile
from pathlib import Path

import numpy as np

import helpers

from content import maps_registry
from engine import bots
from engine import config as C
from engine import maps
from engine.controller_3d import KinematicState, step_kinematics

BUILT_IN = ("datacore", "skybridge", "crucible", "terminal", "atrium")

# A complete arena as an author would write it: floor, walls, a ramp up to a
# platform, some cover and one spawn row (the other team is mirrored).
TEST_YARD = {
    "name": "TEST YARD",
    "description": "Injected by the test-suite.",
    "environment": {"clouds": 0.2, "sun_dir": (0.3, -0.8, 0.5)},
    "mirror": True,
    "center": [
        (0.0, -1.0, 0.0, 28.0, 1.0, 20.0, "floor", -1),
        (0.0, 0.0, 10.5, 30.0, 4.0, 1.0, "brick", -1),
        (0.0, 0.0, -10.5, 30.0, 4.0, 1.0, "brick", -1),
        (-14.5, 0.0, 0.0, 1.0, 4.0, 22.0, "concrete", -1),
        (14.5, 0.0, 0.0, 1.0, 4.0, 22.0, "concrete", -1),
        (0.0, 2.0, 0.0, 6.0, 0.2, 4.0, "grate", -1),
        (0.0, 3.5, 0.0, 12.0, 0.0, 0.0, "light_warm", -1),
    ],
    "half": [
        (-6.0, 0.0, 0.0, 6.0, 2.2, 3.0, "ramp_+x", -1),
        (-9.0, 0.0, 6.0, 1.4, 1.2, 1.4, "crate", -1),
        (-7.0, 0.0, -6.0, 0.8, 0.9, 0.8, "barrel", -1),
        (-12.0, 0.0, -4.0, 1.0, 1.0, 1.0, "spawn", 0),
        (-12.0, 0.0, 0.0, 1.0, 1.0, 1.0, "spawn", 0),
        (-12.0, 0.0, 4.0, 1.0, 1.0, 1.0, "spawn", 0),
    ],
}


def test_registry_entries_follow_the_tuple_format():
    known = set(maps.known_texture_types())
    assert len(maps_registry.MAPS) >= 5 and set(BUILT_IN) <= set(maps_registry.MAPS)
    for key, definition in maps_registry.MAPS.items():
        assert isinstance(definition, dict) and isinstance(definition["name"], str), key
        entries = list(definition["center"]) + list(definition["half"])
        assert len(entries) >= 20, key
        for entry in entries:
            assert isinstance(entry, tuple) and len(entry) == 8, (key, entry)
            x, y, z, sx, sy, sz, texture_type, spawn_team = entry
            assert all(isinstance(v, (int, float)) for v in (x, y, z, sx, sy, sz)), (key, entry)
            assert texture_type in known and spawn_team in (-1, 0, 1), (key, entry)
        assert set(definition.get("environment", {})) <= set(maps_registry.ENVIRONMENT_DEFAULTS), key


def test_all_arenas_validate():
    assert maps.MAP_COUNT >= 5 and maps.MAP_KEYS[:5] == BUILT_IN and not maps.REGISTRY_ERRORS
    assert len(maps.MAP_NAMES) == maps.MAP_COUNT
    for key in maps.MAP_KEYS:
        loaded = maps.load_map(key)
        assert maps.validate_map(loaded) == [], key
        data = loaded.data
        assert data.index == maps.map_index(key) and data.key == key
        assert data.spawn_pos.shape[0] >= 12 and (data.spawn_team == 0).sum() == (data.spawn_team == 1).sum()
        assert data.lights_pos.shape[0] <= C.MAX_STATIC_LIGHTS
        assert np.all(np.abs(data.bounds_min) < C.WORLD_LIMIT) and np.all(np.abs(data.bounds_max) < C.WORLD_LIMIT)
        count = data.box_min.shape[0]
        for name in ("box_max", "box_color", "box_layer", "box_visible", "box_texture", "box_uv_mode", "box_tile",
                     "box_emissive"):
            assert getattr(data, name).shape[0] == count, (key, name)
        assert np.all(data.box_max >= data.box_min)
        assert data.box_texture.min() >= 0 and data.box_texture.max() < len(maps.TEXTURE_LAYERS)
        triangles = data.mesh_tris.shape[0]
        assert data.mesh_normal.shape == (triangles, 3, 3) and data.mesh_uv.shape == (triangles, 3, 2)
        assert np.allclose(np.linalg.norm(data.mesh_normal, axis=2), 1.0, atol=1e-6)


def test_arenas_are_point_symmetric():
    for key in maps.MAP_KEYS:
        data = maps.load_map(key).data
        team0 = data.spawn_pos[data.spawn_team == 0]
        team1 = data.spawn_pos[data.spawn_team == 1]
        mirrored = team0 * np.array([-1.0, 1.0, -1.0])
        gaps = np.linalg.norm(mirrored[:, None, :] - team1[None, :, :], axis=2).min(axis=1)
        assert gaps.max() < 1e-9, key                                           # both sides are identical
        facing = np.stack([np.sin(data.spawn_yaw), np.cos(data.spawn_yaw)], axis=1)
        inward = -data.spawn_pos[:, [0, 2]]
        assert np.all(np.sum(facing * inward, axis=1) > 0.0), key                 # everybody spawns facing the arena


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


def test_navigation_is_generated_for_every_storey():
    atrium = maps.load_map("atrium").nav.nodes
    assert (np.abs(atrium[:, 1] - 3.2) < 1e-6).sum() > 60 and (np.abs(atrium[:, 1]) < 1e-6).sum() > 200
    datacore = maps.load_map("datacore").nav.nodes
    assert (datacore[:, 1] > 2.9).sum() > 15                                     # catwalks, reached over the ramps
    skybridge = maps.load_map("skybridge").nav.nodes
    assert (skybridge[:, 1] < -2.0).sum() > 60                                   # lower deck
    for key in maps.MAP_KEYS:                                                    # every node is a place to stand
        loaded = maps.load_map(key)
        assert maps._standing_room(loaded.world, loaded.nav.nodes).all(), key


def test_cached_navigation_matches_a_fresh_build():
    loaded = maps.load_map("crucible")                                           # first call wrote the cache file
    fresh = maps.build_arena_nav(loaded.data, loaded.world)
    cached = maps._cached_nav(loaded.data, loaded.world)
    assert np.array_equal(fresh.nodes, cached.nodes) and np.array_equal(fresh.adjacency, cached.adjacency)
    assert np.array_equal(fresh.next_hop, cached.next_hop) and np.array_equal(fresh.dist, cached.dist)


def test_runtime_map_injection_and_removal():
    before = maps.MAP_KEYS
    index = maps.register_map("test_yard", TEST_YARD)
    try:
        assert maps.MAP_KEYS == before + ("test_yard",) and index == len(before) and maps.MAP_NAMES[index] == "TEST YARD"
        loaded = maps.load_map("test_yard")
        assert maps.validate_map(loaded) == []
        data = loaded.data
        assert data.spawn_pos.shape[0] == 6 and data.clouds == 0.2 and data.lights_pos.shape[0] == 1
        assert (loaded.nav.nodes[:, 1] > 2.1).sum() >= 4                         # the platform, reached over both ramps
        result = bots.run_bot_match("test_yard", bots=4, ticks=600, difficulty=1, mode=C.MODE_FFA, seed=3)
        assert result["finite"] and result["shots"] > 5
        crc = data.checksum
        changed = dict(TEST_YARD, half=TEST_YARD["half"] + [(-9.0, 0.0, -6.0, 1.4, 1.2, 1.4, "crate", -1)])
        maps.register_map("test_yard", changed)                                  # hot replace
        assert maps.load_map("test_yard").data.checksum != crc
        assert maps.load_map("test_yard").data.box_min.shape[0] > data.box_min.shape[0]
    finally:
        maps.unregister_map("test_yard")
    assert maps.MAP_KEYS == before


def test_json_drop_in_arenas_are_discovered():
    original = C.MAPS_DIR
    with tempfile.TemporaryDirectory() as folder:
        C.MAPS_DIR = Path(folder)
        try:
            (C.MAPS_DIR / "yard.json").write_text(json.dumps(dict(TEST_YARD, key="json_yard")))
            (C.MAPS_DIR / "broken.json").write_text('{"name": "BROKEN", "center": [[0, 0, 0, 1, 1, 1, "lava", -1]]}')
            keys = maps.reload_registry()
            assert keys[:5] == BUILT_IN and "json_yard" in keys and "broken" not in keys
            assert len(maps.REGISTRY_ERRORS) == 1 and "lava" in maps.REGISTRY_ERRORS[0]
            assert maps.validate_map(maps.load_map("json_yard")) == []
            reference = maps.build_map_data("json_yard", TEST_YARD)
            assert maps.get_map_data("json_yard").checksum == reference.checksum   # JSON lists and tuples hash alike
        finally:
            C.MAPS_DIR = original
            maps.reload_registry()
    assert maps.MAP_KEYS[:5] == BUILT_IN and "json_yard" not in maps.MAP_KEYS


def test_malformed_definitions_are_rejected():
    with helpers.raises(ValueError):
        maps.build_map_data("bad", {"center": [(0, 0, 0, 1, 1, 1, "lava", -1)]})
    with helpers.raises(ValueError):
        maps.build_map_data("bad", {"center": [(0, 0, 0, 1, 1, "floor", -1)]})
    with helpers.raises(ValueError):
        maps.build_map_data("bad", {"center": []})
    with helpers.raises(ValueError):
        maps.build_map_data("bad", {"center": [(0, 0, 0, 1, 1, 1, "floor", -1)], "environment": {"gravity": 1}})


def test_prefabs_ramps_and_models_expand_into_primitives():
    data = maps.build_map_data("unit", {"center": [
        (0.0, -1.0, 0.0, 20.0, 1.0, 20.0, "floor", -1),
        (4.0, 0.0, 0.0, 1.0, 2.0, 2.0, "server_rack", -1),
        (0.0, 0.0, 4.0, 4.0, 2.0, 2.0, "ramp_+x", -1),
        (-4.0, 0.0, 0.0, 0.8, 0.9, 0.8, "barrel", -1),
        (0.0, 0.0, -4.0, 0.3, 4.0, 0.3, "lamp_warm", -1),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, "spawn", 1),
    ]})
    glow = data.box_texture == maps.LAYER["glow"]
    assert glow.sum() >= 3 and np.all(data.box_emissive[glow] > 0.0) and np.all(data.box_layer[glow] == 0)
    stairs = (data.box_layer == C.LAYER_PLAYER) & ~data.box_visible
    assert stairs.sum() == int(np.ceil(2.0 / maps.STAIR_RISE))                   # hidden staircase under the ramp
    assert data.mesh_solid.sum() == 6                                            # wedge: slope, back and two sides
    slope = data.mesh_normal[data.mesh_solid][:, 0, :]
    assert np.isclose(slope[:, 1].max(), np.cos(np.arctan2(2.0, 4.0)))
    barrel = data.mesh_tris[~data.mesh_solid]
    assert barrel.shape[0] > 100
    low, high = barrel.reshape(-1, 3).min(axis=0), barrel.reshape(-1, 3).max(axis=0)
    assert np.allclose(low, [-4.4, 0.0, -0.4], atol=1e-6) and np.allclose(high, [-3.6, 0.9, 0.4], atol=1e-6)
    centre = barrel.mean(axis=1) - np.array([-4.0, 0.45, 0.0])
    outward = np.sum(np.cross(barrel[:, 1] - barrel[:, 0], barrel[:, 2] - barrel[:, 0]) * centre, axis=1)
    assert (np.sum(data.mesh_normal[~data.mesh_solid][:, 0, :] * centre, axis=1) > -1e-6).mean() > 0.9
    assert np.abs(outward).max() > 0.0
    assert data.lights_pos.shape[0] == 1 and data.spawn_team.tolist() == [1]
    assert data.mesh_uv.min() >= -1e-6


def test_obj_loader_reads_faces_normals_and_uvs():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "quad.obj"
        path.write_text("# unit quad\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                        "vn 0 0 1\nf 1/1/1 2/2/1 3/3/1 4/4/1\nf -4/-4 -3/-3 -2/-2\n")
        mesh = maps.load_obj(path)
    assert mesh["pos"].shape == (3, 3, 3) and mesh["uv"].shape == (3, 3, 2) and mesh["nrm"].shape == (3, 3, 3)
    assert np.allclose(mesh["nrm"][:2], [0.0, 0.0, 1.0]) and np.allclose(np.abs(mesh["nrm"][2]), [0.0, 0.0, 1.0])
    assert np.allclose(mesh["uv"][0], [[0, 0], [1, 0], [1, 1]]) and np.allclose(mesh["pos"][1, 2], [0, 1, 0])
    for name in ("barrel.obj", "pylon.obj"):
        model = maps.load_obj(C.MODELS_DIR / name)
        assert model["pos"].shape[0] > 50 and np.allclose(np.linalg.norm(model["nrm"], axis=2), 1.0)
        assert model["uv"].min() >= 0.0 and model["uv"].max() <= 4.0                # the texture may wrap round a prop


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


def test_mirrored_ramps_climb_towards_the_centre_too():
    loaded = maps.load_map("atrium")                                             # east stairs are the mirrored copy
    ks = KinematicState(1)
    ks.reset_rows([0], [21.5, 0.0, -10.5])
    for _ in range(112):
        step_kinematics(ks, [0], np.uint16(C.BTN_FORWARD), -np.pi / 2, loaded.world)
    assert abs(ks.pos[0, 1] - 3.2) < 1e-6 and 9.0 < ks.pos[0, 0] < 12.0          # on the landing of the mezzanine


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


def test_bot_match_terminal_and_atrium():
    assert sum(soak("terminal", C.MODE_TDM)["team_scores"]) >= 3
    soak("atrium", C.MODE_FFA)


def test_bot_difficulty_changes_lethality():
    easy = bots.run_bot_match("crucible", bots=6, ticks=1800, difficulty=0, mode=C.MODE_FFA, seed=5)
    hard = bots.run_bot_match("crucible", bots=6, ticks=1800, difficulty=2, mode=C.MODE_FFA, seed=5)
    assert hard["hits"] / max(hard["shots"], 1) > easy["hits"] / max(easy["shots"], 1)
