"""Procedural materials, sky panoramas, world mesh baking and frame pacing (all without a window)."""
import numpy as np

import helpers  # noqa: F401

from engine import config as C
from engine import maps
from engine import pacing
from engine import textures as T
from engine import worldmesh as WM

LIT_ROOM = {
    "name": "LIT ROOM",
    "center": [
        (0.0, -1.0, 0.0, 20.0, 1.0, 12.0, "floor", -1),
        (0.0, 0.0, 0.0, 0.5, 3.0, 12.0, "brick", -1),              # wall splitting the room: x < 0 lit, x > 0 dark
        (-5.0, 0.0, 0.0, 1.4, 1.2, 1.4, "crate", -1),
        (-5.0, 2.5, 3.0, 9.0, 0.0, 0.0, "light_warm", -1),
        (-8.0, 0.0, -4.0, 1.0, 1.0, 1.0, "spawn", 0),
        (8.0, 0.0, 4.0, 1.0, 1.0, 1.0, "spawn", 1),
    ],
}


def test_graphics_presets_are_complete_and_ordered():
    keys = {"render_scale", "shadow_size", "shadow_taps", "anisotropy", "texture_size", "light_cell", "sky_size",
            "particles", "normal_maps", "bloom"}
    assert len(C.GRAPHICS_PRESETS) == len(C.QUALITY_NAMES) == 4
    for preset in C.GRAPHICS_PRESETS:
        assert set(preset) == keys
        assert 0.25 <= preset["render_scale"] <= 1.0 and preset["shadow_taps"] in (1, 4, 9)
    for low, high in zip(C.GRAPHICS_PRESETS, C.GRAPHICS_PRESETS[1:]):
        assert high["render_scale"] >= low["render_scale"] and high["shadow_size"] >= low["shadow_size"]
        assert high["texture_size"] >= low["texture_size"] and high["light_cell"] <= low["light_cell"]
    assert C.GRAPHICS_PRESETS[C.QUALITY_LOW]["shadow_size"] == 0 and not C.GRAPHICS_PRESETS[C.QUALITY_LOW]["bloom"]


def test_material_layers_cover_every_surface_and_tile_seamlessly():
    albedo, relief = T.material_layers(64)
    count = len(maps.TEXTURE_LAYERS)
    assert albedo.shape == relief.shape == (count, 64, 64, 4) and albedo.dtype == relief.dtype == np.uint8
    assert all(maps.SURFACES[name][0] in maps.LAYER for name in maps.SURFACES)
    normal = relief[..., :3].astype(np.float64) / 127.5 - 1.0
    assert np.allclose(np.linalg.norm(normal, axis=-1), 1.0, atol=0.03) and normal[..., 2].min() > 0.05
    assert relief[..., 3].min() >= 80                                           # cavity never blacks a texel out
    for index, name in enumerate(maps.TEXTURE_LAYERS):
        layer = albedo[index, ..., :3].astype(np.float64)
        assert 12.0 < layer.mean() < 245.0 and layer.std() > 1.0, name          # neither blank nor flat
    # Tileable noise: the step across the wrap-around edge is no larger than a step inside the image.
    noise = T.tile_noise(128, 2.4, seed=3)
    inside = np.abs(np.diff(noise, axis=1)).mean()
    assert np.abs(noise[:, 0] - noise[:, -1]).mean() < inside * 1.6
    assert np.abs(noise[0, :] - noise[-1, :]).mean() < inside * 1.6
    again, _ = T.material_layers(64)
    assert np.array_equal(albedo, again)                                         # deterministic: caches stay valid


def test_height_fields_become_correctly_oriented_normal_maps():
    u = (np.arange(64) + 0.5) / 64.0
    ramp_u = np.tile(0.5 + 0.4 * np.sin(u * 2.0 * np.pi), (64, 1))               # height rises along +u in the middle
    normal = T.height_to_normal(ramp_u, 2.0)
    assert np.allclose(np.linalg.norm(normal, axis=-1), 1.0)
    assert normal[32, 0, 0] < -0.05 and abs(normal[32, 0, 1]) < 1e-9             # uphill along +u -> normal leans to -u
    assert normal[32, 32, 0] > 0.05
    assert np.allclose(T.height_to_normal(np.full((16, 16), 0.3), 5.0), [0.0, 0.0, 1.0])
    cavity = T.cavity_from_height(np.where(np.abs(u - 0.5) < 0.05, 0.0, 1.0)[None, :].repeat(64, axis=0))
    assert cavity[32, 32] < cavity[32, 4] <= 1.0                                 # the groove is darker than the plateau


def test_sky_panorama_follows_the_environment():
    day = maps.get_map_data("atrium").environment
    night = maps.get_map_data("datacore").environment
    sky = T.sky_panorama(256, day, seed=1)
    assert sky.shape == (128, 256, 4) and sky.dtype == np.uint8
    zenith, horizon = sky[120, :, :3].mean(axis=0), sky[66, :, :3].mean(axis=0)
    assert zenith[2] > zenith[0] and horizon.mean() > zenith.mean() * 0.9        # blue above, bright haze at the horizon
    assert T.sky_panorama(256, night, seed=1)[100, :, :3].mean() < sky[100, :, :3].mean() * 0.5
    assert np.abs(sky[:, 0, :3].astype(int) - sky[:, -1, :3].astype(int)).mean() < 6.0     # azimuth wraps seamlessly
    clear = T.sky_panorama(256, dict(day, clouds=0.0), seed=1)
    cloudy = T.sky_panorama(256, dict(day, clouds=0.9), seed=1)
    assert clear[..., 3].min() == 255 and cloudy[64:, :, 3].mean() < 235         # alpha = sky left visible by the clouds
    assert np.array_equal(cloudy[:60], T.sky_panorama(256, dict(day, clouds=0.9), seed=1)[:60])
    dx, dy, dz = T.sky_directions(64, 32)
    assert np.allclose(dx * dx + dy * dy + dz * dz, 1.0, atol=1e-5) and dy[0, 0] < -0.99 and dy[-1, 0] > 0.99


def test_world_mesh_is_wound_textured_and_culled():
    for key in ("crucible", "terminal"):
        loaded = maps.load_map(key)
        mesh = WM.build_world_mesh(loaded.data, loaded.world, 2.0)
        v, i = mesh["vertices"], mesh["indices"]
        assert v.dtype == WM.WORLD_VERTEX and i.dtype == np.uint32 and i.max() < v.shape[0] and i.size % 3 == 0
        tri = v["pos"][i.reshape(-1, 3)].astype(np.float64)
        face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal = v["nrm"][i.reshape(-1, 3)].astype(np.float64).mean(axis=1)
        area = np.linalg.norm(face, axis=1)
        assert np.all(np.sum(face * normal, axis=1)[area > 1e-9] < 0.0), key       # every triangle faces its normal
        assert np.allclose(np.linalg.norm(v["nrm"], axis=1), 1.0, atol=1e-4)
        assert np.isfinite(v["uv"]).all() and np.isfinite(v["light"]).all()
        layers = np.unique(v["uv"][:, 2])
        assert layers.min() >= 0 and layers.max() < len(maps.TEXTURE_LAYERS) and len(layers) >= 4
        assert 0.2 <= v["col"][:, 3].min() < 0.9 and v["col"][:, 3].max() == 1.0    # ambient occlusion is in use
        assert mesh["stats"]["faces_culled"] > 20
        sky = mesh["sky_vertices"]
        assert sky.shape[0] > 0 and np.all(sky["uv"][:, 3] >= WM.BACKDROP_FLAG)
        reach = np.abs(sky["pos"][:, [0, 2]]).max(axis=1)
        inner = np.abs(np.concatenate([loaded.data.bounds_min[[0, 2]], loaded.data.bounds_max[[0, 2]]])).max()
        assert reach.reshape(-1, 4).max(axis=1).min() > inner + 40.0               # towers stand well outside the arena
    finer = WM.build_world_mesh(loaded.data, loaded.world, 1.0)["stats"]["triangles"]
    assert finer > mesh["stats"]["triangles"]


def test_hidden_faces_and_texture_coordinates():
    box_min = np.array([[-5.0, -1.0, -5.0], [-1.0, 0.0, -1.0], [3.0, 0.5, 3.0]])
    box_max = np.array([[5.0, 0.0, 5.0], [1.0, 2.0, 1.0], [4.0, 1.5, 4.0]])       # floor, crate on it, floating box
    exposed = WM.exposed_faces(box_min, box_max)
    assert not exposed[1, 3] and exposed[1, [0, 1, 2, 4, 5]].all()                # only the crate's underside is buried
    assert exposed[0].all() and exposed[2].all()
    data = maps.build_map_data("uv", {"center": [(0.0, 0.0, 0.0, 8.0, 4.0, 2.0, "brick", -1),
                                                 (6.0, 0.0, 0.0, 1.5, 1.5, 1.5, "crate", -1)]})
    v, i, samples = WM.tessellate_boxes(data, np.arange(2), 1.0)
    brick = v[v["uv"][:, 2] == maps.LAYER["brick"]]
    side = brick[brick["nrm"][:, 2] > 0.5]                                         # the +z wall face
    tile = maps.SURFACES["brick"][2]
    assert np.allclose(side["uv"][:, 0], side["pos"][:, 0] / tile) and np.allclose(side["uv"][:, 1], side["pos"][:, 1] / tile)
    crate = v[v["uv"][:, 2] == maps.LAYER["crate"]]
    assert crate["uv"][:, :2].min() >= -1e-6 and crate["uv"][:, :2].max() <= 1.0 + 1e-6   # one repeat per crate face
    assert samples.shape == (v.shape[0], 3) and i.max() < v.shape[0]
    assert side.shape[0] == (8 + 1) * (4 + 1)                                      # 1 m cells on an 8 x 4 m face


def test_static_lights_are_baked_with_occlusion():
    data = maps.build_map_data("lit_room", LIT_ROOM)
    world = maps.build_collision_world(data)
    mesh = WM.build_world_mesh(data, world, 1.0)
    v = mesh["vertices"]
    floor = v[(v["nrm"][:, 1] > 0.9) & (np.abs(v["pos"][:, 1]) < 1e-4)]
    lit = floor[(floor["pos"][:, 0] < -1.5)]
    dark = floor[(floor["pos"][:, 0] > 1.5)]
    assert lit["light"].sum(axis=1).max() > 0.5 and lit["light"][:, 0].max() >= lit["light"][:, 2].max()   # warm lamp
    assert dark["light"].sum(axis=1).max() < 1e-6                                  # the wall blocks the lamp completely
    middle = np.abs(floor["pos"][:, 2]) < 4.0
    under_wall = floor[(np.abs(floor["pos"][:, 0]) < 0.1) & middle]
    beside_wall = floor[(np.abs(floor["pos"][:, 0] + 1.0) < 0.1) & middle]
    open_floor = floor[(floor["pos"][:, 0] < -7.0) & (np.abs(floor["pos"][:, 2]) < 2.0)]
    assert under_wall.shape[0] and beside_wall.shape[0] and open_floor.shape[0]
    assert under_wall["col"][:, 3].max() < 0.5                                     # buried vertices are fully occluded
    assert beside_wall["col"][:, 3].mean() < open_floor["col"][:, 3].mean() - 0.02  # contact shadow along the wall
    points = np.array([[-5.0, 0.1, 3.0], [5.0, 0.1, 3.0]])
    normals = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    energy = WM.bake_static_lights(world, points, normals, data.lights_pos, data.lights_color, data.lights_radius)
    assert energy[0].sum() > 1.0 and energy[1].sum() == 0.0


def test_world_mesh_cache_round_trip():
    key = "lit_room_cache"
    maps.register_map(key, LIT_ROOM)
    try:
        data = maps.get_map_data(key)
        first = WM.load_world_mesh(data, None, 2.0)
        second = WM.load_world_mesh(data, None, 2.0)
        assert not first["stats"]["cached"] and second["stats"]["cached"]
        assert np.array_equal(first["vertices"], second["vertices"]) and np.array_equal(first["indices"], second["indices"])
        assert list(C.BAKE_DIR.glob("mesh*_%s_*.npz" % key))
    finally:
        maps.unregister_map(key)
    assert not list(C.BAKE_DIR.glob("*_%s_*.npz" % key))                          # removing an arena removes its bakes


def run_frames(controller, milliseconds, seconds):
    for _ in range(int(seconds * 1000.0 / milliseconds)):
        controller.update(milliseconds / 1000.0)
    return controller.scale


def test_dynamic_resolution_drops_and_recovers():
    controller = pacing.ResolutionController(1.0, True, 0)
    assert controller.scale == 1.0 and controller.steps[0] == C.DYNRES_STEPS[0]
    assert run_frames(controller, 12.0, 5.0) == 1.0                              # comfortably inside the budget: no change
    assert run_frames(controller, 30.0, 3.0) < 1.0                               # 33 fps: step down ...
    assert run_frames(controller, 30.0, 20.0) == controller.steps[0]             # ... all the way if it stays slow
    assert run_frames(controller, 9.0, 30.0) == 1.0                              # headroom: climbs back to full resolution
    fixed = pacing.ResolutionController(0.85, False, 0)
    assert run_frames(fixed, 40.0, 10.0) == 0.85 and fixed.changes == 0          # switched off: never moves
    capped = pacing.ResolutionController(0.70, True, 0)
    assert capped.steps[-1] == 0.70 and max(capped.steps) == 0.70                # never exceeds the preset's ceiling


def test_dynamic_resolution_probes_upwards_under_a_frame_cap():
    controller = pacing.ResolutionController(1.0, True, 60.0)                    # a 60 fps limiter hides any headroom
    run_frames(controller, 30.0, 6.0)
    dropped = controller.scale
    assert dropped < 1.0
    run_frames(controller, 16.7, controller.probe_wait + C.DYNRES_COOLDOWN + 1.0)
    assert controller.scale > dropped                                            # quiet period over: one step up to probe
    wait = controller.probe_wait
    run_frames(controller, 30.0, 4.0)                                            # the probe fails ...
    assert controller.scale <= dropped + 1e-9 and controller.probe_wait > wait   # ... back down, and wait longer next time


def test_quality_governor_only_acts_when_resolution_is_exhausted():
    controller = pacing.ResolutionController(1.0, True, 0)
    governor = pacing.QualityGovernor()
    fired = [governor.update(controller, C.QUALITY_HIGH, 0.012) for _ in range(600)]
    assert not any(fired)
    run_frames(controller, 30.0, 20.0)
    assert controller.index == 0
    fired = [governor.update(controller, C.QUALITY_HIGH, 0.030) for _ in range(int(C.QUALITY_FALLBACK_SECONDS / 0.030) + 2)]
    assert fired.count(True) == 1
    assert not any(governor.update(controller, C.QUALITY_LOW, 0.030) for _ in range(400))   # nothing below LOW
