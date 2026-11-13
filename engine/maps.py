"""
NEXUS-3D-EXTREME :: engine/maps.py
==================================
Hot-pluggable map injector.

This module contains no level geometry.  It reads arena definitions -- nested
dictionaries of ``(x, y, z, scale_x, scale_y, scale_z, texture_type,
spawn_team)`` tuples -- from ``content/maps_registry.py`` and from any
``content/maps/*.json`` file, and turns them into the array bundle
(:class:`MapData`) that the collision world, the renderer and the bots read.

Pipeline for one arena::

    definition (data)  ->  expand prefabs / mirror the team half  ->  MapData arrays
                       ->  CollisionWorld (octree)
                       ->  navigation graph, generated automatically by sampling
                           walkable ground on a grid (no hand placed waypoints)

``register_map`` / ``reload_registry`` add or refresh arenas at run time; the
map list (``MAP_KEYS``, ``MAP_NAMES``, ``MAP_COUNT``) is evaluated on access so
menus and servers always see the current registry.

Ramps are stored twice: a visible wedge of triangles that stops bullets on its
true surface, and an invisible staircase of thin boxes that only the player
layer sees (the controller's stair stepping turns it into a smooth walk).
"""
from __future__ import annotations

import importlib
import json
import math
import zlib
from pathlib import Path

import numpy as np

from . import config as C
from . import mathx as M
from .controller_3d import CollisionWorld

# --------------------------------------------------------------------------- #
#  Materials (the order of TEXTURE_LAYERS is the layer order of the texture array)
# --------------------------------------------------------------------------- #
TEXTURE_LAYERS = ("concrete", "floor", "metal", "rack", "crate", "hazard", "grate", "roof", "brick",
                  "paint", "glow", "glass", "container", "tower", "armor")
LAYER = {name: index for index, name in enumerate(TEXTURE_LAYERS)}
UV_TILE, UV_FIT = 0, 1

# surface name -> (texture layer, uv mode, metres per texture repeat, tint)
SURFACES = {
    "concrete": ("concrete", UV_TILE, 4.0, (0.82, 0.84, 0.88)),
    "floor": ("floor", UV_TILE, 4.0, (0.80, 0.82, 0.88)),
    "metal": ("metal", UV_TILE, 2.0, (0.80, 0.83, 0.90)),
    "rack": ("rack", UV_TILE, 2.0, (0.60, 0.64, 0.72)),
    "crate": ("crate", UV_FIT, 1.0, (1.00, 0.93, 0.82)),
    "hazard": ("hazard", UV_TILE, 1.0, (1.00, 1.00, 1.00)),
    "grate": ("grate", UV_TILE, 1.0, (0.76, 0.80, 0.86)),
    "roof": ("roof", UV_TILE, 4.0, (0.72, 0.73, 0.76)),
    "brick": ("brick", UV_TILE, 2.0, (0.90, 0.78, 0.70)),
    "paint": ("paint", UV_TILE, 2.0, (0.86, 0.88, 0.92)),
    "glass": ("glass", UV_TILE, 4.0, (0.78, 0.90, 1.00)),
    "container": ("container", UV_TILE, 2.5, (0.78, 0.44, 0.32)),
    "tower": ("tower", UV_TILE, 12.0, (0.90, 0.94, 1.00)),
    "armor": ("armor", UV_TILE, 1.0, (0.70, 0.74, 0.80)),
}
GLOW_COLORS = {"cyan": (0.25, 0.95, 1.00), "warm": (1.00, 0.75, 0.35), "red": (1.00, 0.30, 0.22), "ice": (0.55, 0.85, 1.00)}
LIGHT_COLORS = {"ice": (0.55, 0.85, 1.00), "warm": (1.00, 0.72, 0.42), "red": (1.00, 0.35, 0.25), "cyan": (0.25, 0.80, 0.95)}
CONTAINER_COLORS = ((0.78, 0.34, 0.26), (0.26, 0.48, 0.74), (0.34, 0.62, 0.42), (0.86, 0.68, 0.28), (0.60, 0.62, 0.66))
MODEL_PROPS = {"barrel": ("barrel.obj", "metal", (0.62, 0.30, 0.22)), "pylon": ("pylon.obj", "hazard", (1.0, 1.0, 1.0))}
PREFABS = ("server_rack", "pillar", "barrier", "railing", "container_unit", "lamp_warm", "lamp_ice")
RAMPS = {"ramp_+x": "+x", "ramp_-x": "-x", "ramp_+z": "+z", "ramp_-z": "-z"}
_OPPOSITE = {"+x": "-x", "-x": "+x", "+z": "-z", "-z": "+z"}
TEAM_RGB = (tuple(float(v) for v in C.TEAM_COLORS[0]), tuple(float(v) for v in C.TEAM_COLORS[1]))
STAIR_RISE = 0.14
NAV_REVISION = 2                # bump when the automatic graph generation changes (invalidates cached graphs)


def known_texture_types() -> tuple:
    """Every ``texture_type`` string the injector understands."""
    return (tuple(SURFACES) + PREFABS + tuple(MODEL_PROPS) + tuple(RAMPS)
            + tuple("glow_" + name for name in GLOW_COLORS) + ("glow_team", "paint_team")
            + tuple("light_" + name for name in LIGHT_COLORS) + ("light_team", "spawn"))


# --------------------------------------------------------------------------- #
#  Arena data bundle
# --------------------------------------------------------------------------- #
class MapData:
    """Array bundle describing one arena (filled by :class:`MapAssembler`)."""

    def __init__(self):
        self.key = ""
        self.name = ""
        self.description = ""
        self.index = 0
        self.checksum = 0
        self.box_min = np.zeros((0, 3))
        self.box_max = np.zeros((0, 3))
        self.box_color = np.zeros((0, 3), dtype=np.float32)
        self.box_layer = np.zeros(0, dtype=np.uint8)          # collision layers (0 = decor)
        self.box_visible = np.zeros(0, dtype=bool)
        self.box_texture = np.zeros(0, dtype=np.int32)        # index into TEXTURE_LAYERS
        self.box_uv_mode = np.zeros(0, dtype=np.int32)
        self.box_tile = np.ones(0, dtype=np.float32)
        self.box_emissive = np.zeros(0, dtype=np.float32)
        self.mesh_tris = np.zeros((0, 3, 3))                  # wedges and loaded models
        self.mesh_normal = np.zeros((0, 3, 3))                # per-vertex normals
        self.mesh_uv = np.zeros((0, 3, 2), dtype=np.float32)
        self.mesh_color = np.zeros((0, 3), dtype=np.float32)
        self.mesh_texture = np.zeros(0, dtype=np.int32)
        self.mesh_solid = np.zeros(0, dtype=bool)             # True: triangle stops bullets (ramps)
        self.spawn_pos = np.zeros((0, 3))
        self.spawn_yaw = np.zeros(0)
        self.spawn_team = np.zeros(0, dtype=np.int64)
        self.waypoints = np.zeros((0, 3))                     # optional hand placed nodes (normally empty)
        self.nav_hints = np.zeros((0, 3))                     # centre line samples of ramps (seed the auto graph)
        self.links = np.zeros((0, 2), dtype=np.int64)
        self.bounds_min = np.zeros(3)
        self.bounds_max = np.zeros(3)
        self.lights_pos = np.zeros((0, 3))
        self.lights_color = np.zeros((0, 3), dtype=np.float32)
        self.lights_radius = np.zeros(0)
        self.environment = {}
        for name, value in _environment_defaults().items():
            setattr(self, name, value)


def _environment_defaults() -> dict:
    from content import maps_registry
    return dict(maps_registry.ENVIRONMENT_DEFAULTS)


# --------------------------------------------------------------------------- #
#  OBJ mesh loader
# --------------------------------------------------------------------------- #
_OBJ_CACHE: dict = {}


def load_obj(path) -> dict:
    """
    Minimal Wavefront OBJ reader (positions, texture coordinates, normals,
    polygon faces).  Returns per-triangle arrays ``pos (T, 3, 3)``,
    ``nrm (T, 3, 3)`` and ``uv (T, 3, 2)``; missing normals are generated flat.
    """
    path = Path(path)
    stamp = (str(path), path.stat().st_mtime)
    if stamp in _OBJ_CACHE:
        return _OBJ_CACHE[stamp]
    positions, texcoords, normals, corners = [], [], [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "v":
            positions.append([float(v) for v in parts[1:4]])
        elif parts[0] == "vt":
            texcoords.append([float(v) for v in parts[1:3]])
        elif parts[0] == "vn":
            normals.append([float(v) for v in parts[1:4]])
        elif parts[0] == "f":
            face = []
            for token in parts[1:]:
                fields = (token.split("/") + ["", ""])[:3]
                index = [int(f) if f else 0 for f in fields]
                face.append([i - 1 if i > 0 else (len(src) + i if i < 0 else -1)
                             for i, src in zip(index, (positions, texcoords, normals))])
            for k in range(1, len(face) - 1):
                corners.append((face[0], face[k], face[k + 1]))
    if not corners:
        raise ValueError("%s contains no faces" % path)
    idx = np.array(corners, dtype=np.int64)                              # (T, 3, 3): position, uv, normal index
    pos = np.array(positions, dtype=np.float64)[idx[:, :, 0]]
    if texcoords and np.all(idx[:, :, 1] >= 0):
        uv = np.array(texcoords, dtype=np.float32)[idx[:, :, 1]]
    else:
        uv = np.zeros(idx.shape[:2] + (2,), dtype=np.float32)
    if normals and np.all(idx[:, :, 2] >= 0):
        nrm = M.normalize(np.array(normals, dtype=np.float64)[idx[:, :, 2]])
    else:
        flat = M.normalize(M.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0]))
        centre = pos.reshape(-1, 3).mean(axis=0)
        outward = M.dot(flat, pos.mean(axis=1) - centre) < 0.0
        flat[outward] *= -1.0
        nrm = np.repeat(flat[:, None, :], 3, axis=1)
    mesh = {"pos": pos, "nrm": nrm, "uv": uv}
    _OBJ_CACHE[stamp] = mesh
    return mesh


# --------------------------------------------------------------------------- #
#  Definition -> MapData
# --------------------------------------------------------------------------- #
class MapAssembler:
    """Expands registry objects (surfaces, prefabs, ramps, lights, spawns) into primitive arrays."""

    def __init__(self):
        self.boxes: list = []
        self.tris: list = []
        self.spawns: list = []
        self.lights: list = []
        self.hints: list = []

    # ---- primitives ---------------------------------------------------------------
    def box(self, cx, y0, cz, sx, sy, sz, surface="concrete", tint=None, collide=C.LAYER_ALL, visible=True,
            emissive=0.0, tile=None, uv_mode=None) -> None:
        layer_name, mode, tiling, base_tint = SURFACES.get(surface, (surface, UV_TILE, 2.0, (1.0, 1.0, 1.0)))
        lo = (cx - sx * 0.5, y0, cz - sz * 0.5)
        hi = (cx + sx * 0.5, y0 + sy, cz + sz * 0.5)
        self.boxes.append((lo, hi, tuple(tint if tint is not None else base_tint), int(collide), bool(visible),
                           LAYER[layer_name], mode if uv_mode is None else uv_mode,
                           float(tile if tile is not None else tiling), float(emissive)))

    def triangle(self, corners, normals, uvs, tint, texture, solid) -> None:
        self.tris.append((np.asarray(corners, dtype=np.float64), np.asarray(normals, dtype=np.float64),
                          np.asarray(uvs, dtype=np.float32), tuple(tint), int(texture), bool(solid)))

    def ramp(self, cx, y0, cz, sx, rise, sz, direction, tint=(0.80, 0.80, 0.84)) -> None:
        """Walkable wedge over the footprint, climbing ``rise`` towards ``direction``."""
        x0, x1, z0, z1, y1 = cx - sx * 0.5, cx + sx * 0.5, cz - sz * 0.5, cz + sz * 0.5, y0 + rise
        axis = 0 if direction[1] == "x" else 2
        lo, hi = (x0, x1) if axis == 0 else (z0, z1)
        r_low, r_high = (lo, hi) if direction[0] == "+" else (hi, lo)

        def point(run, y, across):
            r = r_low + (r_high - r_low) * run
            return (r, y, z0 + (z1 - z0) * across) if axis == 0 else (x0 + (x1 - x0) * across, y, r)

        a0, a1, b0, b1 = point(0, y0, 0), point(0, y0, 1), point(1, y1, 0), point(1, y1, 1)
        c0, c1 = point(1, y0, 0), point(1, y0, 1)
        centroid = np.mean(np.array([a0, a1, b0, b1, c0, c1], dtype=np.float64), axis=0)
        run_length = abs(r_high - r_low)
        slope_length = math.hypot(run_length, rise)
        width = (z1 - z0) if axis == 0 else (x1 - x0)
        faces = (((a0, a1, b1, b0), ((0, 0), (0, width), (slope_length, width), (slope_length, 0)), "hazard"),
                 ((b0, b1, c1, c0), ((0, rise), (width, rise), (width, 0), (0, 0)), "metal"),
                 ((a0, b0, c0), ((0, 0), (run_length, rise), (run_length, 0)), "metal"),
                 ((a1, c1, b1), ((0, 0), (run_length, 0), (run_length, rise)), "metal"))
        for corners, uvs, surface in faces:
            quad = np.array(corners, dtype=np.float64)
            uv = np.array(uvs, dtype=np.float32) / (1.0 if surface == "hazard" else 2.0)
            order = ((0, 1, 2), (0, 2, 3)) if len(corners) == 4 else ((0, 1, 2),)
            for tri in order:
                p = quad[list(tri)]
                normal = np.cross(p[1] - p[0], p[2] - p[0])
                area = float(np.linalg.norm(normal))
                if area < 1.0e-9:
                    continue
                normal /= area
                if float(np.dot(normal, p.mean(axis=0) - centroid)) < 0.0:
                    normal = -normal
                self.triangle(p, np.tile(normal, (3, 1)), uv[list(tri)], tint, LAYER[SURFACES[surface][0]], True)
        segments = max(2, int(math.ceil((run_length + 1.8) / 2.5)))
        for i in range(segments + 1):
            run = (-0.9 + (run_length + 1.8) * i / segments) / run_length
            px, _, pz = point(run, 0.0, 0.5)
            self.hints.append((px, y0 + rise * min(max(run, 0.0), 1.0), pz))
        steps = max(1, int(math.ceil(rise / STAIR_RISE)))
        for i in range(steps):
            top = y0 + (i + 0.5) / steps * rise
            ra = r_low + (r_high - r_low) * (i / steps)
            rb = r_low + (r_high - r_low) * ((i + 1) / steps)
            if axis == 0:
                self.box((ra + rb) * 0.5, y0, cz, abs(rb - ra), top - y0, sz, collide=C.LAYER_PLAYER, visible=False)
            else:
                self.box(cx, y0, (ra + rb) * 0.5, sx, top - y0, abs(rb - ra), collide=C.LAYER_PLAYER, visible=False)

    def model(self, cx, y0, cz, sx, sy, sz, name) -> None:
        """Instance an OBJ prop scaled into the box; the box itself is the collision volume."""
        filename, surface, tint = MODEL_PROPS[name]
        mesh = load_obj(C.MODELS_DIR / filename)
        pos = mesh["pos"]
        lo, hi = pos.reshape(-1, 3).min(axis=0), pos.reshape(-1, 3).max(axis=0)
        scale = np.array([sx, sy, sz]) / np.maximum(hi - lo, 1.0e-9)
        origin = np.array([cx, y0, cz]) - np.array([(lo[0] + hi[0]) * 0.5, lo[1], (lo[2] + hi[2]) * 0.5]) * scale
        world = pos * scale + origin
        normal = M.normalize(mesh["nrm"] / scale)
        texture = LAYER[SURFACES[surface][0]]
        for t in range(world.shape[0]):
            self.triangle(world[t], normal[t], mesh["uv"][t], tint, texture, False)
        self.box(cx, y0, cz, sx * 0.9, sy, sz * 0.9, collide=C.LAYER_ALL, visible=False)

    # ---- prefabs -----------------------------------------------------------------------
    def server_rack(self, cx, y0, cz, sx, sy, sz) -> None:
        self.box(cx, y0, cz, sx, sy, sz, "rack")
        self.box(cx, y0, cz, sx + 0.06, 0.08, sz + 0.06, "metal", tint=(0.35, 0.37, 0.42), collide=0)
        self.box(cx, y0 + sy - 0.05, cz, sx + 0.05, 0.06, sz + 0.05, "metal", tint=(0.35, 0.37, 0.42), collide=0)
        self.box(cx, y0 + sy * 0.68, cz, sx + 0.05, 0.07, sz + 0.05, "glow", tint=GLOW_COLORS["cyan"], collide=0,
                 emissive=1.0)
        self.box(cx, y0 + sy * 0.30, cz, sx + 0.03, 0.03, sz + 0.03, "glow", tint=(0.20, 1.00, 0.45), collide=0,
                 emissive=0.8)

    def pillar(self, cx, y0, cz, sx, sy, sz) -> None:
        self.box(cx, y0, cz, sx, sy, sz, "concrete")
        cap = min(0.30, sy * 0.12)
        self.box(cx, y0, cz, sx + 0.14, cap, sz + 0.14, "concrete", tint=(0.66, 0.68, 0.72), collide=0)
        self.box(cx, y0 + sy - cap, cz, sx + 0.14, cap, sz + 0.14, "concrete", tint=(0.66, 0.68, 0.72), collide=0)
        self.box(cx, y0 + cap, cz, sx + 0.02, 0.10, sz + 0.02, "hazard", collide=0)

    def barrier(self, cx, y0, cz, sx, sy, sz) -> None:
        base = sy * 0.46
        self.box(cx, y0, cz, sx, base, sz, "concrete", tint=(0.74, 0.76, 0.80))
        if sx < sz:
            self.box(cx, y0 + base, cz, sx * 0.55, sy - base, sz, "concrete", tint=(0.84, 0.86, 0.90))
            self.box(cx, y0 + base * 0.35, cz, sx + 0.02, 0.10, sz * 0.96, "hazard", collide=0)
        else:
            self.box(cx, y0 + base, cz, sx, sy - base, sz * 0.55, "concrete", tint=(0.84, 0.86, 0.90))
            self.box(cx, y0 + base * 0.35, cz, sx * 0.96, 0.10, sz + 0.02, "hazard", collide=0)

    def railing(self, cx, y0, cz, sx, sy, sz) -> None:
        self.box(cx, y0, cz, sx, sy, sz, "metal", tint=(0.52, 0.56, 0.64))
        along_x = sx >= sz
        length = sx if along_x else sz
        thick = (sz if along_x else sx) + 0.06
        self.box(cx, y0 + sy, cz, sx + (0.0 if along_x else 0.06), 0.06, sz + (0.06 if along_x else 0.0), "metal",
                 tint=(0.88, 0.90, 0.95), collide=0)
        posts = max(2, int(round(length / 2.0)) + 1)
        for i in range(posts):
            offset = -length * 0.5 + length * i / (posts - 1)
            px, pz = (cx + offset, cz) if along_x else (cx, cz + offset)
            self.box(px, y0, pz, thick if not along_x else 0.10, sy + 0.08, thick if along_x else 0.10, "metal",
                     tint=(0.30, 0.32, 0.37), collide=0)

    def container_unit(self, cx, y0, cz, sx, sy, sz) -> None:
        tint = CONTAINER_COLORS[int(abs(round(cx * 3.1 + cz * 7.3 + y0))) % len(CONTAINER_COLORS)]
        self.box(cx, y0, cz, sx, sy, sz, "container", tint=tint)
        dark = tuple(v * 0.55 for v in tint)
        for dx in (-0.5, 0.5):
            for dz in (-0.5, 0.5):
                self.box(cx + dx * sx, y0, cz + dz * sz, 0.14, sy, 0.14, "metal", tint=dark, collide=0)
        for level in (0.0, sy - 0.10):
            self.box(cx, y0 + level, cz, sx + 0.08, 0.10, sz + 0.08, "metal", tint=dark, collide=0)

    def lamp(self, cx, y0, cz, sx, sy, sz, color_name) -> None:
        color = LIGHT_COLORS[color_name]
        self.box(cx, y0, cz, sx, sy, sz, "metal", tint=(0.30, 0.32, 0.36))
        self.box(cx, y0 + sy, cz, 0.9, 0.12, 0.9, "metal", tint=(0.25, 0.26, 0.30), collide=0)
        self.box(cx, y0 + sy - 0.05, cz, 0.7, 0.05, 0.7, "glow", tint=color, collide=0, emissive=1.0)
        self.lights.append(((cx, y0 + sy - 0.4, cz), color, 12.0))

    # ---- one registry object ----------------------------------------------------------------
    def place(self, entry, team: int, flipped: bool) -> None:
        x, y, z, sx, sy, sz, kind, spawn_team = entry
        if flipped:
            x, z = -x, -z
        if kind == "spawn":
            owner = (team if spawn_team == 0 else 1 - team) if spawn_team in (0, 1) else team
            self.spawns.append(((x, y, z), math.atan2(-x, -z), owner))
        elif kind in SURFACES:
            self.box(x, y, z, sx, sy, sz, kind)
        elif kind in RAMPS:
            direction = RAMPS[kind]
            self.ramp(x, y, z, sx, sy, sz, _OPPOSITE[direction] if flipped else direction)
        elif kind in MODEL_PROPS:
            self.model(x, y, z, sx, sy, sz, kind)
        elif kind.startswith("lamp_"):
            self.lamp(x, y, z, sx, sy, sz, kind[5:])
        elif kind in PREFABS:
            getattr(self, kind)(x, y, z, sx, sy, sz)
        elif kind == "paint_team":
            self.box(x, y, z, sx, sy, sz, "paint", tint=TEAM_RGB[team], collide=0)
        elif kind.startswith("glow_"):
            color = TEAM_RGB[team] if kind == "glow_team" else GLOW_COLORS[kind[5:]]
            self.box(x, y, z, sx, sy, sz, "glow", tint=color, collide=0, emissive=1.0)
        elif kind.startswith("light_"):
            color = TEAM_RGB[team] if kind == "light_team" else LIGHT_COLORS[kind[6:]]
            self.lights.append(((x, y, z), color, float(sx)))
        else:
            raise ValueError("unknown texture_type %r" % (kind,))

    # ---- output ---------------------------------------------------------------------------------
    def finish(self, data: MapData) -> MapData:
        b = self.boxes
        data.box_min = np.array([r[0] for r in b], dtype=np.float64).reshape(-1, 3)
        data.box_max = np.array([r[1] for r in b], dtype=np.float64).reshape(-1, 3)
        data.box_color = np.array([r[2] for r in b], dtype=np.float32).reshape(-1, 3)
        data.box_layer = np.array([r[3] for r in b], dtype=np.uint8)
        data.box_visible = np.array([r[4] for r in b], dtype=bool)
        data.box_texture = np.array([r[5] for r in b], dtype=np.int32)
        data.box_uv_mode = np.array([r[6] for r in b], dtype=np.int32)
        data.box_tile = np.array([r[7] for r in b], dtype=np.float32)
        data.box_emissive = np.array([r[8] for r in b], dtype=np.float32)
        t = self.tris
        data.mesh_tris = np.array([r[0] for r in t], dtype=np.float64).reshape(-1, 3, 3)
        data.mesh_normal = np.array([r[1] for r in t], dtype=np.float64).reshape(-1, 3, 3)
        data.mesh_uv = np.array([r[2] for r in t], dtype=np.float32).reshape(-1, 3, 2)
        data.mesh_color = np.array([r[3] for r in t], dtype=np.float32).reshape(-1, 3)
        data.mesh_texture = np.array([r[4] for r in t], dtype=np.int32)
        data.mesh_solid = np.array([r[5] for r in t], dtype=bool)
        data.spawn_pos = np.array([s[0] for s in self.spawns], dtype=np.float64).reshape(-1, 3)
        data.spawn_yaw = M.wrap_angle(np.array([s[1] for s in self.spawns], dtype=np.float64))
        data.spawn_team = np.array([s[2] for s in self.spawns], dtype=np.int64)
        data.nav_hints = np.array(self.hints, dtype=np.float64).reshape(-1, 3)
        lights = self.lights[: C.MAX_STATIC_LIGHTS]
        data.lights_pos = np.array([l[0] for l in lights], dtype=np.float64).reshape(-1, 3)
        data.lights_color = np.array([l[1] for l in lights], dtype=np.float32).reshape(-1, 3)
        data.lights_radius = np.array([l[2] for l in lights], dtype=np.float64)
        solid = data.box_layer != 0
        data.bounds_min = data.box_min[solid].min(axis=0)
        data.bounds_max = data.box_max[solid].max(axis=0)
        return data


def _normalise_definition(key: str, definition: dict) -> dict:
    """Validate a registry entry and convert JSON lists to tuples."""
    if not isinstance(definition, dict):
        raise ValueError("map %r: definition must be a dictionary" % key)
    known = set(known_texture_types())
    clean = {"name": str(definition.get("name", key.upper())), "description": str(definition.get("description", "")),
             "mirror": bool(definition.get("mirror", False)), "environment": dict(definition.get("environment", {})),
             "center": [], "half": []}
    for section in ("center", "half"):
        for position, entry in enumerate(definition.get(section, ())):
            if len(entry) != 8:
                raise ValueError("map %r: %s[%d] needs 8 values (x, y, z, sx, sy, sz, texture_type, spawn_team)"
                                 % (key, section, position))
            numbers = tuple(float(v) for v in entry[:6])
            kind, team = str(entry[6]), int(entry[7])
            if kind not in known:
                raise ValueError("map %r: %s[%d] has unknown texture_type %r" % (key, section, position, kind))
            clean[section].append(numbers + (kind, team))
    if not clean["center"] and not clean["half"]:
        raise ValueError("map %r contains no objects" % key)
    unknown = set(clean["environment"]) - set(_environment_defaults())
    if unknown:
        raise ValueError("map %r: unknown environment keys %s" % (key, sorted(unknown)))
    return clean


def build_map_data(key: str, definition: dict, index: int = 0) -> MapData:
    """Turn one registry definition into the arrays the engine consumes."""
    spec = _normalise_definition(key, definition)
    assembler = MapAssembler()
    for entry in spec["center"]:
        assembler.place(entry, 0, False)
    for entry in spec["half"]:
        assembler.place(entry, 0, False)
    if spec["mirror"]:
        for entry in spec["half"]:
            assembler.place(entry, 1, True)
    data = assembler.finish(MapData())
    data.key, data.name, data.description, data.index = key, spec["name"], spec["description"], index
    environment = _environment_defaults()
    environment.update(spec["environment"])
    data.environment = environment
    for name, value in environment.items():
        setattr(data, name, tuple(value) if isinstance(value, (list, tuple)) else value)
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    data.checksum = zlib.crc32(canonical.encode("utf-8")) & 0xFFFFFFFF
    return data


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict = {}
_MAP_CACHE: dict = {}
_DATA_CACHE: dict = {}
REGISTRY_ERRORS: list = []


def reload_registry() -> tuple:
    """
    (Re)read ``content/maps_registry.py`` and every ``content/maps/*.json``.
    Returns the registered keys.  Broken external files are skipped and
    reported in ``REGISTRY_ERRORS`` instead of taking the game down.
    """
    from content import maps_registry
    importlib.reload(maps_registry)
    _REGISTRY.clear()
    _MAP_CACHE.clear()
    _DATA_CACHE.clear()
    REGISTRY_ERRORS.clear()
    ordered = [key for key in maps_registry.MAP_ORDER if key in maps_registry.MAPS]
    ordered += sorted(key for key in maps_registry.MAPS if key not in ordered)
    for key in ordered:
        _REGISTRY[key] = _normalise_definition(key, maps_registry.MAPS[key])
    if C.MAPS_DIR.is_dir():
        for path in sorted(C.MAPS_DIR.glob("*.json")):
            try:
                definition = json.loads(path.read_text(encoding="utf-8"))
                key = str(definition.get("key", path.stem)).lower()
                _REGISTRY[key] = _normalise_definition(key, definition)
            except (ValueError, OSError, AttributeError) as error:
                REGISTRY_ERRORS.append("%s: %s" % (path.name, error))
    return tuple(_REGISTRY)


def register_map(key: str, definition: dict) -> int:
    """Inject (or replace) an arena at run time; returns its index."""
    _ensure_registry()
    key = key.lower()
    _REGISTRY[key] = _normalise_definition(key, definition)
    _MAP_CACHE.pop(key, None)
    _DATA_CACHE.pop(key, None)
    return tuple(_REGISTRY).index(key)


def unregister_map(key: str) -> None:
    """Remove an arena from the registry together with its cached navigation, mesh and sky bakes."""
    key = key.lower()
    _REGISTRY.pop(key, None)
    _MAP_CACHE.pop(key, None)
    _DATA_CACHE.pop(key, None)
    purge_cache(key)


def purge_cache(key: str) -> int:
    """Delete every baked file of the arena ``key``; returns how many were removed."""
    removed = 0
    if C.BAKE_DIR.is_dir():
        for path in C.BAKE_DIR.glob("*_%s_*.npz" % key.lower()):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def _ensure_registry() -> None:
    if not _REGISTRY:
        reload_registry()


def map_keys() -> tuple:
    _ensure_registry()
    return tuple(_REGISTRY)


def __getattr__(name: str):
    """``MAP_KEYS`` / ``MAP_NAMES`` / ``MAP_COUNT`` always reflect the live registry."""
    if name == "MAP_KEYS":
        return map_keys()
    if name == "MAP_NAMES":
        _ensure_registry()
        return tuple(definition["name"] for definition in _REGISTRY.values())
    if name == "MAP_COUNT":
        return len(map_keys())
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def map_index(key_or_index) -> int:
    keys = map_keys()
    if isinstance(key_or_index, str):
        return keys.index(key_or_index.lower())
    return int(key_or_index) % len(keys)


def get_map_data(key_or_index) -> MapData:
    """Array bundle of an arena (cached; geometry only, no collision or navigation)."""
    index = map_index(key_or_index)
    key = map_keys()[index]
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = build_map_data(key, _REGISTRY[key], index)
    data = _DATA_CACHE[key]
    data.index = index
    return data


# --------------------------------------------------------------------------- #
#  Navigation graph
# --------------------------------------------------------------------------- #
class NavGraph:
    """
    Directed waypoint graph with all-pairs shortest paths.

    ``next_hop[i, j]`` is the node to walk to from ``i`` in order to reach
    ``j`` (-1 when unreachable); ``dist[i, j]`` the path length (float32).
    Both tables come from a vectorised Floyd-Warshall pass, so path queries by
    any number of bots are a single fancy-index.
    """

    def __init__(self, nodes: np.ndarray, adjacency: np.ndarray, dist=None, next_hop=None):
        self.nodes = np.asarray(nodes, dtype=np.float64).reshape(-1, 3)
        count = self.nodes.shape[0]
        self.adjacency = np.asarray(adjacency, dtype=bool).reshape(count, count)
        self._nodes_t = np.ascontiguousarray(self.nodes.T)                 # (3, N) for the matrix product in nearest()
        self._half_norm = 0.5 * np.sum(self.nodes * self.nodes, axis=1)
        if dist is not None and next_hop is not None:
            self.dist = np.asarray(dist, dtype=np.float32).reshape(count, count)
            self.next_hop = np.asarray(next_hop, dtype=np.int32).reshape(count, count)
            return
        delta = self.nodes[:, None, :] - self.nodes[None, :, :]
        length = np.sqrt(np.sum(delta * delta, axis=2))
        dist = np.where(self.adjacency, length, np.inf)
        np.fill_diagonal(dist, 0.0)
        nxt = np.where(self.adjacency, np.arange(count, dtype=np.int32)[None, :], np.int32(-1)).astype(np.int32)
        np.fill_diagonal(nxt, np.arange(count, dtype=np.int32))
        alt = np.empty_like(dist)
        better = np.empty(dist.shape, dtype=bool)
        for k in range(count):
            np.add(dist[:, k:k + 1], dist[k:k + 1, :], out=alt)
            np.less(alt, dist, out=better)
            np.copyto(dist, alt, where=better)
            np.copyto(nxt, nxt[:, k:k + 1], where=better)
        self.dist = dist.astype(np.float32)
        self.next_hop = nxt

    @property
    def count(self) -> int:
        return self.nodes.shape[0]

    def nearest(self, points, max_height: float = 2.2) -> np.ndarray:
        """Index of the closest node for each point, preferring nodes on the same storey."""
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        score = self._half_norm[None, :] - points @ self._nodes_t          # |p - n|^2 / 2 without the constant |p|^2 / 2
        score[np.abs(points[:, 1:2] - self._nodes_t[1][None, :]) > max_height] += 1.0e6
        return np.argmin(score, axis=1)

    def step_towards(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Next node on the shortest path ``src -> dst`` (``src`` itself when unreachable)."""
        hop = self.next_hop[src, dst]
        return np.where(hop < 0, src, hop)

    def is_connected(self) -> bool:
        return bool(np.all(np.isfinite(self.dist)))


def _standing_room(world: CollisionWorld, points: np.ndarray) -> np.ndarray:
    """True where a standing player fits at ``points`` (obstacles below stair height are ignored)."""
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    radius = C.PLAYER_RADIUS + 0.10
    qmin = points + np.array([-radius, C.STEP_HEIGHT + 0.05, -radius])
    qmax = points + np.array([radius, C.PLAYER_HEIGHT_STAND, radius])
    blocked, _ = world.query_player_boxes(qmin, qmax)
    free = np.ones(points.shape[0], dtype=bool)
    free[np.unique(blocked)] = False
    return free


def _drop_to_ground(world: CollisionWorld, points: np.ndarray, reach: float = 2.0) -> np.ndarray:
    """Project candidate nodes onto the walkable surface right below them; rejects the rest."""
    if points.shape[0] == 0:
        return np.zeros((0, 3))
    origin = points + np.array([0.0, 0.9, 0.0])
    down = np.tile(np.array([0.0, -1.0, 0.0]), (points.shape[0], 1))
    t, item, _ = world.raycast(origin, down, 0.9 + reach, C.LAYER_PLAYER)
    hit = (item >= 0) & (t > 1.0e-6)
    ground = origin[hit]
    ground[:, 1] -= t[hit]
    return ground[_standing_room(world, ground)]


def _far_from(points: np.ndarray, others: np.ndarray, distance: float) -> np.ndarray:
    """True for every point that is at least ``distance`` away from all ``others``."""
    if points.shape[0] == 0 or others.shape[0] == 0:
        return np.ones(points.shape[0], dtype=bool)
    delta = points[:, None, :] - others[None, :, :]
    return np.min(np.sum(delta * delta, axis=2), axis=1) >= distance * distance


def generate_waypoints(data: MapData, world: CollisionWorld, spacing: float = 2.0, levels: int = 4) -> np.ndarray:
    """
    Sample walkable ground automatically.

    * A regular grid (symmetric about the origin).  For every grid column a ray
      is cast down repeatedly, so storeys stacked on top of each other
      (catwalks, mezzanines, lower decks) each get their node.  A sample that
      is too close to an obstacle is retried a short step to the side, so
      corridors whose walls happen to lie on grid lines are still covered.
    * The centre line of every ramp and of every narrow walkable box (bridges,
      catwalks), which a coarse grid can miss entirely.

    A sample is kept when a standing player fits there.
    """
    lo, hi = data.bounds_min, data.bounds_max
    nx = int(max(abs(lo[0]), abs(hi[0])) // spacing)
    nz = int(max(abs(lo[2]), abs(hi[2])) // spacing)
    gx, gz = np.meshgrid(np.arange(-nx, nx + 1) * spacing, np.arange(-nz, nz + 1) * spacing, indexing="ij")
    gx, gz = gx.ravel(), gz.ravel()
    inside = (gx > lo[0]) & (gx < hi[0]) & (gz > lo[2]) & (gz < hi[2])
    gx, gz = gx[inside], gz[inside]
    start = np.full(gx.shape[0], hi[1] + 1.0)
    alive = np.ones(gx.shape[0], dtype=bool)
    floor_limit = max(float(data.kill_y), float(lo[1]) - 1.0)
    down = np.array([0.0, -1.0, 0.0])
    layers = []
    for _ in range(levels):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        origin = np.stack([gx[idx], start[idx], gz[idx]], axis=1)
        t, item, _ = world.raycast(origin, np.tile(down, (idx.size, 1)), start[idx] - floor_limit, C.LAYER_PLAYER)
        hit = item >= 0
        alive[idx[~hit]] = False
        idx, t, item, origin = idx[hit], t[hit], item[hit], origin[hit]
        if idx.size == 0:
            break
        point = origin.copy()
        point[:, 1] -= t
        surface = t > 1.0e-6
        free = surface & _standing_room(world, point)
        layers.append(point[free])
        retry = point[surface & ~free]
        if retry.shape[0]:
            sign = np.where(retry[:, [0, 2]] < 0.0, -1.0, 1.0)             # keeps the node set point symmetric
            placed = np.zeros(retry.shape[0], dtype=bool)
            for ox, oz in ((0.35, 0.0), (0.0, 0.35), (-0.35, 0.0), (0.0, -0.35),
                           (0.35, 0.35), (-0.35, 0.35), (0.35, -0.35), (-0.35, -0.35)):
                todo = np.flatnonzero(~placed)
                if todo.size == 0:
                    break
                moved = retry[todo].copy()
                moved[:, 0] += ox * spacing * sign[todo, 0]
                moved[:, 2] += oz * spacing * sign[todo, 1]
                origin2 = moved + np.array([0.0, 0.5, 0.0])
                t2, item2, _ = world.raycast(origin2, np.tile(down, (todo.size, 1)), 0.5 + C.STEP_HEIGHT,
                                             C.LAYER_PLAYER)
                moved[:, 1] = origin2[:, 1] - t2
                good = (item2 >= 0) & (t2 > 1.0e-6)
                good[good] = _standing_room(world, moved[good])
                layers.append(moved[good])
                placed[todo[good]] = True
        start[idx] = world.octree.item_min[item, 1] - 0.05          # continue below this surface
        alive[idx[start[idx] <= floor_limit]] = False
    grid = np.concatenate(layers) if layers else np.zeros((0, 3))

    ramp_nodes = _drop_to_ground(world, data.nav_hints)
    grid = grid[_far_from(grid, ramp_nodes, 1.0)]

    walkable = (data.box_layer & C.LAYER_PLAYER) != 0
    size = data.box_max[walkable] - data.box_min[walkable]
    centre = (data.box_max[walkable] + data.box_min[walkable]) * 0.5
    top = data.box_max[walkable][:, 1]
    small, large = np.minimum(size[:, 0], size[:, 2]), np.maximum(size[:, 0], size[:, 2])
    narrow = np.flatnonzero((small >= 0.8) & (small < 4.0) & (large >= 1.5))
    samples = []
    for b in narrow.tolist():
        count = max(1, int(math.ceil(large[b] / 2.5)))
        offset = ((np.arange(count) + 0.5) / count - 0.5) * large[b]
        line = np.tile(np.array([centre[b, 0], top[b], centre[b, 2]]), (count, 1))
        line[:, 0 if size[b, 0] >= size[b, 2] else 2] += offset
        samples.append(line)
    extra = _drop_to_ground(world, np.concatenate(samples), reach=0.05) if samples else np.zeros((0, 3))
    extra = extra[_far_from(extra, np.concatenate([grid, ramp_nodes]), 1.2)]
    if extra.shape[0] > 1:                                               # thin the centre line samples among themselves
        keep = np.ones(extra.shape[0], dtype=bool)
        for i in range(1, extra.shape[0]):
            keep[i] = bool(_far_from(extra[i:i + 1], extra[:i][keep[:i]], 1.2)[0])
        extra = extra[keep]
    return np.concatenate([grid, ramp_nodes, extra])


def link_nodes(nodes: np.ndarray, world: CollisionWorld, links=(), link_radius: float = 14.0,
               sample_step: float = 0.4, max_drop: float = 3.6) -> np.ndarray:
    """
    Derive the walkable adjacency matrix over ``nodes``.

    A directed edge ``i -> j`` exists when three parallel rays at knee height
    and three at head height (the silhouette of a player) are unobstructed and
    the ground profile sampled under the segment never rises by more than a
    stair step, nor falls by more than ``max_drop``, in the direction of
    travel.  Every ray of every candidate pair goes through the octree in
    batches.
    """
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1, 3)
    count = nodes.shape[0]
    adjacency = np.zeros((count, count), dtype=bool)
    if count == 0:
        return adjacency
    gap = nodes[:, None, :] - nodes[None, :, :]
    near = (np.sum(gap * gap, axis=2) <= link_radius * link_radius) & (np.abs(gap[:, :, 1]) <= max_drop)
    ii, jj = np.nonzero(np.triu(near, k=1))
    delta = nodes[jj] - nodes[ii]
    length = np.sqrt(np.sum(delta * delta, axis=1))
    keep = length > 1.0e-6
    ii, jj, delta, length = ii[keep], jj[keep], delta[keep], length[keep]
    if ii.size:
        side = M.normalize(np.stack([delta[:, 2], np.zeros(ii.size), -delta[:, 0]], axis=1)) * (C.PLAYER_RADIUS - 0.05)
        clear = np.ones(ii.size, dtype=bool)
        for height in (0.6, C.PLAYER_HEIGHT_STAND - 0.3):
            lift = np.array([0.0, height, 0.0])
            for offset in (0.0, 1.0, -1.0):
                clear &= world.line_of_sight(nodes[ii] + lift + side * offset, nodes[jj] + lift + side * offset,
                                             C.LAYER_PLAYER)
        ii, jj, delta, length = ii[clear], jj[clear], delta[clear], length[clear]
    if ii.size:
        samples = np.maximum(np.ceil(length / sample_step).astype(np.int64), 1) + 1
        total = int(samples.sum())
        pair = np.repeat(np.arange(ii.size), samples)
        first = np.cumsum(samples) - samples
        local = np.arange(total) - np.repeat(first, samples)
        frac = local / np.repeat(samples - 1, samples)
        origin = nodes[ii][pair] + delta[pair] * frac[:, None]
        origin[:, 1] += 0.6
        down = np.zeros_like(origin)
        down[:, 1] = -1.0
        t, item, _ = world.raycast(origin, down, 0.6 + max_drop + 0.6, C.LAYER_PLAYER)
        gap = item < 0
        ground = np.where(gap, 0.0, origin[:, 1] - t)
        inside = local[1:] > 0
        rise = np.where(inside, np.diff(ground), 0.0)
        pair_gap = np.bincount(pair, weights=gap, minlength=ii.size) > 0
        up_max = np.zeros(ii.size)
        down_max = np.zeros(ii.size)
        np.maximum.at(up_max, pair[1:], rise)
        np.maximum.at(down_max, pair[1:], -rise)
        limit = C.STEP_HEIGHT + 0.04
        forward = ~pair_gap & (up_max <= limit) & (down_max <= max_drop)
        backward = ~pair_gap & (down_max <= limit) & (up_max <= max_drop)
        adjacency[ii[forward], jj[forward]] = True
        adjacency[jj[backward], ii[backward]] = True
    for a, c in links:
        adjacency[a, c] = True
        adjacency[c, a] = True
    return adjacency


def build_nav(nodes: np.ndarray, world: CollisionWorld, links=(), link_radius: float = 14.0,
              sample_step: float = 0.4, max_drop: float = 3.6) -> NavGraph:
    """Walkable graph over ``nodes`` with all-pairs shortest paths (see :func:`link_nodes`)."""
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1, 3)
    return NavGraph(nodes, link_nodes(nodes, world, links, link_radius, sample_step, max_drop))


def _reachable(adjacency: np.ndarray, start: int) -> np.ndarray:
    """Breadth-first reachability from ``start`` along directed edges."""
    seen = np.zeros(adjacency.shape[0], dtype=bool)
    seen[start] = True
    frontier = seen.copy()
    while frontier.any():
        frontier = adjacency[frontier].any(axis=0) & ~seen
        seen |= frontier
    return seen


def build_arena_nav(data: MapData, world: CollisionWorld, spacing: float = 2.0) -> NavGraph:
    """
    Navigation graph for an arena: hand placed waypoints when the definition
    provides them, otherwise an automatically generated grid.  Only the part
    of the graph that is mutually reachable from the first spawn point is
    kept, so crate tops and other islands never become bot destinations.
    """
    if data.waypoints.shape[0]:
        return build_nav(data.waypoints, world, data.links)
    nodes = generate_waypoints(data, world, spacing)
    adjacency = link_nodes(nodes, world, (), link_radius=spacing * 2.3)
    if nodes.shape[0] and data.spawn_pos.shape[0]:
        offset = nodes - data.spawn_pos[0]
        anchor = int(np.argmin(np.sum(offset * offset, axis=1)))
        kept = np.flatnonzero(_reachable(adjacency, anchor) & _reachable(adjacency.T, anchor))
        nodes, adjacency = nodes[kept], adjacency[np.ix_(kept, kept)]
    return NavGraph(nodes, adjacency)


def _cached_nav(data: MapData, world: CollisionWorld) -> NavGraph:
    """Navigation graphs are deterministic per definition, so they are cached on disk by checksum."""
    path = C.BAKE_DIR / ("nav%d_%s_%08x.npz" % (NAV_REVISION, data.key, data.checksum))
    if path.exists():
        try:
            with np.load(path) as stored:
                return NavGraph(stored["nodes"], stored["adjacency"], stored["dist"], stored["next_hop"])
        except (OSError, KeyError, ValueError):
            pass
    graph = build_arena_nav(data, world)
    try:
        C.BAKE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, nodes=graph.nodes, adjacency=graph.adjacency, dist=graph.dist,
                            next_hop=graph.next_hop.astype(np.int16 if graph.count < 32000 else np.int32))
    except OSError:
        pass
    return graph


# --------------------------------------------------------------------------- #
#  Loading, caching and validation
# --------------------------------------------------------------------------- #
def build_collision_world(data: MapData) -> CollisionWorld:
    solid = data.box_layer != 0
    triangles = data.mesh_tris[data.mesh_solid]
    return CollisionWorld(data.box_min[solid], data.box_max[solid], data.box_layer[solid],
                          triangles=triangles if triangles.shape[0] else None)


class LoadedMap:
    """An arena together with its collision world and navigation graph."""

    def __init__(self, data: MapData, use_cache: bool = True):
        self.data = data
        self.world = build_collision_world(data)
        self.nav = _cached_nav(data, self.world) if use_cache else build_arena_nav(data, self.world)


def load_map(key_or_index) -> LoadedMap:
    """Build (once per process) and return an arena with collision and navigation data."""
    index = map_index(key_or_index)
    key = map_keys()[index]
    if key not in _MAP_CACHE:
        _MAP_CACHE[key] = LoadedMap(get_map_data(index))
    _MAP_CACHE[key].data.index = index
    return _MAP_CACHE[key]


def validate_map(loaded: LoadedMap) -> list:
    """Return a list of human readable problems (empty when the arena is sound)."""
    data, world, nav = loaded.data, loaded.world, loaded.nav
    problems: list = []
    limit = C.WORLD_LIMIT
    if np.any(data.bounds_min < -limit) or np.any(data.bounds_max > limit):
        problems.append("geometry leaves the +-%.0f quantisation range" % limit)
    radius, height = C.PLAYER_RADIUS, C.PLAYER_HEIGHT_STAND
    if data.spawn_pos.shape[0] == 0:
        problems.append("no spawn points")
        return problems
    lo = data.spawn_pos + np.array([-radius, 0.02, -radius])
    hi = data.spawn_pos + np.array([radius, height, radius])
    qi, _ = world.query_player_boxes(lo + 1.0e-4, hi - 1.0e-4)
    for index in np.unique(qi).tolist():
        problems.append("spawn %d %s intersects solid geometry" % (index, np.round(data.spawn_pos[index], 2)))
    down = np.tile(np.array([0.0, -1.0, 0.0]), (data.spawn_pos.shape[0], 1))
    t, item, _ = world.raycast(data.spawn_pos + np.array([0.0, 0.5, 0.0]), down, 1.0, C.LAYER_PLAYER)
    for index in np.flatnonzero((item < 0) | (np.abs(t - 0.5) > 0.05)).tolist():
        problems.append("spawn %d %s is not standing on the ground" % (index, np.round(data.spawn_pos[index], 2)))
    for team in (0, 1):
        if not np.any(data.spawn_team == team):
            problems.append("team %d has no spawn point" % team)
    if nav.count < 8:
        problems.append("navigation graph has only %d nodes" % nav.count)
        return problems
    if not nav.is_connected():
        problems.append("navigation graph is not strongly connected")
    nearest = nav.nearest(data.spawn_pos)
    gap = np.sqrt(np.sum((nav.nodes[nearest] - data.spawn_pos) ** 2, axis=1))
    for index in np.flatnonzero(gap > 4.5).tolist():
        problems.append("spawn %d %s is %.1f m from the navigation graph" % (index, np.round(data.spawn_pos[index], 2),
                                                                             gap[index]))
    return problems
