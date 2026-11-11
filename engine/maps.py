"""
NEXUS-3D-EXTREME :: engine/maps.py
==================================
Level construction, the three shipped arenas and bot navigation data.

A level is authored with :class:`MapBuilder` (axis aligned boxes, ramps, spawn
points, navigation waypoints, mood lights) and frozen into :class:`MapData`,
a bundle of contiguous arrays that the collision world, the renderer and the
bots all read directly.

Ramps deserve a note.  The kinematic controller only knows axis aligned boxes,
so a ramp is stored twice: as a visible wedge made of triangles that stops
bullets and grenades on its true surface, and as an invisible staircase of
thin boxes (rise ~0.14) that only the player layer sees.  The controller's
stair stepping turns that staircase into a smooth walk.

Arenas are laid out for one half and stamped a second time rotated by 180
degrees about the origin, which gives both teams an identical side.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np

from . import config as C
from . import mathx as M
from .controller_3d import CollisionWorld

MAT_CONCRETE, MAT_METAL, MAT_ACCENT, MAT_GLOW = range(4)

# palette ---------------------------------------------------------------------
COL_FLOOR = (0.17, 0.19, 0.23)
COL_WALL = (0.25, 0.28, 0.34)
COL_DARK = (0.10, 0.12, 0.16)
COL_PLATFORM = (0.31, 0.33, 0.38)
COL_RAMP = (0.38, 0.36, 0.31)
COL_CRATE = (0.62, 0.45, 0.18)
COL_PIPE = (0.42, 0.44, 0.47)
COL_GLOW = (0.25, 0.95, 1.00)
COL_TEAM = (tuple(float(v) for v in C.TEAM_COLORS[0]), tuple(float(v) for v in C.TEAM_COLORS[1]))

_OPPOSITE = {"+x": "-x", "-x": "+x", "+z": "-z", "-z": "+z"}


class MapData:
    """Immutable array bundle describing one arena."""

    def __init__(self):
        self.key = ""
        self.name = ""
        self.description = ""
        self.index = 0
        self.box_min = np.zeros((0, 3))
        self.box_max = np.zeros((0, 3))
        self.box_color = np.zeros((0, 3), dtype=np.float32)
        self.box_layer = np.zeros(0, dtype=np.uint8)
        self.box_visible = np.zeros(0, dtype=bool)
        self.box_material = np.zeros(0, dtype=np.uint8)
        self.mesh_tris = np.zeros((0, 3, 3))
        self.mesh_normal = np.zeros((0, 3))
        self.mesh_color = np.zeros((0, 3), dtype=np.float32)
        self.spawn_pos = np.zeros((0, 3))
        self.spawn_yaw = np.zeros(0)
        self.spawn_team = np.zeros(0, dtype=np.int64)
        self.waypoints = np.zeros((0, 3))
        self.links = np.zeros((0, 2), dtype=np.int64)
        self.bounds_min = np.zeros(3)
        self.bounds_max = np.zeros(3)
        self.kill_y = C.KILL_PLANE_Y
        self.sky_top = (0.03, 0.05, 0.10)
        self.sky_horizon = (0.20, 0.26, 0.36)
        self.fog_color = (0.12, 0.15, 0.20)
        self.fog_density = 0.012
        self.sun_dir = (0.35, -0.80, 0.48)
        self.sun_color = (1.00, 0.95, 0.86)
        self.ambient = (0.30, 0.33, 0.40)
        self.lights_pos = np.zeros((0, 3))
        self.lights_color = np.zeros((0, 3), dtype=np.float32)
        self.lights_radius = np.zeros(0)


class MapBuilder:
    """Imperative level authoring helper; call :meth:`finish` to obtain a :class:`MapData`."""

    def __init__(self, key: str, name: str, description: str):
        self.key = key
        self.name = name
        self.description = description
        self._boxes: list = []
        self._tris: list = []
        self._tri_normals: list = []
        self._tri_colors: list = []
        self._spawns: list = []
        self._waypoints: list = []
        self._links: list = []
        self._lights: list = []
        self._flip = False

    # ---- symmetry -------------------------------------------------------------
    @contextmanager
    def mirrored(self):
        """Everything authored inside is rotated 180 degrees about the origin."""
        previous = self._flip
        self._flip = not previous
        try:
            yield self
        finally:
            self._flip = previous

    def both(self, layout) -> None:
        """Stamp ``layout(builder, team)`` for team 0 and, mirrored, for team 1."""
        layout(self, 0)
        with self.mirrored():
            layout(self, 1)

    # ---- primitives --------------------------------------------------------------
    def _raw_box(self, x0, y0, z0, x1, y1, z1, color, layer, visible, material) -> None:
        lo = (min(x0, x1), min(y0, y1), min(z0, z1))
        hi = (max(x0, x1), max(y0, y1), max(z0, z1))
        self._boxes.append((lo, hi, tuple(color), int(layer), bool(visible), int(material)))

    def box(self, x0, y0, z0, x1, y1, z1, color=COL_WALL, layer=C.LAYER_ALL, visible=True,
            material=MAT_CONCRETE) -> None:
        """Solid axis aligned box given by two opposite corners."""
        if self._flip:
            x0, x1, z0, z1 = -x1, -x0, -z1, -z0
        self._raw_box(x0, y0, z0, x1, y1, z1, color, layer, visible, material)

    def block(self, cx, cz, sx, sz, height, base=0.0, color=COL_CRATE, material=MAT_CONCRETE) -> None:
        """Box of footprint ``sx`` x ``sz`` centred on ``(cx, cz)`` standing on ``base``."""
        self.box(cx - sx * 0.5, base, cz - sz * 0.5, cx + sx * 0.5, base + height, cz + sz * 0.5,
                 color=color, material=material)

    def glow(self, x0, y0, z0, x1, y1, z1, color=COL_GLOW) -> None:
        """Purely decorative emissive strip (no collision)."""
        self.box(x0, y0, z0, x1, y1, z1, color=color, layer=0, visible=True, material=MAT_GLOW)

    def ramp(self, x0, z0, x1, z1, y0, y1, direction: str, color=COL_RAMP, base=None,
             step_rise: float = 0.14) -> None:
        """
        Ramp over the footprint ``[x0, x1] x [z0, z1]`` climbing from ``y0`` to
        ``y1`` towards ``direction`` ('+x', '-x', '+z' or '-z').  ``base`` is
        the height the solid wedge extends down to (defaults to ``y0``).
        """
        if x0 > x1:
            x0, x1 = x1, x0
        if z0 > z1:
            z0, z1 = z1, z0
        if self._flip:
            x0, x1, z0, z1 = -x1, -x0, -z1, -z0
            direction = _OPPOSITE[direction]
        if base is None:
            base = y0
        rise = float(y1 - y0)
        axis = 0 if direction[1] == "x" else 2
        positive = direction[0] == "+"
        lo, hi = (x0, x1) if axis == 0 else (z0, z1)
        r_low, r_high = (lo, hi) if positive else (hi, lo)

        def point(run_fraction, y, across_fraction):
            run = r_low + (r_high - r_low) * run_fraction
            if axis == 0:
                return (run, y, z0 + (z1 - z0) * across_fraction)
            return (x0 + (x1 - x0) * across_fraction, y, run)

        a0, a1 = point(0.0, y0, 0.0), point(0.0, y0, 1.0)
        b0, b1 = point(1.0, y1, 0.0), point(1.0, y1, 1.0)
        c0, c1 = point(1.0, base, 0.0), point(1.0, base, 1.0)
        d0, d1 = point(0.0, base, 0.0), point(0.0, base, 1.0)
        quads = [(a0, a1, b1, b0), (b0, b1, c1, c0), (a0, b0, c0, d0), (a1, d1, c1, b1), (a0, d0, d1, a1)]
        centroid = np.mean(np.array([a0, a1, b0, b1, c0, c1, d0, d1], dtype=np.float64), axis=0)
        for quad in quads:
            q = np.array(quad, dtype=np.float64)
            for tri in (q[[0, 1, 2]], q[[0, 2, 3]]):
                normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                area = float(np.linalg.norm(normal))
                if area < 1.0e-9:
                    continue
                normal = normal / area
                if float(np.dot(normal, tri.mean(axis=0) - centroid)) < 0.0:
                    normal = -normal
                self._tris.append(tri)
                self._tri_normals.append(normal)
                self._tri_colors.append(tuple(color))

        steps = max(1, int(math.ceil(rise / step_rise)))
        for i in range(steps):
            top = y0 + (i + 0.5) / steps * rise
            if top - base < 0.01:
                continue
            ra = r_low + (r_high - r_low) * (i / steps)
            rb = r_low + (r_high - r_low) * ((i + 1) / steps)
            if axis == 0:
                self._raw_box(ra, base, z0, rb, top, z1, color, C.LAYER_PLAYER, False, MAT_CONCRETE)
            else:
                self._raw_box(x0, base, ra, x1, top, rb, color, C.LAYER_PLAYER, False, MAT_CONCRETE)

    # ---- gameplay markers ---------------------------------------------------------
    def spawn(self, team: int, x, z, yaw_deg, y=0.0) -> None:
        if self._flip:
            x, z, yaw_deg = -x, -z, yaw_deg + 180.0
        self._spawns.append(((float(x), float(y), float(z)), math.radians(yaw_deg), int(team)))

    def waypoint(self, x, z, y=0.0) -> int:
        if self._flip:
            x, z = -x, -z
        self._waypoints.append((float(x), float(y), float(z)))
        return len(self._waypoints) - 1

    def link(self, a: int, b: int) -> None:
        self._links.append((int(a), int(b)))

    def light(self, x, y, z, color, radius: float) -> None:
        if self._flip:
            x, z = -x, -z
        self._lights.append(((float(x), float(y), float(z)), tuple(color), float(radius)))

    # ---- output ----------------------------------------------------------------------
    def finish(self, **environment) -> MapData:
        data = MapData()
        data.key, data.name, data.description = self.key, self.name, self.description
        data.box_min = np.array([b[0] for b in self._boxes], dtype=np.float64).reshape(-1, 3)
        data.box_max = np.array([b[1] for b in self._boxes], dtype=np.float64).reshape(-1, 3)
        data.box_color = np.array([b[2] for b in self._boxes], dtype=np.float32).reshape(-1, 3)
        data.box_layer = np.array([b[3] for b in self._boxes], dtype=np.uint8)
        data.box_visible = np.array([b[4] for b in self._boxes], dtype=bool)
        data.box_material = np.array([b[5] for b in self._boxes], dtype=np.uint8)
        data.mesh_tris = np.array(self._tris, dtype=np.float64).reshape(-1, 3, 3)
        data.mesh_normal = np.array(self._tri_normals, dtype=np.float64).reshape(-1, 3)
        data.mesh_color = np.array(self._tri_colors, dtype=np.float32).reshape(-1, 3)
        data.spawn_pos = np.array([s[0] for s in self._spawns], dtype=np.float64).reshape(-1, 3)
        data.spawn_yaw = M.wrap_angle(np.array([s[1] for s in self._spawns], dtype=np.float64))
        data.spawn_team = np.array([s[2] for s in self._spawns], dtype=np.int64)
        data.waypoints = np.array(self._waypoints, dtype=np.float64).reshape(-1, 3)
        data.links = np.array(self._links, dtype=np.int64).reshape(-1, 2)
        solid = data.box_layer != 0
        data.bounds_min = data.box_min[solid].min(axis=0)
        data.bounds_max = data.box_max[solid].max(axis=0)
        lights = self._lights[: C.MAX_DYNAMIC_LIGHTS]
        data.lights_pos = np.array([l[0] for l in lights], dtype=np.float64).reshape(-1, 3)
        data.lights_color = np.array([l[1] for l in lights], dtype=np.float32).reshape(-1, 3)
        data.lights_radius = np.array([l[2] for l in lights], dtype=np.float64)
        for name, value in environment.items():
            if not hasattr(data, name):
                raise AttributeError("unknown map environment field %r" % name)
            setattr(data, name, value)
        return data


# =========================================================================== #
#  Arena 1: DATACORE  (three-lane server hall with an overlook catwalk)
# =========================================================================== #
def _build_datacore() -> MapData:
    b = MapBuilder("datacore", "DATACORE", "Three lanes through a humming server hall. Hold the centre console.")
    b.box(-32, -1, -22, 32, 0, 22, color=COL_FLOOR)
    b.box(-33, 0, 22, 33, 8, 23, color=COL_WALL)
    b.box(-33, 0, -23, 33, 8, -22, color=COL_WALL)
    b.box(-33, 0, -23, -32, 8, 23, color=COL_WALL)
    b.box(32, 0, -23, 33, 8, 23, color=COL_WALL)
    # centre console platform
    b.box(-5, 0, -4, 5, 1.2, 4, color=COL_PLATFORM, material=MAT_METAL)
    b.box(-0.8, 1.2, -0.8, 0.8, 3.0, 0.8, color=COL_DARK, material=MAT_METAL)
    b.glow(-0.85, 2.2, -0.85, 0.85, 2.35, 0.85)
    b.waypoint(0.0, 0.0 + 2.6, 1.2)
    b.waypoint(0.0, -2.6, 1.2)
    b.light(0.0, 5.0, 0.0, (0.55, 0.85, 1.00), 16.0)

    def half(m: MapBuilder, team: int) -> None:
        accent = COL_TEAM[team]
        # ramp up to the console
        m.ramp(-9, -2, -5, 2, 0.0, 1.2, "+x")
        # rack rows that separate the lanes (doorways left open)
        m.box(-22, 0, 7.4, -16, 2.8, 8.6, color=COL_DARK, material=MAT_METAL)
        m.box(-13, 0, 7.4, -8, 2.8, 8.6, color=COL_DARK, material=MAT_METAL)
        m.box(-22, 0, -8.6, -14, 2.8, -7.4, color=COL_DARK, material=MAT_METAL)
        m.box(-11, 0, -8.6, -8, 2.8, -7.4, color=COL_DARK, material=MAT_METAL)
        m.glow(-22, 1.9, 7.35, -16, 2.0, 8.65)
        m.glow(-13, 1.9, 7.35, -8, 2.0, 8.65)
        m.glow(-22, 1.9, -8.65, -14, 2.0, -7.35)
        m.glow(-11, 1.9, -8.65, -8, 2.0, -7.35)
        # spawn screen with a team stripe
        m.box(-25, 0, -6, -24.4, 3.0, 6, color=COL_WALL)
        m.glow(-24.38, 1.4, -6, -24.3, 1.7, 6, color=accent)
        m.glow(-25.1, 1.4, -6, -25.02, 1.7, 6, color=accent)
        # mid lane cover
        m.block(-14, 0, 1.4, 1.4, 1.0)
        m.block(-18, 3.4, 1.2, 2.4, 2.4, color=COL_DARK, material=MAT_METAL)
        m.block(-18, -3.4, 1.2, 2.4, 2.4, color=COL_DARK, material=MAT_METAL)
        # north lane cover
        m.block(-20, 13.5, 1.4, 1.4, 1.0)
        m.block(-10, 16.0, 2.4, 1.2, 1.0)
        m.block(-5, 11.5, 1.2, 1.2, 2.2, color=COL_DARK, material=MAT_METAL)
        # south lane cover
        m.block(-18, -15, 1.4, 1.4, 1.0)
        m.block(-9, -13, 1.4, 2.6, 2.2, color=COL_DARK, material=MAT_METAL)
        m.block(-3, -17, 2.4, 1.2, 1.0)
        # overlook catwalk along the north wall
        m.ramp(-24, 18, -16, 21, 0.0, 3.0, "+x")
        m.box(-16, 2.8, 17, -2, 3.0, 22, color=COL_PLATFORM, material=MAT_METAL)
        m.box(-16, 3.0, 17, -2, 3.9, 17.2, color=COL_PIPE, material=MAT_METAL)
        m.box(-15.8, 0, 17.2, -15.4, 2.8, 17.6, color=COL_PIPE, material=MAT_METAL)
        m.box(-2.6, 0, 17.2, -2.2, 2.8, 17.6, color=COL_PIPE, material=MAT_METAL)
        m.box(-9.2, 0, 17.2, -8.8, 2.8, 17.6, color=COL_PIPE, material=MAT_METAL)
        # spawns
        for z in (-9.0, -4.5, 0.0, 4.5, 9.0):
            m.spawn(team, -29.5, z, 90.0)
        for z in (-13.0, 13.0, -16.5):
            m.spawn(team, -27.0, z, 90.0)
        # navigation
        for x, z in ((-28, 0), (-28, 9), (-28, -9), (-24, 12.5), (-24, -12.5), (-22, 0),
                     (-17, 11.5), (-11, 13), (-3, 14.5), (-14, -12), (-5.5, -14.5),
                     (-14.5, 8), (-12.5, -8), (-10.5, 3.5), (-10.5, -3.5), (-9.8, 0), (0, 15.5)):
            m.waypoint(x, z)
        m.waypoint(-4.0, 0.0, 1.2)
        low = m.waypoint(-25.2, 19.5)
        top = m.waypoint(-15.0, 19.5, 3.0)
        end = m.waypoint(-4.0, 19.5, 3.0)
        m.link(low, top)
        m.link(top, end)
        m.light(-27.0, 4.0, 0.0, accent, 13.0)
        m.light(-14.0, 3.5, 12.0, (0.25, 0.80, 0.95), 11.0)
        m.light(-14.0, 3.5, -12.0, (0.25, 0.80, 0.95), 11.0)

    b.both(half)
    return b.finish(sky_top=(0.02, 0.03, 0.07), sky_horizon=(0.10, 0.16, 0.26), fog_color=(0.07, 0.10, 0.15),
                    fog_density=0.011, sun_dir=(0.30, -0.85, 0.42), sun_color=(0.85, 0.92, 1.00),
                    ambient=(0.30, 0.34, 0.42))


# =========================================================================== #
#  Arena 2: SKYBRIDGE  (two rooftops, a bridge and lower service decks)
# =========================================================================== #
def _build_skybridge() -> MapData:
    b = MapBuilder("skybridge", "SKYBRIDGE", "Rooftop duel. Cross the bridge or flank along the service decks.")
    # centre island
    b.box(-6, -1, -7, 6, 0, 7, color=COL_PLATFORM)
    b.box(-1, 0, -1, 1, 5.0, 1, color=COL_DARK, material=MAT_METAL)
    b.glow(-1.05, 3.8, -1.05, 1.05, 4.0, 1.05, color=(1.0, 0.35, 0.25))
    b.waypoint(0.0, 0.0 + 3.6)
    b.waypoint(0.0, -3.6)
    b.light(0.0, 4.5, 0.0, (1.00, 0.55, 0.35), 15.0)

    def half(m: MapBuilder, team: int) -> None:
        accent = COL_TEAM[team]
        # rooftop
        m.box(-34, -1, -10, -12, 0, 10, color=COL_FLOOR)
        m.box(-34, -1, 10, -32, 0, 15, color=COL_FLOOR)
        m.box(-34, -1, -15, -32, 0, -10, color=COL_FLOOR)
        # parapets (waist high, can be vaulted with a jump)
        m.box(-34.4, 0, -15, -34, 1.0, 15, color=COL_WALL)
        m.box(-32, 0, 10, -12, 1.0, 10.4, color=COL_WALL)
        m.box(-32, 0, -10.4, -12, 1.0, -10, color=COL_WALL)
        m.box(-12.4, 0, 2.2, -12, 1.0, 10, color=COL_WALL)
        m.box(-12.4, 0, -10, -12, 1.0, -2.2, color=COL_WALL)
        m.glow(-34.02, 0.45, -15, -33.96, 0.6, 15, color=accent)
        # bridge to the island with railings
        m.box(-12, -0.4, -2, -6, 0, 2, color=COL_PLATFORM, material=MAT_METAL)
        m.box(-12, 0, 2, -6, 0.9, 2.2, color=COL_PIPE, material=MAT_METAL)
        m.box(-12, 0, -2.2, -6, 0.9, -2, color=COL_PIPE, material=MAT_METAL)
        # island cover
        m.box(-4.5, 0, 3.0, -1.5, 1.0, 3.4, color=COL_WALL)
        m.block(-4.0, -4.0, 1.4, 1.4, 1.0)
        # service decks 2.5 below, north and south
        m.box(-26, -3.0, 11, 0, -2.5, 17, color=COL_FLOOR, material=MAT_METAL)
        m.box(-26, -3.0, -17, 0, -2.5, -11, color=COL_FLOOR, material=MAT_METAL)
        m.box(-26, -2.5, 17, 0, -1.6, 17.3, color=COL_PIPE, material=MAT_METAL)
        m.box(-26, -2.5, -17.3, 0, -1.6, -17, color=COL_PIPE, material=MAT_METAL)
        m.ramp(-32, 11, -26, 15, -2.5, 0.0, "-x", base=-3.0)
        m.ramp(-32, -15, -26, -11, -2.5, 0.0, "-x", base=-3.0)
        m.block(-14, 14, 1.4, 1.4, 1.0, base=-2.5)
        m.block(-8, -14, 2.6, 1.2, 1.0, base=-2.5)
        # deck -> island stairs (north side of this half; the mirror supplies the south)
        m.ramp(-2, 7, 2, 11, -2.5, 0.0, "-z", base=-3.0)
        # rooftop furniture
        m.block(-22, 0, 4.0, 4.0, 3.0, color=COL_WALL)
        m.glow(-20.02, 1.55, -1.0, -19.96, 1.80, 1.0, color=accent)
        m.block(-27, 6, 2.4, 1.4, 1.0, color=COL_PIPE, material=MAT_METAL)
        m.block(-27, -6, 2.4, 1.4, 1.0, color=COL_PIPE, material=MAT_METAL)
        m.block(-16, 6.5, 1.4, 1.4, 1.0)
        m.block(-16, -6.5, 1.4, 1.4, 1.0)
        m.block(-14.5, 0, 0.6, 2.4, 1.0, color=COL_WALL)
        for z in (-7.5, -4.5, -1.5, 1.5, 4.5, 7.5):
            m.spawn(team, -31.5, z, 90.0)
        m.spawn(team, -29.0, 8.5, 90.0)
        m.spawn(team, -29.0, -8.5, 90.0)
        for x, z in ((-30, 0), (-30, 7), (-30, -7), (-25.5, 3.5), (-25.5, -3.5), (-18.5, 4), (-18.5, -4),
                     (-13.2, 1.2), (-13.2, -1.2), (-9, 0), (-4.5, 0), (-3.5, 5.2), (-3.5, -5.2)):
            m.waypoint(x, z)
        n_top = m.waypoint(-33.0, 12.5)
        n_low = m.waypoint(-25.0, 13.0, -2.5)
        s_top = m.waypoint(-33.0, -12.5)
        s_low = m.waypoint(-25.0, -13.0, -2.5)
        m.link(n_top, n_low)
        m.link(s_top, s_low)
        for x, z in ((-18, 14), (-10, 13), (-3, 14), (-18, -14), (-10, -15.5), (-3, -14)):
            m.waypoint(x, z, -2.5)
        st_low = m.waypoint(0.0, 12.2, -2.5)
        st_top = m.waypoint(0.0, 6.0)
        m.link(st_low, st_top)
        m.light(-28.0, 3.5, 0.0, accent, 14.0)
        m.light(-12.0, -0.5, 14.0, (0.95, 0.80, 0.45), 10.0)
        m.light(-12.0, -0.5, -14.0, (0.95, 0.80, 0.45), 10.0)

    b.both(half)
    return b.finish(kill_y=-9.0, sky_top=(0.05, 0.09, 0.22), sky_horizon=(0.82, 0.48, 0.30),
                    fog_color=(0.40, 0.30, 0.28), fog_density=0.006, sun_dir=(0.55, -0.45, 0.70),
                    sun_color=(1.00, 0.80, 0.60), ambient=(0.36, 0.34, 0.40))


# =========================================================================== #
#  Arena 3: CRUCIBLE  (compact arena built for free-for-all)
# =========================================================================== #
def _build_crucible() -> MapData:
    b = MapBuilder("crucible", "CRUCIBLE", "A tight kill box around a raised core. Nowhere to rest.")
    b.box(-18, -1, -18, 18, 0, 18, color=COL_FLOOR)
    b.box(-19, 0, 18, 19, 7, 19, color=COL_WALL)
    b.box(-19, 0, -19, 19, 7, -18, color=COL_WALL)
    b.box(-19, 0, -19, -18, 7, 19, color=COL_WALL)
    b.box(18, 0, -19, 19, 7, 19, color=COL_WALL)
    b.box(-3, 0, -3, 3, 2.0, 3, color=COL_PLATFORM, material=MAT_METAL)
    b.box(-0.6, 2.0, -0.6, 0.6, 4.2, 0.6, color=COL_DARK, material=MAT_METAL)
    b.glow(-0.65, 3.4, -0.65, 0.65, 3.55, 0.65, color=(1.0, 0.8, 0.2))
    b.waypoint(0.0, 2.0, 2.0)
    b.waypoint(0.0, -2.0, 2.0)
    b.light(0.0, 5.5, 0.0, (1.00, 0.85, 0.50), 15.0)

    def half(m: MapBuilder, team: int) -> None:
        accent = COL_TEAM[team]
        m.ramp(-9, -1.5, -3, 1.5, 0.0, 2.0, "+x")
        # corner bunkers (L shaped)
        m.box(-13, 0, 9, -9, 2.6, 9.6, color=COL_WALL)
        m.box(-13, 0, 9.6, -12.4, 2.6, 13, color=COL_WALL)
        m.box(-13, 0, -9.6, -9, 2.6, -9, color=COL_WALL)
        m.box(-13, 0, -13, -12.4, 2.6, -9.6, color=COL_WALL)
        m.glow(-13, 1.2, 8.94, -9, 1.4, 9.0, color=accent)
        m.glow(-13, 1.2, -9.0, -9, 1.4, -8.94, color=accent)
        # scattered cover
        m.block(-7, 7, 1.4, 1.4, 1.0)
        m.block(-7, -7, 1.4, 1.4, 1.0)
        m.block(0, 10.5, 2.0, 2.0, 4.0, color=COL_DARK, material=MAT_METAL)
        m.box(-14.3, 0, -2, -13.7, 1.0, 2, color=COL_WALL)
        m.block(-4.5, 14.5, 2.6, 1.2, 1.0)
        m.spawn(team, -16.0, 0.0, 90.0)
        m.spawn(team, -16.0, 15.5, 135.0)
        m.spawn(team, -16.0, -15.5, 45.0)
        m.spawn(team, -10.5, 16.0, 150.0)
        m.spawn(team, -10.5, -16.0, 30.0)
        m.spawn(team, -16.0, 7.0, 100.0)
        for x, z in ((-15.8, 0), (-15.5, 15.5), (-15.5, -15.5), (-10.5, 5), (-10.5, -5), (-10.5, 11.5),
                     (-10.5, -11.5), (-6, 12), (-6, -12), (-4.5, 6), (-4.5, -6), (0, 14.5), (-9.8, 0), (-15.5, 8)):
            m.waypoint(x, z)
        m.waypoint(-2.2, 0.0, 2.0)
        m.light(-12.0, 3.5, 11.0, accent, 11.0)
        m.light(-12.0, 3.5, -11.0, accent, 11.0)

    b.both(half)
    return b.finish(sky_top=(0.08, 0.03, 0.03), sky_horizon=(0.42, 0.18, 0.10), fog_color=(0.18, 0.09, 0.07),
                    fog_density=0.014, sun_dir=(-0.40, -0.75, 0.52), sun_color=(1.00, 0.72, 0.50),
                    ambient=(0.34, 0.28, 0.28))


MAP_BUILDERS = (("datacore", _build_datacore), ("skybridge", _build_skybridge), ("crucible", _build_crucible))
MAP_KEYS = tuple(name for name, _ in MAP_BUILDERS)
MAP_NAMES = ("DATACORE", "SKYBRIDGE", "CRUCIBLE")
MAP_COUNT = len(MAP_KEYS)


# =========================================================================== #
#  Navigation graph
# =========================================================================== #
class NavGraph:
    """
    Directed waypoint graph with all-pairs shortest paths.

    ``next_hop[i, j]`` is the node to walk to from ``i`` in order to reach
    ``j`` (-1 when unreachable); ``dist[i, j]`` the path length.  Both tables
    come from a vectorised Floyd-Warshall pass, so path queries by any number
    of bots are a single fancy-index.
    """

    def __init__(self, nodes: np.ndarray, adjacency: np.ndarray):
        self.nodes = np.asarray(nodes, dtype=np.float64).reshape(-1, 3)
        count = self.nodes.shape[0]
        self.adjacency = np.asarray(adjacency, dtype=bool).reshape(count, count)
        delta = self.nodes[:, None, :] - self.nodes[None, :, :]
        length = np.sqrt(np.sum(delta * delta, axis=2))
        dist = np.where(self.adjacency, length, np.inf)
        np.fill_diagonal(dist, 0.0)
        nxt = np.where(self.adjacency, np.arange(count)[None, :], -1)
        np.fill_diagonal(nxt, np.arange(count))
        for k in range(count):
            alt = dist[:, k:k + 1] + dist[k:k + 1, :]
            better = alt < dist
            dist = np.where(better, alt, dist)
            nxt = np.where(better, nxt[:, k:k + 1], nxt)
        self.dist = dist
        self.next_hop = nxt.astype(np.int64)

    @property
    def count(self) -> int:
        return self.nodes.shape[0]

    def nearest(self, points, max_height: float = 2.2) -> np.ndarray:
        """Index of the closest node for each point, preferring nodes on the same storey."""
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        delta = points[:, None, :] - self.nodes[None, :, :]
        d2 = np.sum(delta * delta, axis=2)
        penalty = np.where(np.abs(delta[:, :, 1]) > max_height, 1.0e6, 0.0)
        return np.argmin(d2 + penalty, axis=1)

    def step_towards(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Next node on the shortest path ``src -> dst`` (``src`` itself when unreachable)."""
        hop = self.next_hop[src, dst]
        return np.where(hop < 0, src, hop)

    def is_connected(self) -> bool:
        return bool(np.all(np.isfinite(self.dist)))


def build_nav(data: MapData, world: CollisionWorld, link_radius: float = 14.0, sample_step: float = 0.4,
              max_drop: float = 3.6) -> NavGraph:
    """
    Derive the walkable waypoint graph of a level.

    A directed edge ``i -> j`` exists when the straight segment between the two
    waypoints is unobstructed at knee height and the ground profile sampled
    underneath never rises by more than a stair step (or falls by more than
    ``max_drop``) in the direction of travel.  All ray casts of all candidate
    pairs are issued as two octree batches.
    """
    nodes = data.waypoints
    count = nodes.shape[0]
    adjacency = np.zeros((count, count), dtype=bool)
    if count == 0:
        return NavGraph(nodes, adjacency)
    ii, jj = np.triu_indices(count, k=1)
    delta = nodes[jj] - nodes[ii]
    length = np.sqrt(np.sum(delta * delta, axis=1))
    near = (length <= link_radius) & (length > 1.0e-6)
    ii, jj, delta, length = ii[near], jj[near], delta[near], length[near]
    if ii.size:
        lift = np.array([0.0, 0.6, 0.0])
        clear = world.line_of_sight(nodes[ii] + lift, nodes[jj] + lift, C.LAYER_PLAYER)
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
        reach = 0.6 + max_drop + 0.6
        t, item, _ = world.raycast(origin, down, reach, C.LAYER_PLAYER)
        gap = item < 0
        ground = np.where(gap, 0.0, origin[:, 1] - t)
        inside = local[1:] > 0                       # differences that stay within one pair
        rise_fwd = np.where(inside, np.diff(ground), 0.0)
        pair_gap = np.bincount(pair, weights=gap, minlength=ii.size) > 0
        seg_pair = pair[1:]
        up_max = np.zeros(ii.size)
        down_max = np.zeros(ii.size)
        finite = np.isfinite(rise_fwd)
        np.maximum.at(up_max, seg_pair[finite], rise_fwd[finite])
        np.maximum.at(down_max, seg_pair[finite], -rise_fwd[finite])
        limit = C.STEP_HEIGHT + 0.04
        forward = ~pair_gap & (up_max <= limit) & (down_max <= max_drop)
        backward = ~pair_gap & (down_max <= limit) & (up_max <= max_drop)
        adjacency[ii[forward], jj[forward]] = True
        adjacency[jj[backward], ii[backward]] = True
    for a, c in data.links:
        adjacency[a, c] = True
        adjacency[c, a] = True
    return NavGraph(nodes, adjacency)


# =========================================================================== #
#  Loading, caching and validation
# =========================================================================== #
class LoadedMap:
    """A level together with its collision world and navigation graph."""

    def __init__(self, data: MapData):
        self.data = data
        self.world = build_collision_world(data)
        self.nav = build_nav(data, self.world)


def build_collision_world(data: MapData) -> CollisionWorld:
    solid = data.box_layer != 0
    return CollisionWorld(data.box_min[solid], data.box_max[solid], data.box_layer[solid],
                          triangles=data.mesh_tris if data.mesh_tris.shape[0] else None)


_MAP_CACHE: dict = {}


def map_index(key_or_index) -> int:
    if isinstance(key_or_index, str):
        return MAP_KEYS.index(key_or_index)
    return int(key_or_index) % MAP_COUNT


def get_map_data(key_or_index) -> MapData:
    index = map_index(key_or_index)
    data = MAP_BUILDERS[index][1]()
    data.index = index
    return data


def load_map(key_or_index) -> LoadedMap:
    """Build (once per process) and return the arena with its collision and nav data."""
    index = map_index(key_or_index)
    if index not in _MAP_CACHE:
        _MAP_CACHE[index] = LoadedMap(get_map_data(index))
    return _MAP_CACHE[index]


def validate_map(loaded: LoadedMap) -> list:
    """Return a list of human readable problems (empty when the level is sound)."""
    data, world = loaded.data, loaded.world
    problems: list = []
    limit = C.WORLD_LIMIT
    if np.any(data.bounds_min < -limit) or np.any(data.bounds_max > limit):
        problems.append("geometry leaves the +-%.0f quantisation range" % limit)
    radius, height = C.PLAYER_RADIUS, C.PLAYER_HEIGHT_STAND
    if data.spawn_pos.shape[0] == 0:
        problems.append("no spawn points")
    else:
        lo = data.spawn_pos + np.array([-radius, 0.02, -radius])
        hi = data.spawn_pos + np.array([radius, height, radius])
        qi, _ = world.query_player_boxes(lo + 1.0e-4, hi - 1.0e-4)
        for index in np.unique(qi).tolist():
            problems.append("spawn %d intersects solid geometry" % index)
        down = np.tile(np.array([0.0, -1.0, 0.0]), (data.spawn_pos.shape[0], 1))
        t, item, _ = world.raycast(data.spawn_pos + np.array([0.0, 0.5, 0.0]), down, 1.0, C.LAYER_PLAYER)
        for index in np.flatnonzero((item < 0) | (np.abs(t - 0.5) > 0.05)).tolist():
            problems.append("spawn %d is not standing on the ground" % index)
        for team in (0, 1):
            if not np.any(data.spawn_team == team):
                problems.append("team %d has no spawn point" % team)
    if data.waypoints.shape[0]:
        down = np.tile(np.array([0.0, -1.0, 0.0]), (data.waypoints.shape[0], 1))
        t, item, _ = world.raycast(data.waypoints + np.array([0.0, 0.5, 0.0]), down, 1.2, C.LAYER_PLAYER)
        for index in np.flatnonzero((item < 0) | (np.abs(t - 0.5) > 0.12)).tolist():
            problems.append("waypoint %d is not on the ground (%s)" % (index, np.round(data.waypoints[index], 2)))
        lo = data.waypoints + np.array([-radius, 0.1, -radius])
        hi = data.waypoints + np.array([radius, height, radius])
        qi, _ = world.query_player_boxes(lo, hi)
        for index in np.unique(qi).tolist():
            problems.append("waypoint %d is inside geometry (%s)" % (index, np.round(data.waypoints[index], 2)))
        if not loaded.nav.is_connected():
            unreachable = np.argwhere(~np.isfinite(loaded.nav.dist))
            sample = ", ".join("%d->%d" % (a, c) for a, c in unreachable[:6].tolist())
            problems.append("navigation graph is not strongly connected (%d pairs, e.g. %s)"
                            % (unreachable.shape[0], sample))
    else:
        problems.append("no navigation waypoints")
    return problems
