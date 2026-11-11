"""
NEXUS-3D-EXTREME :: engine/renderer.py                 (MODULE C)
=================================================================
Fragment shaders, real-time HUD and vectorised "juice".

Everything the player sees is produced here, on top of raw Panda3D geometry
attached to Ursina's scene graph:

* custom GLSL vertex / fragment programs compiled straight into the Panda3D
  pipeline (world lighting with a 1/d^2 point-light array, sky, effect sprites,
  view-model, and a screen-space post process pass);
* the whole level as one static vertex buffer built with NumPy;
* every dynamic effect -- remote players, particles, tracers, bullet holes,
  floating damage numbers -- as structure-of-arrays pools that are turned into
  vertex data with broadcasting and uploaded as one buffer per pool;
* an orthographic HUD (dynamic crosshair, hit markers, kill feed, ...).

World coordinates follow Ursina 8 (Y-up, left-handed).  With that coordinate
system a triangle faces the viewer when ``dot(cross(p1 - p0, p2 - p0), n) < 0``
for its outward normal ``n``; the box tables below are laid out accordingly.
"World up" is +Y, the axis stock (Z-up) Panda3D calls the global Z axis.
"""
from __future__ import annotations

import math

import numpy as np
from direct.filter.FilterManager import FilterManager
from panda3d.core import (Camera as PandaCamera, ColorBlendAttrib, Geom, GeomEnums, GeomNode, GeomTriangles,
                          GeomVertexArrayFormat, GeomVertexData, GeomVertexFormat, InternalName, LVecBase4f, NodePath,
                          OmniBoundingVolume, PerspectiveLens, PTA_LVecBase4f, Shader, Texture)
from ursina import Entity, Text, application, camera, color, scene, window

from . import config as C
from . import mathx as M
from . import maps
from . import simulation as S
from . import weapons as W
from .controller_3d import (HITBOX_COUNT, HITBOX_CROUCH_CENTER, HITBOX_CROUCH_HALF, HITBOX_STAND_CENTER,
                            HITBOX_STAND_HALF)

# =========================================================================== #
#  GLSL programs
# =========================================================================== #
LIGHT_SLOTS = C.MAX_DYNAMIC_LIGHTS * 2          # first half: muzzle / explosion flashes, second half: level lights

WORLD_VERT = """#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;
out vec3 v_world;
out vec3 v_normal;
out vec4 v_color;
out vec2 v_uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_world = (p3d_ModelMatrix * p3d_Vertex).xyz;
    v_normal = mat3(p3d_ModelMatrix) * p3d_Normal;
    v_color = p3d_Color;
    v_uv = p3d_MultiTexCoord0;
}
"""

WORLD_FRAG = """#version 140
const int LIGHTS = %d;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_ambient;
uniform vec3 u_fog_color;
uniform float u_fog_density;
uniform vec3 u_camera_pos;
uniform vec4 u_light_pos[LIGHTS];      // xyz = position, w = I0
uniform vec4 u_light_col[LIGHTS];      // rgb = colour, a = 1 / radius^2 window (0 = unbounded)
in vec3 v_world;
in vec3 v_normal;
in vec4 v_color;
in vec2 v_uv;                          // x = material id, y = additive flash
out vec4 fragColor;

void main() {
    vec3 n = normalize(v_normal);
    vec3 an = abs(n);
    vec2 p = an.y > 0.5 ? v_world.xz : (an.x > 0.5 ? v_world.zy : v_world.xy);
    float material = v_uv.x;
    vec3 albedo = v_color.rgb;
    vec3 to_cam = u_camera_pos - v_world;
    float dist = length(to_cam);
    if (material > 2.5) {                                   // emissive strips ignore scene lighting
        vec3 glow = albedo * (1.35 + 0.25 * sin(v_world.x * 0.7 + v_world.z * 0.9));
        float haze = 1.0 - exp(-u_fog_density * dist * 0.5);
        fragColor = vec4(mix(glow, u_fog_color, haze), v_color.a);
        return;
    }
    if (material < 1.5) {                                   // procedural panel seams, anti-aliased with fwidth
        vec2 q = p * (material > 0.5 ? 1.0 : 0.5);
        vec2 g = abs(fract(q) - 0.5);
        float fw = max(fwidth(q.x), fwidth(q.y));
        float seam = smoothstep(0.47 - fw, 0.47, max(g.x, g.y)) * (1.0 - clamp(fw * 6.0, 0.0, 1.0));
        albedo *= 1.0 - 0.22 * seam;
        albedo *= 0.96 + 0.04 * sin(p.x * 0.37) * sin(p.y * 0.29);
    }
    float ndl = max(dot(n, -u_sun_dir), 0.0);
    vec3 light = u_ambient * (0.60 + 0.40 * n.y) + u_sun_color * ndl * 0.80;
    for (int i = 0; i < LIGHTS; i++) {
        vec3 d = u_light_pos[i].xyz - v_world;
        float d2 = dot(d, d);
        float fall = u_light_pos[i].w / max(d2, 0.20);      // Intensity = I0 / d^2
        float window = clamp(1.0 - d2 * u_light_col[i].a, 0.0, 1.0);
        float facing = max(dot(n, d * inversesqrt(max(d2, 0.0001))), 0.0);
        light += u_light_col[i].rgb * (fall * window * window * (0.25 + 0.75 * facing));
    }
    vec3 col = albedo * light;
    if (material > 0.5 && material < 1.5) {                 // brushed metal: a small sun glint
        vec3 v = to_cam / max(dist, 0.001);
        vec3 h = normalize(v - u_sun_dir);
        col += u_sun_color * pow(max(dot(n, h), 0.0), 28.0) * 0.25;
    }
    col += vec3(v_uv.y);
    float fog = 1.0 - exp(-u_fog_density * dist);
    fragColor = vec4(mix(col, u_fog_color, fog), v_color.a);
}
""" % LIGHT_SLOTS

SKY_VERT = """#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
out vec3 v_dir;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_dir = p3d_Vertex.xyz;
}
"""

SKY_FRAG = """#version 140
uniform vec3 u_sky_top;
uniform vec3 u_sky_horizon;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
in vec3 v_dir;
out vec4 fragColor;
void main() {
    vec3 d = normalize(v_dir);
    vec3 col = mix(u_sky_horizon, u_sky_top, smoothstep(-0.08, 0.65, d.y));
    float sun = max(dot(d, -u_sun_dir), 0.0);
    col += u_sun_color * (pow(sun, 350.0) * 1.5 + pow(sun, 12.0) * 0.18);
    fragColor = vec4(col, 1.0);
}
"""

# Effect sprites.  normal.x selects the shape: 0 soft disc, 1 beam (soft across v), 2 solid quad.
FX_VERT = """#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;
out vec4 v_color;
out vec2 v_uv;
out float v_shape;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_color = p3d_Color;
    v_uv = p3d_MultiTexCoord0;
    v_shape = p3d_Normal.x;
}
"""

FX_FRAG = """#version 140
in vec4 v_color;
in vec2 v_uv;
in float v_shape;
out vec4 fragColor;
void main() {
    float a = v_color.a;
    if (v_shape < 0.5) {
        a *= smoothstep(1.0, 0.2, length(v_uv - 0.5) * 2.0);
    } else if (v_shape < 1.5) {
        a *= smoothstep(1.0, 0.1, abs(v_uv.y - 0.5) * 2.0);
    }
    fragColor = vec4(v_color.rgb * a, a);                   // premultiplied alpha
}
"""

VIEWMODEL_FRAG = """#version 140
uniform float u_flash;
uniform vec3 u_tint;
in vec3 v_world;
in vec3 v_normal;
in vec4 v_color;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec3 n = normalize(v_normal);
    if (v_uv.x > 2.5) {
        fragColor = vec4(v_color.rgb * 1.5, 1.0);
        return;
    }
    float key = max(dot(n, normalize(vec3(-0.35, 0.80, -0.45))), 0.0);
    vec3 col = v_color.rgb * (u_tint * 0.95 + vec3(1.05) * key + vec3(0.18));
    col += v_color.rgb * vec3(1.9, 1.35, 0.75) * u_flash * (0.35 + 0.65 * max(n.z, 0.0));
    fragColor = vec4(col, 1.0);
}
"""

POST_VERT = """#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uv = p3d_MultiTexCoord0;
}
"""

# Screen-space pass: recoil kick-back, chromatic aberration, shake, damage vignette.
POST_FRAG = """#version 140
uniform sampler2D tex;
uniform vec4 u_kick;        // xy = recoil kick-back offset (uv), z = punch zoom, w = white flash
uniform vec4 u_impact;      // x = chromatic aberration, y = damage vignette, z = desaturation, w = static vignette
uniform vec2 u_shake;       // explosion shake jitter (uv)
in vec2 uv;
out vec4 fragColor;
void main() {
    vec2 c = uv - 0.5;
    float r2 = dot(c, c);
    vec2 p = clamp(c * (1.0 - u_kick.z) + 0.5 + u_kick.xy + u_shake, 0.001, 0.999);
    vec2 split = c * (u_impact.x * (0.6 + 2.4 * r2));
    vec3 col = vec3(texture(tex, clamp(p + split, 0.001, 0.999)).r,
                    texture(tex, p).g,
                    texture(tex, clamp(p - split, 0.001, 0.999)).b);
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, vec3(luma), u_impact.z);
    float edge = smoothstep(0.08, 0.55, r2 * 2.0);
    col = mix(col, vec3(0.45, 0.01, 0.01) + col * vec3(0.9, 0.30, 0.30), u_impact.y * edge * 0.75);
    col *= 1.0 - u_impact.w * r2 * 1.6;
    col += vec3(u_kick.w);
    fragColor = vec4(col, 1.0);
}
"""

# =========================================================================== #
#  Geometry helpers
# =========================================================================== #
VERTEX = np.dtype([("pos", "<f4", 3), ("nrm", "<f4", 3), ("col", "<f4", 4), ("uv", "<f4", 2)])
_FORMAT = None

_AXES = np.eye(3)
FACE_N = np.array([_AXES[0], -_AXES[0], _AXES[1], -_AXES[1], _AXES[2], -_AXES[2]])
FACE_A = np.array([_AXES[2], _AXES[1], _AXES[0], _AXES[2], _AXES[1], _AXES[0]])     # chosen so that A x B = -N
FACE_B = np.array([_AXES[1], _AXES[2], _AXES[2], _AXES[0], _AXES[0], _AXES[1]])
_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
BOX_LOCAL = (FACE_N[:, None, :] + FACE_A[:, None, :] * _CORNERS[None, :, 0:1]
             + FACE_B[:, None, :] * _CORNERS[None, :, 1:2])                         # (6, 4, 3) corners of [-1, 1]^3
BOX_NORMAL = np.repeat(FACE_N[:, None, :], 4, axis=1)                               # (6, 4, 3)
QUAD_UV = (_CORNERS + 1.0) * 0.5
_QUAD_INDEX = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)


def vertex_format() -> GeomVertexFormat:
    """Interleaved float32 layout shared by every mesh: position, normal, colour, uv (48 bytes)."""
    global _FORMAT
    if _FORMAT is None:
        array = GeomVertexArrayFormat()
        array.addColumn(InternalName.getVertex(), 3, Geom.NTFloat32, Geom.CPoint)
        array.addColumn(InternalName.getNormal(), 3, Geom.NTFloat32, Geom.CNormal)
        array.addColumn(InternalName.getColor(), 4, Geom.NTFloat32, Geom.CColor)
        array.addColumn(InternalName.getTexcoord(), 2, Geom.NTFloat32, Geom.CTexcoord)
        fmt = GeomVertexFormat()
        fmt.addArray(array)
        _FORMAT = GeomVertexFormat.registerFormat(fmt)
        if _FORMAT.getArray(0).getStride() != VERTEX.itemsize:
            raise RuntimeError("vertex layout mismatch between NumPy and Panda3D")
    return _FORMAT


def quad_indices(quads: int) -> np.ndarray:
    """Index buffer for ``quads`` four-vertex quads (two triangles each)."""
    return (np.arange(quads, dtype=np.uint32)[:, None] * 4 + _QUAD_INDEX[None, :]).ravel()


def box_vertices(centers, halves, colors, uv) -> np.ndarray:
    """
    Vertex data of ``B`` axis aligned boxes: 24 vertices each (flat shaded
    faces), generated by broadcasting the unit-cube tables.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    halves = np.asarray(halves, dtype=np.float64).reshape(-1, 3)
    count = centers.shape[0]
    out = np.zeros(count * 24, dtype=VERTEX)
    out["pos"] = (centers[:, None, None, :] + BOX_LOCAL[None] * halves[:, None, None, :]).reshape(-1, 3)
    out["nrm"] = np.broadcast_to(BOX_NORMAL[None], (count, 6, 4, 3)).reshape(-1, 3)
    out["col"] = np.repeat(np.asarray(colors, dtype=np.float32).reshape(count, 4), 24, axis=0)
    out["uv"] = np.repeat(np.asarray(uv, dtype=np.float32).reshape(count, 2), 24, axis=0)
    return out


def make_geom_node(name: str, vertices: np.ndarray, indices: np.ndarray, dynamic: bool = False):
    """Upload a vertex / index pair into a new GeomNode; returns ``(NodePath, GeomVertexData)``."""
    usage = Geom.UHDynamic if dynamic else Geom.UHStatic
    vdata = GeomVertexData(name, vertex_format(), usage)
    vdata.uncleanSetNumRows(len(vertices))
    memoryview(vdata.modifyArray(0)).cast("B")[:] = np.ascontiguousarray(vertices).view(np.uint8)
    prim = GeomTriangles(Geom.UHStatic)
    prim.setIndexType(GeomEnums.NTUint32)
    handle = prim.modifyVertices()
    handle.uncleanSetNumRows(len(indices))
    memoryview(handle).cast("B")[:] = np.ascontiguousarray(indices, dtype=np.uint32).view(np.uint8)
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node), vdata


def set_blend(node: NodePath, mode: str) -> None:
    """Configure blending for effect meshes ('add' or 'alpha'; both expect premultiplied colour)."""
    if mode == "add":
        node.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd, ColorBlendAttrib.OOne, ColorBlendAttrib.OOne))
    else:
        node.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd, ColorBlendAttrib.OOne,
                                             ColorBlendAttrib.OOneMinusIncomingAlpha))
    node.setDepthWrite(False)
    node.setTwoSided(True)


class DynamicMesh:
    """
    Fixed capacity quad buffer that is rewritten from a NumPy array every
    frame.  Unused quads are zeroed (degenerate triangles), so a single static
    index buffer serves any fill level.
    """

    def __init__(self, name: str, max_quads: int, parent: NodePath, shader: Shader):
        self.capacity = int(max_quads) * 4
        self.buffer = np.zeros(self.capacity, dtype=VERTEX)
        self.node, self.vdata = make_geom_node(name, self.buffer, quad_indices(max_quads), dynamic=True)
        self.node.node().setBounds(OmniBoundingVolume())
        self.node.node().setFinal(True)
        self.node.reparentTo(parent)
        self.node.setShader(shader)
        self.node.hide()
        self.used = 0

    def upload(self, vertices: np.ndarray) -> None:
        """Replace the buffer contents with ``vertices`` (a VERTEX array, 4 per quad)."""
        count = min(len(vertices), self.capacity)
        if count == 0 and self.used == 0:
            return
        buffer = self.buffer
        buffer[:count] = vertices[:count]
        if count < self.used:
            buffer[count:self.used] = 0
        self.used = count
        memoryview(self.vdata.modifyArray(0)).cast("B")[:] = buffer.view(np.uint8)
        if count:
            self.node.show()
        else:
            self.node.hide()


# =========================================================================== #
#  Dynamic point lights (muzzle flashes, explosions) + level mood lights
# =========================================================================== #
class LightPool:
    """
    Pool of short-lived point lights fed to the world shader as uniform
    arrays.  A muzzle flash is a light of intensity ``I0`` that exists for
    exactly ``MUZZLE_FLASH_TIME`` (0.03 s) and then terminates itself; the
    shader attenuates it with ``I0 / d^2``.
    """

    def __init__(self, count: int = C.MAX_DYNAMIC_LIGHTS):
        self.count = int(count)
        self.pos = np.zeros((self.count, 3), dtype=np.float64)
        self.color = np.zeros((self.count, 3), dtype=np.float64)
        self.intensity = np.zeros(self.count, dtype=np.float64)
        self.life = np.zeros(self.count, dtype=np.float64)
        self.ttl = np.ones(self.count, dtype=np.float64)
        self.fade = np.zeros(self.count, dtype=bool)
        self.pta_pos = PTA_LVecBase4f()
        self.pta_col = PTA_LVecBase4f()
        for _ in range(self.count * 2):
            self.pta_pos.pushBack(LVecBase4f(0, 0, 0, 0))
            self.pta_col.pushBack(LVecBase4f(0, 0, 0, 0))
        self.flash_level = 0.0

    def bind(self, node: NodePath) -> None:
        node.setShaderInput("u_light_pos", self.pta_pos)
        node.setShaderInput("u_light_col", self.pta_col)

    def set_static(self, positions, colors, radii) -> None:
        """Install the level's mood lights in the second half of the arrays."""
        base = self.count
        for i in range(self.count):
            if i < len(radii):
                radius = float(radii[i])
                p, c = positions[i], colors[i]
                self.pta_pos[base + i] = LVecBase4f(float(p[0]), float(p[1]), float(p[2]), radius * radius / 15.0)
                self.pta_col[base + i] = LVecBase4f(float(c[0]), float(c[1]), float(c[2]), 1.0 / (radius * radius))
            else:
                self.pta_pos[base + i] = LVecBase4f(0, 0, 0, 0)
                self.pta_col[base + i] = LVecBase4f(0, 0, 0, 0)

    def add(self, position, rgb, intensity: float, duration: float = C.MUZZLE_FLASH_TIME, fade: bool = False) -> None:
        slot = int(np.argmin(self.life))
        self.pos[slot] = position
        self.color[slot] = rgb
        self.intensity[slot] = intensity
        self.life[slot] = duration
        self.ttl[slot] = max(duration, 1.0e-4)
        self.fade[slot] = fade

    def muzzle_flash(self, position, rgb=(1.0, 0.78, 0.45), intensity: float = 10.0) -> None:
        """Instantaneous muzzle light: full intensity for 0.03 s, then gone."""
        self.add(position, rgb, intensity, C.MUZZLE_FLASH_TIME, False)

    def explosion(self, position, rgb=(1.0, 0.55, 0.20), intensity: float = 90.0, duration: float = 0.35) -> None:
        self.add(position, rgb, intensity, duration, True)

    def update(self, dt: float) -> None:
        """Age the lights and refresh the uniform arrays (call once per frame, before new flashes are added)."""
        self.life = np.maximum(self.life - dt, 0.0)
        alive = self.life > 0.0
        gain = np.where(self.fade, (self.life / self.ttl) ** 2, 1.0) * alive * self.intensity
        self.flash_level = float(np.max(np.where(self.fade, 0.0, gain))) if alive.any() else 0.0
        self.write()

    def write(self) -> None:
        alive = self.life > 0.0
        gain = np.where(self.fade, (self.life / self.ttl) ** 2, 1.0) * alive * self.intensity
        for i in range(self.count):
            p, c = self.pos[i], self.color[i]
            self.pta_pos[i] = LVecBase4f(float(p[0]), float(p[1]), float(p[2]), float(gain[i]))
            self.pta_col[i] = LVecBase4f(float(c[0]), float(c[1]), float(c[2]), 0.0)


# =========================================================================== #
#  World (static level mesh + sky)
# =========================================================================== #
class WorldRenderer:
    """Builds the level as a single static mesh and owns the world shader inputs."""

    def __init__(self, lights: LightPool):
        self.root = NodePath("nexus_world")
        self.root.reparentTo(scene)
        self.shader = Shader.make(Shader.SL_GLSL, WORLD_VERT, WORLD_FRAG)
        self.sky_shader = Shader.make(Shader.SL_GLSL, SKY_VERT, SKY_FRAG)
        self.fx_shader = Shader.make(Shader.SL_GLSL, FX_VERT, FX_FRAG)
        self.lights = lights
        self.static = None
        self.sky = None
        self.map_key = ""
        self.triangle_count = 0
        self.ambient = (0.3, 0.3, 0.35)
        lights.bind(self.root)
        self.root.setShaderInput("u_camera_pos", (0.0, 0.0, 0.0))

    def load_map(self, data: maps.MapData) -> None:
        """(Re)build the static geometry and environment for an arena."""
        if self.static is not None:
            self.static.removeNode()
        if self.sky is not None:
            self.sky.removeNode()
        visible = data.box_visible
        bmin, bmax = data.box_min[visible], data.box_max[visible]
        count = bmin.shape[0]
        colors = np.concatenate([data.box_color[visible], np.ones((count, 1), dtype=np.float32)], axis=1)
        uv = np.stack([data.box_material[visible].astype(np.float32), np.zeros(count, dtype=np.float32)], axis=1)
        box_v = box_vertices((bmin + bmax) * 0.5, (bmax - bmin) * 0.5, colors, uv)
        box_i = quad_indices(count * 6)
        tris = data.mesh_tris.copy()
        normals = data.mesh_normal
        if tris.shape[0]:
            winding = M.dot(M.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), normals)
            flip = winding > 0.0                         # must be negative to face the viewer
            tris[flip] = tris[flip][:, [0, 2, 1]]
        tri_v = np.zeros(tris.shape[0] * 3, dtype=VERTEX)
        tri_v["pos"] = tris.reshape(-1, 3)
        tri_v["nrm"] = np.repeat(normals, 3, axis=0)
        tri_v["col"][:, :3] = np.repeat(data.mesh_color, 3, axis=0)
        tri_v["col"][:, 3] = 1.0
        tri_i = np.arange(tri_v.shape[0], dtype=np.uint32) + box_v.shape[0]
        vertices = np.concatenate([box_v, tri_v])
        indices = np.concatenate([box_i, tri_i])
        self.triangle_count = indices.shape[0] // 3
        self.static, _ = make_geom_node("static_world", vertices, indices)
        self.static.reparentTo(self.root)
        self.static.setShader(self.shader)
        sun = M.normalize(np.asarray(data.sun_dir, dtype=np.float64))
        self.root.setShaderInput("u_sun_dir", (float(sun[0]), float(sun[1]), float(sun[2])))
        self.root.setShaderInput("u_sun_color", tuple(float(v) for v in data.sun_color))
        self.root.setShaderInput("u_ambient", tuple(float(v) for v in data.ambient))
        self.root.setShaderInput("u_fog_color", tuple(float(v) for v in data.fog_color))
        self.root.setShaderInput("u_fog_density", float(data.fog_density))
        self.ambient = tuple(float(v) for v in data.ambient)
        self.lights.set_static(data.lights_pos, data.lights_color, data.lights_radius)
        sky_v = box_vertices(np.zeros((1, 3)), np.full((1, 3), 400.0), np.ones((1, 4)), np.zeros((1, 2)))
        self.sky, _ = make_geom_node("sky", sky_v, quad_indices(6))
        self.sky.reparentTo(self.root)
        self.sky.setShader(self.sky_shader)
        self.sky.setTwoSided(True)
        self.sky.setDepthWrite(False)
        self.sky.setDepthTest(False)
        self.sky.setBin("background", 0)
        self.sky.setShaderInput("u_sky_top", tuple(float(v) for v in data.sky_top))
        self.sky.setShaderInput("u_sky_horizon", tuple(float(v) for v in data.sky_horizon))
        window.color = color.rgba(*[float(v) for v in data.fog_color], 1.0)
        self.map_key = data.key

    def update(self, cam_pos) -> None:
        position = (float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        self.root.setShaderInput("u_camera_pos", position)
        if self.sky is not None:
            self.sky.setPos(*position)


# =========================================================================== #
#  Vectorised effect pools
# =========================================================================== #
EMPTY = np.zeros(0, dtype=VERTEX)


class ParticlePool:
    """
    Structure-of-arrays particle system.  Simulation (gravity, drag, ageing)
    and billboard construction are single NumPy expressions over all live
    particles; the result is one vertex buffer and one draw call.
    """

    def __init__(self, name: str, capacity: int, parent: NodePath, shader: Shader, blend: str, seed: int = 1):
        n = int(capacity)
        self.capacity = n
        self.pos = np.zeros((n, 3), dtype=np.float64)
        self.vel = np.zeros((n, 3), dtype=np.float64)
        self.life = np.zeros(n, dtype=np.float64)
        self.ttl = np.ones(n, dtype=np.float64)
        self.size0 = np.zeros(n, dtype=np.float64)
        self.size1 = np.zeros(n, dtype=np.float64)
        self.color = np.zeros((n, 4), dtype=np.float64)
        self.gravity = np.zeros(n, dtype=np.float64)
        self.drag = np.zeros(n, dtype=np.float64)
        self.cursor = 0
        self.rng = np.random.default_rng(seed)
        self.mesh = DynamicMesh(name, n, parent, shader)
        set_blend(self.mesh.node, blend)
        self.mesh.node.setBin("fixed", 40 if blend == "add" else 35)

    def emit(self, position, count: int, direction=None, spread: float = 1.0, speed=(1.0, 3.0),
             rgba=(1.0, 0.8, 0.4, 1.0), size=(0.05, 0.02), life=(0.2, 0.5), gravity: float = 1.0,
             drag: float = 0.0, jitter: float = 0.0) -> None:
        """Spawn ``count`` particles at ``position`` (oldest slots are recycled)."""
        count = int(min(count, self.capacity))
        if count <= 0:
            return
        idx = (self.cursor + np.arange(count)) % self.capacity
        self.cursor = int((self.cursor + count) % self.capacity)
        rng = self.rng
        heading = M.normalize(rng.normal(size=(count, 3)))
        if direction is not None:
            heading = M.normalize(np.asarray(direction, dtype=np.float64)[None, :] + heading * spread)
        self.pos[idx] = np.asarray(position, dtype=np.float64)[None, :] + rng.normal(size=(count, 3)) * jitter
        self.vel[idx] = heading * rng.uniform(speed[0], speed[1], count)[:, None]
        ttl = rng.uniform(life[0], life[1], count)
        self.life[idx] = ttl
        self.ttl[idx] = ttl
        self.size0[idx] = size[0] * rng.uniform(0.7, 1.3, count)
        self.size1[idx] = size[1]
        self.color[idx] = rgba
        self.gravity[idx] = gravity
        self.drag[idx] = drag

    def update(self, dt: float) -> None:
        alive = self.life > 0.0
        if not alive.any():
            return
        self.vel[alive, 1] -= C.GRAVITY * self.gravity[alive] * dt
        self.vel[alive] *= np.exp(-self.drag[alive] * dt)[:, None]
        self.pos[alive] += self.vel[alive] * dt
        self.life[alive] -= dt

    def build(self, right, up) -> None:
        """Camera facing quads for every live particle -> vertex buffer."""
        idx = np.flatnonzero(self.life > 0.0)
        count = idx.size
        if count == 0:
            self.mesh.upload(EMPTY)
            return
        age = 1.0 - self.life[idx] / self.ttl[idx]
        size = M.lerp(self.size0[idx], self.size1[idx], age)
        corners = (self.pos[idx][:, None, :]
                   + (right[None, None, :] * _CORNERS[None, :, 0:1] + up[None, None, :] * _CORNERS[None, :, 1:2])
                   * size[:, None, None])
        rgba = self.color[idx].copy()
        rgba[:, 3] *= np.clip(1.0 - age, 0.0, 1.0)
        out = np.zeros(count * 4, dtype=VERTEX)
        out["pos"] = corners.reshape(-1, 3)
        out["col"] = np.repeat(rgba, 4, axis=0)
        out["uv"] = np.tile(QUAD_UV, (count, 1))
        self.mesh.upload(out)


class TracerPool:
    """Bullet streaks: a short bright segment racing from the muzzle to the impact point."""

    def __init__(self, capacity: int, parent: NodePath, shader: Shader):
        n = int(capacity)
        self.capacity = n
        self.origin = np.zeros((n, 3), dtype=np.float64)
        self.direction = np.zeros((n, 3), dtype=np.float64)
        self.length = np.zeros(n, dtype=np.float64)
        self.travel = np.zeros(n, dtype=np.float64)
        self.streak = np.zeros(n, dtype=np.float64)
        self.speed = np.zeros(n, dtype=np.float64)
        self.width = np.zeros(n, dtype=np.float64)
        self.color = np.zeros((n, 4), dtype=np.float64)
        self.alive = np.zeros(n, dtype=bool)
        self.cursor = 0
        self.mesh = DynamicMesh("tracers", n, parent, shader)
        set_blend(self.mesh.node, "add")
        self.mesh.node.setBin("fixed", 42)

    def add(self, start, end, rgba=(1.0, 0.85, 0.5, 1.0), width: float = 0.022, speed: float = 260.0,
            streak: float = 7.0) -> None:
        start = np.asarray(start, dtype=np.float64)
        delta = np.asarray(end, dtype=np.float64) - start
        length = float(np.sqrt(np.dot(delta, delta)))
        if length < 0.3:
            return
        i = self.cursor
        self.cursor = (i + 1) % self.capacity
        self.origin[i] = start
        self.direction[i] = delta / length
        self.length[i] = length
        self.travel[i] = min(1.5, length)
        self.streak[i] = min(streak, length)
        self.speed[i] = speed
        self.width[i] = width
        self.color[i] = rgba
        self.alive[i] = True

    def update(self, dt: float) -> None:
        self.travel[self.alive] += self.speed[self.alive] * dt
        self.alive &= (self.travel - self.streak) < self.length

    def build(self, cam_pos) -> None:
        idx = np.flatnonzero(self.alive)
        count = idx.size
        if count == 0:
            self.mesh.upload(EMPTY)
            return
        head = np.minimum(self.travel[idx], self.length[idx])
        tail = np.clip(self.travel[idx] - self.streak[idx], 0.0, self.length[idx])
        p_head = self.origin[idx] + self.direction[idx] * head[:, None]
        p_tail = self.origin[idx] + self.direction[idx] * tail[:, None]
        to_cam = np.asarray(cam_pos, dtype=np.float64)[None, :] - (p_head + p_tail) * 0.5
        side = M.normalize(M.cross(self.direction[idx], to_cam))
        distance = np.sqrt(np.sum(to_cam * to_cam, axis=1))
        side = side * (self.width[idx] * np.clip(distance * 0.06, 1.0, 6.0))[:, None]
        corners = np.stack([p_tail - side, p_head - side, p_head + side, p_tail + side], axis=1)
        out = np.zeros(count * 4, dtype=VERTEX)
        out["pos"] = corners.reshape(-1, 3)
        out["nrm"][:, 0] = 1.0
        out["col"] = np.repeat(self.color[idx], 4, axis=0)
        out["uv"] = np.tile(QUAD_UV, (count, 1))
        self.mesh.upload(out)


class DecalPool:
    """Ring buffer of bullet scorch marks glued to the surfaces they hit."""

    def __init__(self, capacity: int, parent: NodePath, shader: Shader):
        self.capacity = int(capacity)
        self.vertices = np.zeros(self.capacity * 4, dtype=VERTEX)
        self.cursor = 0
        self.dirty = False
        self.mesh = DynamicMesh("decals", self.capacity, parent, shader)
        set_blend(self.mesh.node, "alpha")
        self.mesh.node.setBin("fixed", 20)
        self.mesh.node.setDepthOffset(2)

    def add(self, position, normal, size: float = 0.07, rgba=(0.02, 0.02, 0.025, 0.85)) -> None:
        normal = M.normalize(np.asarray(normal, dtype=np.float64))
        if float(np.dot(normal, normal)) < 0.5:
            return
        u, v = M.perpendicular_basis(normal)
        center = np.asarray(position, dtype=np.float64) + normal * 0.012
        corners = center[None, :] + (u[None, :] * _CORNERS[:, 0:1] + v[None, :] * _CORNERS[:, 1:2]) * size
        block = self.vertices[self.cursor * 4: self.cursor * 4 + 4]
        block["pos"] = corners
        block["nrm"] = 0.0
        block["col"] = rgba
        block["uv"] = QUAD_UV
        self.cursor = (self.cursor + 1) % self.capacity
        self.dirty = True

    def clear(self) -> None:
        self.vertices[:] = 0
        self.dirty = True

    def build(self) -> None:
        if self.dirty:
            self.mesh.upload(self.vertices)
            self.dirty = False


# =========================================================================== #
#  Remote players (one dynamic mesh for everybody)
# =========================================================================== #
PART_HEAD, PART_CHEST, PART_ABDOMEN, PART_ARM_L, PART_ARM_R, PART_LEG_L, PART_LEG_R, PART_WEAPON, PART_VISOR = range(9)
PART_COUNT = 9
PART_STAND_CENTER = np.concatenate([HITBOX_STAND_CENTER, [[0.19, 1.20, 0.40], [0.0, 1.64, 0.145]]])
PART_STAND_HALF = np.concatenate([HITBOX_STAND_HALF, [[0.045, 0.06, 0.34], [0.10, 0.035, 0.02]]])
PART_CROUCH_CENTER = np.concatenate([HITBOX_CROUCH_CENTER, [[0.19, 0.70, 0.44], [0.0, 0.99, 0.185]]])
PART_CROUCH_HALF = np.concatenate([HITBOX_CROUCH_HALF, [[0.045, 0.06, 0.34], [0.10, 0.035, 0.02]]])
PART_TINT = np.array([1.15, 1.00, 0.82, 0.72, 0.72, 0.58, 0.58, 0.0, 0.0])
FFA_ENEMY_COLOR = np.array([0.95, 0.30, 0.24])
assert PART_STAND_CENTER.shape[0] == PART_COUNT and HITBOX_COUNT == 7


class CharacterBatch:
    """
    Draws every visible remote player (and live grenades) from one vertex
    buffer.  The body parts are the anatomical hitboxes themselves, so the
    silhouette on screen is exactly the volume that can be hit; legs and arms
    swing about their joints and the upper body follows the aim pitch.
    """

    def __init__(self, parent: NodePath, shader: Shader, capacity: int = C.MAX_PLAYERS):
        self.capacity = int(capacity)
        self.phase = np.zeros(self.capacity, dtype=np.float64)
        self.flash = np.zeros(self.capacity, dtype=np.float64)
        quads = (self.capacity * PART_COUNT + C.GRENADE_CAPACITY) * 6
        self.mesh = DynamicMesh("characters", quads, parent, shader)
        self.drawn = 0

    def hit(self, player: int, strength: float = 0.55) -> None:
        if 0 <= player < self.capacity:
            self.flash[player] = strength

    def build(self, view, skip: int, dt: float, time_now: float, grenade_pos=None) -> None:
        self.flash = np.maximum(self.flash - dt * 4.0, 0.0)
        mask = view.alive & view.active
        if 0 <= skip < self.capacity:
            mask = mask.copy()
            mask[skip] = False
        rows = np.flatnonzero(mask)
        n = rows.size
        self.drawn = n
        blocks = []
        if n:
            blend = M.ease_in_out_cos(view.crouch[rows])[:, None, None]
            center = PART_STAND_CENTER[None] + (PART_CROUCH_CENTER - PART_STAND_CENTER)[None] * blend      # (n, P, 3)
            half = PART_STAND_HALF[None] + (PART_CROUCH_HALF - PART_STAND_HALF)[None] * blend
            local = center[:, :, None, None, :] + BOX_LOCAL[None, None] * half[:, :, None, None, :]         # (n, P, 6, 4, 3)
            normal = np.broadcast_to(BOX_NORMAL[None, None], local.shape).copy()

            speed = np.hypot(view.vel[rows, 0], view.vel[rows, 2])
            self.phase[rows] += speed * dt * 1.9
            grounded = (view.flags[rows] & 2) != 0
            sliding = (view.flags[rows] & 8) != 0
            swing = np.sin(self.phase[rows]) * np.clip(speed / 6.0, 0.0, 1.0) * (grounded & ~sliding)
            pitch = view.pitch[rows]
            angle = np.zeros((n, PART_COUNT), dtype=np.float64)
            angle[:, PART_LEG_L] = swing * 0.60
            angle[:, PART_LEG_R] = -swing * 0.60
            angle[:, PART_ARM_L] = -swing * 0.35 - pitch * 0.5
            angle[:, PART_ARM_R] = -pitch * 0.8
            angle[:, PART_WEAPON] = -pitch * 0.8
            angle[:, PART_HEAD] = -pitch * 0.6
            angle[:, PART_VISOR] = -pitch * 0.6
            pivot = center.copy()
            pivot[:, :, 1] += half[:, :, 1]                                           # limbs hinge at their top edge
            shoulder = center[:, PART_CHEST, :].copy()
            shoulder[:, 1] += half[:, PART_CHEST, 1]
            for part in (PART_HEAD, PART_VISOR, PART_WEAPON):
                pivot[:, part, :] = shoulder
            pivot[:, PART_ABDOMEN, :] = center[:, PART_ABDOMEN, :]
            rel = local - pivot[:, :, None, None, :]
            cos_a = np.cos(angle)[:, :, None, None]
            sin_a = np.sin(angle)[:, :, None, None]
            local = np.stack([rel[..., 0], rel[..., 1] * cos_a - rel[..., 2] * sin_a,
                              rel[..., 1] * sin_a + rel[..., 2] * cos_a], axis=-1) + pivot[:, :, None, None, :]
            normal = np.stack([normal[..., 0], normal[..., 1] * cos_a - normal[..., 2] * sin_a,
                               normal[..., 1] * sin_a + normal[..., 2] * cos_a], axis=-1)

            cos_y = np.cos(view.yaw[rows])[:, None, None, None]
            sin_y = np.sin(view.yaw[rows])[:, None, None, None]
            origin = view.pos[rows][:, None, None, None, :]
            world = np.stack([local[..., 0] * cos_y + local[..., 2] * sin_y + origin[..., 0],
                              local[..., 1] + origin[..., 1],
                              -local[..., 0] * sin_y + local[..., 2] * cos_y + origin[..., 2]], axis=-1)
            wnormal = np.stack([normal[..., 0] * cos_y + normal[..., 2] * sin_y, normal[..., 1],
                                -normal[..., 0] * sin_y + normal[..., 2] * cos_y], axis=-1)

            team = view.team[rows]
            base = np.where((team < 2)[:, None], C.TEAM_COLORS[np.minimum(team, 1)], FFA_ENEMY_COLOR[None, :])
            rgb = base[:, None, :] * PART_TINT[None, :, None]
            rgb[:, PART_WEAPON, :] = (0.11, 0.12, 0.14)
            rgb[:, PART_VISOR, :] = np.minimum(base * 1.5 + 0.25, 1.6)
            material = np.full((n, PART_COUNT), 2.0)
            material[:, PART_VISOR] = 3.0
            flash = np.broadcast_to(self.flash[rows][:, None], (n, PART_COUNT))
            out = np.zeros(n * PART_COUNT * 24, dtype=VERTEX)
            out["pos"] = world.reshape(-1, 3)
            out["nrm"] = wnormal.reshape(-1, 3)
            out["col"][:, :3] = np.repeat(rgb.reshape(-1, 3), 24, axis=0)
            out["col"][:, 3] = 1.0
            out["uv"][:, 0] = np.repeat(material.reshape(-1), 24)
            out["uv"][:, 1] = np.repeat(flash.reshape(-1), 24)
            blocks.append(out)
        if grenade_pos is not None and len(grenade_pos):
            g = np.asarray(grenade_pos, dtype=np.float64).reshape(-1, 3)[: C.GRENADE_CAPACITY]
            pulse = 0.75 + 0.25 * math.sin(time_now * 22.0)
            colors = np.tile(np.array([1.0 * pulse, 0.35 * pulse, 0.12, 1.0]), (g.shape[0], 1))
            blocks.append(box_vertices(g, np.full((g.shape[0], 3), 0.09), colors, np.tile([3.0, 0.0], (g.shape[0], 1))))
        if not blocks:
            self.mesh.upload(EMPTY)
        else:
            self.mesh.upload(np.concatenate(blocks) if len(blocks) > 1 else blocks[0])


# =========================================================================== #
#  3D damage numbers: pooled seven-segment vector text
# =========================================================================== #
SEGMENT_RECT = np.array([      # x0, y0, x1, y1 inside a 0.6 x 1.0 glyph cell
    [0.08, 0.90, 0.52, 1.00],  # A top
    [0.50, 0.52, 0.60, 0.96],  # B upper right
    [0.50, 0.04, 0.60, 0.48],  # C lower right
    [0.08, 0.00, 0.52, 0.10],  # D bottom
    [0.00, 0.04, 0.10, 0.48],  # E lower left
    [0.00, 0.52, 0.10, 0.96],  # F upper left
    [0.08, 0.45, 0.52, 0.55],  # G middle
])
DIGIT_SEGMENTS = np.array([
    [1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1],
], dtype=bool)
SEGMENT_X = SEGMENT_RECT[:, [0, 2, 2, 0]]            # (7, 4) corner x of each segment quad
SEGMENT_Y = SEGMENT_RECT[:, [1, 1, 3, 3]] - 0.5      # (7, 4) corner y, vertically centred
MAX_DIGITS = 4
_POW10 = np.array([1000, 100, 10, 1], dtype=np.int64)
DAMAGE_TEXT_LIFETIME = 0.75
DAMAGE_TEXT_RISE = 1.7
GLYPH_ADVANCE = 0.78


class DamageTextPool:
    """
    Object pool of floating damage numbers rendered as vector geometry.

    Each number is anchored at the exact world coordinates of the impact, is
    built from seven-segment digit quads (no textures, no font atlas), rises
    along world up and fades its alpha linearly to zero over a 0.75 second
    lifetime.  Spawning recycles the slot with the least life left, so no
    allocation ever happens during play, and all live numbers are converted
    to vertices in one broadcast expression.
    """

    def __init__(self, capacity: int, parent: NodePath, shader: Shader, seed: int = 5):
        n = int(capacity)
        self.capacity = n
        self.pos = np.zeros((n, 3), dtype=np.float64)
        self.drift = np.zeros((n, 3), dtype=np.float64)
        self.value = np.zeros(n, dtype=np.int64)
        self.life = np.zeros(n, dtype=np.float64)
        self.scale = np.ones(n, dtype=np.float64)
        self.color = np.ones((n, 3), dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.mesh = DynamicMesh("damage_text", n * MAX_DIGITS * 7, parent, shader)
        set_blend(self.mesh.node, "add")
        self.mesh.node.setDepthTest(False)
        self.mesh.node.setBin("fixed", 60)

    def spawn(self, position, value: float, rgb=(1.0, 1.0, 1.0), scale: float = 1.0) -> int:
        slot = int(np.argmin(self.life))
        self.pos[slot] = position
        self.drift[slot] = (self.rng.uniform(-0.35, 0.35), DAMAGE_TEXT_RISE, self.rng.uniform(-0.35, 0.35))
        self.value[slot] = int(np.clip(round(value), 0, 9999))
        self.life[slot] = DAMAGE_TEXT_LIFETIME
        self.scale[slot] = scale
        self.color[slot] = rgb
        return slot

    def update(self, dt: float) -> None:
        alive = self.life > 0.0
        if alive.any():
            self.pos[alive] += self.drift[alive] * dt          # upward translation along world up (+Y)
            self.life[alive] -= dt

    def build(self, cam_pos, right, up) -> None:
        idx = np.flatnonzero(self.life > 0.0)
        count = idx.size
        if count == 0:
            self.mesh.upload(EMPTY)
            return
        value = self.value[idx]
        digits = (value[:, None] // _POW10[None, :]) % 10                                   # (n, 4)
        ndigits = np.maximum(np.floor(np.log10(np.maximum(value, 1))).astype(np.int64) + 1, 1)
        first = MAX_DIGITS - ndigits
        slot = np.arange(MAX_DIGITS)[None, :]
        lit = DIGIT_SEGMENTS[digits] & (slot >= first[:, None])[:, :, None]                 # (n, 4, 7)
        slot_x = (slot - first[:, None]) * GLYPH_ADVANCE - ndigits[:, None] * GLYPH_ADVANCE * 0.5
        lx = slot_x[:, :, None, None] + SEGMENT_X[None, None, :, :]                         # (n, 4, 7, 4)
        ly = np.broadcast_to(SEGMENT_Y[None, None, :, :], lx.shape)
        remaining = np.clip(self.life[idx] / DAMAGE_TEXT_LIFETIME, 0.0, 1.0)
        pop = 1.0 + 0.45 * np.clip((remaining - 0.84) / 0.16, 0.0, 1.0)
        offset = self.pos[idx] - np.asarray(cam_pos, dtype=np.float64)[None, :]
        distance = np.sqrt(np.sum(offset * offset, axis=1))
        size = self.scale[idx] * pop * np.clip(distance * 0.045, 0.22, 2.4)
        world = (self.pos[idx][:, None, None, None, :]
                 + (right[None, None, None, None, :] * lx[..., None] + up[None, None, None, None, :] * ly[..., None])
                 * size[:, None, None, None, None])
        alpha = remaining[:, None, None, None] * lit[:, :, :, None] * np.ones((1, 1, 1, 4))
        out = np.zeros(count * MAX_DIGITS * 7 * 4, dtype=VERTEX)
        out["pos"] = world.reshape(-1, 3)
        out["nrm"][:, 0] = 2.0
        out["col"][:, :3] = np.repeat(self.color[idx], MAX_DIGITS * 7 * 4, axis=0)
        out["col"][:, 3] = alpha.reshape(-1)
        self.mesh.upload(out)


# =========================================================================== #
#  Screen-space post processing
# =========================================================================== #
class PostFX:
    """
    Renders the 3D scene into an off-screen buffer and resolves it through
    ``POST_FRAG``: recoil kick-back (screen offset + punch zoom driven by the
    camera rig's recoil spring), chromatic aberration that spikes when damage
    is taken, explosion shake jitter, damage vignette and low-health
    desaturation.  If the driver cannot create the buffer the game simply
    runs without the pass.
    """

    def __init__(self):
        base = application.base
        self.manager = FilterManager(base.win, base.cam)
        self.color_tex = Texture()
        self.depth_tex = Texture()
        self.quad = self.manager.renderSceneInto(colortex=self.color_tex, depthtex=self.depth_tex)
        self.enabled = self.quad is not None
        self.buffer = self.manager.buffers[-1] if self.enabled and self.manager.buffers else None
        self.aberration = 0.0
        self.damage = 0.0
        self.flash = 0.0
        self.desaturation = 0.0
        self.vignette = 0.55
        if self.enabled:
            self.quad.setShader(Shader.make(Shader.SL_GLSL, POST_VERT, POST_FRAG))
            self.quad.setShaderInput("tex", self.color_tex)
            self._push(0.0, 0.0, 0.0, (0.0, 0.0))

    def add_damage(self, strength: float) -> None:
        """Spike the chromatic aberration and the red vignette (strength ~ damage / 40)."""
        self.aberration = min(1.0, self.aberration + 0.55 * strength)
        self.damage = min(1.0, self.damage + 0.75 * strength)

    def add_flash(self, strength: float) -> None:
        self.flash = min(0.8, self.flash + strength)

    def _push(self, kick_x: float, kick_y: float, zoom: float, shake) -> None:
        self.quad.setShaderInput("u_kick", LVecBase4f(kick_x, kick_y, zoom, self.flash))
        self.quad.setShaderInput("u_impact", LVecBase4f(self.aberration * 0.035, self.damage, self.desaturation,
                                                        self.vignette))
        self.quad.setShaderInput("u_shake", (float(shake[0]), float(shake[1])))

    def update(self, dt: float, rig, health_fraction: float, alive: bool) -> None:
        self.aberration *= math.exp(-5.0 * dt)
        self.damage *= math.exp(-4.5 * dt)
        self.flash *= math.exp(-9.0 * dt)
        target = min(max(1.0 - health_fraction * 2.4, 0.0), 0.75) if alive else 0.85
        self.desaturation += (target - self.desaturation) * (1.0 - math.exp(-6.0 * dt))
        if not self.enabled:
            return
        kick_x = float(rig.punch[1]) * 0.35 + float(rig.punch[2]) * 0.10
        kick_y = float(rig.punch[0]) * 0.55
        zoom = min(float(rig.kickback) * 0.30 + rig.trauma * rig.trauma * 0.02, 0.08)
        self._push(kick_x, kick_y, zoom, rig.shake_uv)


# =========================================================================== #
#  First-person view-model
# =========================================================================== #
_GUN = (0.13, 0.14, 0.16)
_STEEL = (0.26, 0.27, 0.30)
_POLY = (0.24, 0.20, 0.14)
_SLEEVE = (0.20, 0.24, 0.21)
_GLOVE = (0.10, 0.10, 0.11)
_DOT = (1.00, 0.25, 0.20)
_RAIL = (0.30, 0.90, 1.00)

# Per weapon: list of (centre, size, colour, material) boxes in gun space (x right, y up, z forward, metres).
WEAPON_MODELS = (
    {"muzzle": 0.61, "sight": 0.096, "support": 0.30, "parts": (            # VK-9 Vanguard
        ((0, 0.020, 0.10), (0.050, 0.080, 0.36), _GUN, 1), ((0, 0.030, 0.36), (0.046, 0.056, 0.20), _POLY, 0),
        ((0, 0.036, 0.53), (0.022, 0.022, 0.16), _STEEL, 1), ((0, 0.005, -0.19), (0.040, 0.070, 0.22), _POLY, 0),
        ((0, -0.090, 0.07), (0.034, 0.130, 0.060), _STEEL, 1), ((0, -0.070, -0.045), (0.034, 0.090, 0.040), _GUN, 0),
        ((0, 0.078, 0.06), (0.020, 0.030, 0.070), _GUN, 1), ((0, 0.082, 0.024), (0.008, 0.008, 0.004), _DOT, 3))},
    {"muzzle": 0.46, "sight": 0.088, "support": 0.22, "parts": (            # HX-4 Hornet
        ((0, 0.015, 0.06), (0.048, 0.075, 0.28), _GUN, 1), ((0, 0.028, 0.28), (0.040, 0.046, 0.16), _STEEL, 1),
        ((0, 0.030, 0.40), (0.020, 0.020, 0.10), _GUN, 1), ((0, -0.010, -0.14), (0.030, 0.030, 0.14), _STEEL, 1),
        ((0, -0.100, 0.04), (0.030, 0.150, 0.045), _STEEL, 1), ((0, -0.065, -0.045), (0.034, 0.085, 0.040), _GUN, 0),
        ((0, 0.070, 0.04), (0.018, 0.026, 0.050), _GUN, 1), ((0, 0.074, 0.012), (0.007, 0.007, 0.004), _DOT, 3))},
    {"muzzle": 0.66, "sight": 0.072, "support": 0.34, "parts": (            # Breacher-12
        ((0, 0.015, 0.08), (0.056, 0.076, 0.32), _GUN, 1), ((0, 0.036, 0.42), (0.034, 0.034, 0.44), _STEEL, 1),
        ((0, -0.010, 0.40), (0.030, 0.030, 0.36), _GUN, 1), ((0, -0.006, 0.34), (0.052, 0.048, 0.14), _POLY, 0),
        ((0, 0.000, -0.20), (0.042, 0.074, 0.22), _POLY, 0), ((0, -0.070, -0.050), (0.034, 0.090, 0.040), _POLY, 0),
        ((0, 0.062, 0.60), (0.008, 0.016, 0.012), _DOT, 3))},
    {"muzzle": 0.76, "sight": 0.118, "support": 0.34, "parts": (            # Longbow DMR
        ((0, 0.020, 0.10), (0.046, 0.076, 0.40), _GUN, 1), ((0, 0.030, 0.40), (0.040, 0.050, 0.22), _POLY, 0),
        ((0, 0.034, 0.62), (0.020, 0.020, 0.28), _STEEL, 1), ((0, 0.004, -0.22), (0.038, 0.066, 0.26), _POLY, 0),
        ((0, -0.080, 0.08), (0.032, 0.100, 0.070), _STEEL, 1), ((0, -0.070, -0.045), (0.034, 0.090, 0.040), _GUN, 0),
        ((0, 0.092, 0.10), (0.034, 0.038, 0.22), _GUN, 1), ((0, 0.092, -0.012), (0.024, 0.024, 0.004), _RAIL, 3))},
    {"muzzle": 0.24, "sight": 0.064, "support": 0.02, "parts": (            # P9 Sidekick
        ((0, 0.030, 0.06), (0.036, 0.040, 0.22), _STEEL, 1), ((0, 0.000, 0.03), (0.034, 0.028, 0.16), _GUN, 1),
        ((0, -0.060, -0.035), (0.034, 0.100, 0.046), _GUN, 0), ((0, 0.054, -0.03), (0.010, 0.010, 0.010), _DOT, 3),
        ((0, 0.054, 0.15), (0.006, 0.012, 0.008), _STEEL, 1))},
    {"muzzle": 0.72, "sight": 0.100, "support": 0.34, "parts": (            # Bulwark LMG
        ((0, 0.020, 0.10), (0.064, 0.094, 0.42), _GUN, 1), ((0, 0.034, 0.44), (0.050, 0.056, 0.22), _STEEL, 1),
        ((0, 0.038, 0.62), (0.026, 0.026, 0.20), _GUN, 1), ((0, 0.004, -0.22), (0.046, 0.080, 0.22), _POLY, 0),
        ((-0.012, -0.100, 0.12), (0.110, 0.110, 0.130), _POLY, 0), ((0, -0.070, -0.045), (0.036, 0.090, 0.042), _GUN, 0),
        ((0, 0.082, 0.04), (0.022, 0.030, 0.080), _GUN, 1), ((0, 0.086, -0.002), (0.008, 0.008, 0.004), _DOT, 3),
        ((0, -0.040, 0.52), (0.012, 0.080, 0.012), _STEEL, 1))},
    {"muzzle": 0.86, "sight": 0.122, "support": 0.36, "parts": (            # Aurora Rail
        ((0, 0.018, 0.10), (0.050, 0.086, 0.44), _GUN, 1), ((0.022, 0.034, 0.56), (0.010, 0.030, 0.56), _STEEL, 1),
        ((-0.022, 0.034, 0.56), (0.010, 0.030, 0.56), _STEEL, 1), ((0, 0.034, 0.56), (0.016, 0.012, 0.54), _RAIL, 3),
        ((0, 0.002, -0.24), (0.040, 0.070, 0.26), _GUN, 0), ((0, -0.070, -0.045), (0.034, 0.090, 0.040), _GUN, 0),
        ((0, 0.096, 0.12), (0.038, 0.042, 0.26), _GUN, 1), ((0, 0.096, -0.012), (0.026, 0.026, 0.004), _RAIL, 3),
        ((0, -0.070, 0.10), (0.034, 0.090, 0.090), _STEEL, 1))},
)
VM_HIP = np.array([0.150, -0.150, 0.40])


class ViewModel:
    """
    Procedural first-person weapon.  It lives in its own tiny scene with its
    own camera and is composited into the post-process buffer after the world
    (depth cleared first), so it never clips into walls yet still receives
    the screen-space effects.
    """

    def __init__(self, post: PostFX, fx_shader: Shader):
        base = application.base
        self.root = NodePath("viewmodel_root")
        self.shader = Shader.make(Shader.SL_GLSL, WORLD_VERT, VIEWMODEL_FRAG)
        self.lens = PerspectiveLens()
        self.lens.setNearFar(0.02, 30.0)
        self.aspect = 0.0
        cam_node = PandaCamera("viewmodel_camera")
        cam_node.setLens(self.lens)
        self.camera = self.root.attachNewNode(cam_node)
        target = post.buffer if post.enabled and post.buffer is not None else base.win
        self.region = target.makeDisplayRegion()
        self.region.setSort(10)
        self.region.setClearDepthActive(True)
        self.region.setClearColorActive(False)
        self.region.setCamera(self.camera)
        self.pivot = self.root.attachNewNode("viewmodel_pivot")
        self.pivot.setShader(self.shader)
        self.pivot.setShaderInput("u_flash", 0.0)
        self.pivot.setShaderInput("u_tint", (0.4, 0.4, 0.45))
        self.models = []
        for spec in WEAPON_MODELS:
            parts = list(spec["parts"])
            support = spec["support"]
            parts.append(((0.030, -0.115, -0.13), (0.078, 0.078, 0.30), _SLEEVE, 2))          # right forearm
            parts.append(((0.004, -0.060, -0.045), (0.050, 0.070, 0.060), _GLOVE, 2))         # right hand
            parts.append(((-0.095, -0.075, support - 0.12), (0.070, 0.070, 0.30), _SLEEVE, 2))  # left forearm
            parts.append(((-0.034, -0.030, support + 0.03), (0.062, 0.050, 0.085), _GLOVE, 2))  # left hand
            centers = np.array([p[0] for p in parts], dtype=np.float64)
            halves = np.array([p[1] for p in parts], dtype=np.float64) * 0.5
            colors = np.array([tuple(p[2]) + (1.0,) for p in parts], dtype=np.float32)
            uv = np.array([(float(p[3]), 0.0) for p in parts], dtype=np.float32)
            node, _ = make_geom_node("viewmodel_weapon", box_vertices(centers, halves, colors, uv),
                                     quad_indices(len(parts) * 6))
            node.reparentTo(self.pivot)
            node.hide()
            self.models.append(node)
        self.flash_mesh = DynamicMesh("viewmodel_flash", 3, self.pivot, fx_shader)
        set_blend(self.flash_mesh.node, "add")
        self.flash_mesh.node.setBin("fixed", 50)
        self.weapon = -1
        self.flash_time = 0.0
        self.flash_seed = 0
        self.visible = True
        self.set_weapon(0)
        self.set_fov(C.VIEWMODEL_FOV)

    def set_fov(self, fov: float) -> None:
        self.fov = float(fov)
        self.aspect = 0.0

    def set_tint(self, ambient) -> None:
        self.pivot.setShaderInput("u_tint", tuple(float(v) for v in ambient))

    def set_weapon(self, index: int) -> None:
        index = int(index) % len(self.models)
        if index == self.weapon:
            return
        if self.weapon >= 0:
            self.models[self.weapon].hide()
        self.weapon = index
        if self.visible:
            self.models[index].show()

    def set_visible(self, visible: bool) -> None:
        if visible == self.visible:
            return
        self.visible = visible
        if visible:
            self.models[self.weapon].show()
        else:
            self.models[self.weapon].hide()
            self.flash_mesh.upload(EMPTY)

    def fire(self) -> None:
        self.flash_time = 0.05
        self.flash_seed += 1

    def muzzle_offset(self, ads_blend: float) -> np.ndarray:
        """Muzzle position in camera space (right, up, forward) for tracers and the flash light."""
        spec = WEAPON_MODELS[self.weapon]
        hip = VM_HIP + np.array([0.0, 0.036, spec["muzzle"]])
        ads = np.array([0.0, -spec["sight"] + 0.036, 0.30 + spec["muzzle"]])
        return hip + (ads - hip) * ads_blend

    def update(self, dt: float, rig, flash_level: float) -> None:
        aspect = float(window.aspect_ratio)
        if abs(aspect - self.aspect) > 1.0e-4:
            self.aspect = aspect
            self.lens.setAspectRatio(aspect)
            self.lens.setFov(self.fov)
        spec = WEAPON_MODELS[self.weapon]
        ads = np.array([0.0, -spec["sight"], 0.30])
        base = VM_HIP + (ads - VM_HIP) * rig.ads_blend
        offset = base + rig.vm_offset
        self.pivot.setPos(float(offset[0]), float(offset[1]), float(offset[2]))
        rx, ry, rz = rig.vm_rotation
        self.pivot.setHpr(-math.degrees(ry), math.degrees(rx), -math.degrees(rz))
        self.flash_time = max(self.flash_time - dt, 0.0)
        glow = min(flash_level * 0.12, 1.2) if self.flash_time > 0.0 or flash_level > 0.0 else 0.0
        self.pivot.setShaderInput("u_flash", float(glow))
        if self.flash_time > 0.0 and self.visible:
            rng = np.random.default_rng(self.flash_seed)
            size = 0.085 * float(rng.uniform(0.8, 1.3)) * (self.flash_time / 0.05 + 0.4)
            roll = float(rng.uniform(0.0, math.pi))
            center = np.array([0.0, 0.036, spec["muzzle"] + 0.05])
            ax = np.array([math.cos(roll), math.sin(roll), 0.0])
            ay = np.array([-math.sin(roll), math.cos(roll), 0.0])
            az = np.array([0.0, 0.0, 1.0])
            quads = []
            for u, v in ((ax, ay), (ax, az * 1.6), (ay, az * 1.6)):
                quads.append(center[None, :] + (u[None, :] * _CORNERS[:, 0:1] + v[None, :] * _CORNERS[:, 1:2]) * size)
            out = np.zeros(12, dtype=VERTEX)
            out["pos"] = np.concatenate(quads)
            out["col"] = (1.0, 0.82, 0.45, 0.9)
            out["uv"] = np.tile(QUAD_UV, (3, 1))
            self.flash_mesh.upload(out)
        else:
            self.flash_mesh.upload(EMPTY)


# =========================================================================== #
#  Scene facade: owns every pool and exposes gameplay-level effect calls
# =========================================================================== #
class SceneRenderer:
    """One object the game talks to: load a map, trigger effects, draw a frame."""

    def __init__(self):
        self.lights = LightPool()
        self.world = WorldRenderer(self.lights)
        root, fx = self.world.root, self.world.fx_shader
        self.sparks = ParticlePool("sparks", 1024, root, fx, "add", seed=11)
        self.smoke = ParticlePool("smoke", 384, root, fx, "alpha", seed=12)
        self.tracers = TracerPool(160, root, fx)
        self.decals = DecalPool(256, root, fx)
        self.characters = CharacterBatch(root, self.world.shader)
        self.damage_text = DamageTextPool(40, root, fx)
        self.post = PostFX()
        self.viewmodel = ViewModel(self.post, fx)
        self.right = np.array([1.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.forward = np.array([0.0, 0.0, 1.0])

    def load_map(self, data: maps.MapData) -> None:
        self.world.load_map(data)
        self.decals.clear()
        self.viewmodel.set_tint(self.world.ambient)

    def begin_frame(self, dt: float) -> None:
        """Age the dynamic lights first, so flashes added by this frame's events get their full life."""
        self.lights.update(dt)

    # ---- gameplay effects ---------------------------------------------------------------
    def fx_muzzle(self, position, rgb=(1.0, 0.78, 0.45), intensity: float = 10.0) -> None:
        self.lights.muzzle_flash(position, rgb, intensity)

    def fx_tracer(self, start, end, rgba=(1.0, 0.85, 0.5, 1.0), width: float = 0.022) -> None:
        self.tracers.add(start, end, rgba, width)

    def fx_impact(self, position, normal) -> None:
        self.sparks.emit(position, 7, direction=normal, spread=0.8, speed=(1.5, 5.5), rgba=(1.0, 0.82, 0.45, 1.0),
                         size=(0.035, 0.008), life=(0.10, 0.32), gravity=0.6)
        self.smoke.emit(position, 2, direction=normal, spread=0.5, speed=(0.3, 1.0), rgba=(0.55, 0.55, 0.58, 0.30),
                        size=(0.06, 0.26), life=(0.45, 0.95), gravity=-0.02, drag=1.6)
        self.decals.add(position, normal)

    def fx_hit(self, position, headshot: bool = False) -> None:
        rgba = (1.0, 0.16, 0.14, 1.0) if headshot else (1.0, 0.40, 0.30, 1.0)
        self.sparks.emit(position, 12 if headshot else 8, speed=(1.2, 4.5), rgba=rgba, size=(0.045, 0.01),
                         life=(0.15, 0.40), gravity=0.5)

    def fx_death(self, position, rgb) -> None:
        center = np.asarray(position, dtype=np.float64) + np.array([0.0, 1.0, 0.0])
        self.sparks.emit(center, 34, speed=(1.5, 6.5), rgba=(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0),
                         size=(0.07, 0.01), life=(0.3, 0.8), gravity=0.5, jitter=0.25)
        self.smoke.emit(center, 8, speed=(0.3, 1.4), rgba=(0.4, 0.4, 0.44, 0.35), size=(0.20, 0.70), life=(0.6, 1.3),
                        gravity=-0.03, drag=1.2, jitter=0.3)

    def fx_explosion(self, position) -> None:
        position = np.asarray(position, dtype=np.float64)
        self.sparks.emit(position, 80, speed=(4.0, 17.0), rgba=(1.0, 0.62, 0.22, 1.0), size=(0.11, 0.02),
                         life=(0.25, 0.75), gravity=0.7)
        self.sparks.emit(position, 22, speed=(0.5, 3.0), rgba=(1.0, 0.85, 0.55, 1.0), size=(0.9, 0.2),
                         life=(0.10, 0.24), gravity=0.0, jitter=0.3)
        self.smoke.emit(position, 30, speed=(0.8, 4.5), rgba=(0.32, 0.31, 0.33, 0.55), size=(0.45, 1.9),
                        life=(0.9, 2.1), gravity=-0.05, drag=1.4, jitter=0.4)
        self.lights.explosion(position)
        self.decals.add(position - np.array([0.0, 0.05, 0.0]), (0.0, 1.0, 0.0), size=1.1, rgba=(0.02, 0.02, 0.02, 0.7))

    def fx_damage_number(self, position, amount: float, headshot: bool = False, killed: bool = False) -> None:
        rgb = (1.0, 0.10, 0.16) if headshot else ((1.0, 0.78, 0.20) if killed else (1.0, 1.0, 1.0))
        self.damage_text.spawn(position, amount, rgb, 1.35 if headshot else 1.0)

    # ---- frame ----------------------------------------------------------------------------------
    def render(self, dt: float, time_now: float, cam_pos, yaw: float, pitch: float, view, skip_player: int,
               grenade_pos=None) -> None:
        """Advance every pool and rebuild the dynamic vertex buffers for this frame."""
        self.right = M.right_vector(yaw)
        self.up = M.up_vector(yaw, pitch)
        self.forward = M.forward_vector(yaw, pitch)
        self.world.update(cam_pos)
        self.lights.write()
        self.sparks.update(dt)
        self.smoke.update(dt)
        self.tracers.update(dt)
        self.damage_text.update(dt)
        self.sparks.build(self.right, self.up)
        self.smoke.build(self.right, self.up)
        self.tracers.build(cam_pos)
        self.decals.build()
        self.characters.build(view, skip_player, dt, time_now, grenade_pos)
        self.damage_text.build(cam_pos, self.right, self.up)


# =========================================================================== #
#  Orthographic HUD
# =========================================================================== #
def _rgba(r: float, g: float, b: float, a: float = 1.0):
    return color.rgba(float(r), float(g), float(b), float(a))


HUD_WHITE = (0.94, 0.96, 1.00)
HUD_CRIMSON = (1.00, 0.08, 0.16)
HUD_GOLD = (1.00, 0.80, 0.25)
HUD_DIM = (0.62, 0.68, 0.76)
HITMARKER_TIME = 0.24
CROSSHAIR_ARM = 0.011


class HUD:
    """
    In-game heads-up display on Ursina's orthographic UI layer (y spans
    [-0.5, 0.5], x spans +-aspect/2).  Text is only re-typeset when its content
    changes; everything animated per frame is plain quad transforms.
    """

    def __init__(self):
        self.root = Entity(parent=camera.ui, name="nexus_hud")
        self.half_width = float(window.aspect_ratio) * 0.5
        hw = self.half_width
        self._cache: dict = {}
        # ---- dynamic crosshair: four arms that open with the bullet spread ------------------
        self.cross_gap = 0.01
        self.cross = [self._quad(0, 0, CROSSHAIR_ARM, 0.0022, HUD_WHITE, 0.9), self._quad(0, 0, CROSSHAIR_ARM, 0.0022, HUD_WHITE, 0.9),
                      self._quad(0, 0, 0.0022, CROSSHAIR_ARM, HUD_WHITE, 0.9), self._quad(0, 0, 0.0022, CROSSHAIR_ARM, HUD_WHITE, 0.9)]
        self.dot = self._quad(0, 0, 0.0026, 0.0026, HUD_WHITE, 0.95)
        # ---- hit marker ----------------------------------------------------------------------
        self.hit_time = 0.0
        self.hit_rgb = HUD_WHITE
        self.hit_big = False
        self.hit = [self._quad(0, 0, 0.017, 0.003, HUD_WHITE, 0.0, rotation_z=angle) for angle in (45, -45, -45, 45)]
        # ---- damage direction indicators -------------------------------------------------------
        self.indicators = [self._quad(0, 0, 0.07, 0.009, HUD_CRIMSON, 0.0) for _ in range(6)]
        self.indicator_ttl = np.zeros(6)
        self.indicator_angle = np.zeros(6)
        # ---- vitals ----------------------------------------------------------------------------
        self._quad(-hw + 0.045, -0.445, 0.25, 0.018, (0.05, 0.06, 0.08), 0.70, origin=(-0.5, 0))
        self.health_fill = self._quad(-hw + 0.047, -0.445, 0.246, 0.013, (0.30, 0.95, 0.55), 0.95, origin=(-0.5, 0), z=-0.01)
        self.health_text = self._text("100", -hw + 0.305, -0.445, 1.25, (-0.5, 0), HUD_WHITE)
        self._text("VITALS", -hw + 0.045, -0.418, 0.62, (-0.5, 0), HUD_DIM)
        # ---- weapon ------------------------------------------------------------------------------
        self.ammo_text = self._text("30", hw - 0.115, -0.432, 2.7, (0.5, 0), HUD_WHITE)
        self.mag_text = self._text("/ 30", hw - 0.045, -0.442, 1.0, (0.5, 0), HUD_DIM)
        self.weapon_text = self._text("", hw - 0.045, -0.388, 0.78, (0.5, 0), HUD_DIM)
        self.reload_text = self._text("", 0.0, -0.11, 0.95, (0, 0), HUD_GOLD)
        self._quad(hw - 0.345, -0.470, 0.30, 0.007, (0.05, 0.06, 0.08), 0.70, origin=(-0.5, 0))
        self.ability_fill = self._quad(hw - 0.345, -0.470, 0.30, 0.005, (1.0, 0.55, 0.20), 0.95, origin=(-0.5, 0), z=-0.01)
        self.ability_text = self._text("PULSE GRENADE [G]", hw - 0.345, -0.487, 0.6, (-0.5, 0), HUD_DIM)
        # ---- match -----------------------------------------------------------------------------------
        self.score_text = self._text("", 0.0, 0.462, 1.25, (0, 0), HUD_WHITE)
        self.mode_text = self._text("", 0.0, 0.432, 0.68, (0, 0), HUD_DIM)
        self.center_text = self._text("", 0.0, 0.17, 2.3, (0, 0), HUD_WHITE)
        self.sub_text = self._text("", 0.0, 0.115, 1.05, (0, 0), HUD_DIM)
        self.hint_text = self._text("", 0.0, -0.36, 0.66, (0, 0), HUD_DIM)
        # ---- feeds ------------------------------------------------------------------------------------
        self.kill_lines = [self._text("", hw - 0.03, 0.445 - i * 0.031, 0.80, (0.5, 0), HUD_WHITE) for i in range(5)]
        self.kill_feed: list = []                  # [text, rgb, ttl]
        self.reward_lines = [self._text("", 0.13, -0.035 - i * 0.03, 0.85, (-0.5, 0), HUD_GOLD) for i in range(4)]
        self.reward_feed: list = []
        # ---- economy / diagnostics ----------------------------------------------------------------------
        self.credit_text = self._text("", -hw + 0.03, 0.462, 0.95, (-0.5, 0), HUD_GOLD)
        self.level_text = self._text("", -hw + 0.03, 0.434, 0.70, (-0.5, 0), HUD_DIM)
        self.net_text = self._text("", -hw + 0.03, 0.408, 0.60, (-0.5, 0), HUD_DIM)
        # ---- scoreboard ----------------------------------------------------------------------------------
        self.board = Entity(parent=self.root, enabled=False, z=-2.0)
        Entity(parent=self.board, model="quad", color=_rgba(0.02, 0.03, 0.05, 0.86), scale=(1.06, 0.70), z=0.5)
        self.board_title = Text("SCOREBOARD", parent=self.board, position=(0, 0.315), scale=1.3, origin=(0, 0),
                                color=_rgba(*HUD_WHITE))
        self.board_names = Text("", parent=self.board, position=(-0.48, 0.265), scale=0.85, origin=(-0.5, 0.5),
                                color=_rgba(*HUD_WHITE))
        self.board_stats = Text("", parent=self.board, position=(0.10, 0.265), scale=0.85, origin=(-0.5, 0.5),
                                color=_rgba(*HUD_WHITE))
        self._board_stamp = None

    # ---- construction helpers ---------------------------------------------------------------
    def _quad(self, x, y, w, h, rgb, alpha, origin=(0, 0), z=0.0, rotation_z=0.0) -> Entity:
        return Entity(parent=self.root, model="quad", color=_rgba(rgb[0], rgb[1], rgb[2], alpha), position=(x, y, z),
                      scale=(w, h), origin=origin, rotation_z=rotation_z)

    def _text(self, value, x, y, scale, origin, rgb) -> Text:
        return Text(value, parent=self.root, position=(x, y), scale=scale, origin=origin, color=_rgba(*rgb))

    def _set(self, entity: Text, value: str) -> None:
        """Assign text only when it changed (typesetting is the expensive part)."""
        if self._cache.get(id(entity)) != value:
            self._cache[id(entity)] = value
            entity.text = value

    def set_visible(self, visible: bool) -> None:
        self.root.enabled = visible

    # ---- crosshair / feedback vectors -----------------------------------------------------------
    def set_crosshair(self, spread: float, vertical_fov_deg: float, alpha: float = 1.0, dt: float = 0.016) -> None:
        """
        Open the crosshair to the current accuracy cone: the gap is the cone
        half-angle projected onto the screen, ``tan(spread) / tan(vfov / 2) * 0.5``.
        """
        target = math.tan(max(spread, 0.0)) / math.tan(math.radians(vertical_fov_deg) * 0.5) * 0.5 + 0.004
        self.cross_gap += (target - self.cross_gap) * (1.0 - math.exp(-22.0 * dt))
        reach = self.cross_gap + CROSSHAIR_ARM * 0.5
        left, right, up, down = self.cross
        left.x, right.x, up.y, down.y = -reach, reach, reach, -reach
        tint = _rgba(HUD_WHITE[0], HUD_WHITE[1], HUD_WHITE[2], 0.9 * alpha)
        for arm in self.cross:
            arm.color = tint

    def hitmarker(self, headshot: bool = False, kill: bool = False) -> None:
        """Instant hit confirmation; crimson when the head multiplier was applied."""
        self.hit_time = HITMARKER_TIME * (1.5 if kill else 1.0)
        self.hit_rgb = HUD_CRIMSON if headshot else ((1.0, 0.45, 0.15) if kill else HUD_WHITE)
        self.hit_big = kill

    def damage_from(self, relative_angle: float) -> None:
        """Flash an arc towards the attacker (angle clockwise from straight ahead, radians)."""
        slot = int(np.argmin(self.indicator_ttl))
        self.indicator_ttl[slot] = 1.3
        self.indicator_angle[slot] = relative_angle

    def push_kill(self, text: str, rgb=HUD_WHITE) -> None:
        self.kill_feed.insert(0, [text, rgb, 6.0])
        del self.kill_feed[len(self.kill_lines):]
        self._refresh_feed(self.kill_lines, self.kill_feed)

    def push_reward(self, text: str, rgb=HUD_GOLD) -> None:
        self.reward_feed.insert(0, [text, rgb, 2.6])
        del self.reward_feed[len(self.reward_lines):]
        self._refresh_feed(self.reward_lines, self.reward_feed)

    def _refresh_feed(self, lines, feed) -> None:
        for index, line in enumerate(lines):
            self._set(line, feed[index][0] if index < len(feed) else "")

    # ---- value setters -----------------------------------------------------------------------------
    def set_health(self, health: float, max_health: float) -> None:
        fraction = min(max(health / max(max_health, 1.0), 0.0), 1.0)
        self.health_fill.scale_x = 0.246 * fraction
        if fraction > 0.5:
            rgb = (0.30, 0.95, 0.55)
        elif fraction > 0.25:
            rgb = (1.00, 0.80, 0.25)
        else:
            rgb = HUD_CRIMSON
        self.health_fill.color = _rgba(rgb[0], rgb[1], rgb[2], 0.95)
        self._set(self.health_text, "%d" % math.ceil(max(health, 0.0)))

    def set_weapon(self, name: str, ammo: int, magazine: int, reloading: bool) -> None:
        self._set(self.ammo_text, "%d" % ammo)
        self._set(self.mag_text, "/ %d" % magazine)
        self._set(self.weapon_text, name)
        self._set(self.reload_text, "RELOADING" if reloading else ("PRESS R TO RELOAD" if ammo == 0 else ""))
        low = ammo * 4 <= magazine
        self.ammo_text.color = _rgba(*(HUD_CRIMSON if low else HUD_WHITE))

    def set_ability(self, ready_fraction: float) -> None:
        fraction = min(max(ready_fraction, 0.0), 1.0)
        self.ability_fill.scale_x = 0.30 * fraction
        self.ability_fill.color = _rgba(1.0, 0.55, 0.20, 0.95) if fraction >= 1.0 else _rgba(0.55, 0.60, 0.68, 0.9)

    def set_match(self, mode: int, scores, seconds_left: int, own_team: int, own_kills: int, limit: int) -> None:
        minutes, seconds = divmod(max(int(seconds_left), 0), 60)
        clock = "%02d:%02d" % (minutes, seconds)
        if mode == C.MODE_TDM:
            self._set(self.score_text, "%s  %d   |   %s   |   %d  %s" % (C.TEAM_NAMES[0], scores[0], clock, scores[1],
                                                                        C.TEAM_NAMES[1]))
            self._set(self.mode_text, "%s  -  FIRST TO %d  -  YOU FIGHT FOR %s" % (C.MODE_NAMES[mode], limit,
                                                                                 C.TEAM_NAMES[own_team % 2]))
        else:
            self._set(self.score_text, "ELIMINATIONS  %d / %d   |   %s" % (own_kills, limit, clock))
            self._set(self.mode_text, C.MODE_NAMES[mode])

    def set_center(self, text: str, sub: str = "", rgb=HUD_WHITE) -> None:
        self._set(self.center_text, text)
        self._set(self.sub_text, sub)
        self.center_text.color = _rgba(*rgb)

    def set_hint(self, text: str) -> None:
        self._set(self.hint_text, text)

    def set_economy(self, credits_text: str, level_text: str) -> None:
        self._set(self.credit_text, credits_text)
        self._set(self.level_text, level_text)

    def set_net(self, text: str) -> None:
        self._set(self.net_text, text)

    def set_scoreboard(self, visible: bool, rows=None, title: str = "SCOREBOARD") -> None:
        """``rows`` is a list of ``(name, kills, deaths, score, ping, tag)`` already sorted."""
        if self.board.enabled != visible:
            self.board.enabled = visible
        if not visible or rows is None:
            return
        stamp = (title, tuple(rows))
        if stamp == self._board_stamp:
            return
        self._board_stamp = stamp
        self.board_title.text = title
        self.board_names.text = "\n".join(["PLAYER", ""] + ["%s%s" % (row[5], row[0]) for row in rows])
        self.board_stats.text = "\n".join(["K  /  D        SCORE        PING", ""] + [
            "%-3d /  %-3d        %-6d        %s" % (row[1], row[2], row[3], ("%d ms" % row[4]) if row[4] else "-")
            for row in rows])

    # ---- per frame animation --------------------------------------------------------------------------
    def update(self, dt: float) -> None:
        if self.hit_time > 0.0:
            self.hit_time = max(self.hit_time - dt, 0.0)
        phase = min(self.hit_time / HITMARKER_TIME, 1.0)
        spread = (0.017 if self.hit_big else 0.013) * (1.0 + 0.7 * phase)
        tint = _rgba(self.hit_rgb[0], self.hit_rgb[1], self.hit_rgb[2], min(phase * 1.6, 1.0))
        for arm, (sx, sy) in zip(self.hit, ((-1, 1), (1, 1), (-1, -1), (1, -1))):
            arm.x, arm.y = sx * spread, sy * spread
            arm.color = tint
        self.indicator_ttl = np.maximum(self.indicator_ttl - dt, 0.0)
        for index, marker in enumerate(self.indicators):
            ttl = float(self.indicator_ttl[index])
            if ttl <= 0.0:
                if marker.color.a != 0.0:
                    marker.color = _rgba(1, 0, 0, 0)
                continue
            angle = float(self.indicator_angle[index])
            marker.x, marker.y = math.sin(angle) * 0.17, math.cos(angle) * 0.17
            marker.rotation_z = math.degrees(angle)
            marker.color = _rgba(HUD_CRIMSON[0], HUD_CRIMSON[1], HUD_CRIMSON[2], min(ttl, 0.85))
        for lines, feed in ((self.kill_lines, self.kill_feed), (self.reward_lines, self.reward_feed)):
            expired = False
            for index, entry in enumerate(feed):
                entry[2] -= dt
                expired |= entry[2] <= 0.0
                rgb = entry[1]
                lines[index].color = _rgba(rgb[0], rgb[1], rgb[2], min(max(entry[2], 0.0) * 1.5, 1.0))
            if expired:
                feed[:] = [entry for entry in feed if entry[2] > 0.0]
                self._refresh_feed(lines, feed)


def apply_camera(rig) -> None:
    """Copy a :class:`engine.controller_3d.CameraRig` pose onto Ursina's camera."""
    camera.position = (float(rig.cam_pos[0]), float(rig.cam_pos[1]), float(rig.cam_pos[2]))
    camera.rotation = (-math.degrees(rig.cam_pitch), math.degrees(rig.cam_yaw), -math.degrees(rig.cam_roll))
    camera.fov = rig.fov


def vertical_fov(horizontal_fov_deg: float) -> float:
    """Vertical field of view (degrees) for Ursina's horizontal ``camera.fov``."""
    aspect = max(float(window.aspect_ratio), 0.1)
    return math.degrees(2.0 * math.atan(math.tan(math.radians(horizontal_fov_deg) * 0.5) / aspect))
