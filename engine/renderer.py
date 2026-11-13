"""
NEXUS-3D-EXTREME :: engine/renderer.py                 (MODULE C)
=================================================================
Textured, shadowed, batched real-time renderer and HUD.

Everything the player sees is produced here from raw Panda3D geometry attached
to Ursina's scene graph:

* **Materials.**  Every surface samples two ``sampler2DArray`` textures (albedo
  + gloss, normal + cavity).  The layers are synthesised procedurally at load
  (``engine/textures.py``) and any of them can be replaced by an image dropped
  into ``content/textures``.  Normal mapping needs no tangent attribute: the
  tangent frame is rebuilt per pixel from screen-space derivatives.
* **Lighting.**  Hemisphere ambient with baked ambient occlusion, the sun with
  PCF shadow maps, the arena's static lamps baked per vertex with real
  occlusion (``engine/worldmesh.py``), and up to eight dynamic point lights
  (muzzle flashes, explosions) evaluated per pixel with ``I0 / d^2``.
* **Shadows.**  The static level is rendered into the sun's depth map once per
  arena; only the characters are re-rendered into a second, small depth map
  every frame.  The shader takes the minimum of both.
* **Sky.**  A camera-centred dome textured with an equirectangular panorama
  (procedural, or an image from ``content/sky``), an analytic sun disc and
  stars, and a ring of distant towers rendered as real geometry.
* **Batching.**  The whole arena is one static vertex buffer and one draw
  call; every dynamic effect -- players, particles, tracers, bullet holes,
  damage numbers, HUD vectors -- is a structure-of-arrays pool converted to
  vertices with broadcasting and uploaded as one buffer per pool.  Buffers
  are trimmed to the live vertex count, so idle pools cost nothing.
* **Frame pacing.**  The 3D scene renders into an off-screen buffer whose
  resolution follows the measured frame time (dynamic resolution), and is
  resolved through a post-process pass (recoil kick, chromatic aberration,
  shake, damage vignette, bloom).

World coordinates follow Ursina 8 (Y-up, left-handed).  With that coordinate
system a triangle faces the viewer when ``dot(cross(p1 - p0, p2 - p0), n) < 0``
for its outward normal ``n``; the box tables are laid out accordingly.
"""
from __future__ import annotations

import math

import numpy as np
from direct.filter.FilterManager import FilterManager
from panda3d.core import (BitMask32, Camera as PandaCamera, ColorBlendAttrib, CullFaceAttrib, Filename,
                          FrameBufferProperties, Geom, GeomEnums, GeomNode, GeomTriangles, GeomVertexArrayFormat,
                          GeomVertexData, GeomVertexFormat, GraphicsOutput, GraphicsPipe, InternalName, LMatrix4f,
                          LVecBase4f, NodePath, OmniBoundingVolume, OrthographicLens, PerspectiveLens, PNMImage,
                          PTA_LVecBase4f, SamplerState, Shader, Texture, WindowProperties)
from ursina import Entity, Text, application, camera, color, scene, window

from . import config as C
from . import mathx as M
from . import maps
from . import textures
from . import worldmesh
from .pacing import ResolutionController
from .controller_3d import (HITBOX_COUNT, HITBOX_CROUCH_CENTER, HITBOX_CROUCH_HALF, HITBOX_STAND_CENTER,
                            HITBOX_STAND_HALF)
from .worldmesh import FACE_A, FACE_B, FACE_N, WORLD_VERTEX

MATERIAL_REVISION = 4          # bump when the procedural materials change (invalidates the cached texture arrays)
SKY_REVISION = 2
SHADOW_STATIC_MASK = BitMask32.bit(1)
SHADOW_DYNAMIC_MASK = BitMask32.bit(2)
SHADOW_MASKS = SHADOW_STATIC_MASK | SHADOW_DYNAMIC_MASK

# =========================================================================== #
#  GLSL programs
# =========================================================================== #
WORLD_VERT = """#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat4 u_light_vp;
uniform vec4 u_shadow;                 // x = shadow texel (uv), y = depth bias, z = normal offset (m), w = unused
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
in vec4 p3d_MultiTexCoord0;
in vec3 baked_light;
out vec3 v_world;
out vec3 v_normal;
out vec4 v_color;                      // rgb = tint, a = baked ambient occlusion
out vec4 v_uv;                         // xy = texture coordinates, z = texture layer, w = emissive
out vec3 v_light;                      // baked static lights
out vec3 v_shadow;                     // position in the sun's shadow map
void main() {
    vec4 world = p3d_ModelMatrix * p3d_Vertex;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_world = world.xyz;
    v_normal = mat3(p3d_ModelMatrix) * p3d_Normal;
    v_color = p3d_Color;
    v_uv = p3d_MultiTexCoord0;
    v_light = baked_light;
    vec4 lit = u_light_vp * vec4(world.xyz + v_normal * u_shadow.z, 1.0);
    v_shadow = lit.xyz * 0.5 + 0.5;
}
"""

WORLD_FRAG_TEMPLATE = """#version 140
#define SHADOWS @SHADOWS@
#define TAPS @TAPS@
#define NORMAL_MAPS @NORMAL_MAPS@
const int LIGHTS = @LIGHTS@;
uniform sampler2DArray u_albedo;       // rgb = albedo, a = gloss
uniform sampler2DArray u_relief;       // xyz = tangent space normal, a = cavity
#if SHADOWS
uniform sampler2DShadow u_shadow_static;
uniform sampler2DShadow u_shadow_dynamic;
#endif
uniform vec4 u_shadow;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_ambient;
uniform vec3 u_fog_color;
uniform float u_fog_density;
uniform vec3 u_camera_pos;
uniform int u_light_count;
uniform vec4 u_light_pos[LIGHTS];      // xyz = position, w = I0
uniform vec4 u_light_col[LIGHTS];      // rgb = colour
in vec3 v_world;
in vec3 v_normal;
in vec4 v_color;
in vec4 v_uv;
in vec3 v_light;
in vec3 v_shadow;
out vec4 fragColor;

#if SHADOWS
float shadow_tap(vec2 uv, float depth) {
    return min(texture(u_shadow_static, vec3(uv, depth)), texture(u_shadow_dynamic, vec3(uv, depth)));
}

float sun_visibility(vec3 sc, float slope) {
    if (sc.x <= 0.002 || sc.x >= 0.998 || sc.y <= 0.002 || sc.y >= 0.998 || sc.z >= 1.0) return 1.0;
    float depth = sc.z - u_shadow.y * (1.0 + 2.0 * slope);
    float texel = u_shadow.x;
#if TAPS == 1
    return shadow_tap(sc.xy, depth);
#elif TAPS == 4
    return 0.25 * (shadow_tap(sc.xy + vec2(-0.5, -0.5) * texel, depth) + shadow_tap(sc.xy + vec2(0.5, -0.5) * texel, depth)
                 + shadow_tap(sc.xy + vec2(-0.5, 0.5) * texel, depth) + shadow_tap(sc.xy + vec2(0.5, 0.5) * texel, depth));
#else
    float lit = 0.0;
    for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++)
            lit += shadow_tap(sc.xy + vec2(float(x), float(y)) * texel, depth);
    return lit / 9.0;
#endif
}
#endif

void main() {
    vec3 ng = normalize(v_normal);
    vec3 uvw = vec3(v_uv.xy, v_uv.z);
    vec4 base = texture(u_albedo, uvw);
    vec3 n = ng;
    float cavity = 1.0;
#if NORMAL_MAPS
    vec4 relief = texture(u_relief, uvw);
    vec3 tn = relief.xyz * 2.0 - 1.0;
    vec3 dp1 = dFdx(v_world);
    vec3 dp2 = dFdy(v_world);
    vec2 duv1 = dFdx(v_uv.xy);
    vec2 duv2 = dFdy(v_uv.xy);
    vec3 dp2perp = cross(dp2, ng);
    vec3 dp1perp = cross(ng, dp1);
    vec3 tangent = dp2perp * duv1.x + dp1perp * duv2.x;          // world direction of +u
    vec3 bitangent = dp2perp * duv1.y + dp1perp * duv2.y;        // world direction of +v
    float handed = dot(dp1, dp2perp) < 0.0 ? -1.0 : 1.0;
    float scale = handed * inversesqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1.0e-12));
    n = normalize(tangent * (tn.x * scale) + bitangent * (tn.y * scale) + ng * tn.z);
    cavity = relief.a;
#endif
    vec3 albedo = base.rgb * v_color.rgb;
    float gloss = base.a;
    float emissive = v_uv.w;
    float haze = 1.0;
    bool backdrop = emissive >= 3.5;                              // distant skyline: no shadows, thinner fog
    if (backdrop) { emissive -= 4.0; haze = 0.55; }

    vec3 to_cam = u_camera_pos - v_world;
    float dist = length(to_cam);
    vec3 view = to_cam / max(dist, 1.0e-4);
    float ndl = max(dot(n, -u_sun_dir), 0.0);
    float sun = ndl;
#if SHADOWS
    if (ndl > 0.0 && !backdrop) sun *= sun_visibility(v_shadow, 1.0 - max(dot(ng, -u_sun_dir), 0.0));
#endif
    float occlusion = v_color.a * cavity;
    vec3 light = u_ambient * (mix(0.60, 1.0, n.y * 0.5 + 0.5) * occlusion)
               + u_sun_color * sun
               + v_light * (0.5 + 0.5 * cavity);
    float shininess = mix(6.0, 90.0, gloss);
    vec3 specular = u_sun_color * (pow(max(dot(n, normalize(view - u_sun_dir)), 0.0), shininess) * gloss * 0.55
                                   * (ndl > 0.0 ? sun / ndl : 0.0));
    for (int i = 0; i < u_light_count; i++) {
        vec3 d = u_light_pos[i].xyz - v_world;
        float d2 = dot(d, d);
        float fall = u_light_pos[i].w / max(d2, 0.20);          // Intensity = I0 / d^2
        vec3 l = d * inversesqrt(max(d2, 1.0e-4));
        light += u_light_col[i].rgb * (fall * (0.25 + 0.75 * max(dot(n, l), 0.0)));
        specular += u_light_col[i].rgb * (min(fall, 1.5) * pow(max(dot(n, normalize(view + l)), 0.0), shininess) * gloss * 0.5);
    }
    vec3 col = albedo * light + specular * occlusion + albedo * (emissive * 1.6);
    float fog = 1.0 - exp(-u_fog_density * dist * haze);
    if (backdrop) fog = max(fog, 0.35);                           // aerial perspective keeps the skyline in the distance
    vec3 fog_col = u_fog_color + u_sun_color * (pow(max(dot(view, u_sun_dir), 0.0), 8.0) * 0.22);
    col = mix(col, fog_col, fog);
    vec3 over = max(col - 0.75, 0.0);                             // soft highlight roll-off instead of clipping
    col = min(col, vec3(0.75)) + 0.25 * (1.0 - exp(-over * 4.0));
    fragColor = vec4(col, 1.0);
}
"""

SHADOW_VERT = """#version 140
uniform mat4 p3d_ModelMatrix;
uniform mat4 u_light_vp;
in vec4 p3d_Vertex;
void main() {
    gl_Position = u_light_vp * (p3d_ModelMatrix * p3d_Vertex);
}
"""

SHADOW_FRAG = """#version 140
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0);
}
"""

SKY_VERT = """#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_MultiTexCoord0;
out vec3 v_dir;
out vec2 v_uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_dir = p3d_Vertex.xyz;
    v_uv = p3d_MultiTexCoord0.xy;          // equirectangular coordinates, exact at the dome's vertices
}
"""

SKY_FRAG = """#version 140
uniform sampler2D u_panorama;          // equirectangular: rgb = sky, a = clear sky fraction
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec2 u_sky;                    // x = star amount, y = time (s)
in vec3 v_dir;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec3 d = normalize(v_dir);
    vec4 sky = textureLod(u_panorama, v_uv, 0.0);
    vec3 col = sky.rgb;
    float facing = max(dot(d, -u_sun_dir), 0.0);
    col += u_sun_color * ((smoothstep(0.99935, 0.99975, facing) * 1.4 + pow(facing, 700.0) * 0.5) * sky.a);
    if (u_sky.x > 0.01 && d.y > -0.02) {
        vec3 grid = d * 190.0;
        vec3 cell = floor(grid);
        float h = fract(sin(dot(cell, vec3(127.1, 311.7, 74.7))) * 43758.5453);
        float point = smoothstep(0.30, 0.0, length(fract(grid) - 0.5));
        float twinkle = 0.70 + 0.30 * sin(u_sky.y * (1.5 + h * 4.0) + h * 50.0);
        float lit = step(1.0 - 0.035 * u_sky.x, h);
        col += vec3(0.85, 0.92, 1.0) * (lit * point * twinkle * sky.a * smoothstep(-0.02, 0.15, d.y));
    }
    fragColor = vec4(col, 1.0);
}
"""

# Effect sprites and HUD vectors.  texcoord.z selects the shape: 0 soft disc, 1 beam (soft across v), 2 solid quad.
FX_VERT = """#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec4 p3d_Color;
in vec3 p3d_MultiTexCoord0;
out vec4 v_color;
out vec3 v_uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_color = p3d_Color;
    v_uv = p3d_MultiTexCoord0;
}
"""

FX_FRAG = """#version 140
in vec4 v_color;
in vec3 v_uv;
out vec4 fragColor;
void main() {
    float a = v_color.a;
    if (v_uv.z < 0.5) {
        a *= smoothstep(1.0, 0.2, length(v_uv.xy - 0.5) * 2.0);
    } else if (v_uv.z < 1.5) {
        a *= smoothstep(1.0, 0.1, abs(v_uv.y - 0.5) * 2.0);
    }
    fragColor = vec4(v_color.rgb * a, a);                   // premultiplied alpha
}
"""

VIEWMODEL_VERT = """#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
in vec4 p3d_MultiTexCoord0;
out vec3 v_normal;
out vec4 v_color;
out vec4 v_uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_normal = mat3(p3d_ModelMatrix) * p3d_Normal;
    v_color = p3d_Color;
    v_uv = p3d_MultiTexCoord0;
}
"""

VIEWMODEL_FRAG = """#version 140
uniform sampler2DArray u_albedo;
uniform float u_flash;
uniform vec3 u_tint;                   // ambient colour of the arena
uniform vec3 u_key;                    // sun colour of the arena
in vec3 v_normal;
in vec4 v_color;
in vec4 v_uv;
out vec4 fragColor;
void main() {
    vec3 n = normalize(v_normal);
    vec4 base = texture(u_albedo, vec3(v_uv.xy, v_uv.z));
    vec3 albedo = base.rgb * v_color.rgb;
    if (v_uv.w > 0.5) {
        fragColor = vec4(albedo * 1.7, 1.0);
        return;
    }
    vec3 key_dir = normalize(vec3(-0.35, 0.80, -0.45));
    float key = max(dot(n, key_dir), 0.0);
    vec3 light = u_tint * 0.95 + (u_key * 0.55 + 0.45) * key + vec3(0.14);
    vec3 col = albedo * light;
    col += (u_key * 0.5 + 0.5) * (pow(max(dot(n, normalize(key_dir + vec3(0.0, 0.0, -1.0))), 0.0), mix(8.0, 70.0, base.a)) * base.a * 0.5);
    col += albedo * vec3(1.9, 1.35, 0.75) * u_flash * (0.35 + 0.65 * max(n.z, 0.0));
    vec3 over = max(col - 0.75, 0.0);
    col = min(col, vec3(0.75)) + 0.25 * (1.0 - exp(-over * 4.0));
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

# Final resolve: recoil kick-back, chromatic aberration, shake, bloom, damage vignette.
POST_FRAG = """#version 140
uniform sampler2D tex;
uniform sampler2D u_bloom_tex;
uniform vec4 u_kick;        // xy = recoil kick-back offset (uv), z = punch zoom, w = white flash
uniform vec4 u_impact;      // x = chromatic aberration, y = damage vignette, z = desaturation, w = static vignette
uniform vec3 u_shake;       // xy = explosion shake jitter (uv), z = bloom strength
in vec2 uv;
out vec4 fragColor;
void main() {
    vec2 c = uv - 0.5;
    float r2 = dot(c, c);
    vec2 p = clamp(c * (1.0 - u_kick.z) + 0.5 + u_kick.xy + u_shake.xy, 0.001, 0.999);
    vec3 col = texture(tex, p).rgb;
    if (u_impact.x > 0.0004) {                              // chromatic aberration only while it is visible
        vec2 split = c * (u_impact.x * (0.6 + 2.4 * r2));
        col.r = texture(tex, clamp(p + split, 0.001, 0.999)).r;
        col.b = texture(tex, clamp(p - split, 0.001, 0.999)).b;
    }
    if (u_shake.z > 0.0) col += texture(u_bloom_tex, p).rgb * u_shake.z;
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, vec3(luma), u_impact.z);
    float edge = smoothstep(0.08, 0.55, r2 * 2.0);
    col = mix(col, vec3(0.45, 0.01, 0.01) + col * vec3(0.9, 0.30, 0.30), u_impact.y * edge * 0.75);
    col *= 1.0 - u_impact.w * r2 * 1.6;
    col += vec3(u_kick.w);
    fragColor = vec4(col, 1.0);
}
"""

BLOOM_BRIGHT_FRAG = """#version 140
uniform sampler2D tex;
in vec2 uv;
out vec4 fragColor;
void main() {
    vec2 px = 1.0 / vec2(textureSize(tex, 0));
    vec3 c = 0.25 * (texture(tex, uv + px * vec2(-1.0, -1.0)).rgb + texture(tex, uv + px * vec2(1.0, -1.0)).rgb
                   + texture(tex, uv + px * vec2(-1.0, 1.0)).rgb + texture(tex, uv + px * vec2(1.0, 1.0)).rgb);
    float peak = max(c.r, max(c.g, c.b));
    fragColor = vec4(c * smoothstep(0.70, 0.98, peak), 1.0);
}
"""

BLOOM_BLUR_FRAG = """#version 140
uniform sampler2D tex;
uniform vec2 u_axis;
in vec2 uv;
out vec4 fragColor;
void main() {
    vec2 px = u_axis / vec2(textureSize(tex, 0));
    vec3 c = texture(tex, uv).rgb * 0.2270270270;
    c += (texture(tex, uv + px * 1.3846153846).rgb + texture(tex, uv - px * 1.3846153846).rgb) * 0.3162162162;
    c += (texture(tex, uv + px * 3.2307692308).rgb + texture(tex, uv - px * 3.2307692308).rgb) * 0.0702702703;
    fragColor = vec4(c, 1.0);
}
"""


def world_fragment_source(shadows: bool, taps: int, normal_maps: bool) -> str:
    """World fragment program specialised for a quality level."""
    return (WORLD_FRAG_TEMPLATE.replace("@SHADOWS@", "1" if shadows else "0").replace("@TAPS@", str(int(taps)))
            .replace("@NORMAL_MAPS@", "1" if normal_maps else "0").replace("@LIGHTS@", str(C.MAX_DYNAMIC_LIGHTS)))


# =========================================================================== #
#  Vertex formats and geometry helpers
# =========================================================================== #
FX_VERTEX = np.dtype([("pos", "<f4", 3), ("col", "<f4", 4), ("uv", "<f4", 3)])     # uv.z = sprite shape
_FORMATS: dict = {}

_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
BOX_LOCAL = (FACE_N[:, None, :] + FACE_A[:, None, :] * _CORNERS[None, :, 0:1]
             + FACE_B[:, None, :] * _CORNERS[None, :, 1:2])                         # (6, 4, 3) corners of [-1, 1]^3
BOX_NORMAL = np.repeat(FACE_N[:, None, :], 4, axis=1)                               # (6, 4, 3)
QUAD_UV = (_CORNERS + 1.0) * 0.5
_QUAD_INDEX = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
# Texture axes of a box face in the same layout as BOX_LOCAL: u horizontal, v up the wall.
_BOX_U = np.stack([BOX_LOCAL[f, :, worldmesh.UV_AXIS_U[f]] for f in range(6)])       # (6, 4)
_BOX_V = np.stack([BOX_LOCAL[f, :, worldmesh.UV_AXIS_V[f]] for f in range(6)])


def vertex_format(dtype: np.dtype) -> GeomVertexFormat:
    """Interleaved float32 Panda3D layout matching ``WORLD_VERTEX`` (68 bytes) or ``FX_VERTEX`` (40 bytes)."""
    key = "world" if dtype == WORLD_VERTEX else "fx"
    if key not in _FORMATS:
        array = GeomVertexArrayFormat()
        array.addColumn(InternalName.getVertex(), 3, Geom.NTFloat32, Geom.CPoint)
        if key == "world":
            array.addColumn(InternalName.getNormal(), 3, Geom.NTFloat32, Geom.CNormal)
            array.addColumn(InternalName.getColor(), 4, Geom.NTFloat32, Geom.CColor)
            array.addColumn(InternalName.getTexcoord(), 4, Geom.NTFloat32, Geom.CTexcoord)
            array.addColumn(InternalName.make("baked_light"), 3, Geom.NTFloat32, Geom.COther)
        else:
            array.addColumn(InternalName.getColor(), 4, Geom.NTFloat32, Geom.CColor)
            array.addColumn(InternalName.getTexcoord(), 3, Geom.NTFloat32, Geom.CTexcoord)
        fmt = GeomVertexFormat()
        fmt.addArray(array)
        registered = GeomVertexFormat.registerFormat(fmt)
        if registered.getArray(0).getStride() != dtype.itemsize:
            raise RuntimeError("vertex layout mismatch between NumPy and Panda3D")
        _FORMATS[key] = registered
    return _FORMATS[key]


def quad_indices(quads: int) -> np.ndarray:
    """Index buffer for ``quads`` four-vertex quads (two triangles each)."""
    return (np.arange(quads, dtype=np.uint32)[:, None] * 4 + _QUAD_INDEX[None, :]).ravel()


def box_vertices(centers, halves, colors, layer, emissive=0.0, uv_scale=1.0) -> np.ndarray:
    """
    ``WORLD_VERTEX`` data of ``B`` axis aligned boxes: 24 vertices each (flat
    shaded faces), generated by broadcasting the unit-cube tables.  Texture
    coordinates are the face extents in metres times ``uv_scale``.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    halves = np.asarray(halves, dtype=np.float64).reshape(-1, 3)
    count = centers.shape[0]
    out = np.zeros(count * 24, dtype=WORLD_VERTEX)
    out["pos"] = (centers[:, None, None, :] + BOX_LOCAL[None] * halves[:, None, None, :]).reshape(-1, 3)
    out["nrm"] = np.broadcast_to(BOX_NORMAL[None], (count, 6, 4, 3)).reshape(-1, 3)
    out["col"] = np.repeat(np.asarray(colors, dtype=np.float32).reshape(count, 4), 24, axis=0)
    half_u = halves[:, worldmesh.UV_AXIS_U][:, :, None]                                # (B, 6, 1)
    half_v = halves[:, worldmesh.UV_AXIS_V][:, :, None]
    scale = np.broadcast_to(np.asarray(uv_scale, dtype=np.float64), (count,))[:, None, None]
    out["uv"][:, 0] = ((_BOX_U[None] + 1.0) * half_u * scale).reshape(-1)
    out["uv"][:, 1] = ((_BOX_V[None] + 1.0) * half_v * scale).reshape(-1)
    out["uv"][:, 2] = np.repeat(np.broadcast_to(np.asarray(layer, dtype=np.float32), (count,)), 24)
    out["uv"][:, 3] = np.repeat(np.broadcast_to(np.asarray(emissive, dtype=np.float32), (count,)), 24)
    return out


def _index_primitive(indices: np.ndarray) -> GeomTriangles:
    prim = GeomTriangles(Geom.UHStatic)
    prim.setIndexType(GeomEnums.NTUint32)
    handle = prim.modifyVertices()
    handle.uncleanSetNumRows(len(indices))
    memoryview(handle).cast("B")[:] = np.ascontiguousarray(indices, dtype=np.uint32).view(np.uint8)
    return prim


def make_geom_node(name: str, vertices: np.ndarray, indices: np.ndarray, dynamic: bool = False):
    """Upload a vertex / index pair into a new GeomNode; returns ``(NodePath, GeomVertexData)``."""
    usage = Geom.UHDynamic if dynamic else Geom.UHStatic
    vdata = GeomVertexData(name, vertex_format(vertices.dtype), usage)
    vdata.uncleanSetNumRows(len(vertices))
    memoryview(vdata.modifyArray(0)).cast("B")[:] = np.ascontiguousarray(vertices).view(np.uint8)
    geom = Geom(vdata)
    geom.addPrimitive(_index_primitive(indices))
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
    node.hide(SHADOW_MASKS)


class DynamicMesh:
    """
    Quad buffer that is rewritten from a NumPy array every frame.

    The GPU buffer is sized to the live vertex count rounded up to a power of
    two (index primitives are cached per size), so an idle pool is neither
    uploaded nor drawn and a busy one only moves the bytes it uses.
    """
    MIN_ROWS = 64

    def __init__(self, name: str, max_quads: int, parent: NodePath, shader: Shader, dtype: np.dtype = FX_VERTEX):
        self.dtype = dtype
        self.capacity = int(max_quads) * 4
        self.rows = min(self.MIN_ROWS, self.capacity)
        self._primitives: dict = {}
        self._zeros = bytes(self.capacity * dtype.itemsize)
        self.vdata = GeomVertexData(name, vertex_format(dtype), Geom.UHDynamic)
        self.vdata.uncleanSetNumRows(self.rows)
        memoryview(self.vdata.modifyArray(0)).cast("B")[:] = self._zeros[: self.rows * dtype.itemsize]
        geom = Geom(self.vdata)
        geom.addPrimitive(self._primitive(self.rows))
        self.geom_node = GeomNode(name)
        self.geom_node.addGeom(geom)
        self.geom_node.setBounds(OmniBoundingVolume())
        self.geom_node.setFinal(True)
        self.node = NodePath(self.geom_node)
        self.node.reparentTo(parent)
        if shader is not None:                       # None: inherit the parent's (world) shader
            self.node.setShader(shader)
        self.node.hide()
        self.used = 0

    def _primitive(self, rows: int) -> GeomTriangles:
        if rows not in self._primitives:
            self._primitives[rows] = _index_primitive(quad_indices(rows // 4))
        return self._primitives[rows]

    def upload(self, vertices: np.ndarray) -> None:
        """Replace the buffer contents with ``vertices`` (four per quad)."""
        count = min(len(vertices), self.capacity)
        if count == 0:
            if self.used:
                self.used = 0
                self.node.hide()
            return
        rows = min(max(self.MIN_ROWS, 1 << (count - 1).bit_length()), self.capacity)
        if rows != self.rows:
            if rows > self.rows:
                self.vdata.uncleanSetNumRows(rows)
                self.geom_node.modifyGeom(0).setPrimitive(0, self._primitive(rows))
            else:
                self.geom_node.modifyGeom(0).setPrimitive(0, self._primitive(rows))
                self.vdata.uncleanSetNumRows(rows)
            self.rows = rows
        payload = np.ascontiguousarray(vertices[:count]).view(np.uint8)
        target = memoryview(self.vdata.modifyArray(0)).cast("B")
        target[: payload.size] = payload
        if count < rows:
            target[payload.size: rows * self.dtype.itemsize] = self._zeros[: (rows - count) * self.dtype.itemsize]
        if not self.used:
            self.node.show()
        self.used = count


EMPTY = np.zeros(0, dtype=FX_VERTEX)
EMPTY_WORLD = np.zeros(0, dtype=WORLD_VERTEX)


# =========================================================================== #
#  Texture loading: procedural material arrays, image overrides, panoramic sky
# =========================================================================== #
def load_image_rgba(path, width: int = 0, height: int = 0):
    """
    Read an image file (PNG, JPG, TGA, ...) as an ``(H, W, 4)`` uint8 array
    with row 0 at the bottom, optionally resampled.  Returns ``None`` when
    the file cannot be decoded.
    """
    image = PNMImage()
    if not image.read(Filename.fromOsSpecific(str(path))):
        return None
    if width and height and (image.getXSize() != width or image.getYSize() != height):
        scaled = PNMImage(width, height, image.getNumChannels())
        scaled.quickFilterFrom(image)
        image = scaled
    texture = Texture()
    texture.load(image)
    data = np.frombuffer(bytes(texture.getRamImageAs("RGBA")), dtype=np.uint8)
    return data.reshape(image.getYSize(), image.getXSize(), 4).copy()


def prune_stale_bakes() -> int:
    """Delete cached bakes written by older revisions of the generators; returns how many files went."""
    current = {"nav": maps.NAV_REVISION, "mesh": worldmesh.MESH_REVISION, "materials": MATERIAL_REVISION,
               "sky": SKY_REVISION}
    removed = 0
    if C.BAKE_DIR.is_dir():
        for path in C.BAKE_DIR.glob("*.npz"):
            head = path.name.split("_", 1)[0]
            family = head.rstrip("0123456789")
            revision = head[len(family):]
            if family in current and revision.isdigit() and int(revision) != current[family]:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
    return removed


def _find_image(folder, stem: str):
    for extension in (".png", ".jpg", ".jpeg", ".tga", ".bmp"):
        path = folder / (stem + extension)
        if path.exists():
            return path
    return None


class MaterialLibrary:
    """
    The two texture arrays every lit surface samples.  Layers follow
    ``maps.TEXTURE_LAYERS``.  ``content/textures/<layer>.png`` replaces a
    layer's albedo (its alpha channel, if any, is the gloss) and
    ``<layer>_n.png`` its normal map.
    """

    def __init__(self):
        self.albedo = Texture("materials_albedo")
        self.relief = Texture("materials_relief")
        self.size = 0
        self.overrides: list = []

    def build(self, size: int, anisotropy: int) -> None:
        if size != self.size:
            albedo, relief = self._generate(size)
            self.overrides = []
            for index, name in enumerate(maps.TEXTURE_LAYERS):
                path = _find_image(C.TEXTURES_DIR, name) if C.TEXTURES_DIR.is_dir() else None
                image = load_image_rgba(path, size, size) if path is not None else None
                if image is not None:
                    if int(image[..., 3].min()) == 255:
                        image[..., 3] = albedo[index, ..., 3]            # opaque file: keep the procedural gloss
                    albedo[index] = image
                    self.overrides.append(path.name)
                path = _find_image(C.TEXTURES_DIR, name + "_n") if C.TEXTURES_DIR.is_dir() else None
                image = load_image_rgba(path, size, size) if path is not None else None
                if image is not None:
                    relief[index] = image
                    self.overrides.append(path.name)
            for texture, data in ((self.albedo, albedo), (self.relief, relief)):
                texture.setup2dTextureArray(size, size, data.shape[0], Texture.TUnsignedByte, Texture.FRgba8)
                texture.setRamImageAs(data.tobytes(), "RGBA")
                texture.setWrapU(SamplerState.WMRepeat)
                texture.setWrapV(SamplerState.WMRepeat)
                texture.setMinfilter(SamplerState.FTLinearMipmapLinear)
                texture.setMagfilter(SamplerState.FTLinear)
            self.size = size
        self.albedo.setAnisotropicDegree(max(int(anisotropy), 1))
        self.relief.setAnisotropicDegree(max(int(anisotropy), 1))

    @staticmethod
    def _generate(size: int):
        """Procedural layers, cached on disk because synthesis takes a second or two."""
        path = C.BAKE_DIR / ("materials%d_%d_%d.npz" % (MATERIAL_REVISION, len(maps.TEXTURE_LAYERS), size))
        if path.exists():
            try:
                with np.load(path) as stored:
                    albedo, relief = stored["albedo"], stored["relief"]
                if albedo.shape == (len(maps.TEXTURE_LAYERS), size, size, 4) and relief.shape == albedo.shape:
                    return albedo.copy(), relief.copy()
            except (OSError, KeyError, ValueError):
                pass
        albedo, relief = textures.material_layers(size)
        try:
            C.BAKE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, albedo=albedo, relief=relief)
        except OSError:
            pass
        return albedo, relief

    def bind(self, node: NodePath) -> None:
        node.setShaderInput("u_albedo", self.albedo)
        node.setShaderInput("u_relief", self.relief)


def sky_texture(data: maps.MapData, width: int) -> Texture:
    """
    Panoramic sky for an arena: ``content/sky/<key>.png`` (or ``default``) when
    present, otherwise the procedural panorama (cached on disk).
    """
    image = None
    if C.SKY_DIR.is_dir():
        path = _find_image(C.SKY_DIR, data.key) or _find_image(C.SKY_DIR, "default")
        image = load_image_rgba(path) if path is not None else None
    if image is None:
        cache = C.BAKE_DIR / ("sky%d_%s_%08x_%d.npz" % (SKY_REVISION, data.key, data.checksum, width))
        if cache.exists():
            try:
                with np.load(cache) as stored:
                    image = stored["sky"]
            except (OSError, KeyError, ValueError):
                image = None
        if image is None or image.shape != (width // 2, width, 4):
            image = textures.sky_panorama(width, data.environment, seed=data.checksum & 0xFFFF)
            try:
                C.BAKE_DIR.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, sky=image)
            except OSError:
                pass
    texture = Texture("sky_panorama")
    texture.setup2dTexture(image.shape[1], image.shape[0], Texture.TUnsignedByte, Texture.FRgba8)
    texture.setRamImageAs(np.ascontiguousarray(image).tobytes(), "RGBA")
    texture.setWrapU(SamplerState.WMRepeat)
    texture.setWrapV(SamplerState.WMClamp)
    texture.setMinfilter(SamplerState.FTLinear)
    texture.setMagfilter(SamplerState.FTLinear)
    return texture


def sky_dome(segments: int = 64, rings: int = 32):
    """
    Unit sphere seen from the inside (FX vertex layout).  Positions double as
    view directions and the texture coordinates are the equirectangular
    panorama coordinates, so the fragment shader needs no trigonometry.
    """
    azimuth = np.linspace(-math.pi, math.pi, segments + 1)
    elevation = np.linspace(-math.pi * 0.5, math.pi * 0.5, rings + 1)
    az, el = np.meshgrid(azimuth, elevation, indexing="xy")
    vertices = np.zeros(az.size, dtype=FX_VERTEX)
    vertices["pos"] = np.stack([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], axis=-1).reshape(-1, 3)
    vertices["col"] = 1.0
    vertices["uv"][:, 0] = (az / (2.0 * math.pi) + 0.5).ravel()
    vertices["uv"][:, 1] = (el / math.pi + 0.5).ravel()
    return vertices, worldmesh._grid_indices(segments, rings).astype(np.uint32)


# =========================================================================== #
#  Dynamic point lights (muzzle flashes, explosions)
# =========================================================================== #
class LightPool:
    """
    Pool of short-lived point lights fed to the world shader as uniform
    arrays.  A muzzle flash is a light of intensity ``I0`` that exists for
    exactly ``MUZZLE_FLASH_TIME`` (0.03 s) and then terminates itself; the
    shader attenuates it with ``I0 / d^2``.  Only the live lights are sent
    (``u_light_count``), so most frames skip the per-pixel loop entirely.
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
        for _ in range(self.count):
            self.pta_pos.pushBack(LVecBase4f(0, 0, 0, 0))
            self.pta_col.pushBack(LVecBase4f(0, 0, 0, 0))
        self.flash_level = 0.0
        self.active = 0
        self._bound: list = []
        self._sent = -1

    def bind(self, node: NodePath) -> None:
        node.setShaderInput("u_light_pos", self.pta_pos)
        node.setShaderInput("u_light_col", self.pta_col)
        node.setShaderInput("u_light_count", 0)
        self._bound.append(node)

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

    def explosion(self, position, rgb=(1.0, 0.55, 0.20), intensity: float = 55.0, duration: float = 0.35) -> None:
        self.add(position, rgb, intensity, duration, True)

    def gains(self) -> np.ndarray:
        alive = self.life > 0.0
        return np.where(self.fade, (self.life / self.ttl) ** 2, 1.0) * alive * self.intensity

    def irradiance(self, points: np.ndarray) -> np.ndarray:
        """Unshadowed light of the live dynamic lights at ``points`` (used for characters' rim light)."""
        gain = self.gains()
        delta = self.pos[None, :, :] - points[:, None, :]
        d2 = np.maximum(np.sum(delta * delta, axis=2), 0.20)
        return np.sum(self.color[None, :, :] * (gain[None, :] / d2)[:, :, None], axis=1)

    def update(self, dt: float) -> None:
        """Age the lights (call once per frame, before new flashes are added)."""
        self.life = np.maximum(self.life - dt, 0.0)
        gain = self.gains()
        self.flash_level = float(np.max(np.where(self.fade, 0.0, gain))) if (self.life > 0.0).any() else 0.0

    def write(self) -> None:
        """Pack the live lights to the front of the uniform arrays."""
        gain = self.gains()
        live = np.flatnonzero(gain > 0.0)
        for slot, i in enumerate(live.tolist()):
            p, c = self.pos[i], self.color[i]
            self.pta_pos[slot] = LVecBase4f(float(p[0]), float(p[1]), float(p[2]), float(gain[i]))
            self.pta_col[slot] = LVecBase4f(float(c[0]), float(c[1]), float(c[2]), 0.0)
        self.active = int(live.size)
        if self.active != self._sent:
            self._sent = self.active
            for node in self._bound:
                node.setShaderInput("u_light_count", self.active)


# =========================================================================== #
#  Sun shadows: static depth map rendered once per arena + dynamic map per frame
# =========================================================================== #
class SunShadows:
    """
    Orthographic shadow maps for the sun.

    The level never moves, so its depth map is rendered for two frames after
    an arena loads and then frozen.  Characters go into a second, smaller map
    that is redrawn every frame.  Both use one light matrix, built here with
    NumPy and handed to the shaders as ``u_light_vp`` (the cameras exist only
    to drive the passes; their own lenses are bypassed).
    """

    def __init__(self, root: NodePath):
        self.root = root
        self.size = 0
        self.buffers: list = []
        self.cameras: list = []
        self.static_tex = self._depth_texture("shadow_static")
        self.dynamic_tex = self._depth_texture("shadow_dynamic")
        self.pending = 0
        self.matrix = np.eye(4)
        self.texel_world = 0.05
        caster = NodePath("shadow_caster_state")
        caster.setShader(Shader.make(Shader.SL_GLSL, SHADOW_VERT, SHADOW_FRAG), 100)
        caster.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone), 100)
        self.caster_state = caster.getState()
        root.setShaderInput("u_light_vp", LMatrix4f.identMat())
        root.setShaderInput("u_shadow", LVecBase4f(0.001, 0.001, 0.0, 0.0))
        root.setShaderInput("u_shadow_static", self.static_tex)
        root.setShaderInput("u_shadow_dynamic", self.dynamic_tex)

    @staticmethod
    def _depth_texture(name: str) -> Texture:
        texture = Texture(name)
        texture.setMinfilter(SamplerState.FTShadow)
        texture.setMagfilter(SamplerState.FTShadow)
        texture.setWrapU(SamplerState.WMClamp)
        texture.setWrapV(SamplerState.WMClamp)
        return texture

    def _make_pass(self, name: str, size: int, texture: Texture, mask: BitMask32, sort: int):
        base = application.base
        props = FrameBufferProperties()
        props.setRgbColor(False)
        props.setRgbaBits(0, 0, 0, 0)
        props.setDepthBits(24)
        buffer = base.graphicsEngine.makeOutput(base.pipe, name, sort, props, WindowProperties.size(size, size),
                                                GraphicsPipe.BFRefuseWindow, base.win.getGsg(), base.win)
        if buffer is None:
            return None
        buffer.addRenderTexture(texture, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPDepth)
        buffer.setClearColorActive(False)
        buffer.setClearDepthActive(True)
        node = PandaCamera(name + "_camera")
        lens = OrthographicLens()
        lens.setFilmSize(10, 10)
        lens.setNearFar(1, 100)
        node.setLens(lens)
        node.setCullBounds(OmniBoundingVolume())
        node.setInitialState(self.caster_state)
        node.setScene(self.root)
        node.setCameraMask(mask)
        path = self.root.attachNewNode(node)
        buffer.makeDisplayRegion().setCamera(path)
        self.buffers.append(buffer)
        self.cameras.append(path)
        return buffer

    def configure(self, size: int) -> bool:
        """(Re)create the depth passes for a shadow map edge of ``size`` (0 switches shadows off)."""
        if size == self.size:
            return self.enabled
        base = application.base
        for buffer in self.buffers:
            base.graphicsEngine.removeWindow(buffer)
        for path in self.cameras:
            path.removeNode()
        self.buffers, self.cameras = [], []
        self.size = int(size)
        if size > 0:
            static = self._make_pass("sun_shadow_static", size, self.static_tex, SHADOW_STATIC_MASK, -2000)
            dynamic = self._make_pass("sun_shadow_dynamic", min(size, C.SHADOW_DYNAMIC_SIZE), self.dynamic_tex,
                                      SHADOW_DYNAMIC_MASK, -1999)
            if static is None or dynamic is None:
                self.size = 0
                return self.configure(0)
            self.pending = 3
        return self.enabled

    @property
    def enabled(self) -> bool:
        return self.size > 0 and len(self.buffers) == 2

    def fit(self, bounds_min, bounds_max, sun_dir) -> None:
        """Fit the light frustum to the arena's bounding box and schedule the static map for rendering."""
        forward = M.normalize(np.asarray(sun_dir, dtype=np.float64))
        helper = np.array([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.98 else np.array([1.0, 0.0, 0.0])
        right = M.normalize(np.cross(helper, forward))
        up = np.cross(forward, right)
        lo = np.asarray(bounds_min, dtype=np.float64) - 1.0
        hi = np.asarray(bounds_max, dtype=np.float64) + np.array([1.0, 3.0, 1.0])
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        local = corners @ np.stack([right, up, forward]).T
        mn, mx = local.min(axis=0), local.max(axis=0)
        mn[2] -= 2.0
        matrix = np.eye(4)
        for row, axis in enumerate((right, up, forward)):
            matrix[row, :3] = axis * 2.0 / (mx[row] - mn[row])
            matrix[row, 3] = -(mx[row] + mn[row]) / (mx[row] - mn[row])
        self.matrix = matrix
        size = max(self.size, 1)
        self.texel_world = float(max(mx[0] - mn[0], mx[1] - mn[1])) / size
        self.root.setShaderInput("u_light_vp", LMatrix4f(*matrix.T.reshape(-1).tolist()))   # Panda stores row vectors
        self.root.setShaderInput("u_shadow", LVecBase4f(1.0 / size, C.SHADOW_DEPTH_BIAS / float(mx[2] - mn[2]),
                                                        C.SHADOW_NORMAL_OFFSET * self.texel_world, 0.0))
        self.pending = 3

    def update(self) -> None:
        """Keep the static pass alive only while its map is being (re)rendered."""
        if not self.enabled:
            return
        static = self.buffers[0]
        if self.pending > 0:
            self.pending -= 1
            if not static.isActive():
                static.setActive(True)
        elif static.isActive():
            static.setActive(False)


# =========================================================================== #
#  World (static level mesh, skyline, sky dome)
# =========================================================================== #
class WorldRenderer:
    """Builds the level as a single static mesh and owns the world shader inputs."""

    def __init__(self, lights: LightPool, materials: MaterialLibrary, preset: dict):
        self.root = NodePath("nexus_world")
        self.root.reparentTo(scene)
        self.sky_shader = Shader.make(Shader.SL_GLSL, SKY_VERT, SKY_FRAG)
        self.fx_shader = Shader.make(Shader.SL_GLSL, FX_VERT, FX_FRAG)
        self.lights = lights
        self.materials = materials
        self.shadows = SunShadows(self.root)
        self.shader = None
        self.static = None
        self.backdrop = None
        self.sky = None
        self.data = None
        self.preset = dict(preset)
        self.stats = {"vertices": 0, "triangles": 0, "seconds": 0.0, "cached": False}
        self.ambient = (0.3, 0.3, 0.35)
        self.sun_color = (1.0, 1.0, 1.0)
        self.stars = 0.0
        self.static_lights = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
        lights.bind(self.root)
        materials.bind(self.root)
        for name, value in (("u_camera_pos", (0.0, 0.0, 0.0)), ("u_sun_dir", (0.0, -1.0, 0.0)), ("u_sun_color", (1.0, 1.0, 1.0)),
                            ("u_ambient", self.ambient), ("u_fog_color", (0.5, 0.5, 0.5)), ("u_fog_density", 0.0)):
            self.root.setShaderInput(name, value)
        self.apply_preset(preset)

    def apply_preset(self, preset: dict) -> None:
        """Compile the world shader for a quality level and resize textures and shadow maps to match."""
        self.preset = dict(preset)
        self.materials.build(int(preset["texture_size"]), int(preset["anisotropy"]))
        shadows = self.shadows.configure(int(preset["shadow_size"]))
        source = world_fragment_source(shadows, int(preset["shadow_taps"]), bool(preset["normal_maps"]))
        self.shader = Shader.make(Shader.SL_GLSL, WORLD_VERT, source)
        self.root.setShader(self.shader)
        if self.data is not None:
            self.load_map(self.data)

    def load_map(self, data: maps.MapData) -> None:
        """(Re)build the static geometry and environment for an arena."""
        for node in (self.static, self.backdrop, self.sky):
            if node is not None:
                node.removeNode()
        self.data = data
        mesh = worldmesh.load_world_mesh(data, None, float(self.preset["light_cell"]))
        self.stats = dict(mesh["stats"])
        self.static, _ = make_geom_node("static_world", mesh["vertices"], mesh["indices"])
        self.static.reparentTo(self.root)
        self.static.hide(SHADOW_DYNAMIC_MASK)
        self.backdrop = None
        if mesh["sky_vertices"].shape[0]:
            self.backdrop, _ = make_geom_node("skyline", mesh["sky_vertices"], mesh["sky_indices"])
            self.backdrop.reparentTo(self.root)
            self.backdrop.hide(SHADOW_MASKS)
            self.backdrop.setBin("background", 5)

        sun = M.normalize(np.asarray(data.sun_dir, dtype=np.float64))
        self.ambient = tuple(float(v) for v in data.ambient)
        self.sun_color = tuple(float(v) for v in data.sun_color)
        self.stars = float(data.environment.get("stars", 0.0))
        self.root.setShaderInput("u_sun_dir", (float(sun[0]), float(sun[1]), float(sun[2])))
        self.root.setShaderInput("u_sun_color", self.sun_color)
        self.root.setShaderInput("u_ambient", self.ambient)
        self.root.setShaderInput("u_fog_color", tuple(float(v) for v in data.fog_color))
        self.root.setShaderInput("u_fog_density", float(data.fog_density))
        self.static_lights = (np.asarray(data.lights_pos, dtype=np.float64).reshape(-1, 3),
                              np.asarray(data.lights_color, dtype=np.float64).reshape(-1, 3),
                              np.asarray(data.lights_radius, dtype=np.float64).reshape(-1))
        self.shadows.fit(data.bounds_min, data.bounds_max, sun)

        dome_vertices, dome_indices = sky_dome()
        self.sky, _ = make_geom_node("sky", dome_vertices, dome_indices)
        self.sky.reparentTo(self.root)
        self.sky.setScale(C.SKY_DOME_RADIUS)
        self.sky.setShader(self.sky_shader)
        self.sky.setShaderInput("u_panorama", sky_texture(data, int(self.preset["sky_size"])))
        self.sky.setShaderInput("u_sky", (self.stars, 0.0))
        self.sky.setTwoSided(True)
        self.sky.setDepthWrite(False)
        self.sky.setDepthTest(False)
        self.sky.setBin("background", 0)
        self.sky.hide(SHADOW_MASKS)
        window.color = color.rgba(*[float(v) for v in data.fog_color], 1.0)

    def static_irradiance(self, points: np.ndarray) -> np.ndarray:
        """Unshadowed light of the arena's lamps at ``points``; keeps moving objects in tune with baked lighting."""
        positions, colours, radii = self.static_lights
        if positions.shape[0] == 0 or points.shape[0] == 0:
            return np.zeros((points.shape[0], 3))
        delta = positions[None, :, :] - points[:, None, :]
        d2 = np.sum(delta * delta, axis=2)
        r2 = (radii * radii)[None, :]
        window_ = np.clip(1.0 - d2 / r2, 0.0, 1.0)
        energy = (r2 / 15.0) / np.maximum(d2, 0.20) * window_ * window_ * 0.7
        return np.sum(colours[None, :, :] * energy[:, :, None], axis=1)

    def update(self, cam_pos, time_now: float) -> None:
        position = (float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        self.root.setShaderInput("u_camera_pos", position)
        if self.sky is not None:
            self.sky.setPos(*position)
            if self.stars > 0.01:
                self.sky.setShaderInput("u_sky", (self.stars, float(time_now % 3600.0)))
        self.shadows.update()


# =========================================================================== #
#  Vectorised effect pools
# =========================================================================== #
def fx_quads(corners: np.ndarray, rgba: np.ndarray, shape: float) -> np.ndarray:
    """``FX_VERTEX`` data for ``(Q, 4, 3)`` quad corners with one colour per quad."""
    count = corners.shape[0]
    out = np.zeros(count * 4, dtype=FX_VERTEX)
    out["pos"] = corners.reshape(-1, 3)
    out["col"] = np.repeat(rgba, 4, axis=0)
    out["uv"][:, :2] = np.tile(QUAD_UV, (count, 1))
    out["uv"][:, 2] = shape
    return out


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
        self.live = 0
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
        self.live = self.capacity

    def update(self, dt: float) -> None:
        if not self.live:
            return
        alive = self.life > 0.0
        self.live = int(np.count_nonzero(alive))
        if not self.live:
            return
        self.vel[alive, 1] -= C.GRAVITY * self.gravity[alive] * dt
        self.vel[alive] *= np.exp(-self.drag[alive] * dt)[:, None]
        self.pos[alive] += self.vel[alive] * dt
        self.life[alive] -= dt

    def build(self, right, up) -> None:
        """Camera facing quads for every live particle -> vertex buffer."""
        if not self.live:
            self.mesh.upload(EMPTY)
            return
        idx = np.flatnonzero(self.life > 0.0)
        if idx.size == 0:
            self.mesh.upload(EMPTY)
            return
        age = 1.0 - self.life[idx] / self.ttl[idx]
        size = M.lerp(self.size0[idx], self.size1[idx], age)
        corners = (self.pos[idx][:, None, :]
                   + (right[None, None, :] * _CORNERS[None, :, 0:1] + up[None, None, :] * _CORNERS[None, :, 1:2])
                   * size[:, None, None])
        rgba = self.color[idx].copy()
        rgba[:, 3] *= np.clip(1.0 - age, 0.0, 1.0)
        self.mesh.upload(fx_quads(corners, rgba, 0.0))


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
        if idx.size == 0:
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
        self.mesh.upload(fx_quads(corners, self.color[idx], 1.0))


class DecalPool:
    """Ring buffer of bullet scorch marks glued to the surfaces they hit."""

    def __init__(self, capacity: int, parent: NodePath, shader: Shader):
        self.capacity = int(capacity)
        self.vertices = np.zeros(self.capacity * 4, dtype=FX_VERTEX)
        self.cursor = 0
        self.filled = 0
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
        block["col"] = rgba
        block["uv"][:, :2] = QUAD_UV
        block["uv"][:, 2] = 0.0
        self.cursor = (self.cursor + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)
        self.dirty = True

    def clear(self) -> None:
        self.vertices[:] = 0
        self.cursor = 0
        self.filled = 0
        self.dirty = True

    def build(self) -> None:
        if self.dirty:
            self.mesh.upload(self.vertices[: self.filled * 4])
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
PART_LAYER = np.full(PART_COUNT, float(maps.LAYER["armor"]))
PART_LAYER[PART_WEAPON] = float(maps.LAYER["metal"])
PART_LAYER[PART_VISOR] = float(maps.LAYER["glow"])
# Texture coordinates of every part face in metres (armour plating repeats every 0.5 m).
PART_UV = np.stack([(_BOX_U[None] + 1.0) * PART_STAND_HALF[:, worldmesh.UV_AXIS_U][:, :, None],
                    (_BOX_V[None] + 1.0) * PART_STAND_HALF[:, worldmesh.UV_AXIS_V][:, :, None]], axis=-1) * 2.0


class CharacterBatch:
    """
    Draws every visible remote player (and live grenades) from one vertex
    buffer.  The body parts are the anatomical hitboxes themselves, so the
    silhouette on screen is exactly the volume that can be hit; legs and arms
    swing about their joints and the upper body follows the aim pitch.  The
    batch is textured and lit like the level and casts shadows.
    """

    def __init__(self, parent: NodePath, capacity: int = C.MAX_PLAYERS):
        self.capacity = int(capacity)
        self.phase = np.zeros(self.capacity, dtype=np.float64)
        self.flash = np.zeros(self.capacity, dtype=np.float64)
        quads = (self.capacity * PART_COUNT + C.GRENADE_CAPACITY) * 6
        self.mesh = DynamicMesh("characters", quads, parent, None, WORLD_VERTEX)
        self.mesh.node.hide(SHADOW_STATIC_MASK)
        self.drawn = 0

    def hit(self, player: int, strength: float = 0.55) -> None:
        if 0 <= player < self.capacity:
            self.flash[player] = strength

    def build(self, view, skip: int, dt: float, time_now: float, grenade_pos=None, irradiance=None) -> None:
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
            rgb[:, PART_WEAPON, :] = (0.30, 0.32, 0.36)
            rgb[:, PART_VISOR, :] = np.minimum(base * 1.5 + 0.25, 1.6)
            glow = np.broadcast_to(self.flash[rows][:, None], (n, PART_COUNT)).copy()
            glow[:, PART_VISOR] = 1.0
            per_player = PART_COUNT * 24
            out = np.zeros(n * per_player, dtype=WORLD_VERTEX)
            out["pos"] = world.reshape(-1, 3)
            out["nrm"] = wnormal.reshape(-1, 3)
            out["col"][:, :3] = np.repeat(rgb.reshape(-1, 3), 24, axis=0)
            out["col"][:, 3] = 1.0
            out["uv"][:, :2] = np.tile(PART_UV.reshape(-1, 2), (n, 1))
            out["uv"][:, 2] = np.tile(np.repeat(PART_LAYER, 24), n)
            out["uv"][:, 3] = np.repeat(glow.reshape(-1), 24)
            if irradiance is not None:
                out["light"] = np.repeat(irradiance(view.pos[rows] + np.array([0.0, 1.1, 0.0])), per_player, axis=0)
            blocks.append(out)
        if grenade_pos is not None and len(grenade_pos):
            g = np.asarray(grenade_pos, dtype=np.float64).reshape(-1, 3)[: C.GRENADE_CAPACITY]
            pulse = 0.75 + 0.25 * math.sin(time_now * 22.0)
            colors = np.tile(np.array([1.0 * pulse, 0.35 * pulse, 0.12, 1.0]), (g.shape[0], 1))
            blocks.append(box_vertices(g, np.full((g.shape[0], 3), 0.09), colors, maps.LAYER["glow"], 1.2, 4.0))
        if not blocks:
            self.mesh.upload(EMPTY_WORLD)
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
    to vertices in one broadcast expression (only lit segments are emitted).
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
                 * size[:, None, None, None, None])                                         # (n, 4, 7, 4, 3)
        rgba = np.concatenate([np.broadcast_to(self.color[idx][:, None, None, :], lit.shape + (3,)),
                               np.broadcast_to(remaining[:, None, None, None], lit.shape + (1,))], axis=-1)
        self.mesh.upload(fx_quads(world[lit], rgba[lit], 2.0))


# =========================================================================== #
#  Screen-space post processing, render scale and dynamic resolution
# =========================================================================== #
class ScaledFilterManager(FilterManager):
    """A FilterManager whose off-screen buffers are ``render_scale`` times the window size."""

    def __init__(self, win, cam):
        self.render_scale = 1.0
        FilterManager.__init__(self, win, cam)

    def getScaledSize(self, mul, div, align):
        width, height = FilterManager.getScaledSize(self, mul, div, align)
        return max(int(width * self.render_scale), 16), max(int(height * self.render_scale), 16)


def _screen_texture(name: str) -> Texture:
    texture = Texture(name)
    texture.setMinfilter(SamplerState.FTLinear)
    texture.setMagfilter(SamplerState.FTLinear)
    texture.setWrapU(SamplerState.WMClamp)
    texture.setWrapV(SamplerState.WMClamp)
    return texture


class PostFX:
    """
    Renders the 3D scene into an off-screen buffer (at the current render
    scale) and resolves it through ``POST_FRAG``: recoil kick-back (screen
    offset + punch zoom driven by the camera rig's recoil spring), chromatic
    aberration that spikes when damage is taken, explosion shake jitter,
    damage vignette, low-health desaturation and an optional bloom chain
    (bright pass at quarter resolution, separable blur).  If the driver cannot
    create the buffer the game simply runs without the pass.
    """

    def __init__(self, preset: dict):
        base = application.base
        self.manager = ScaledFilterManager(base.win, base.cam)
        self.manager.render_scale = float(preset["render_scale"])
        self.color_tex = _screen_texture("scene_color")
        self.depth_tex = Texture("scene_depth")
        self.quad = self.manager.renderSceneInto(colortex=self.color_tex, depthtex=self.depth_tex)
        self.enabled = self.quad is not None
        self.buffer = self.manager.buffers[0] if self.enabled and self.manager.buffers else None
        self.aberration = 0.0
        self.damage = 0.0
        self.flash = 0.0
        self.desaturation = 0.0
        self.vignette = 0.55
        self.bloom = False
        self.bloom_buffers: list = []
        self.black = Texture("black")
        self.black.setup2dTexture(1, 1, Texture.TUnsignedByte, Texture.FRgba8)
        self.black.setRamImageAs(bytes(4), "RGBA")
        self.bloom_tex = self.black
        if self.enabled:
            vertex = POST_VERT
            self.quad.setShader(Shader.make(Shader.SL_GLSL, vertex, POST_FRAG))
            self.quad.setShaderInput("tex", self.color_tex)
            self.quad.setShaderInput("u_bloom_tex", self.black)
            stages = (("bloom_bright", BLOOM_BRIGHT_FRAG, None), ("bloom_blur_h", BLOOM_BLUR_FRAG, (1.6, 0.0)),
                      ("bloom_blur_v", BLOOM_BLUR_FRAG, (0.0, 1.6)))
            source = self.color_tex
            first = len(self.manager.buffers)
            for name, fragment, axis in stages:
                target = _screen_texture(name)
                quad = self.manager.renderQuadInto(name, div=4, colortex=target)
                if quad is None:
                    source = None
                    break
                quad.setShader(Shader.make(Shader.SL_GLSL, vertex, fragment))
                quad.setShaderInput("tex", source)
                if axis is not None:
                    quad.setShaderInput("u_axis", axis)
                source = target
            self.bloom_buffers = list(self.manager.buffers[first:])
            if source is not None and len(self.bloom_buffers) == len(stages):
                self.bloom_tex = source
            self.set_bloom(bool(preset["bloom"]))
            self._push(0.0, 0.0, 0.0, (0.0, 0.0))

    @property
    def render_scale(self) -> float:
        return float(self.manager.render_scale)

    def set_scale(self, scale: float) -> None:
        """Resize every off-screen buffer to ``scale`` times the window."""
        scale = min(max(float(scale), 0.25), 1.0)
        if self.enabled and abs(scale - self.manager.render_scale) > 1.0e-4:
            self.manager.render_scale = scale
            self.manager.resizeBuffers()

    def set_bloom(self, enabled: bool) -> None:
        self.bloom = bool(enabled) and self.bloom_tex is not self.black
        for buffer in self.bloom_buffers:
            buffer.setActive(self.bloom)
        if self.enabled:
            self.quad.setShaderInput("u_bloom_tex", self.bloom_tex if self.bloom else self.black)

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
        self.quad.setShaderInput("u_shake", (float(shake[0]), float(shake[1]), C.BLOOM_STRENGTH if self.bloom else 0.0))

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
_GUN = (0.42, 0.44, 0.50)
_STEEL = (0.62, 0.64, 0.70)
_POLY = (0.36, 0.30, 0.22)
_SLEEVE = (0.26, 0.31, 0.27)
_GLOVE = (0.13, 0.13, 0.14)
_DOT = (1.00, 0.25, 0.20)
_RAIL = (0.30, 0.90, 1.00)
# view-model surface id -> (texture layer, emissive, texture repeats per metre)
_VM_SURFACE = {0: (maps.LAYER["armor"], 0.0, 5.0), 1: (maps.LAYER["metal"], 0.0, 3.0),
               2: (maps.LAYER["armor"], 0.0, 9.0), 3: (maps.LAYER["glow"], 1.0, 8.0)}

# Per weapon: list of (centre, size, colour, surface) boxes in gun space (x right, y up, z forward, metres).
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
    the screen-space effects.  Its parts sample the same material arrays as
    the level.
    """

    def __init__(self, post: PostFX, fx_shader: Shader, materials: MaterialLibrary):
        base = application.base
        self.root = NodePath("viewmodel_root")
        self.shader = Shader.make(Shader.SL_GLSL, VIEWMODEL_VERT, VIEWMODEL_FRAG)
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
        self.pivot.setShaderInput("u_albedo", materials.albedo)
        self.pivot.setShaderInput("u_flash", 0.0)
        self.pivot.setShaderInput("u_tint", (0.4, 0.4, 0.45))
        self.pivot.setShaderInput("u_key", (1.0, 1.0, 1.0))
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
            surface = [_VM_SURFACE[p[3]] for p in parts]
            vertices = box_vertices(centers, halves, colors, [s[0] for s in surface], [s[1] for s in surface],
                                    [s[2] for s in surface])
            node, _ = make_geom_node("viewmodel_weapon", vertices, quad_indices(len(parts) * 6))
            node.reparentTo(self.pivot)
            node.hide()
            self.models.append(node)
        self.flash_mesh = DynamicMesh("viewmodel_flash", 3, self.pivot, fx_shader)
        set_blend(self.flash_mesh.node, "add")
        self.flash_mesh.node.setBin("fixed", 50)
        self.weapon = -1
        self.flash_time = 0.0
        self.flash_seed = 0
        self.flash_drawn = False
        self.glow = -1.0
        self.pose = None
        self.visible = True
        self.set_weapon(0)
        self.set_fov(C.VIEWMODEL_FOV)

    def set_fov(self, fov: float) -> None:
        self.fov = float(fov)
        self.aspect = 0.0

    def set_environment(self, ambient, sun_color) -> None:
        self.pivot.setShaderInput("u_tint", tuple(float(v) for v in ambient))
        self.pivot.setShaderInput("u_key", tuple(float(v) for v in sun_color))

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
            self.flash_drawn = False

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
        self.flash_time = max(self.flash_time - dt, 0.0)
        if not self.visible:
            return
        spec = WEAPON_MODELS[self.weapon]
        ads = np.array([0.0, -spec["sight"], 0.30])
        base = VM_HIP + (ads - VM_HIP) * rig.ads_blend
        offset = base + rig.vm_offset
        rx, ry, rz = rig.vm_rotation
        pose = (float(offset[0]), float(offset[1]), float(offset[2]), float(rx), float(ry), float(rz))
        if pose != self.pose:
            self.pose = pose
            self.pivot.setPosHpr(pose[0], pose[1], pose[2], -math.degrees(ry), math.degrees(rx), -math.degrees(rz))
        glow = min(flash_level * 0.12, 1.2) if self.flash_time > 0.0 or flash_level > 0.0 else 0.0
        if glow != self.glow:
            self.glow = glow
            self.pivot.setShaderInput("u_flash", float(glow))
        if self.flash_time > 0.0:
            rng = np.random.default_rng(self.flash_seed)
            size = 0.085 * float(rng.uniform(0.8, 1.3)) * (self.flash_time / 0.05 + 0.4)
            roll = float(rng.uniform(0.0, math.pi))
            center = np.array([0.0, 0.036, spec["muzzle"] + 0.05])
            ax = np.array([math.cos(roll), math.sin(roll), 0.0])
            ay = np.array([-math.sin(roll), math.cos(roll), 0.0])
            az = np.array([0.0, 0.0, 1.0])
            corners = np.stack([center[None, :] + (u[None, :] * _CORNERS[:, 0:1] + v[None, :] * _CORNERS[:, 1:2]) * size
                                for u, v in ((ax, ay), (ax, az * 1.6), (ay, az * 1.6))])
            self.flash_mesh.upload(fx_quads(corners, np.tile(np.array([1.0, 0.82, 0.45, 0.9]), (3, 1)), 0.0))
            self.flash_drawn = True
        elif self.flash_drawn:
            self.flash_mesh.upload(EMPTY)
            self.flash_drawn = False


# =========================================================================== #
#  Scene facade: owns every pool and exposes gameplay-level effect calls
# =========================================================================== #
class SceneRenderer:
    """One object the game talks to: load a map, trigger effects, draw a frame."""

    def __init__(self, quality: int = C.QUALITY_DEFAULT, dynamic_resolution: bool = C.DYNAMIC_RESOLUTION_DEFAULT,
                 fps_limit: float = C.FPS_LIMIT_DEFAULT):
        self.quality = int(min(max(quality, 0), len(C.GRAPHICS_PRESETS) - 1))
        preset = C.GRAPHICS_PRESETS[self.quality]
        prune_stale_bakes()
        self.materials = MaterialLibrary()
        self.lights = LightPool()
        self.world = WorldRenderer(self.lights, self.materials, preset)
        root, fx = self.world.root, self.world.fx_shader
        self.sparks = ParticlePool("sparks", 1024, root, fx, "add", seed=11)
        self.smoke = ParticlePool("smoke", 384, root, fx, "alpha", seed=12)
        self.tracers = TracerPool(160, root, fx)
        self.decals = DecalPool(256, root, fx)
        self.characters = CharacterBatch(root)
        self.damage_text = DamageTextPool(40, root, fx)
        self.post = PostFX(preset)
        self.viewmodel = ViewModel(self.post, fx, self.materials)
        self.fps_limit = float(fps_limit)
        self.resolution = ResolutionController(float(preset["render_scale"]), dynamic_resolution, fps_limit)
        self.post.set_scale(self.resolution.scale)
        self.particle_scale = float(preset["particles"])
        self.right = np.array([1.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.forward = np.array([0.0, 0.0, 1.0])

    # ---- configuration ---------------------------------------------------------------------
    def set_quality(self, quality: int) -> None:
        """Switch the graphics preset at run time (textures, shadows, shader variant, bake density, bloom)."""
        quality = int(min(max(quality, 0), len(C.GRAPHICS_PRESETS) - 1))
        if quality == self.quality:
            return
        self.quality = quality
        preset = C.GRAPHICS_PRESETS[quality]
        self.world.apply_preset(preset)
        self.post.set_bloom(bool(preset["bloom"]))
        self.particle_scale = float(preset["particles"])
        self.resolution.configure(float(preset["render_scale"]), self.resolution.enabled, self.fps_limit)
        self.post.set_scale(self.resolution.scale)
        self.viewmodel.set_environment(self.world.ambient, self.world.sun_color)

    def set_pacing(self, dynamic_resolution: bool, fps_limit: float) -> None:
        """Tell the dynamic resolution controller how the frame rate is capped."""
        self.fps_limit = float(fps_limit)
        self.resolution.configure(float(C.GRAPHICS_PRESETS[self.quality]["render_scale"]), dynamic_resolution, fps_limit)
        self.post.set_scale(self.resolution.scale)

    def describe(self) -> dict:
        """Numbers for the diagnostics line and the test-suite."""
        return {"quality": C.QUALITY_NAMES[self.quality], "render_scale": self.post.render_scale,
                "frame_ms": self.resolution.average_ms, "triangles": int(self.world.stats.get("triangles", 0)),
                "vertices": int(self.world.stats.get("vertices", 0)), "shadows": self.world.shadows.enabled,
                "shadow_size": self.world.shadows.size, "bloom": self.post.bloom, "lights": self.lights.active,
                "texture_size": self.materials.size, "texture_overrides": list(self.materials.overrides),
                "post": self.post.enabled}

    def load_map(self, data: maps.MapData) -> None:
        self.world.load_map(data)
        self.decals.clear()
        self.viewmodel.set_environment(self.world.ambient, self.world.sun_color)
        self.resolution.settle()

    def begin_frame(self, dt: float) -> None:
        """Age the dynamic lights first, so flashes added by this frame's events get their full life."""
        self.lights.update(dt)

    def end_frame(self, frame_seconds: float) -> float:
        """Feed the measured frame time to the dynamic resolution controller; returns the render scale."""
        scale = self.resolution.update(frame_seconds)
        self.post.set_scale(scale)
        return scale

    # ---- gameplay effects ---------------------------------------------------------------
    def _amount(self, count: int) -> int:
        return max(1, int(math.ceil(count * self.particle_scale)))

    def fx_muzzle(self, position, rgb=(1.0, 0.78, 0.45), intensity: float = 10.0) -> None:
        self.lights.muzzle_flash(position, rgb, intensity)

    def fx_tracer(self, start, end, rgba=(1.0, 0.85, 0.5, 1.0), width: float = 0.022) -> None:
        self.tracers.add(start, end, rgba, width)

    def fx_impact(self, position, normal) -> None:
        self.sparks.emit(position, self._amount(7), direction=normal, spread=0.8, speed=(1.5, 5.5),
                         rgba=(1.0, 0.82, 0.45, 1.0), size=(0.035, 0.008), life=(0.10, 0.32), gravity=0.6)
        self.smoke.emit(position, self._amount(2), direction=normal, spread=0.5, speed=(0.3, 1.0),
                        rgba=(0.55, 0.55, 0.58, 0.30), size=(0.06, 0.26), life=(0.45, 0.95), gravity=-0.02, drag=1.6)
        self.decals.add(position, normal)

    def fx_hit(self, position, headshot: bool = False) -> None:
        rgba = (1.0, 0.16, 0.14, 1.0) if headshot else (1.0, 0.40, 0.30, 1.0)
        self.sparks.emit(position, self._amount(12 if headshot else 8), speed=(1.2, 4.5), rgba=rgba, size=(0.045, 0.01),
                         life=(0.15, 0.40), gravity=0.5)

    def fx_death(self, position, rgb) -> None:
        center = np.asarray(position, dtype=np.float64) + np.array([0.0, 1.0, 0.0])
        self.sparks.emit(center, self._amount(34), speed=(1.5, 6.5), rgba=(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0),
                         size=(0.07, 0.01), life=(0.3, 0.8), gravity=0.5, jitter=0.25)
        self.smoke.emit(center, self._amount(8), speed=(0.3, 1.4), rgba=(0.4, 0.4, 0.44, 0.35), size=(0.20, 0.70),
                        life=(0.6, 1.3), gravity=-0.03, drag=1.2, jitter=0.3)

    def fx_explosion(self, position) -> None:
        position = np.asarray(position, dtype=np.float64)
        self.sparks.emit(position, self._amount(80), speed=(4.0, 17.0), rgba=(1.0, 0.62, 0.22, 1.0), size=(0.11, 0.02),
                         life=(0.25, 0.75), gravity=0.7)
        self.sparks.emit(position, self._amount(22), speed=(0.5, 3.0), rgba=(1.0, 0.85, 0.55, 1.0), size=(0.9, 0.2),
                         life=(0.10, 0.24), gravity=0.0, jitter=0.3)
        self.smoke.emit(position, self._amount(30), speed=(0.8, 4.5), rgba=(0.32, 0.31, 0.33, 0.55), size=(0.45, 1.9),
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
        self.world.update(cam_pos, time_now)
        self.lights.write()
        self.sparks.update(dt)
        self.smoke.update(dt)
        self.tracers.update(dt)
        self.damage_text.update(dt)
        self.sparks.build(self.right, self.up)
        self.smoke.build(self.right, self.up)
        self.tracers.build(cam_pos)
        self.decals.build()
        self.characters.build(view, skip_player, dt, time_now, grenade_pos, self.world.static_irradiance)
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
HEALTH_WIDTH = 0.246
ABILITY_WIDTH = 0.30
(Q_CROSS, Q_DOT, Q_HIT, Q_INDICATOR, Q_HEALTH_BACK, Q_HEALTH, Q_ABILITY_BACK, Q_ABILITY, HUD_QUADS) = (0, 4, 5, 9, 15, 16, 17, 18, 19)


class HUD:
    """
    In-game heads-up display on Ursina's orthographic UI layer (y spans
    [-0.5, 0.5], x spans +-aspect/2).

    Every vector element -- the dynamic crosshair, hit markers, damage
    direction arcs and the bars -- lives in one structure-of-arrays table that
    becomes a single vertex buffer and a single draw call per frame.  Text is
    only re-typeset when its content changes, empty lines are switched off and
    colours are only touched when they actually change.
    """

    def __init__(self, fx_shader: Shader | None = None):
        self.root = Entity(parent=camera.ui, name="nexus_hud")
        self.half_width = float(window.aspect_ratio) * 0.5
        hw = self.half_width
        self._cache: dict = {}
        self._tint: dict = {}
        # ---- vector table: centre, half size, rotation (rad, clockwise), colour ---------------
        self.q_pos = np.zeros((HUD_QUADS, 2))
        self.q_half = np.zeros((HUD_QUADS, 2))
        self.q_rot = np.zeros(HUD_QUADS)
        self.q_rgba = np.zeros((HUD_QUADS, 4))
        shader = fx_shader if fx_shader is not None else Shader.make(Shader.SL_GLSL, FX_VERT, FX_FRAG)
        self.mesh = DynamicMesh("hud_vectors", HUD_QUADS, self.root, shader)
        set_blend(self.mesh.node, "alpha")
        self.mesh.node.setDepthTest(False)
        self.mesh.node.setBin("background", 10)
        # ---- dynamic crosshair: four arms that open with the bullet spread ------------------
        self.cross_gap = 0.01
        self.q_half[Q_CROSS: Q_CROSS + 2] = (CROSSHAIR_ARM * 0.5, 0.0011)
        self.q_half[Q_CROSS + 2: Q_CROSS + 4] = (0.0011, CROSSHAIR_ARM * 0.5)
        self.q_rgba[Q_CROSS: Q_CROSS + 4] = HUD_WHITE + (0.9,)
        self.q_half[Q_DOT] = (0.0013, 0.0013)
        self.q_rgba[Q_DOT] = HUD_WHITE + (0.95,)
        # ---- hit marker ----------------------------------------------------------------------
        self.hit_time = 0.0
        self.hit_rgb = HUD_WHITE
        self.hit_big = False
        self.q_half[Q_HIT: Q_HIT + 4] = (0.0085, 0.0015)
        self.q_rot[Q_HIT: Q_HIT + 4] = np.radians((45.0, -45.0, -45.0, 45.0))
        # ---- damage direction indicators -------------------------------------------------------
        self.q_half[Q_INDICATOR: Q_INDICATOR + 6] = (0.035, 0.0045)
        self.indicator_ttl = np.zeros(6)
        self.indicator_angle = np.zeros(6)
        # ---- vitals ----------------------------------------------------------------------------
        self.health_left = -hw + 0.047
        self._bar(Q_HEALTH_BACK, -hw + 0.045, -0.445, 0.25, 0.018, (0.05, 0.06, 0.08), 0.70)
        self._bar(Q_HEALTH, self.health_left, -0.445, HEALTH_WIDTH, 0.013, (0.30, 0.95, 0.55), 0.95)
        self.health_text = self._text("100", -hw + 0.305, -0.445, 1.25, (-0.5, 0), HUD_WHITE)
        self._text("VITALS", -hw + 0.045, -0.418, 0.62, (-0.5, 0), HUD_DIM)
        # ---- weapon ------------------------------------------------------------------------------
        self.ammo_text = self._text("30", hw - 0.115, -0.432, 2.7, (0.5, 0), HUD_WHITE)
        self.mag_text = self._text("/ 30", hw - 0.045, -0.442, 1.0, (0.5, 0), HUD_DIM)
        self.weapon_text = self._text("", hw - 0.045, -0.388, 0.78, (0.5, 0), HUD_DIM)
        self.reload_text = self._text("", 0.0, -0.11, 0.95, (0, 0), HUD_GOLD)
        self.ability_left = hw - 0.345
        self._bar(Q_ABILITY_BACK, self.ability_left, -0.470, ABILITY_WIDTH, 0.007, (0.05, 0.06, 0.08), 0.70)
        self._bar(Q_ABILITY, self.ability_left, -0.470, ABILITY_WIDTH, 0.005, (1.0, 0.55, 0.20), 0.95)
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
        self._upload()

    # ---- construction helpers ---------------------------------------------------------------
    def _bar(self, slot: int, left: float, y: float, width: float, height: float, rgb, alpha: float) -> None:
        self.q_pos[slot] = (left + width * 0.5, y)
        self.q_half[slot] = (width * 0.5, height * 0.5)
        self.q_rgba[slot] = tuple(rgb) + (alpha,)

    def _text(self, value, x, y, scale, origin, rgb) -> Text:
        entity = Text(value, parent=self.root, position=(x, y), scale=scale, origin=origin, color=_rgba(*rgb))
        self._cache[id(entity)] = value
        if value == "":
            entity.enabled = False
        return entity

    def _set(self, entity: Text, value: str) -> None:
        """Assign text only when it changed (typesetting is the expensive part); empty lines are switched off."""
        if self._cache.get(id(entity)) != value:
            self._cache[id(entity)] = value
            if value == "":
                entity.enabled = False
            else:
                entity.text = value
                if not entity.enabled:
                    entity.enabled = True

    def _color(self, entity: Text, rgb, alpha: float = 1.0) -> None:
        """Tint a text; the engine call is skipped unless the (quantised) colour changed."""
        key = (round(rgb[0], 3), round(rgb[1], 3), round(rgb[2], 3), round(alpha * 24.0))
        if self._tint.get(id(entity)) != key:
            self._tint[id(entity)] = key
            entity.color = _rgba(rgb[0], rgb[1], rgb[2], key[3] / 24.0)

    def _upload(self) -> None:
        """Vector table -> one vertex buffer."""
        live = np.flatnonzero(self.q_rgba[:, 3] > 0.004)
        if live.size == 0:
            self.mesh.upload(EMPTY)
            return
        local = _CORNERS[None, :, :] * self.q_half[live][:, None, :]                       # (Q, 4, 2)
        cos_r, sin_r = np.cos(self.q_rot[live])[:, None], np.sin(self.q_rot[live])[:, None]
        corners = np.zeros((live.size, 4, 3))
        corners[..., 0] = self.q_pos[live][:, None, 0] + local[..., 0] * cos_r + local[..., 1] * sin_r
        corners[..., 1] = self.q_pos[live][:, None, 1] - local[..., 0] * sin_r + local[..., 1] * cos_r
        self.mesh.upload(fx_quads(corners, self.q_rgba[live], 2.0))

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
        self.q_pos[Q_CROSS: Q_CROSS + 4] = ((-reach, 0.0), (reach, 0.0), (0.0, reach), (0.0, -reach))
        self.q_rgba[Q_CROSS: Q_CROSS + 4, 3] = 0.9 * alpha
        self.q_rgba[Q_DOT, 3] = 0.95 * alpha

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
        if fraction > 0.5:
            rgb = (0.30, 0.95, 0.55)
        elif fraction > 0.25:
            rgb = (1.00, 0.80, 0.25)
        else:
            rgb = HUD_CRIMSON
        self._bar(Q_HEALTH, self.health_left, -0.445, HEALTH_WIDTH * fraction, 0.013, rgb, 0.95 if fraction > 0.0 else 0.0)
        self._set(self.health_text, "%d" % math.ceil(max(health, 0.0)))

    def set_weapon(self, name: str, ammo: int, magazine: int, reloading: bool) -> None:
        self._set(self.ammo_text, "%d" % ammo)
        self._set(self.mag_text, "/ %d" % magazine)
        self._set(self.weapon_text, name)
        self._set(self.reload_text, "RELOADING" if reloading else ("PRESS R TO RELOAD" if ammo == 0 else ""))
        self._color(self.ammo_text, HUD_CRIMSON if ammo * 4 <= magazine else HUD_WHITE)

    def set_ability(self, ready_fraction: float) -> None:
        fraction = min(max(ready_fraction, 0.0), 1.0)
        rgb, alpha = ((1.0, 0.55, 0.20), 0.95) if fraction >= 1.0 else ((0.55, 0.60, 0.68), 0.9)
        self._bar(Q_ABILITY, self.ability_left, -0.470, ABILITY_WIDTH * fraction, 0.005, rgb, alpha if fraction > 0.0 else 0.0)

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
        if text:
            self._color(self.center_text, rgb)

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
        self.q_pos[Q_HIT: Q_HIT + 4] = np.array(((-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))) * spread
        self.q_rgba[Q_HIT: Q_HIT + 4] = tuple(self.hit_rgb) + (min(phase * 1.6, 1.0),)
        self.indicator_ttl = np.maximum(self.indicator_ttl - dt, 0.0)
        span = slice(Q_INDICATOR, Q_INDICATOR + 6)
        self.q_pos[span, 0] = np.sin(self.indicator_angle) * 0.17
        self.q_pos[span, 1] = np.cos(self.indicator_angle) * 0.17
        self.q_rot[span] = self.indicator_angle
        self.q_rgba[span, :3] = HUD_CRIMSON
        self.q_rgba[span, 3] = np.minimum(self.indicator_ttl, 0.85)
        for lines, feed in ((self.kill_lines, self.kill_feed), (self.reward_lines, self.reward_feed)):
            expired = False
            for index, entry in enumerate(feed):
                entry[2] -= dt
                expired |= entry[2] <= 0.0
                self._color(lines[index], entry[1], min(max(entry[2], 0.0) * 1.5, 1.0))
            if expired:
                feed[:] = [entry for entry in feed if entry[2] > 0.0]
                self._refresh_feed(lines, feed)
        self._upload()


def apply_camera(rig) -> None:
    """Copy a :class:`engine.controller_3d.CameraRig` pose onto Ursina's camera."""
    camera.position = (float(rig.cam_pos[0]), float(rig.cam_pos[1]), float(rig.cam_pos[2]))
    camera.rotation = (-math.degrees(rig.cam_pitch), math.degrees(rig.cam_yaw), -math.degrees(rig.cam_roll))
    camera.fov = rig.fov


def vertical_fov(horizontal_fov_deg: float) -> float:
    """Vertical field of view (degrees) for Ursina's horizontal ``camera.fov``."""
    aspect = max(float(window.aspect_ratio), 0.1)
    return math.degrees(2.0 * math.atan(math.tan(math.radians(horizontal_fov_deg) * 0.5) / aspect))
