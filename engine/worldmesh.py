"""
NEXUS-3D-EXTREME :: engine/worldmesh.py
=======================================
Static level mesh construction (pure NumPy, no window needed).

Turns the array bundle of an arena (:class:`engine.maps.MapData`) into one
vertex / index buffer for the renderer:

* faces that are completely buried in other geometry are dropped;
* every remaining box face is tessellated into a grid, so that lighting baked
  into the vertices has spatial resolution;
* texture coordinates are generated per face (world-space tiling, or one
  repeat per face for props such as crates);
* the arena's static lights are baked per vertex with real occlusion (rays
  through the collision octree), together with a short-range ambient
  occlusion term that grounds objects on the floor;
* a ring of distant towers gives the sky dome a three-dimensional skyline.

The result is cached on disk per arena checksum and quality level.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import config as C
from . import mathx as M
from . import maps
from .controller_3d import CollisionWorld

MESH_REVISION = 5            # bump when the generated mesh or the bake changes (invalidates cached meshes)
MAX_CELLS = 96               # tessellation limit per face edge
AO_RANGE = 2.4               # metres an ambient occlusion ray travels
AO_STRENGTH = 0.80
BACKDROP_FLAG = 4.0          # added to uv.w for skyline geometry (the shader thins the fog for it)

# position, normal, tint rgb + ambient occlusion, uv + texture layer + emissive, baked static light
WORLD_VERTEX = np.dtype([("pos", "<f4", 3), ("nrm", "<f4", 3), ("col", "<f4", 4), ("uv", "<f4", 4), ("light", "<f4", 3)])

_AXES = np.eye(3)
FACE_N = np.array([_AXES[0], -_AXES[0], _AXES[1], -_AXES[1], _AXES[2], -_AXES[2]])
FACE_A = np.array([_AXES[2], _AXES[1], _AXES[0], _AXES[2], _AXES[1], _AXES[0]])     # chosen so that A x B = -N
FACE_B = np.array([_AXES[1], _AXES[2], _AXES[2], _AXES[0], _AXES[0], _AXES[1]])
FACE_AXIS_N = np.array([0, 0, 1, 1, 2, 2])
FACE_AXIS_A = np.array([2, 1, 0, 2, 1, 0])
FACE_AXIS_B = np.array([1, 2, 2, 0, 0, 1])
UV_AXIS_U = np.array([2, 2, 0, 0, 0, 0])            # texture u runs horizontally on walls,
UV_AXIS_V = np.array([1, 1, 2, 2, 1, 1])            # texture v runs up the wall
_FACE_SAMPLES = np.array([[0.0, 0.0], [-0.6, -0.6], [0.6, -0.6], [0.6, 0.6], [-0.6, 0.6]])


# --------------------------------------------------------------------------- #
#  Hidden face removal
# --------------------------------------------------------------------------- #
def exposed_faces(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """
    ``(B, 6)`` mask of box faces worth drawing.  A face is dropped when five
    probe points just outside it all lie inside other boxes (crate bottoms on
    the floor, wall segments butted against each other, ...).
    """
    count = box_min.shape[0]
    centre = (box_min + box_max) * 0.5
    half = (box_max - box_min) * 0.5
    exposed = np.ones((count, 6), dtype=bool)
    if count < 2:
        return exposed
    index = np.arange(count)
    for face in range(6):
        n, a, b = FACE_AXIS_N[face], FACE_AXIS_A[face], FACE_AXIS_B[face]
        probes = np.repeat(centre[:, None, :], len(_FACE_SAMPLES), axis=1)                       # (B, 5, 3)
        probes[:, :, n] += FACE_N[face, n] * (half[:, n] + 0.02)[:, None]
        probes[:, :, a] += _FACE_SAMPLES[None, :, 0] * half[:, a][:, None]
        probes[:, :, b] += _FACE_SAMPLES[None, :, 1] * half[:, b][:, None]
        inside = np.all((probes[:, :, None, :] > box_min[None, None, :, :] + 1.0e-4)
                        & (probes[:, :, None, :] < box_max[None, None, :, :] - 1.0e-4), axis=3)  # (B, 5, B)
        inside[index, :, index] = False
        exposed[:, face] = ~np.all(inside.any(axis=2), axis=1)
    return exposed


# --------------------------------------------------------------------------- #
#  Tessellation and texture coordinates
# --------------------------------------------------------------------------- #
def _grid_indices(na: int, nb: int) -> np.ndarray:
    """Triangle indices of one ``na x nb`` cell grid whose vertices are stored row by row."""
    i, j = np.meshgrid(np.arange(na), np.arange(nb), indexing="xy")
    v00 = (j * (na + 1) + i).ravel()
    v10, v01 = v00 + 1, v00 + na + 1
    v11 = v01 + 1
    return np.stack([v00, v10, v11, v00, v11, v01], axis=1).ravel()


def tessellate_boxes(data: maps.MapData, rows: np.ndarray, cell: float, exposed: np.ndarray | None = None):
    """
    Vertex and index arrays for the boxes ``rows`` of ``data``.  Returns
    ``(vertices, indices, sample_points)``; ``sample_points`` are the positions
    (lifted off the surface and pulled in from the face edges) at which baked
    lighting should be evaluated.
    """
    bmin, bmax = data.box_min[rows], data.box_max[rows]
    centre, half = (bmin + bmax) * 0.5, (bmax - bmin) * 0.5
    count = rows.shape[0]
    if exposed is None:
        exposed = np.ones((count, 6), dtype=bool)
    tile = np.maximum(data.box_tile[rows].astype(np.float64) * C.TEXTURE_WORLD_SCALE, 1.0e-3)
    fit = data.box_uv_mode[rows] == maps.UV_FIT
    flat = (data.box_emissive[rows] > 0.0) | (np.min(half, axis=1) * 2.0 < 0.02)
    tint = data.box_color[rows]
    layer = data.box_texture[rows].astype(np.float32)
    emissive = data.box_emissive[rows]

    box_id, face_id = np.nonzero(exposed)
    ea = half[box_id, FACE_AXIS_A[face_id]] * 2.0
    eb = half[box_id, FACE_AXIS_B[face_id]] * 2.0
    na = np.clip(np.ceil(ea / cell - 1.0e-6), 1, MAX_CELLS).astype(np.int64)
    nb = np.clip(np.ceil(eb / cell - 1.0e-6), 1, MAX_CELLS).astype(np.int64)
    na[flat[box_id]] = 1
    nb[flat[box_id]] = 1

    vertex_blocks, index_blocks, sample_blocks = [], [], []
    offset = 0
    for key in np.unique(na * (MAX_CELLS + 1) + nb).tolist():
        ga, gb = divmod(key, MAX_CELLS + 1)
        sel = np.flatnonzero((na == ga) & (nb == gb))
        b, f = box_id[sel], face_id[sel]
        s = np.linspace(-1.0, 1.0, ga + 1)
        t = np.linspace(-1.0, 1.0, gb + 1)
        ss, tt = np.meshgrid(s, t, indexing="xy")                       # (gb + 1, ga + 1), rows run along B
        ss, tt = ss.ravel()[None, :, None], tt.ravel()[None, :, None]
        per_face = ss.shape[1]
        normal, axis_a, axis_b = FACE_N[f][:, None, :], FACE_A[f][:, None, :], FACE_B[f][:, None, :]
        h = half[b]
        hn = np.sum(h * np.abs(FACE_N[f]), axis=1)[:, None, None]
        ha = np.sum(h * FACE_A[f], axis=1)[:, None, None]
        hb = np.sum(h * FACE_B[f], axis=1)[:, None, None]
        pos = centre[b][:, None, :] + normal * hn + axis_a * ss * ha + axis_b * tt * hb          # (K, per_face, 3)
        inset_a = np.minimum(0.08, ha * 0.5)
        inset_b = np.minimum(0.08, hb * 0.5)
        sample = pos + normal * 0.07 - axis_a * ss * inset_a - axis_b * tt * inset_b

        k = sel.size
        pu = np.take_along_axis(pos, UV_AXIS_U[f][:, None, None].repeat(per_face, axis=1), axis=2)[..., 0]
        pv = np.take_along_axis(pos, UV_AXIS_V[f][:, None, None].repeat(per_face, axis=1), axis=2)[..., 0]
        lo_u, lo_v = bmin[b, UV_AXIS_U[f]][:, None], bmin[b, UV_AXIS_V[f]][:, None]
        size_u = np.maximum(bmax[b, UV_AXIS_U[f]] - bmin[b, UV_AXIS_U[f]], 1.0e-6)[:, None]
        size_v = np.maximum(bmax[b, UV_AXIS_V[f]] - bmin[b, UV_AXIS_V[f]], 1.0e-6)[:, None]
        fitted = fit[b][:, None]
        u = np.where(fitted, (pu - lo_u) / size_u, pu / tile[b][:, None])
        v = np.where(fitted, (pv - lo_v) / size_v, pv / tile[b][:, None])

        block = np.zeros(k * per_face, dtype=WORLD_VERTEX)
        block["pos"] = pos.reshape(-1, 3)
        block["nrm"] = np.broadcast_to(normal, pos.shape).reshape(-1, 3)
        block["col"][:, :3] = np.repeat(tint[b], per_face, axis=0)
        block["col"][:, 3] = 1.0
        block["uv"][:, 0] = u.ravel()
        block["uv"][:, 1] = v.ravel()
        block["uv"][:, 2] = np.repeat(layer[b], per_face)
        block["uv"][:, 3] = np.repeat(emissive[b], per_face)
        vertex_blocks.append(block)
        sample_blocks.append(sample.reshape(-1, 3))
        grid = _grid_indices(ga, gb)
        index_blocks.append((offset + np.arange(k)[:, None] * per_face + grid[None, :]).ravel())
        offset += k * per_face
    if not vertex_blocks:
        return np.zeros(0, dtype=WORLD_VERTEX), np.zeros(0, dtype=np.uint32), np.zeros((0, 3))
    return (np.concatenate(vertex_blocks), np.concatenate(index_blocks).astype(np.uint32),
            np.concatenate(sample_blocks))


def triangle_vertices(data: maps.MapData):
    """Wedges and loaded models as vertices (three per triangle), correctly wound for the renderer."""
    tris = data.mesh_tris.copy()
    normals = data.mesh_normal.copy()
    uv = data.mesh_uv.copy()
    count = tris.shape[0]
    if count:
        winding = M.dot(M.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), normals.mean(axis=1))
        flip = winding > 0.0                                   # must be negative to face the viewer
        for array in (tris, normals, uv):
            array[flip] = array[flip][:, [0, 2, 1]]
    out = np.zeros(count * 3, dtype=WORLD_VERTEX)
    out["pos"] = tris.reshape(-1, 3)
    out["nrm"] = normals.reshape(-1, 3)
    out["col"][:, :3] = np.repeat(data.mesh_color, 3, axis=0)
    out["col"][:, 3] = 1.0
    out["uv"][:, :2] = uv.reshape(-1, 2)
    out["uv"][:, 2] = np.repeat(data.mesh_texture.astype(np.float32), 3)
    centroid = np.repeat(tris.mean(axis=1), 3, axis=0)
    sample = tris.reshape(-1, 3) + normals.reshape(-1, 3) * 0.07 + (centroid - tris.reshape(-1, 3)) * 0.12
    return out, np.arange(count * 3, dtype=np.uint32), sample


# --------------------------------------------------------------------------- #
#  Baked lighting
# --------------------------------------------------------------------------- #
def bake_static_lights(world: CollisionWorld, points: np.ndarray, normals: np.ndarray, lights_pos, lights_color,
                       lights_radius) -> np.ndarray:
    """
    Irradiance of the arena's static lights at ``points``: ``I0 / d^2`` with a
    smooth radius window, Lambert-wrapped and shadowed by the level geometry.
    """
    out = np.zeros((points.shape[0], 3), dtype=np.float32)
    for position, colour, radius in zip(np.asarray(lights_pos, dtype=np.float64).reshape(-1, 3),
                                        np.asarray(lights_color, dtype=np.float64).reshape(-1, 3),
                                        np.asarray(lights_radius, dtype=np.float64).reshape(-1)):
        delta = position[None, :] - points
        d2 = np.sum(delta * delta, axis=1)
        idx = np.flatnonzero(d2 < radius * radius)
        if idx.size == 0:
            continue
        distance = np.sqrt(d2[idx])
        direction = delta[idx] / np.maximum(distance, 1.0e-6)[:, None]
        facing = np.sum(normals[idx] * direction, axis=1)
        window = np.clip(1.0 - d2[idx] / (radius * radius), 0.0, 1.0)
        energy = (radius * radius / 15.0) / np.maximum(d2[idx], 0.20) * window * window * (0.25 + 0.75 * np.maximum(facing, 0.0))
        lit = np.flatnonzero((energy > 1.0e-3) & (facing > -0.35))
        if lit.size == 0:
            continue
        rows = idx[lit]
        reach = np.maximum(distance[lit] - 0.45, 0.0)             # stop short: lamps sit inside their own fixture
        seen = world.line_of_sight(points[rows], points[rows] + direction[lit] * reach[:, None], C.LAYER_BULLET)
        out[rows[seen]] += (colour[None, :] * energy[lit][seen][:, None]).astype(np.float32)
    return out


def bake_ambient_occlusion(world: CollisionWorld, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Short-range ambient occlusion in [0, 1] (1 = open) from seven rays over the hemisphere."""
    count = points.shape[0]
    if count == 0:
        return np.ones(0, dtype=np.float32)
    helper = np.where(np.abs(normals[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    tangent = M.normalize(M.cross(helper, normals))
    bitangent = M.cross(normals, tangent)
    blocked = np.zeros(count)
    weight = 0.0
    for phase in range(6):
        angle = phase * math.pi / 3.0
        direction = M.normalize(normals * 0.62 + (tangent * math.cos(angle) + bitangent * math.sin(angle)) * 0.78)
        t, item, _ = world.raycast(points, direction, AO_RANGE, C.LAYER_BULLET)
        blocked += np.where(item >= 0, (1.0 - t / AO_RANGE) ** 2, 0.0)
        weight += 1.0
    t, item, _ = world.raycast(points, normals, AO_RANGE, C.LAYER_BULLET)
    blocked += np.where(item >= 0, (1.0 - t / AO_RANGE) ** 2, 0.0) * 1.5
    weight += 1.5
    return np.clip(1.0 - AO_STRENGTH * blocked / weight, 0.22, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Three-dimensional skyline
# --------------------------------------------------------------------------- #
def skyline_boxes(data: maps.MapData):
    """
    Distant towers around the arena as ``(box_min, box_max, tint)``.  Density
    and height follow the environment's ``skyline`` value; the layout is
    seeded by the arena checksum, so it is identical on every machine.
    """
    amount = float(data.environment.get("skyline", 0.0))
    if amount <= 0.05:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(data.checksum ^ 0x5F3759DF)
    count = int(12 + 34 * amount)
    reach = float(np.max(np.abs(np.concatenate([data.bounds_min[[0, 2]], data.bounds_max[[0, 2]]])))) + 110.0
    angle = (np.arange(count) + rng.uniform(-0.4, 0.4, count)) / count * 2.0 * math.pi
    distance = reach + rng.uniform(0.0, 170.0, count)
    width = rng.uniform(16.0, 36.0, count)
    depth = rng.uniform(16.0, 36.0, count)
    height = (14.0 + rng.random(count) ** 2 * 120.0) * (0.40 + 0.60 * amount) * (0.70 + 0.30 * distance / reach)
    base = min(float(data.bounds_min[1]), 0.0) - 30.0
    cx, cz = np.sin(angle) * distance, np.cos(angle) * distance
    lo = np.stack([cx - width * 0.5, np.full(count, base), cz - depth * 0.5], axis=1)
    hi = np.stack([cx + width * 0.5, base + 30.0 + height, cz + depth * 0.5], axis=1)
    shade = rng.uniform(0.55, 1.0, count)
    tint = np.stack([shade * 0.92, shade * 0.96, shade * 1.04], axis=1).astype(np.float32)
    return lo, hi, tint


def _skyline_vertices(data: maps.MapData):
    lo, hi, tint = skyline_boxes(data)
    count = lo.shape[0]
    if count == 0:
        return np.zeros(0, dtype=WORLD_VERTEX), np.zeros(0, dtype=np.uint32)
    proxy = maps.MapData()
    proxy.box_min, proxy.box_max, proxy.box_color = lo, hi, tint
    proxy.box_texture = np.full(count, maps.LAYER["tower"], dtype=np.int32)
    proxy.box_uv_mode = np.full(count, maps.UV_TILE, dtype=np.int32)
    proxy.box_tile = np.full(count, 12.0, dtype=np.float32)
    night = float(data.environment.get("stars", 0.0))
    proxy.box_emissive = np.zeros(count, dtype=np.float32)
    exposed = np.ones((count, 6), dtype=bool)
    exposed[:, 3] = False                                         # nobody sees the undersides
    vertices, indices, _ = tessellate_boxes(proxy, np.arange(count), 1.0e9, exposed)
    vertices["uv"][:, 3] = BACKDROP_FLAG + 0.10 + 0.75 * night      # lit windows glow at night
    return vertices, indices


# --------------------------------------------------------------------------- #
#  Whole arena
# --------------------------------------------------------------------------- #
def build_world_mesh(data: maps.MapData, world: CollisionWorld, cell: float = 2.0, ambient_occlusion: bool = True) -> dict:
    """
    Assemble and light an arena.  Returns a dictionary with the arena mesh
    (``vertices`` / ``indices``), the distant skyline as a separate mesh
    (``sky_vertices`` / ``sky_indices``; it must not cast shadows) and
    ``stats``.
    """
    started = time.perf_counter()
    rows = np.flatnonzero(data.box_visible)
    exposed = exposed_faces(data.box_min[rows], data.box_max[rows])
    box_v, box_i, box_s = tessellate_boxes(data, rows, cell, exposed)
    tri_v, tri_i, tri_s = triangle_vertices(data)
    vertices = np.concatenate([box_v, tri_v])
    indices = np.concatenate([box_i, tri_i + box_v.shape[0]]).astype(np.uint32)
    samples = np.concatenate([box_s, tri_s])
    normals = vertices["nrm"].astype(np.float64)
    shaded = vertices["uv"][:, 3] <= 0.0                            # emissive surfaces ignore scene lighting
    vertices["light"][shaded] = bake_static_lights(world, samples[shaded], normals[shaded], data.lights_pos,
                                                   data.lights_color, data.lights_radius)
    if ambient_occlusion:
        vertices["col"][shaded, 3] = bake_ambient_occlusion(world, samples[shaded], normals[shaded])
    sky_v, sky_i = _skyline_vertices(data)
    stats = {"vertices": int(vertices.shape[0] + sky_v.shape[0]), "triangles": int((indices.shape[0] + sky_i.shape[0]) // 3),
             "faces_culled": int(exposed.size - exposed.sum()), "skyline_towers": int(sky_v.shape[0] // 20),
             "seconds": time.perf_counter() - started, "cached": False}
    return {"vertices": vertices, "indices": indices, "sky_vertices": sky_v, "sky_indices": sky_i, "stats": stats}


def load_world_mesh(data: maps.MapData, world: CollisionWorld | None = None, cell: float = 2.0,
                    ambient_occlusion: bool = True, use_cache: bool = True) -> dict:
    """
    Cached :func:`build_world_mesh` (keyed by arena checksum, tessellation and
    bake revision).  ``world`` may be ``None``; the collision world is then
    built only if the mesh actually has to be baked.
    """
    path = C.BAKE_DIR / ("mesh%d_%s_%08x_%03d_%d.npz" % (MESH_REVISION, data.key, data.checksum,
                                                          int(round(cell * 100.0)), int(ambient_occlusion)))
    if use_cache and path.exists():
        try:
            with np.load(path) as stored:
                mesh = {name: stored[name] for name in ("vertices", "indices", "sky_vertices", "sky_indices")}
            if mesh["vertices"].dtype == WORLD_VERTEX and mesh["sky_vertices"].dtype == WORLD_VERTEX:
                total = mesh["vertices"].shape[0] + mesh["sky_vertices"].shape[0]
                mesh["stats"] = {"vertices": int(total), "seconds": 0.0, "cached": True,
                                 "triangles": int((mesh["indices"].shape[0] + mesh["sky_indices"].shape[0]) // 3)}
                return mesh
        except (OSError, KeyError, ValueError):
            pass
    mesh = build_world_mesh(data, world if world is not None else maps.build_collision_world(data), cell,
                            ambient_occlusion)
    if use_cache:
        try:
            C.BAKE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{name: mesh[name] for name in ("vertices", "indices", "sky_vertices", "sky_indices")})
        except OSError:
            pass
    return mesh
