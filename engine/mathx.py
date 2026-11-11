"""
NEXUS-3D-EXTREME :: engine/mathx.py
===================================
Vectorised linear algebra used by every subsystem.  Every function broadcasts
over leading dimensions so a single call can service one entity or a whole
structure-of-arrays block; nothing in here allocates Python objects per entity.

Conventions
-----------
* Vectors are ``(..., 3)`` float arrays in the Y-up / left-handed world frame
  described in :mod:`engine.config`.
* Quaternions are ``(..., 4)`` arrays laid out ``[w, x, y, z]``.
* Rigid transforms are ``(..., 4, 4)`` matrices acting on column vectors.

Only element-wise NumPy kernels are used (no BLAS reductions) so that a row
simulated alone produces the same result as the same row simulated inside a
larger batch -- a prerequisite for client side prediction to agree with the
authoritative server.
"""
from __future__ import annotations

import numpy as np

PI = np.pi
TAU = 2.0 * np.pi
_MASK32 = np.uint64(0xFFFFFFFF)
_TINY = 1.0e-12


# --------------------------------------------------------------------------- #
#  Scalar / interpolation helpers
# --------------------------------------------------------------------------- #
def lerp(a, b, t):
    """Linear interpolation ``a + (b - a) * t`` (NumPy has no built-in lerp)."""
    return a + (b - a) * t


def inv_lerp(a, b, value):
    """Inverse of :func:`lerp`; returns 0 when ``a == b``."""
    span = np.asarray(b, dtype=np.float64) - a
    safe = np.where(np.abs(span) < _TINY, 1.0, span)
    return np.where(np.abs(span) < _TINY, 0.0, (value - a) / safe)


def clamp(value, lo, hi):
    return np.minimum(np.maximum(value, lo), hi)


def clamp01(value):
    return np.minimum(np.maximum(value, 0.0), 1.0)


def smoothstep(t):
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out_cos(t):
    """Cosine ease-in-out mapping [0, 1] -> [0, 1] with zero slope at both ends."""
    return 0.5 - 0.5 * np.cos(PI * clamp01(t))


def ease_out_cubic(t):
    u = 1.0 - clamp01(t)
    return 1.0 - u * u * u


def exp_decay(value, k, dt):
    """Closed form of dV/dt = -k V over one step: ``V * exp(-k * dt)``."""
    return value * np.exp(-k * dt)


def damp(current, target, rate, dt):
    """Frame-rate independent exponential approach of ``current`` towards ``target``."""
    return target + (current - target) * np.exp(-rate * dt)


def move_towards(current, target, max_delta):
    delta = target - current
    return current + np.clip(delta, -max_delta, max_delta)


def wrap_angle(angle):
    """Wrap radians into [-pi, pi)."""
    return (angle + PI) % TAU - PI


def angle_delta(a, b):
    """Shortest signed rotation that takes angle ``a`` onto angle ``b``."""
    return wrap_angle(b - a)


def lerp_angle(a, b, t):
    return a + angle_delta(a, b) * t


# --------------------------------------------------------------------------- #
#  Vector helpers
# --------------------------------------------------------------------------- #
def dot(a, b):
    return np.sum(a * b, axis=-1)


def length(v):
    return np.sqrt(np.sum(v * v, axis=-1))


def length_sq(v):
    return np.sum(v * v, axis=-1)


def normalize(v, eps: float = _TINY):
    """Unit vectors; zero-length inputs stay zero instead of producing NaN."""
    v = np.asarray(v, dtype=np.float64)
    mag = np.sqrt(np.sum(v * v, axis=-1, keepdims=True))
    return v / np.maximum(mag, eps)


def cross(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.float64)
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def reflect(v, n):
    """Mirror ``v`` about the plane with unit normal ``n``."""
    return v - 2.0 * dot(v, n)[..., None] * n


def safe_inverse(d, eps: float = 1.0e-12):
    """Reciprocal that never divides by zero (keeps the sign of the input)."""
    d = np.asarray(d, dtype=np.float64)
    guarded = np.where(np.abs(d) < eps, np.where(d < 0.0, -eps, eps), d)
    return 1.0 / guarded


# --------------------------------------------------------------------------- #
#  Orientation: yaw / pitch <-> vectors
# --------------------------------------------------------------------------- #
def forward_vector(yaw, pitch=0.0):
    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    cp = np.cos(pitch)
    out = np.empty(np.broadcast_shapes(yaw.shape, pitch.shape) + (3,), dtype=np.float64)
    out[..., 0] = np.sin(yaw) * cp
    out[..., 1] = np.sin(pitch)
    out[..., 2] = np.cos(yaw) * cp
    return out


def right_vector(yaw):
    yaw = np.asarray(yaw, dtype=np.float64)
    out = np.zeros(yaw.shape + (3,), dtype=np.float64)
    out[..., 0] = np.cos(yaw)
    out[..., 2] = -np.sin(yaw)
    return out


def up_vector(yaw, pitch=0.0):
    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    sp = np.sin(pitch)
    out = np.empty(np.broadcast_shapes(yaw.shape, pitch.shape) + (3,), dtype=np.float64)
    out[..., 0] = -np.sin(yaw) * sp
    out[..., 1] = np.cos(pitch)
    out[..., 2] = -np.cos(yaw) * sp
    return out


def yaw_pitch_from_direction(direction):
    """Inverse of :func:`forward_vector` (direction need not be normalised)."""
    d = normalize(direction)
    yaw = np.arctan2(d[..., 0], d[..., 2])
    pitch = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    return yaw, pitch


def yaw_matrix(yaw):
    """Rotation about world up; columns are (right, up, forward)."""
    yaw = np.asarray(yaw, dtype=np.float64)
    c = np.cos(yaw)
    s = np.sin(yaw)
    m = np.zeros(yaw.shape + (3, 3), dtype=np.float64)
    m[..., 0, 0] = c
    m[..., 0, 2] = s
    m[..., 1, 1] = 1.0
    m[..., 2, 0] = -s
    m[..., 2, 2] = c
    return m


def view_matrix3(yaw, pitch):
    """Full first-person rotation; columns are (right, up, forward)."""
    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    shape = np.broadcast_shapes(yaw.shape, pitch.shape)
    m = np.empty(shape + (3, 3), dtype=np.float64)
    m[..., :, 0] = np.broadcast_to(right_vector(yaw), shape + (3,))
    m[..., :, 1] = up_vector(yaw, pitch)
    m[..., :, 2] = forward_vector(yaw, pitch)
    return m


# --------------------------------------------------------------------------- #
#  Quaternions  [w, x, y, z]
# --------------------------------------------------------------------------- #
def quat_identity(shape=()):
    if isinstance(shape, (int, np.integer)):
        shape = (int(shape),)
    q = np.zeros(tuple(shape) + (4,), dtype=np.float64)
    q[..., 0] = 1.0
    return q


def quat_from_axis_angle(axis, angle):
    axis = normalize(axis)
    angle = np.asarray(angle, dtype=np.float64)
    half = angle * 0.5
    s = np.sin(half)
    q = np.empty(np.broadcast_shapes(axis.shape[:-1], angle.shape) + (4,), dtype=np.float64)
    q[..., 0] = np.cos(half)
    q[..., 1] = axis[..., 0] * s
    q[..., 2] = axis[..., 1] * s
    q[..., 3] = axis[..., 2] * s
    return q


def quat_mul(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.float64)
    out[..., 0] = aw * bw - ax * bx - ay * by - az * bz
    out[..., 1] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 2] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


def quat_conjugate(q):
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    mag = np.sqrt(np.sum(q * q, axis=-1, keepdims=True))
    return q / np.maximum(mag, _TINY)


def quat_from_yaw_pitch(yaw, pitch):
    """
    Orientation quaternion of a first-person view: yaw about +Y followed by a
    local pitch about the right axis (positive pitch looks up).
    """
    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    hy = yaw * 0.5
    hp = -pitch * 0.5            # looking up is a negative rotation about +X
    cy, sy = np.cos(hy), np.sin(hy)
    cp, sp = np.cos(hp), np.sin(hp)
    q = np.empty(np.broadcast_shapes(yaw.shape, pitch.shape) + (4,), dtype=np.float64)
    # q = q_yaw (cy, 0, sy, 0) * q_pitch (cp, sp, 0, 0)
    q[..., 0] = cy * cp
    q[..., 1] = cy * sp
    q[..., 2] = sy * cp
    q[..., 3] = -sy * sp
    return q


def quat_rotate(q, v):
    """Rotate vectors ``v`` by unit quaternions ``q``."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., 0:1]
    u = q[..., 1:]
    t = 2.0 * cross(u, v)
    return v + w * t + cross(u, t)


def quat_to_matrix(q):
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    m[..., 0, 1] = 2.0 * (x * y - z * w)
    m[..., 0, 2] = 2.0 * (x * z + y * w)
    m[..., 1, 0] = 2.0 * (x * y + z * w)
    m[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    m[..., 1, 2] = 2.0 * (y * z - x * w)
    m[..., 2, 0] = 2.0 * (x * z - y * w)
    m[..., 2, 1] = 2.0 * (y * z + x * w)
    m[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return m


def quat_to_yaw_pitch(q):
    """Recover yaw / pitch from a view quaternion built by :func:`quat_from_yaw_pitch`."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    fx = 2.0 * (x * z + y * w)
    fy = 2.0 * (y * z - x * w)
    fz = 1.0 - 2.0 * (x * x + y * y)
    yaw = np.arctan2(fx, fz)
    pitch = np.arcsin(np.clip(fy, -1.0, 1.0))
    return yaw, pitch


def quat_slerp(q0, q1, t):
    """
    Spherical linear interpolation, vectorised over leading dimensions.
    Takes the short arc and falls back to normalised lerp for nearly parallel
    inputs so no division by a vanishing sine can occur.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    t = np.expand_dims(np.asarray(t, dtype=np.float64), -1)
    cos_theta = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(cos_theta < 0.0, -q1, q1)
    cos_theta = np.clip(np.abs(cos_theta), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    near = sin_theta < 1.0e-6
    safe_sin = np.where(near, 1.0, sin_theta)
    w0 = np.where(near, 1.0 - t, np.sin((1.0 - t) * theta) / safe_sin)
    w1 = np.where(near, t, np.sin(t * theta) / safe_sin)
    return quat_normalize(w0 * q0 + w1 * q1)


# --------------------------------------------------------------------------- #
#  Rigid transforms
# --------------------------------------------------------------------------- #
def compose_rigid(position, rotation3):
    position = np.asarray(position, dtype=np.float64)
    rotation3 = np.asarray(rotation3, dtype=np.float64)
    shape = np.broadcast_shapes(position.shape[:-1], rotation3.shape[:-2])
    m = np.zeros(shape + (4, 4), dtype=np.float64)
    m[..., :3, :3] = rotation3
    m[..., :3, 3] = position
    m[..., 3, 3] = 1.0
    return m


def rigid_from_pos_yaw(position, yaw):
    """Root transform of an upright entity: translation plus heading."""
    return compose_rigid(position, yaw_matrix(yaw))


def rigid_from_pos_quat(position, q):
    return compose_rigid(position, quat_to_matrix(q))


def rigid_inverse(m):
    """Inverse of a rotation + translation matrix (no scale / shear)."""
    m = np.asarray(m, dtype=np.float64)
    inv = np.zeros_like(m)
    rt = np.swapaxes(m[..., :3, :3], -1, -2)
    inv[..., :3, :3] = rt
    inv[..., :3, 3] = -np.einsum("...ij,...j->...i", rt, m[..., :3, 3])
    inv[..., 3, 3] = 1.0
    return inv


def transform_points(m, p):
    return np.einsum("...ij,...j->...i", m[..., :3, :3], p) + m[..., :3, 3]


def transform_vectors(m, v):
    return np.einsum("...ij,...j->...i", m[..., :3, :3], v)


# --------------------------------------------------------------------------- #
#  Intersection kernels
# --------------------------------------------------------------------------- #
def aabb_overlap(amin, amax, bmin, bmax):
    return np.all((amin <= bmax) & (amax >= bmin), axis=-1)


def ray_aabb_slab(origin, inv_dir, bmin, bmax):
    """
    Slab test.  All arguments broadcast; returns ``(t_near, t_far)``.  The ray
    hits when ``t_far >= max(t_near, 0)``.
    """
    t1 = (bmin - origin) * inv_dir
    t2 = (bmax - origin) * inv_dir
    t_near = np.max(np.minimum(t1, t2), axis=-1)
    t_far = np.min(np.maximum(t1, t2), axis=-1)
    return t_near, t_far


def ray_triangles(origin, direction, v0, e1, e2):
    """
    Two sided Moller-Trumbore test for paired rays / triangles.
    Returns ``(t, hit)``; ``t`` is only meaningful where ``hit`` is True.
    """
    p = cross(direction, e2)
    det = dot(e1, p)
    ok = np.abs(det) > 1.0e-12
    inv_det = 1.0 / np.where(ok, det, 1.0)
    tvec = origin - v0
    u = dot(tvec, p) * inv_det
    q = cross(tvec, e1)
    v = dot(direction, q) * inv_det
    t = dot(e2, q) * inv_det
    hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0)
    return t, hit


def perpendicular_basis(direction):
    """Two unit vectors spanning the plane perpendicular to ``direction``."""
    d = normalize(direction)
    world_up = np.zeros_like(d)
    world_up[..., 1] = 1.0
    alt = np.zeros_like(d)
    alt[..., 0] = 1.0
    ref = np.where(np.abs(d[..., 1:2]) > 0.999, alt, world_up)
    u = normalize(cross(ref, d))
    v = cross(d, u)
    return u, v


def apply_spread(direction, disc_offsets, spread_radians):
    """
    Deflect unit ``direction`` vectors inside a cone.  ``disc_offsets`` are
    points in the unit disc; ``spread_radians`` is the cone half-angle.
    """
    u, v = perpendicular_basis(direction)
    scale = np.tan(np.asarray(spread_radians, dtype=np.float64))
    if np.ndim(scale) > 0:
        scale = scale[..., None]
    deflected = direction + (u * disc_offsets[..., 0:1] + v * disc_offsets[..., 1:2]) * scale
    return normalize(deflected)


# --------------------------------------------------------------------------- #
#  Deterministic hashing (shared by client and server for spread patterns)
# --------------------------------------------------------------------------- #
def hash_u32(x):
    """Avalanching 32-bit integer hash (lowbias32); accepts scalars or arrays."""
    h = np.asarray(x).astype(np.uint64) & _MASK32
    h = h ^ (h >> np.uint64(16))
    h = (h * np.uint64(0x7FEB352D)) & _MASK32
    h = h ^ (h >> np.uint64(15))
    h = (h * np.uint64(0x846CA68B)) & _MASK32
    h = h ^ (h >> np.uint64(16))
    return h


def hash_unit_floats(seed: int, count: int, stream: int = 0):
    """``count`` reproducible floats in [0, 1) derived from an integer seed."""
    base = (int(seed) * 0x2545F491 + int(stream) * 0x3C6EF372 + 0x1B873593) & 0xFFFFFFFF
    index = np.arange(count, dtype=np.uint64)
    keys = (np.uint64(base) + index * np.uint64(0x9E3779B1)) & _MASK32
    return hash_u32(keys).astype(np.float64) / 4294967296.0


def spread_disc(seed: int, count: int):
    """``count`` points uniformly distributed inside the unit disc."""
    radius = np.sqrt(hash_unit_floats(seed, count, 1))
    theta = hash_unit_floats(seed, count, 2) * TAU
    out = np.empty((count, 2), dtype=np.float64)
    out[:, 0] = radius * np.cos(theta)
    out[:, 1] = radius * np.sin(theta)
    return out


def unit_floats_from_keys(keys, stream: int = 0):
    """One reproducible float in [0, 1) per integer key (vectorised)."""
    k = np.asarray(keys).astype(np.uint64) & _MASK32
    k = (k * np.uint64(0x9E3779B1) + np.uint64((int(stream) * 0x85EBCA6B + 0x1B873593) & 0xFFFFFFFF)) & _MASK32
    return hash_u32(k).astype(np.float64) / 4294967296.0


def spread_disc_keys(keys):
    """One point inside the unit disc per integer key; shape ``(len(keys), 2)``."""
    radius = np.sqrt(unit_floats_from_keys(keys, 1))
    theta = unit_floats_from_keys(keys, 2) * TAU
    out = np.empty((radius.shape[0], 2), dtype=np.float64)
    out[:, 0] = radius * np.cos(theta)
    out[:, 1] = radius * np.sin(theta)
    return out


def smooth_noise(t, seed: int = 0):
    """C1 value noise in [-1, 1]; used for camera shake and idle weapon sway."""
    t = np.asarray(t, dtype=np.float64)
    cell = np.floor(t)
    frac = t - cell
    icell = cell.astype(np.int64) & 0x7FFFFFFF
    key = (icell + np.int64(int(seed) * 7919 % 0x3FFFFFFF)) & 0x7FFFFFFF
    a = hash_u32(key).astype(np.float64) / 2147483648.0 - 1.0
    b = hash_u32(key + 1).astype(np.float64) / 2147483648.0 - 1.0
    w = frac * frac * (3.0 - 2.0 * frac)
    return a + (b - a) * w
