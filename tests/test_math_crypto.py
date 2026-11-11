"""Vector math, quaternion and cipher foundations."""
import numpy as np

from helpers import raises

from engine import crypto
from engine import mathx as M


def test_forward_right_up_are_orthonormal():
    rng = np.random.default_rng(1)
    yaw, pitch = rng.uniform(-3, 3, 64), rng.uniform(-1.4, 1.4, 64)
    f, r, u = M.forward_vector(yaw, pitch), M.right_vector(yaw), M.up_vector(yaw, pitch)
    for a, b in ((f, r), (f, u), (r, u)):
        assert np.allclose(M.dot(a, b), 0.0, atol=1e-12)
    for v in (f, r, u):
        assert np.allclose(M.length(v), 1.0)
    assert np.allclose(M.forward_vector(0.0), [0, 0, 1]) and np.allclose(M.right_vector(0.0), [1, 0, 0])
    assert np.allclose(M.forward_vector(np.pi / 2), [1, 0, 0], atol=1e-12)


def test_yaw_pitch_round_trip_through_quaternion_and_matrix():
    rng = np.random.default_rng(2)
    yaw, pitch = rng.uniform(-3, 3, 64), rng.uniform(-1.4, 1.4, 64)
    q = M.quat_from_yaw_pitch(yaw, pitch)
    assert np.allclose(M.quat_rotate(q, np.array([0.0, 0.0, 1.0])), M.forward_vector(yaw, pitch))
    assert np.allclose(M.quat_to_matrix(q), M.view_matrix3(yaw, pitch))
    back_yaw, back_pitch = M.quat_to_yaw_pitch(q)
    assert np.allclose(M.wrap_angle(back_yaw - yaw), 0.0, atol=1e-9) and np.allclose(back_pitch, pitch)
    d_yaw, d_pitch = M.yaw_pitch_from_direction(M.forward_vector(yaw, pitch) * 7.0)
    assert np.allclose(M.wrap_angle(d_yaw - yaw), 0.0, atol=1e-9) and np.allclose(d_pitch, pitch)


def test_slerp_takes_the_short_arc_at_constant_speed():
    a = M.quat_from_yaw_pitch(np.array([0.2, 3.0]), 0.0)
    b = M.quat_from_yaw_pitch(np.array([0.8, -3.0]), 0.0)
    mid_yaw, _ = M.quat_to_yaw_pitch(M.quat_slerp(a, b, 0.5))
    assert np.allclose(mid_yaw[0], 0.5)
    assert abs(abs(mid_yaw[1]) - np.pi) < 1e-9                       # 3.0 -> -3.0 crosses pi, not zero
    same = M.quat_slerp(a, a, np.array([0.3, 0.7]))
    assert np.allclose(same, a)
    quarter, _ = M.quat_to_yaw_pitch(M.quat_slerp(a[0], b[0], 0.25))
    assert np.allclose(quarter, 0.35)


def test_lerp_and_easing():
    assert M.lerp(2.0, 6.0, 0.25) == 3.0
    t = np.linspace(0, 1, 11)
    ease = M.ease_in_out_cos(t)
    assert ease[0] == 0.0 and abs(ease[-1] - 1.0) < 1e-12 and np.all(np.diff(ease) >= 0)
    assert np.allclose(ease + ease[::-1], 1.0)
    assert np.isclose(M.exp_decay(10.0, 2.0, 0.5), 10.0 * np.exp(-1.0))
    assert np.isclose(M.wrap_angle(3 * np.pi + 0.1), -np.pi + 0.1)


def test_rigid_inverse_and_point_transform():
    rng = np.random.default_rng(3)
    m = M.rigid_from_pos_yaw(rng.normal(size=(20, 3)), rng.uniform(-3, 3, 20))
    assert np.allclose(np.einsum("nij,njk->nik", m, M.rigid_inverse(m)), np.eye(4))
    p = rng.normal(size=(20, 3))
    assert np.allclose(M.transform_points(M.rigid_inverse(m), M.transform_points(m, p)), p)


def test_ray_triangle_and_slab_kernels():
    v0, e1, e2 = np.array([0.0, 0, 5]), np.array([2.0, 0, 0]), np.array([0.0, 2, 0])
    t, hit = M.ray_triangles(np.array([0.5, 0.5, 0.0]), np.array([0.0, 0, 1]), v0, e1, e2)
    assert hit and np.isclose(t, 5.0)
    _, miss = M.ray_triangles(np.array([3.0, 3.0, 0.0]), np.array([0.0, 0, 1]), v0, e1, e2)
    assert not miss
    near, far = M.ray_aabb_slab(np.zeros(3), M.safe_inverse(np.array([0.0, 0, 1])), np.array([-1.0, -1, 4]), np.array([1.0, 1, 6]))
    assert np.isclose(near, 4.0) and np.isclose(far, 6.0)


def test_spread_hash_is_deterministic_and_uniform():
    keys = np.arange(4000)
    a, b = M.spread_disc_keys(keys), M.spread_disc_keys(keys)
    assert np.array_equal(a, b)
    radius = np.hypot(a[:, 0], a[:, 1])
    assert radius.max() <= 1.0 and abs(np.mean(radius ** 2) - 0.5) < 0.03 and np.abs(a.mean(axis=0)).max() < 0.03
    assert not np.array_equal(M.spread_disc_keys(keys + 1)[:-1], a[:-1])
    direction = np.tile(M.forward_vector(0.4, 0.1), (4000, 1))
    spread = M.apply_spread(direction, a, 0.05)
    angle = np.arccos(np.clip(M.dot(spread, direction), -1, 1))
    assert angle.max() <= 0.05 + 1e-9


def test_seal_round_trip_and_tamper_detection():
    blob = crypto.seal(b"key", b"lobby beacon payload" * 5)
    assert crypto.unseal(b"key", blob) == b"lobby beacon payload" * 5
    assert b"lobby" not in blob
    for index in (0, 15, len(blob) - 1):
        forged = bytearray(blob)
        forged[index] ^= 0x01
        with raises(crypto.AuthenticationError):
            crypto.unseal(b"key", bytes(forged))
    with raises(crypto.AuthenticationError):
        crypto.unseal(b"other key", blob)
    assert crypto.seal(b"key", b"x") != crypto.seal(b"key", b"x")       # fresh nonce every time
    assert len(crypto.keystream(b"k", b"n" * 12, 100)) == 100
