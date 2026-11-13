"""Module B: octree, collision world, kinematic controller, hitboxes, camera rig."""
import time

import numpy as np

import helpers  # noqa: F401  (puts the project root on sys.path)

from engine import config as C
from engine import mathx as M
from engine.controller_3d import (CameraRig, CollisionWorld, HitboxSet, KinematicState, Octree, eye_height,
                                  step_kinematics, HITBOX_ZONE)


def random_world(seed=7, boxes=500, triangles=30):
    rng = np.random.default_rng(seed)
    center = rng.uniform(-60, 60, (boxes, 3))
    center[:, 1] = rng.uniform(0, 12, boxes)
    half = rng.uniform(0.2, 3.0, (boxes, 3))
    half[:15] *= 8
    tris = rng.uniform(-50, 50, (triangles, 1, 3)) + rng.uniform(-3, 3, (triangles, 3, 3))
    return CollisionWorld(center - half, center + half, triangles=tris), rng


def arena():
    boxes = [([-50, -1, -50], [50, 0, 50]), ([5, 0, -5], [6, 4, 5])]
    boxes += [([-10 - i * 0.5, 0, 10], [-9.5 - i * 0.5, (i + 1) * 0.3, 14]) for i in range(8)]
    boxes += [([-20, 2.4, 10], [-14, 2.6, 14]), ([10, 0, 10], [14, 1.4, 14]), ([10, 1.4, 10], [14, 1.5, 14])]
    return CollisionWorld(np.array([b[0] for b in boxes], float), np.array([b[1] for b in boxes], float))


def test_octree_raycast_matches_brute_force():
    world, rng = random_world()
    origin = rng.uniform(-70, 70, (1500, 3))
    origin[:, 1] = rng.uniform(0, 14, 1500)
    direction = M.normalize(rng.normal(size=(1500, 3)))
    t, item, normal = world.raycast(origin, direction, 150.0)
    assert np.allclose(t, world.octree.raycast_bruteforce(origin, direction, 150.0), atol=1e-9)
    hit = item >= 0
    assert hit.sum() > 500
    assert np.allclose(np.linalg.norm(normal[hit], axis=1), 1.0)
    assert np.all(M.dot(normal[hit], direction[hit]) <= 1e-9)           # normals face the ray


def test_octree_aabb_query_matches_brute_force():
    world, rng = random_world(seed=11)
    qmin = rng.uniform(-60, 60, (300, 3))
    qmax = qmin + rng.uniform(0.5, 6, (300, 3))
    got = set(zip(*[a.tolist() for a in world.octree.query_aabb(qmin, qmax)]))
    want = set(zip(*[a.tolist() for a in world.octree.query_aabb_bruteforce(qmin, qmax)]))
    assert got == want and len(got) > 20


def test_octree_dynamic_insert_remove_and_layers():
    tree = Octree([-20, -20, -20], [20, 20, 20], max_depth=5, leaf_capacity=4)
    rng = np.random.default_rng(3)
    lo = rng.uniform(-18, 16, (60, 3))
    tree.add_boxes(lo, lo + rng.uniform(0.3, 2.0, (60, 3)))
    tree.build()
    pillar = tree.insert_box([-0.5, -19, -0.5], [0.5, -15, 0.5], layer=C.LAYER_PLAYER)
    down = (np.array([[0.0, 19.0, 0.0]]), np.array([[0.0, -1.0, 0.0]]))
    t, item, _ = tree.raycast(*down, 100.0, C.LAYER_PLAYER)
    assert np.isclose(t[0], tree.raycast_bruteforce(*down, 100.0, C.LAYER_PLAYER)[0])
    _, bullet_item, _ = tree.raycast(np.array([[0.0, -17.0, -10.0]]), np.array([[0.0, 0.0, 1.0]]), 9.6, C.LAYER_BULLET)
    assert bullet_item[0] != pillar                                      # player-only box is invisible to bullets
    for _ in range(80):
        p = rng.uniform(-18, 16, 3)
        tree.insert_box(p, p + rng.uniform(0.2, 1.5, 3))
    origin = rng.uniform(-19, 19, (200, 3))
    direction = M.normalize(rng.normal(size=(200, 3)))
    assert np.allclose(tree.raycast(origin, direction, 80.0)[0], tree.raycast_bruteforce(origin, direction, 80.0))
    tree.remove(pillar)
    assert tree.raycast(np.array([[0.0, -17.0, -5.0]]), np.array([[0.0, 0.0, 1.0]]), 4.9, C.LAYER_PLAYER)[1][0] != pillar
    stats = tree.stats()
    assert stats["items"] == 140 and stats["max_depth"] <= 5


def test_single_ray_stays_inside_the_half_millisecond_budget():
    world, rng = random_world()
    origin = rng.uniform(-60, 60, (200, 3))
    direction = M.normalize(rng.normal(size=(200, 3)))
    world.raycast(origin[0], direction[0], 150.0)
    samples = []
    for i in range(200):
        start = time.perf_counter()
        world.raycast(origin[i], direction[i], 150.0)
        samples.append(time.perf_counter() - start)
    assert np.median(samples) < 0.0005, "median single ray %.3f ms" % (np.median(samples) * 1e3)


def test_line_of_sight():
    world = arena()
    assert not world.line_of_sight(np.array([[0.0, 1.0, 0.0]]), np.array([[10.0, 1.0, 0.0]]))[0]     # wall at x = 5
    assert world.line_of_sight(np.array([[0.0, 1.0, 0.0]]), np.array([[4.0, 1.0, 0.0]]))[0]
    assert world.line_of_sight(np.array([[0.0, 5.0, 0.0]]), np.array([[10.0, 5.0, 0.0]]))[0]         # over the wall


def run(ks, world, row, buttons, yaw, ticks):
    for _ in range(ticks):
        step_kinematics(ks, [row], np.uint16(buttons), yaw, world)


def test_walls_floors_and_gravity():
    world, ks = arena(), KinematicState(2)
    ks.reset_rows([0, 1], np.array([[0.0, 0, 0], [0.0, 3.0, 0]]))
    ks.mflags[1] = 0
    for _ in range(120):
        step_kinematics(ks, [0, 1], np.array([C.BTN_FORWARD | C.BTN_SPRINT, 0], np.uint16), np.array([np.pi / 2, 0.0]), world)
    assert abs(ks.pos[0, 0] - (5.0 - C.PLAYER_RADIUS)) < 1e-6 and ks.vel[0, 0] == 0.0
    assert abs(ks.pos[1, 1]) < 1e-9 and ks.mflags[1] & C.MF_ON_GROUND


def test_sprint_speed_and_exponential_slide():
    world, ks = arena(), KinematicState(1)
    ks.reset_rows([0], [0.0, 0, -30.0])
    run(ks, world, 0, C.BTN_FORWARD | C.BTN_SPRINT, 0.0, 90)
    assert abs(np.hypot(ks.vel[0, 0], ks.vel[0, 2]) - C.SPRINT_SPEED) < 1e-6
    speeds, heights, sliding = [], [], []
    for _ in range(80):
        step_kinematics(ks, [0], np.uint16(C.BTN_FORWARD | C.BTN_SPRINT | C.BTN_SLIDE), 0.0, world)
        speeds.append(np.hypot(ks.vel[0, 0], ks.vel[0, 2]))
        heights.append(float(eye_height(ks.crouch_t[0])))
        sliding.append(bool(ks.mflags[0] & C.MF_SLIDING))
    assert sliding[0] and speeds[0] > C.SPRINT_SPEED                                    # launch impulse
    ratio = np.array(speeds[1:10]) / np.array(speeds[0:9])
    assert np.allclose(ratio, np.exp(-C.SLIDE_FRICTION_K * C.TICK_DT), atol=1e-9)       # V_next = V * e^(-k dt)
    assert not sliding[-1]                                                              # the slide ends by itself
    assert abs(min(heights) - C.EYE_HEIGHT_STAND * 0.6) < 1e-9                          # camera 40 % lower
    steps = np.abs(np.diff(heights[:14]))
    assert steps[0] < steps[len(steps) // 2]                                            # eased in, not linear


def test_slide_needs_speed_and_locks_heading():
    world, ks = arena(), KinematicState(1)
    ks.reset_rows([0], [0.0, 0, -30.0])
    run(ks, world, 0, C.BTN_FORWARD, 0.0, 60)                                           # walking is too slow
    step_kinematics(ks, [0], np.uint16(C.BTN_FORWARD | C.BTN_SLIDE), 0.0, world)
    assert not ks.mflags[0] & C.MF_SLIDING and ks.mflags[0] & C.MF_CROUCHED
    ks.reset_rows([0], [0.0, 0, -30.0])
    run(ks, world, 0, C.BTN_FORWARD | C.BTN_SPRINT, 0.0, 60)
    step_kinematics(ks, [0], np.uint16(C.BTN_SLIDE), 0.0, world)
    for _ in range(10):                                                                 # steering input is ignored
        step_kinematics(ks, [0], np.uint16(C.BTN_SLIDE | C.BTN_RIGHT), 1.2, world)
    assert ks.mflags[0] & C.MF_SLIDING and abs(ks.vel[0, 0]) < 1e-9 and ks.vel[0, 2] > 6.0


def test_jump_coyote_time_and_no_double_jump():
    world, ks = arena(), KinematicState(1)
    ks.reset_rows([0], [0.0, 0, -30.0])
    heights = []
    for tick in range(70):
        step_kinematics(ks, [0], np.uint16(C.BTN_JUMP if tick % 2 == 0 else 0), 0.0, world)   # mashing jump
        heights.append(ks.pos[0, 1])
    apex = max(heights)
    assert 0.95 < apex < C.JUMP_SPEED ** 2 / (2 * C.GRAVITY) + 0.01
    first_landing = next(i for i in range(5, 70) if heights[i] == 0.0)
    assert max(heights[:first_landing]) == apex                                        # no second jump in mid-air
    def walk_off_ledge(extra_ticks):
        ks.reset_rows([0], [12.0, 1.5, 9.8])                                            # on the 1.5 high block, at its edge
        while ks.mflags[0] & C.MF_ON_GROUND:
            step_kinematics(ks, [0], np.uint16(C.BTN_BACK), 0.0, world)
        run(ks, world, 0, C.BTN_BACK, 0.0, extra_ticks)
        step_kinematics(ks, [0], np.uint16(C.BTN_BACK | C.BTN_JUMP), 0.0, world)
        return ks.vel[0, 1]

    assert walk_off_ledge(2) > 5.0                                                      # inside the coyote window
    assert walk_off_ledge(10) < 0.0                                                     # too late: keeps falling


def test_stairs_up_and_down_without_leaving_the_ground():
    world, ks = arena(), KinematicState(1)
    ks.reset_rows([0], [-6.0, 0, 12.0])
    run(ks, world, 0, C.BTN_FORWARD, -np.pi / 2, 150)
    assert abs(ks.pos[0, 1] - 2.6) < 1e-6                                               # reached the landing
    airborne = 0
    for _ in range(150):
        step_kinematics(ks, [0], np.uint16(C.BTN_FORWARD), np.pi / 2, world)
        airborne += not ks.mflags[0] & C.MF_ON_GROUND
    assert airborne == 0 and abs(ks.pos[0, 1]) < 1e-9


def test_crouch_blocks_standing_up_under_a_ceiling():
    boxes = [([-20, -1, -20], [20, 0, 20]), ([-2, 1.3, 2], [2, 1.6, 12])]
    world = CollisionWorld(np.array([b[0] for b in boxes], float), np.array([b[1] for b in boxes], float))
    ks = KinematicState(1)
    ks.reset_rows([0], [0.0, 0, 0.0])
    run(ks, world, 0, C.BTN_FORWARD, 0.0, 60)
    assert ks.pos[0, 2] < 2.0 - C.PLAYER_RADIUS + 1e-6                                  # standing: blocked by the beam
    run(ks, world, 0, C.BTN_FORWARD | C.BTN_SLIDE, 0.0, 90)
    assert ks.pos[0, 2] > 3.0                                                           # crouched: passes underneath
    run(ks, world, 0, 0, 0.0, 20)
    assert ks.mflags[0] & C.MF_CROUCHED                                                 # no head room: stays crouched


def test_batch_and_single_row_stepping_are_bit_identical():
    world = arena()
    rng = np.random.default_rng(5)
    a, b = KinematicState(8), KinematicState(8)
    start = rng.uniform(-8, 4, (8, 3))
    start[:, 1] = 0.0
    a.reset_rows(np.arange(8), start)
    b.reset_rows(np.arange(8), start)
    buttons, yaw = np.zeros(8, np.uint16), np.zeros(8)
    for tick in range(500):
        if tick % 7 == 0:
            buttons = rng.integers(0, 1 << 12, 8).astype(np.uint16)
        if tick % 11 == 0:
            yaw = rng.uniform(-3, 3, 8)
        step_kinematics(a, np.arange(8), buttons, yaw, world)
        for row in range(8):
            step_kinematics(b, [row], buttons[row], yaw[row], world)
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.vel, b.vel) and np.array_equal(a.mflags, b.mflags)
    assert np.all(a.pos[:, 1] >= -1e-9)                                                 # nobody fell through the floor


def test_hitbox_zones_follow_the_root_transform():
    hb = HitboxSet(3)
    pos = np.array([[0.0, 0, 10], [4.0, 0, 10], [0.0, 0, 30]])
    hb.update(np.arange(3), pos, np.array([0.0, np.pi / 2, 0.0]), np.array([0.0, 0.0, 1.0]), np.ones(3, bool))
    forward = np.array([[0.0, 0, 1.0]])
    for height, zone in ((1.62, C.ZONE_HEAD), (1.2, C.ZONE_TORSO), (0.3, C.ZONE_LIMB)):
        origin = np.array([[0.0 if zone != C.ZONE_LIMB else 0.12, height, 0.0]])
        t, entity, box = hb.raycast(origin, forward, 100.0, np.array([True, True, False]))
        assert entity[0] == 0 and HITBOX_ZONE[box[0]] == zone and 9.5 < t[0] < 10.0
    _, entity, _ = hb.raycast(np.array([[0.0, 1.62, 0.0]]), forward, 5.0, np.ones(3, bool))
    assert entity[0] == -1                                                              # occluded: max distance 5
    _, entity, box = hb.raycast(np.array([[0.0, 0.97, 20.0]]), forward, 100.0, np.ones(3, bool))
    assert entity[0] == 2 and HITBOX_ZONE[box[0]] == C.ZONE_HEAD                        # crouched head is 40 % lower
    side = hb.world_center[1, 4] - pos[1]
    assert abs(side[2] + 0.33) < 1e-9 and abs(side[0]) < 1e-9                           # right arm rotated with yaw
    saved = hb.snapshot()
    elsewhere = np.tile(np.eye(4), (3, 1, 1))
    elsewhere[:, :3, 3] = (100.0, 0.0, 100.0)
    hb.set_from_matrices(elsewhere, np.zeros(3), np.ones(3, bool))
    assert hb.raycast(np.array([[0.0, 1.62, 0.0]]), forward, 100.0, np.ones(3, bool))[1][0] == -1
    hb.restore(saved)
    assert hb.raycast(np.array([[0.0, 1.62, 0.0]]), forward, 100.0, np.ones(3, bool))[1][0] == 0


def test_camera_rig_feel():
    rig = CameraRig(90.0)
    for _ in range(60):
        rig.update(1 / 60, [0, 0, 0], [C.WALK_SPEED, 0, 0], C.MF_ON_GROUND, 0.0, 0.0, 0.0)
    strafe_right = rig.cam_roll
    for _ in range(60):
        rig.update(1 / 60, [0, 0, 0], [-C.WALK_SPEED, 0, 0], C.MF_ON_GROUND, 0.0, 0.0, 0.0)
    assert strafe_right < -0.02 and rig.cam_roll > 0.02                                 # banks into the strafe
    for _ in range(60):
        rig.update(1 / 60, [0, 0, 0], [0, 0, 9.0], C.MF_ON_GROUND | C.MF_SLIDING, 1.0, 0.0, 0.0)
    assert abs(rig.cam_roll) > np.radians(4.0) and rig.fov > 95.0                       # slide tilt + FOV kick
    assert abs(rig.cam_pos[1] - C.EYE_HEIGHT_CROUCH) < 0.05
    rig.add_recoil(0.03, 0.0)
    climbed = 0.0
    for _ in range(12):
        rig.update(1 / 60, [0, 0, 0], [0, 0, 0], C.MF_ON_GROUND, 0.0, 0.0, 0.0)
        climbed += rig.consume_aim_kick()[0]
    assert 0.01 < climbed < 0.03                                                        # aim climbs, then recovers
    near, far = CameraRig(), CameraRig()
    near.add_explosion_shake([0, 0, 0], [[2.0, 0, 0]], power=0.8)
    far.add_explosion_shake([0, 0, 0], [[4.0, 0, 0]], power=0.8)
    assert np.isclose(near.trauma / far.trauma, 4.0)                                    # inverse-square falloff


def test_camera_springs_are_stable_at_any_frame_time():
    """A 10 fps frame (level load, software rendering) must not make the recoil or landing springs diverge."""
    from engine.controller_3d import CameraRig, spring_step
    feet = np.zeros(3)
    peaks = {}
    for dt in (1.0 / 240.0, 1.0 / 60.0, 0.05, 0.1):
        rig = CameraRig(90.0)
        rig.add_recoil(0.02, 0.006)
        rig.add_landing(9.0)
        peak = 0.0
        for _ in range(int(3.0 / dt)):
            rig.update(dt, feet, np.zeros(3), C.MF_ON_GROUND, 0.0, 0.0, 0.0)
            peak = max(peak, float(np.abs(rig.punch).max()))
            assert np.all(np.isfinite(rig.punch)) and np.isfinite(rig.land_offset)
        peaks[dt] = peak
        assert np.abs(rig.punch).max() < 1e-4 and abs(rig.land_offset) < 1e-3       # settled after three seconds
        assert np.all(np.abs(rig.shake_uv) <= 0.012) and np.isfinite(rig.cam_yaw) and np.isfinite(rig.cam_pitch)
    assert peaks[0.1] <= peaks[1.0 / 60.0] * 1.5 and peaks[1.0 / 60.0] < 0.05         # same feel, never amplified
    x, v = spring_step(np.array([1.0, -2.0]), np.zeros(2), 1.0e6, 0.0, 1.0)             # absurdly stiff, huge step
    assert np.all(np.abs(x) <= np.array([1.0, 2.0])) and np.all(np.isfinite(v))
