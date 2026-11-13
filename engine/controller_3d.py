"""
NEXUS-3D-EXTREME :: engine/controller_3d.py            (MODULE B)
=================================================================
3D kinematics, bounding volumes and spatial partitioning.

Contents
--------
1. ``Octree``            dynamic loose octree stored in flat NumPy arrays with
                         batched, fully vectorised AABB and ray traversal.
2. ``CollisionWorld``    static level geometry indexed by the octree.
3. ``KinematicState`` /  structure-of-arrays first-person controller: sprint,
   ``step_kinematics``   crouch, exponential-friction slide, jump (coyote time
                         and input buffering), stair stepping, swept AABB
                         collision.  One call advances every row passed in.
4. ``HitboxSet``         anatomical oriented bounding boxes (head / torso /
                         limbs) that follow each entity root transform, plus a
                         batched ray test.
5. ``CameraRig``         view feel: cosine eased crouch height, velocity driven
                         double-sine weapon sway, strafe / slide banking,
                         recoil, landing dip and inverse-square explosion shake.

The module never imports Ursina, so the dedicated server and the unit tests
run it without a window.  The exact same ``step_kinematics`` is executed by
the authoritative server and by the predicting client.
"""
from __future__ import annotations

from itertools import chain

import numpy as np

from . import config as C
from . import mathx as M

_FAR = 1.0e30


# =========================================================================== #
#  1. OCTREE
# =========================================================================== #
class Octree:
    """
    Dynamic loose octree over axis aligned boxes and triangles.

    Storage is data oriented: node bounds, child links and item ranges live in
    contiguous arrays, and both query kinds advance a *frontier* of
    ``(query, node)`` pairs one tree level per NumPy call, so thousands of rays
    or boxes are resolved without a Python loop per ray or per node.

    Placement rule (looseness ``k``): an item descends into the child octant
    that contains its centre while its largest half extent still fits inside
    the child's loosened cube (``radius <= child_half * (k - 1)``).  Every item
    is therefore stored in exactly one node, which removes the duplicate
    filtering a clipping octree needs.  A node keeps up to ``leaf_capacity``
    items before it is subdivided, down to ``max_depth`` levels; the smallest
    voxel is ``root_size / 2 ** max_depth``.

    After every structural change the tree is "baked": per-node item lists are
    flattened into CSR arrays and each node receives the tight bounding box of
    its whole subtree, which is what the traversals test against.
    """

    KIND_BOX = 0
    KIND_TRIANGLE = 1

    def __init__(self, bounds_min, bounds_max, max_depth: int = 6, leaf_capacity: int = 8,
                 looseness: float = 2.0):
        bounds_min = np.asarray(bounds_min, dtype=np.float64)
        bounds_max = np.asarray(bounds_max, dtype=np.float64)
        self.max_depth = int(max_depth)
        self.leaf_capacity = int(leaf_capacity)
        self.looseness = float(looseness)
        self.root_center = (bounds_min + bounds_max) * 0.5
        self.root_half = float(np.max(bounds_max - bounds_min)) * 0.5 * 1.001 + 1.0e-6
        self.version = 0

        # ---- item arrays (amortised growth) --------------------------------
        cap = 256
        self.item_count = 0
        self.item_min = np.zeros((cap, 3), dtype=np.float64)
        self.item_max = np.zeros((cap, 3), dtype=np.float64)
        self.item_center = np.zeros((cap, 3), dtype=np.float64)
        self.item_radius = np.zeros(cap, dtype=np.float64)
        self.item_kind = np.zeros(cap, dtype=np.uint8)
        self.item_layer = np.zeros(cap, dtype=np.uint8)
        self.item_ref = np.full(cap, -1, dtype=np.int32)
        self.item_tag = np.zeros(cap, dtype=np.int32)
        self.item_alive = np.zeros(cap, dtype=bool)
        self.item_node = np.full(cap, -1, dtype=np.int32)

        # ---- triangle arrays ------------------------------------------------
        self.tri_count = 0
        self.tri_v0 = np.zeros((16, 3), dtype=np.float64)
        self.tri_e1 = np.zeros((16, 3), dtype=np.float64)
        self.tri_e2 = np.zeros((16, 3), dtype=np.float64)
        self.tri_normal = np.zeros((16, 3), dtype=np.float64)

        # ---- node arrays ----------------------------------------------------
        self._reset_nodes()
        self._built = False

    # ------------------------------------------------------------------ #
    #  storage helpers
    # ------------------------------------------------------------------ #
    def _reset_nodes(self) -> None:
        cap = 64
        self.node_count = 0
        self.node_center = np.zeros((cap, 3), dtype=np.float64)
        self.node_half = np.zeros(cap, dtype=np.float64)
        self.node_depth = np.zeros(cap, dtype=np.int16)
        self.node_child = np.full((cap, 8), -1, dtype=np.int32)
        self._node_items: list[list[int]] = []
        self.node_start = np.zeros(1, dtype=np.int64)
        self.node_len = np.zeros(1, dtype=np.int64)
        self.node_bmin = np.full((1, 3), _FAR, dtype=np.float64)
        self.node_bmax = np.full((1, 3), _FAR, dtype=np.float64)
        self.flat_items = np.zeros(0, dtype=np.int64)
        self._dirty = True
        self._new_node(self.root_center, self.root_half, 0)

    @staticmethod
    def _grown(array: np.ndarray, capacity: int, fill=0) -> np.ndarray:
        shape = (capacity,) + array.shape[1:]
        out = np.full(shape, fill, dtype=array.dtype)
        out[: array.shape[0]] = array
        return out

    def _reserve_items(self, extra: int) -> None:
        need = self.item_count + extra
        cap = self.item_min.shape[0]
        if need <= cap:
            return
        while cap < need:
            cap *= 2
        self.item_min = self._grown(self.item_min, cap)
        self.item_max = self._grown(self.item_max, cap)
        self.item_center = self._grown(self.item_center, cap)
        self.item_radius = self._grown(self.item_radius, cap)
        self.item_kind = self._grown(self.item_kind, cap)
        self.item_layer = self._grown(self.item_layer, cap)
        self.item_ref = self._grown(self.item_ref, cap, -1)
        self.item_tag = self._grown(self.item_tag, cap)
        self.item_alive = self._grown(self.item_alive, cap, False)
        self.item_node = self._grown(self.item_node, cap, -1)

    def _new_node(self, center, half: float, depth: int) -> int:
        index = self.node_count
        if index >= self.node_center.shape[0]:
            cap = self.node_center.shape[0] * 2
            self.node_center = self._grown(self.node_center, cap)
            self.node_half = self._grown(self.node_half, cap)
            self.node_depth = self._grown(self.node_depth, cap)
            self.node_child = self._grown(self.node_child, cap, -1)
        self.node_center[index] = center
        self.node_half[index] = half
        self.node_depth[index] = depth
        self.node_child[index] = -1
        self._node_items.append([])
        self.node_count += 1
        return index

    # ------------------------------------------------------------------ #
    #  item registration
    # ------------------------------------------------------------------ #
    def _register(self, mins, maxs, kind: int, layer, tag, ref) -> np.ndarray:
        mins = np.atleast_2d(np.asarray(mins, dtype=np.float64))
        maxs = np.atleast_2d(np.asarray(maxs, dtype=np.float64))
        count = mins.shape[0]
        self._reserve_items(count)
        ids = np.arange(self.item_count, self.item_count + count, dtype=np.int64)
        self.item_min[ids] = mins
        self.item_max[ids] = maxs
        self.item_center[ids] = (mins + maxs) * 0.5
        self.item_radius[ids] = np.max(maxs - mins, axis=1) * 0.5
        self.item_kind[ids] = kind
        self.item_layer[ids] = layer
        self.item_tag[ids] = tag
        self.item_ref[ids] = ref
        self.item_alive[ids] = True
        self.item_node[ids] = -1
        self.item_count += count
        if self._built:
            for item in ids.tolist():
                self._insert_item(item)
            self._touch()
        return ids

    def add_boxes(self, mins, maxs, layer=C.LAYER_ALL, tag=0) -> np.ndarray:
        """Register axis aligned boxes; returns their item ids."""
        return self._register(mins, maxs, self.KIND_BOX, layer, tag, -1)

    def add_triangles(self, triangles, layer=C.LAYER_BULLET, tag=0) -> np.ndarray:
        """Register a triangle soup ``(T, 3, 3)`` (mesh geometry); returns item ids."""
        tris = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        count = tris.shape[0]
        need = self.tri_count + count
        if need > self.tri_v0.shape[0]:
            cap = self.tri_v0.shape[0]
            while cap < need:
                cap *= 2
            self.tri_v0 = self._grown(self.tri_v0, cap)
            self.tri_e1 = self._grown(self.tri_e1, cap)
            self.tri_e2 = self._grown(self.tri_e2, cap)
            self.tri_normal = self._grown(self.tri_normal, cap)
        refs = np.arange(self.tri_count, need, dtype=np.int32)
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        self.tri_v0[refs] = tris[:, 0]
        self.tri_e1[refs] = e1
        self.tri_e2[refs] = e2
        self.tri_normal[refs] = M.normalize(M.cross(e1, e2))
        self.tri_count = need
        return self._register(tris.min(axis=1), tris.max(axis=1), self.KIND_TRIANGLE, layer, tag, refs)

    def insert_box(self, box_min, box_max, layer=C.LAYER_ALL, tag=0) -> int:
        """Dynamic insertion of one box into an already built tree."""
        if not self._built:
            self.build()
        return int(self.add_boxes(box_min, box_max, layer, tag)[0])

    def remove(self, item: int) -> None:
        """Remove an item (dynamic geometry such as destroyed cover)."""
        item = int(item)
        if not self.item_alive[item]:
            return
        node = int(self.item_node[item])
        if node >= 0:
            self._node_items[node].remove(item)
        self.item_alive[item] = False
        self.item_node[item] = -1
        self._touch()

    def move_box(self, item: int, box_min, box_max) -> None:
        """Re-seat a dynamic box after it moved."""
        item = int(item)
        node = int(self.item_node[item])
        if node >= 0:
            self._node_items[node].remove(item)
        box_min = np.asarray(box_min, dtype=np.float64)
        box_max = np.asarray(box_max, dtype=np.float64)
        self.item_min[item] = box_min
        self.item_max[item] = box_max
        self.item_center[item] = (box_min + box_max) * 0.5
        self.item_radius[item] = float(np.max(box_max - box_min)) * 0.5
        self.item_alive[item] = True
        self._insert_item(item)
        self._touch()

    def _touch(self) -> None:
        self._dirty = True
        self.version += 1

    # ------------------------------------------------------------------ #
    #  construction
    # ------------------------------------------------------------------ #
    def build(self) -> "Octree":
        """(Re)build the hierarchy from every live item with vectorised partitioning."""
        self._reset_nodes()
        alive = np.flatnonzero(self.item_alive[: self.item_count])
        if alive.size:
            lo = self.item_min[alive].min(axis=0)
            hi = self.item_max[alive].max(axis=0)
            center = (lo + hi) * 0.5
            half = float(np.max(hi - lo)) * 0.5 * 1.001 + 1.0e-6
            if np.any(np.abs(center - self.root_center) + half > self.root_half):
                self.root_center = center
                self.root_half = max(half, self.root_half)
                self.node_center[0] = self.root_center
                self.node_half[0] = self.root_half
            self._partition(0, alive)
        self._built = True
        self._touch()
        return self

    def _partition(self, node: int, items: np.ndarray) -> None:
        """Recursively distribute ``items`` below ``node`` (one NumPy pass per node)."""
        depth = int(self.node_depth[node])
        if items.size <= self.leaf_capacity or depth >= self.max_depth:
            self._node_items[node].extend(items.tolist())
            self.item_node[items] = node
            return
        center = self.node_center[node].copy()
        child_half = float(self.node_half[node]) * 0.5
        fits = self.item_radius[items] <= child_half * (self.looseness - 1.0)
        stay = items[~fits]
        if stay.size:
            self._node_items[node].extend(stay.tolist())
            self.item_node[stay] = node
        go = items[fits]
        if go.size == 0:
            return
        ic = self.item_center[go]
        octant = ((ic[:, 0] >= center[0]).astype(np.int64)
                  | ((ic[:, 1] >= center[1]).astype(np.int64) << 1)
                  | ((ic[:, 2] >= center[2]).astype(np.int64) << 2))
        for k in range(8):
            sub = go[octant == k]
            if sub.size == 0:
                continue
            child = int(self.node_child[node, k])
            if child < 0:
                offset = np.array([(k & 1) * 2 - 1, ((k >> 1) & 1) * 2 - 1, ((k >> 2) & 1) * 2 - 1],
                                  dtype=np.float64) * child_half
                child = self._new_node(center + offset, child_half, depth + 1)
                self.node_child[node, k] = child
            self._partition(child, sub)

    def _insert_item(self, item: int) -> None:
        node = 0
        radius = float(self.item_radius[item])
        center = self.item_center[item]
        while True:
            depth = int(self.node_depth[node])
            child_half = float(self.node_half[node]) * 0.5
            fits = depth < self.max_depth and radius <= child_half * (self.looseness - 1.0)
            has_children = bool(np.any(self.node_child[node] >= 0))
            if not fits:
                self._node_items[node].append(item)
                self.item_node[item] = node
                return
            if not has_children:
                here = self._node_items[node]
                if len(here) < self.leaf_capacity:
                    here.append(item)
                    self.item_node[item] = node
                    return
                everything = np.array(here + [item], dtype=np.int64)
                self._node_items[node] = []
                self._partition(node, everything)
                return
            nc = self.node_center[node]
            k = int(center[0] >= nc[0]) | (int(center[1] >= nc[1]) << 1) | (int(center[2] >= nc[2]) << 2)
            child = int(self.node_child[node, k])
            if child < 0:
                offset = np.array([(k & 1) * 2 - 1, ((k >> 1) & 1) * 2 - 1, ((k >> 2) & 1) * 2 - 1],
                                  dtype=np.float64) * child_half
                child = self._new_node(nc + offset, child_half, depth + 1)
                self.node_child[node, k] = child
            node = child

    def _bake(self) -> None:
        """Flatten item lists to CSR and compute tight subtree bounds."""
        n = self.node_count
        lens = np.fromiter((len(lst) for lst in self._node_items), dtype=np.int64, count=n)
        total = int(lens.sum())
        self.node_len = lens
        self.node_start = np.cumsum(lens) - lens
        self.flat_items = np.fromiter(chain.from_iterable(self._node_items), dtype=np.int64, count=total)
        bmin = np.full((n, 3), _FAR, dtype=np.float64)
        bmax = np.full((n, 3), -_FAR, dtype=np.float64)
        filled = np.flatnonzero(lens > 0)
        if filled.size:
            starts = self.node_start[filled]
            bmin[filled] = np.minimum.reduceat(self.item_min[self.flat_items], starts, axis=0)
            bmax[filled] = np.maximum.reduceat(self.item_max[self.flat_items], starts, axis=0)
        child = self.node_child[:n]
        for node in range(n - 1, -1, -1):          # children always have larger indices
            kids = child[node]
            kids = kids[kids >= 0]
            if kids.size:
                bmin[node] = np.minimum(bmin[node], bmin[kids].min(axis=0))
                bmax[node] = np.maximum(bmax[node], bmax[kids].max(axis=0))
        empty = bmax[:, 0] < bmin[:, 0]
        bmin[empty] = _FAR
        bmax[empty] = _FAR
        self.node_bmin = bmin
        self.node_bmax = bmax
        self._dirty = False

    def _ensure_baked(self) -> None:
        if not self._built:
            self.build()
        if self._dirty:
            self._bake()

    # ------------------------------------------------------------------ #
    #  traversal
    # ------------------------------------------------------------------ #
    def _expand(self, acc_q: list, acc_n: list):
        """Turn accumulated (query, node) pairs into (query, item) pairs."""
        if not acc_q:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        nq = np.concatenate(acc_q)
        nn = np.concatenate(acc_n)
        counts = self.node_len[nn]
        total = int(counts.sum())
        out_start = np.cumsum(counts) - counts
        index = np.arange(total, dtype=np.int64) - np.repeat(out_start, counts) + np.repeat(self.node_start[nn], counts)
        return np.repeat(nq, counts), self.flat_items[index]

    def query_aabb(self, qmin, qmax, layer_mask: int = C.LAYER_ALL):
        """
        Batched box query.  ``qmin`` / ``qmax`` are ``(Q, 3)``.  Returns two
        equally long int arrays ``(query_index, item_id)`` listing every item
        whose AABB intersects the corresponding query box.
        """
        self._ensure_baked()
        qmin = np.atleast_2d(np.asarray(qmin, dtype=np.float64))
        qmax = np.atleast_2d(np.asarray(qmax, dtype=np.float64))
        count = qmin.shape[0]
        if self.flat_items.size == 0 or count == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        fq = np.arange(count, dtype=np.int64)
        fn = np.zeros(count, dtype=np.int64)
        acc_q: list = []
        acc_n: list = []
        bmin, bmax, child, nlen = self.node_bmin, self.node_bmax, self.node_child, self.node_len
        while fq.size:
            hit = np.all((qmax[fq] >= bmin[fn]) & (qmin[fq] <= bmax[fn]), axis=1)
            fq = fq[hit]
            fn = fn[hit]
            if fq.size == 0:
                break
            has = nlen[fn] > 0
            if has.any():
                acc_q.append(fq[has])
                acc_n.append(fn[has])
            kids = child[fn]
            rows, cols = np.nonzero(kids >= 0)
            fq = fq[rows]
            fn = kids[rows, cols].astype(np.int64)
        qi, items = self._expand(acc_q, acc_n)
        if qi.size == 0:
            return qi, items
        ok = (np.all((self.item_max[items] >= qmin[qi]) & (self.item_min[items] <= qmax[qi]), axis=1)
              & ((self.item_layer[items] & layer_mask) != 0))
        return qi[ok], items[ok]

    def _ray_pairs(self, origins, inv_dir, max_t):
        fq = np.arange(origins.shape[0], dtype=np.int64)
        fn = np.zeros(origins.shape[0], dtype=np.int64)
        acc_q: list = []
        acc_n: list = []
        bmin, bmax, child, nlen = self.node_bmin, self.node_bmax, self.node_child, self.node_len
        while fq.size:
            oq = origins[fq]
            iq = inv_dir[fq]
            t1 = (bmin[fn] - oq) * iq
            t2 = (bmax[fn] - oq) * iq
            t_near = np.minimum(t1, t2).max(axis=1)
            t_far = np.maximum(t1, t2).min(axis=1)
            hit = (t_far >= np.maximum(t_near, 0.0)) & (t_near <= max_t[fq])
            fq = fq[hit]
            fn = fn[hit]
            if fq.size == 0:
                break
            has = nlen[fn] > 0
            if has.any():
                acc_q.append(fq[has])
                acc_n.append(fn[has])
            kids = child[fn]
            rows, cols = np.nonzero(kids >= 0)
            fq = fq[rows]
            fn = kids[rows, cols].astype(np.int64)
        return self._expand(acc_q, acc_n)

    def raycast(self, origins, directions, max_dist, layer_mask: int = C.LAYER_BULLET):
        """
        Batched nearest-hit ray cast against the indexed geometry.

        Parameters are ``(R, 3)`` origins, ``(R, 3)`` unit directions and a
        scalar or ``(R,)`` maximum distance.  Returns ``(t, item, normal)``:
        ``t`` is the hit distance (``max_dist`` when nothing was hit), ``item``
        the item id or -1, ``normal`` the unit surface normal facing the ray.
        Only the octants pierced by each ray are visited.
        """
        self._ensure_baked()
        origins = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        directions = np.atleast_2d(np.asarray(directions, dtype=np.float64))
        count = origins.shape[0]
        best_t = np.array(np.broadcast_to(np.asarray(max_dist, dtype=np.float64), (count,)), dtype=np.float64)
        best_item = np.full(count, -1, dtype=np.int64)
        normal = np.zeros((count, 3), dtype=np.float64)
        if self.flat_items.size == 0 or count == 0:
            return best_t, best_item, normal
        inv_dir = M.safe_inverse(directions)
        qi, items = self._ray_pairs(origins, inv_dir, best_t)
        if qi.size == 0:
            return best_t, best_item, normal
        keep = (self.item_layer[items] & layer_mask) != 0
        qi = qi[keep]
        items = items[keep]
        if qi.size == 0:
            return best_t, best_item, normal

        pair_t = np.full(qi.size, np.inf, dtype=np.float64)
        pair_axis = np.zeros(qi.size, dtype=np.int64)
        pair_inside = np.zeros(qi.size, dtype=bool)
        kinds = self.item_kind[items]

        box = kinds == self.KIND_BOX
        if box.any():
            q = qi[box]
            it = items[box]
            t1 = (self.item_min[it] - origins[q]) * inv_dir[q]
            t2 = (self.item_max[it] - origins[q]) * inv_dir[q]
            t_low = np.minimum(t1, t2)
            t_near = t_low.max(axis=1)
            t_far = np.maximum(t1, t2).min(axis=1)
            hit = (t_far >= np.maximum(t_near, 0.0)) & (t_near <= best_t[q])
            pair_t[box] = np.where(hit, np.maximum(t_near, 0.0), np.inf)
            pair_axis[box] = t_low.argmax(axis=1)
            pair_inside[box] = t_near < 0.0

        tri = ~box
        if tri.any():
            q = qi[tri]
            ref = self.item_ref[items[tri]]
            t, hit = M.ray_triangles(origins[q], directions[q], self.tri_v0[ref], self.tri_e1[ref], self.tri_e2[ref])
            hit &= t <= best_t[q]
            pair_t[tri] = np.where(hit, t, np.inf)

        valid = np.isfinite(pair_t)
        if not valid.any():
            return best_t, best_item, normal
        idx = np.flatnonzero(valid)
        order = idx[np.lexsort((pair_t[idx], qi[idx]))]
        first = np.ones(order.size, dtype=bool)
        first[1:] = qi[order[1:]] != qi[order[:-1]]
        win = order[first]
        rays = qi[win]
        best_t[rays] = pair_t[win]
        best_item[rays] = items[win]

        win_box = kinds[win] == self.KIND_BOX
        if win_box.any():
            rb = rays[win_box]
            wb = win[win_box]
            axis = pair_axis[wb]
            nb = np.zeros((rb.size, 3), dtype=np.float64)
            nb[np.arange(rb.size), axis] = -np.sign(directions[rb, axis])
            inside = pair_inside[wb]
            nb[inside] = -directions[rb[inside]]
            normal[rb] = nb
        if (~win_box).any():
            rt = rays[~win_box]
            nt = self.tri_normal[self.item_ref[items[win[~win_box]]]]
            facing = M.dot(nt, directions[rt]) > 0.0
            normal[rt] = np.where(facing[:, None], -nt, nt)
        return best_t, best_item, normal

    def any_hit(self, origins, directions, max_dist, layer_mask: int = C.LAYER_BULLET) -> np.ndarray:
        """Boolean occlusion test (line of sight) for a batch of ray segments."""
        t, item, _ = self.raycast(origins, directions, max_dist, layer_mask)
        return item >= 0

    # ------------------------------------------------------------------ #
    #  reference implementations and diagnostics
    # ------------------------------------------------------------------ #
    def query_aabb_bruteforce(self, qmin, qmax, layer_mask: int = C.LAYER_ALL):
        """O(Q * N) reference used by the test-suite to validate the tree."""
        qmin = np.atleast_2d(np.asarray(qmin, dtype=np.float64))
        qmax = np.atleast_2d(np.asarray(qmax, dtype=np.float64))
        n = self.item_count
        ok = (np.all((self.item_max[None, :n] >= qmin[:, None]) & (self.item_min[None, :n] <= qmax[:, None]), axis=2)
              & self.item_alive[None, :n] & ((self.item_layer[None, :n] & layer_mask) != 0))
        qi, items = np.nonzero(ok)
        return qi.astype(np.int64), items.astype(np.int64)

    def raycast_bruteforce(self, origins, directions, max_dist, layer_mask: int = C.LAYER_BULLET):
        """O(R * N) reference ray cast (distance only)."""
        origins = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        directions = np.atleast_2d(np.asarray(directions, dtype=np.float64))
        count = origins.shape[0]
        best = np.array(np.broadcast_to(np.asarray(max_dist, dtype=np.float64), (count,)), dtype=np.float64)
        n = self.item_count
        if n == 0:
            return best
        inv_dir = M.safe_inverse(directions)
        usable = self.item_alive[:n] & ((self.item_layer[:n] & layer_mask) != 0)
        boxes = np.flatnonzero(usable & (self.item_kind[:n] == self.KIND_BOX))
        if boxes.size:
            t_near, t_far = M.ray_aabb_slab(origins[:, None, :], inv_dir[:, None, :],
                                            self.item_min[None, boxes], self.item_max[None, boxes])
            hit = t_far >= np.maximum(t_near, 0.0)
            t = np.where(hit, np.maximum(t_near, 0.0), np.inf).min(axis=1)
            best = np.minimum(best, t)
        tris = np.flatnonzero(usable & (self.item_kind[:n] == self.KIND_TRIANGLE))
        if tris.size:
            ref = self.item_ref[tris]
            t, hit = M.ray_triangles(origins[:, None, :], directions[:, None, :],
                                     self.tri_v0[None, ref], self.tri_e1[None, ref], self.tri_e2[None, ref])
            best = np.minimum(best, np.where(hit, t, np.inf).min(axis=1))
        return best

    def stats(self) -> dict:
        self._ensure_baked()
        depths = self.node_depth[: self.node_count]
        return {
            "items": int(np.count_nonzero(self.item_alive[: self.item_count])),
            "nodes": int(self.node_count),
            "max_depth": int(depths.max()) if self.node_count else 0,
            "max_items_per_node": int(self.node_len.max()) if self.node_count else 0,
            "mean_items_per_filled_node": float(self.node_len[self.node_len > 0].mean()) if np.any(self.node_len > 0) else 0.0,
            "smallest_voxel": float(self.root_half * 2.0 / (2 ** self.max_depth)),
        }


# =========================================================================== #
#  2. COLLISION WORLD
# =========================================================================== #
class CollisionWorld:
    """
    Level geometry indexed by an :class:`Octree`.

    ``box_min`` / ``box_max`` are the solid axis aligned primitives (layer bits
    choose whether they stop players, bullets or both); ``triangles`` is an
    optional mesh soup that only stops bullets (ramps collide with players
    through stair-step boxes and with bullets through their true surface).
    """

    def __init__(self, box_min, box_max, box_layer=None, triangles=None, max_depth: int = 6,
                 leaf_capacity: int = 8):
        box_min = np.asarray(box_min, dtype=np.float64).reshape(-1, 3)
        box_max = np.asarray(box_max, dtype=np.float64).reshape(-1, 3)
        if box_layer is None:
            box_layer = np.full(box_min.shape[0], C.LAYER_ALL, dtype=np.uint8)
        box_layer = np.asarray(box_layer, dtype=np.uint8)
        if box_min.shape[0]:
            lo = box_min.min(axis=0)
            hi = box_max.max(axis=0)
        else:
            lo = np.full(3, -1.0)
            hi = np.full(3, 1.0)
        if triangles is not None and len(triangles):
            tri = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
            lo = np.minimum(lo, tri.reshape(-1, 3).min(axis=0))
            hi = np.maximum(hi, tri.reshape(-1, 3).max(axis=0))
        else:
            tri = np.zeros((0, 3, 3), dtype=np.float64)
        self.bounds_min = lo
        self.bounds_max = hi
        self.octree = Octree(lo, hi, max_depth=max_depth, leaf_capacity=leaf_capacity)
        self.box_items = (self.octree.add_boxes(box_min, box_max, box_layer)
                          if box_min.shape[0] else np.zeros(0, dtype=np.int64))
        self.tri_items = (self.octree.add_triangles(tri, C.LAYER_BULLET)
                          if tri.shape[0] else np.zeros(0, dtype=np.int64))
        self.octree.build()
        self._solid_version = -1
        self.solid_min = np.zeros((1, 3))
        self.solid_max = np.zeros((1, 3))
        self.sentinel = 0
        self._refresh()

    @property
    def version(self) -> int:
        return self.octree.version

    def _refresh(self) -> None:
        """Rebuild the padded gather tables (one sentinel row far away from everything)."""
        n = self.octree.item_count
        self.solid_min = np.vstack([self.octree.item_min[:n], np.full((1, 3), _FAR)])
        self.solid_max = np.vstack([self.octree.item_max[:n], np.full((1, 3), _FAR)])
        self.sentinel = n
        self._solid_version = self.octree.version

    def query_player_boxes(self, qmin, qmax):
        """(query, item) pairs of player-blocking boxes intersecting the query boxes."""
        if self._solid_version != self.octree.version:
            self._refresh()
        return self.octree.query_aabb(qmin, qmax, C.LAYER_PLAYER)

    def tables(self):
        if self._solid_version != self.octree.version:
            self._refresh()
        return self.solid_min, self.solid_max, self.sentinel

    def raycast(self, origins, directions, max_dist, layer_mask: int = C.LAYER_BULLET):
        return self.octree.raycast(origins, directions, max_dist, layer_mask)

    def line_of_sight(self, start, end, layer_mask: int = C.LAYER_BULLET) -> np.ndarray:
        """True where the segment ``start -> end`` is unobstructed (batched)."""
        start = np.atleast_2d(np.asarray(start, dtype=np.float64))
        end = np.atleast_2d(np.asarray(end, dtype=np.float64))
        delta = end - start
        dist = M.length(delta)
        direction = delta / np.maximum(dist, 1.0e-9)[:, None]
        _, item, _ = self.octree.raycast(start, direction, np.maximum(dist - 1.0e-4, 0.0), layer_mask)
        return item < 0

    def add_dynamic_box(self, box_min, box_max, layer=C.LAYER_ALL) -> int:
        item = self.octree.insert_box(box_min, box_max, layer)
        self._refresh()
        return item

    def remove_dynamic_box(self, item: int) -> None:
        self.octree.remove(item)
        self._refresh()


# =========================================================================== #
#  3. KINEMATIC FIRST-PERSON CONTROLLER (structure of arrays)
# =========================================================================== #
class KinematicState:
    """
    Movement state of ``capacity`` bodies, one contiguous array per component.
    ``pos`` is the feet position.  The object also owns the per-row broad-phase
    candidate cache used by :func:`step_kinematics`.
    """

    SCALAR_FIELDS = ("slide_t", "slide_cd", "crouch_t", "air_t", "jump_buf")

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        n = self.capacity
        self.pos = np.zeros((n, 3), dtype=np.float64)
        self.vel = np.zeros((n, 3), dtype=np.float64)
        self.mflags = np.zeros(n, dtype=np.uint16)
        self.prev_buttons = np.zeros(n, dtype=np.uint16)
        self.slide_t = np.zeros(n, dtype=np.float64)
        self.slide_cd = np.zeros(n, dtype=np.float64)
        self.crouch_t = np.zeros(n, dtype=np.float64)
        self.air_t = np.zeros(n, dtype=np.float64)
        self.jump_buf = np.zeros(n, dtype=np.float64)
        self.slide_dir = np.zeros((n, 2), dtype=np.float64)
        self.fall_speed = np.zeros(n, dtype=np.float64)     # impact speed of the latest landing (FX only)
        # broad-phase cache
        self._cache_min = np.full((n, 3), np.inf, dtype=np.float64)
        self._cache_max = np.full((n, 3), -np.inf, dtype=np.float64)
        self._cache_idx = np.zeros((n, 16), dtype=np.int64)
        self._cache_len = np.zeros(n, dtype=np.int64)
        self._cache_version = -1
        self.broadphase_queries = 0

    # -- row utilities --------------------------------------------------------
    def reset_rows(self, rows, position, grounded: bool = True) -> None:
        rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))
        self.pos[rows] = position
        self.vel[rows] = 0.0
        self.mflags[rows] = C.MF_ON_GROUND if grounded else 0
        self.prev_buttons[rows] = 0
        for name in self.SCALAR_FIELDS:
            getattr(self, name)[rows] = 0.0
        self.slide_dir[rows] = 0.0
        self.fall_speed[rows] = 0.0
        self._cache_min[rows] = np.inf
        self._cache_max[rows] = -np.inf

    def save_row(self, row: int) -> tuple:
        return (self.pos[row].copy(), self.vel[row].copy(), int(self.mflags[row]), int(self.prev_buttons[row]),
                float(self.slide_t[row]), float(self.slide_cd[row]), float(self.crouch_t[row]),
                float(self.air_t[row]), float(self.jump_buf[row]), self.slide_dir[row].copy())

    def load_row(self, row: int, saved: tuple) -> None:
        (self.pos[row], self.vel[row], self.mflags[row], self.prev_buttons[row], self.slide_t[row],
         self.slide_cd[row], self.crouch_t[row], self.air_t[row], self.jump_buf[row], self.slide_dir[row]) = saved

    # -- broad phase ----------------------------------------------------------
    def candidates(self, rows: np.ndarray, need_min: np.ndarray, need_max: np.ndarray, world: CollisionWorld):
        """
        Padded ``(n, K, 3)`` min / max tables of the boxes each row may touch
        this tick.  The octree is only consulted for rows whose required region
        left the inflated region cached by an earlier query.
        """
        solid_min, solid_max, sentinel = world.tables()
        if self._cache_version != world.version:
            self._cache_min[:] = np.inf
            self._cache_max[:] = -np.inf
            self._cache_version = world.version
        stale = (np.any(need_min < self._cache_min[rows], axis=1)
                 | np.any(need_max > self._cache_max[rows], axis=1))
        if stale.any():
            srows = rows[stale]
            qmin = need_min[stale] - C.CANDIDATE_MARGIN
            qmax = need_max[stale] + C.CANDIDATE_MARGIN
            qi, items = world.query_player_boxes(qmin, qmax)
            self.broadphase_queries += 1
            counts = np.bincount(qi, minlength=srows.size)
            widest = int(counts.max()) if counts.size else 0
            if widest + 1 > self._cache_idx.shape[1]:
                grown = np.full((self.capacity, max(widest + 1, self._cache_idx.shape[1] * 2)), sentinel, dtype=np.int64)
                keep_rows = np.flatnonzero(np.isfinite(self._cache_min[:, 0]))
                grown[keep_rows, : self._cache_idx.shape[1]] = self._cache_idx[keep_rows]
                for r in keep_rows.tolist():
                    grown[r, self._cache_len[r]:] = sentinel
                self._cache_idx = grown
            self._cache_idx[srows] = sentinel
            if qi.size:
                order = np.argsort(qi, kind="stable")
                qs = qi[order]
                col = np.arange(qs.size, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
                self._cache_idx[srows[qs], col] = items[order]
            self._cache_len[srows] = counts
            self._cache_min[srows] = qmin
            self._cache_max[srows] = qmax
        width = max(int(self._cache_len[rows].max()), 1)
        idx = self._cache_idx[rows, :width]
        return solid_min[idx], solid_max[idx]


def eye_height(crouch_t):
    """
    Camera height above the feet.  ``crouch_t`` in [0, 1] is the linear crouch
    progress; the height follows an ease-in-out cosine and bottoms out 40 %
    below the standing eye level.
    """
    blend = M.ease_in_out_cos(crouch_t)
    return C.EYE_HEIGHT_STAND - (C.EYE_HEIGHT_STAND - C.EYE_HEIGHT_CROUCH) * blend


def _sweep_axis(pos, lo, hi, axis: int, delta, cmin, cmax):
    """
    Exact one dimensional sweep of ``n`` boxes ``[pos + lo, pos + hi]`` along
    ``axis`` by ``delta`` against padded candidate boxes ``(n, K, 3)``.

    A candidate blocks when the two boxes overlap on the other two axes and
    the leading face crosses the candidate's near face during this move, so
    arbitrarily fast bodies cannot tunnel and bodies that are already inside
    geometry are free to leave.  Returns ``(new_coordinate, hit_mask)``.
    """
    eps = C.COLLISION_EPS
    a1, a2 = (axis + 1) % 3, (axis + 2) % 3
    p1_lo = (pos[:, a1] + lo[:, a1])[:, None]
    p1_hi = (pos[:, a1] + hi[:, a1])[:, None]
    p2_lo = (pos[:, a2] + lo[:, a2])[:, None]
    p2_hi = (pos[:, a2] + hi[:, a2])[:, None]
    overlap = ((p1_lo < cmax[:, :, a1] - eps) & (p1_hi > cmin[:, :, a1] + eps)
               & (p2_lo < cmax[:, :, a2] - eps) & (p2_hi > cmin[:, :, a2] + eps))
    forward = delta > 0.0
    backward = delta < 0.0
    lead_offset = np.where(forward, hi[:, axis], lo[:, axis])
    lead0 = pos[:, axis] + lead_offset
    lead1 = lead0 + delta
    fwd2 = forward[:, None]
    face = np.where(fwd2, cmin[:, :, axis], cmax[:, :, axis])
    l0 = lead0[:, None]
    l1 = lead1[:, None]
    crossing = np.where(fwd2, (l0 <= face + eps) & (l1 > face), (l0 >= face - eps) & (l1 < face))
    crossing &= overlap & (forward | backward)[:, None]
    limit_fwd = np.where(crossing, face, _FAR).min(axis=1)
    limit_bwd = np.where(crossing, face, -_FAR).max(axis=1)
    hit = crossing.any(axis=1)
    lead = np.where(hit, np.where(forward, limit_fwd, limit_bwd), lead1)
    return lead - lead_offset, hit


def _blocked(pos, lo, hi, cmin, cmax):
    """True for rows whose box ``[pos + lo, pos + hi]`` penetrates any candidate."""
    eps = C.COLLISION_EPS
    pmin = (pos + lo)[:, None, :]
    pmax = (pos + hi)[:, None, :]
    return np.any(np.all((pmin < cmax - eps) & (pmax > cmin + eps), axis=2), axis=1)


def step_kinematics(ks: KinematicState, rows, buttons, yaw, world: CollisionWorld,
                    dt: float = C.TICK_DT, speed_mul=None) -> None:
    """
    Advance the selected ``rows`` of ``ks`` by one fixed tick.

    ``buttons`` (uint16 bit masks) and ``yaw`` (radians) are per-row inputs for
    this tick; ``speed_mul`` scales the ground speed (weapon mobility).  All
    branches are evaluated with boolean masks, so the cost is independent of
    which rows are sprinting, sliding or airborne.
    """
    rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))
    n = rows.size
    if n == 0:
        return
    buttons = np.broadcast_to(np.asarray(buttons, dtype=np.uint16), (n,))
    yaw = np.broadcast_to(np.asarray(yaw, dtype=np.float64), (n,))
    speed_mul = np.ones(n, dtype=np.float64) if speed_mul is None else np.broadcast_to(
        np.asarray(speed_mul, dtype=np.float64), (n,))

    pos = ks.pos[rows]
    vel = ks.vel[rows]
    vx = vel[:, 0].copy()
    vy = vel[:, 1].copy()
    vz = vel[:, 2].copy()
    flags = ks.mflags[rows]
    prev = ks.prev_buttons[rows]
    slide_t = ks.slide_t[rows]
    slide_cd = ks.slide_cd[rows]
    crouch_t = ks.crouch_t[rows]
    air_t = ks.air_t[rows]
    jump_buf = ks.jump_buf[rows]
    sdx = ks.slide_dir[rows, 0]
    sdz = ks.slide_dir[rows, 1]

    was_ground = (flags & C.MF_ON_GROUND) != 0
    crouched = (flags & C.MF_CROUCHED) != 0
    sliding = (flags & C.MF_SLIDING) != 0

    # ---- decode inputs ------------------------------------------------------
    fwd_in = ((buttons & C.BTN_FORWARD) != 0).astype(np.float64) - ((buttons & C.BTN_BACK) != 0)
    right_in = ((buttons & C.BTN_RIGHT) != 0).astype(np.float64) - ((buttons & C.BTN_LEFT) != 0)
    jump_down = (buttons & C.BTN_JUMP) != 0
    crouch_down = (buttons & C.BTN_SLIDE) != 0
    sprint_down = (buttons & C.BTN_SPRINT) != 0
    ads = (buttons & C.BTN_ADS) != 0
    shoot_down = (buttons & C.BTN_SHOOT) != 0
    jump_pressed = jump_down & ((prev & C.BTN_JUMP) == 0)
    crouch_pressed = crouch_down & ((prev & C.BTN_SLIDE) == 0)

    sin_y = np.sin(yaw)
    cos_y = np.cos(yaw)
    wx = sin_y * fwd_in + cos_y * right_in
    wz = cos_y * fwd_in - sin_y * right_in
    wish_len = np.sqrt(wx * wx + wz * wz)
    has_wish = wish_len > 1.0e-9
    inv_len = np.where(has_wish, 1.0 / np.maximum(wish_len, 1.0e-9), 0.0)
    wx = wx * inv_len
    wz = wz * inv_len

    # ---- broad phase (uses the pre-move state and a conservative reach) ------
    hspeed = np.sqrt(vx * vx + vz * vz)
    reach_h = (np.maximum(hspeed, C.SLIDE_SPEED_CAP) + 2.0) * dt + 0.05
    reach_v = np.maximum(np.abs(vy), C.JUMP_SPEED) * dt + C.GRAVITY * dt * dt + C.STEP_HEIGHT + 0.05
    radius = C.PLAYER_RADIUS
    need_min = pos + np.stack([-radius - reach_h, -reach_v, -radius - reach_h], axis=1)
    need_max = pos + np.stack([radius + reach_h, C.PLAYER_HEIGHT_STAND + reach_v, radius + reach_h], axis=1)
    cmin, cmax = ks.candidates(rows, need_min, need_max, world)

    # ---- timers ---------------------------------------------------------------
    slide_cd = np.maximum(slide_cd - dt, 0.0)
    jump_buf = np.where(jump_pressed, C.JUMP_BUFFER_TIME, np.maximum(jump_buf - dt, 0.0))
    air_t = np.where(was_ground, 0.0, air_t + dt)

    # ---- slide entry ----------------------------------------------------------
    start_slide = crouch_pressed & was_ground & ~sliding & (hspeed >= C.SLIDE_MIN_SPEED) & (slide_cd <= 0.0)
    inv_h = 1.0 / np.maximum(hspeed, 1.0e-9)
    sdx = np.where(start_slide, vx * inv_h, sdx)
    sdz = np.where(start_slide, vz * inv_h, sdz)
    launch = np.maximum(hspeed, np.minimum(np.maximum(hspeed + C.SLIDE_IMPULSE, C.SLIDE_START_SPEED), C.SLIDE_SPEED_CAP))
    vx = np.where(start_slide, sdx * launch, vx)
    vz = np.where(start_slide, sdz * launch, vz)
    sliding = sliding | start_slide
    slide_t = np.where(start_slide, 0.0, slide_t)

    # ---- jump (coyote time + buffered press) ----------------------------------
    do_jump = (was_ground | (air_t <= C.COYOTE_TIME)) & (jump_buf > 0.0)
    vy = np.where(do_jump, C.JUMP_SPEED, vy)
    grounded = was_ground & ~do_jump
    air_t = np.where(do_jump, C.COYOTE_TIME + 1.0, air_t)
    jump_buf = np.where(do_jump, 0.0, jump_buf)

    # ---- slide dynamics: locked heading, V_next = V * exp(-k dt) ---------------
    slide_speed = np.maximum(vx * sdx + vz * sdz, 0.0) * np.exp(-C.SLIDE_FRICTION_K * dt)
    slide_active = sliding & grounded
    vx = np.where(slide_active, sdx * slide_speed, vx)
    vz = np.where(slide_active, sdz * slide_speed, vz)
    slide_t = np.where(slide_active, slide_t + dt, slide_t)
    end_slide = sliding & (do_jump | ~was_ground | (slide_speed < C.SLIDE_END_SPEED)
                           | (slide_t >= C.SLIDE_MAX_TIME)
                           | (~crouch_down & (slide_t >= C.SLIDE_MIN_TIME)))
    slide_cd = np.where(end_slide, C.SLIDE_COOLDOWN, slide_cd)
    sliding = sliding & ~end_slide

    # ---- crouch (standing up needs head room) ----------------------------------
    lo = np.empty((n, 3), dtype=np.float64)
    hi = np.empty((n, 3), dtype=np.float64)
    lo[:, 0] = -radius
    lo[:, 1] = 0.0
    lo[:, 2] = -radius
    hi[:, 0] = radius
    hi[:, 1] = C.PLAYER_HEIGHT_STAND
    hi[:, 2] = radius
    want_crouch = crouch_down | sliding
    needs_room = crouched & ~want_crouch
    can_stand = np.ones(n, dtype=bool)
    if needs_room.any():
        can_stand[needs_room] = ~_blocked(pos[needs_room], lo[needs_room], hi[needs_room],
                                          cmin[needs_room], cmax[needs_room])
    crouched = want_crouch | (crouched & ~can_stand)
    hi[:, 1] = np.where(crouched, C.PLAYER_HEIGHT_CROUCH, C.PLAYER_HEIGHT_STAND)
    crouch_target = crouched.astype(np.float64)
    crouch_t = crouch_t + np.clip(crouch_target - crouch_t, -dt / C.CROUCH_TRANSITION_TIME,
                                  dt / C.CROUCH_TRANSITION_TIME)

    # ---- ground / air locomotion ------------------------------------------------
    sprinting = sprint_down & (fwd_in > 0.0) & grounded & ~crouched & ~sliding & ~ads & ~shoot_down
    wish_speed = (np.where(crouched, C.CROUCH_SPEED, np.where(sprinting, C.SPRINT_SPEED, C.WALK_SPEED))
                  * speed_mul * np.where(ads, C.ADS_SPEED_SCALE, 1.0))
    walking = grounded & ~sliding
    speed = np.sqrt(vx * vx + vz * vz)
    drop = np.maximum(speed, C.STOP_SPEED) * C.GROUND_FRICTION * dt
    keep = np.where(walking & (speed > 1.0e-9), np.maximum(speed - drop, 0.0) / np.maximum(speed, 1.0e-9), 1.0)
    vx = vx * keep
    vz = vz * keep
    speed_before = np.sqrt(vx * vx + vz * vz)
    current = vx * wx + vz * wz
    accel = np.where(walking, C.GROUND_ACCEL, C.AIR_ACCEL)
    gain = np.minimum(accel * dt * wish_speed, np.maximum(wish_speed - current, 0.0))
    gain = np.where(has_wish & ~sliding, gain, 0.0)
    vx = vx + wx * gain
    vz = vz + wz * gain
    speed_after = np.sqrt(vx * vx + vz * vz)
    air_limit = np.maximum(speed_before, wish_speed)
    trim = np.where(~grounded & (speed_after > air_limit), air_limit / np.maximum(speed_after, 1.0e-9), 1.0)
    vx = vx * trim
    vz = vz * trim
    speed_after = np.sqrt(vx * vx + vz * vz)
    hard = np.where(speed_after > C.MAX_HORIZONTAL_SPEED, C.MAX_HORIZONTAL_SPEED / np.maximum(speed_after, 1.0e-9), 1.0)
    vx = vx * hard
    vz = vz * hard
    vy = np.maximum(vy - C.GRAVITY * dt, -C.TERMINAL_VELOCITY)

    # ---- horizontal move with stair stepping --------------------------------------
    start = pos.copy()
    dx = vx * dt
    dz = vz * dt
    pos[:, 0], hit_x = _sweep_axis(pos, lo, hi, 0, dx, cmin, cmax)
    pos[:, 2], hit_z = _sweep_axis(pos, lo, hi, 2, dz, cmin, cmax)
    stepped = np.zeros(n, dtype=bool)
    try_step = (hit_x | hit_z) & grounded
    if try_step.any():
        s = np.flatnonzero(try_step)
        sp = start[s].copy()
        slo, shi, scmin, scmax = lo[s], hi[s], cmin[s], cmax[s]
        base_y = sp[:, 1].copy()
        sp[:, 1], _ = _sweep_axis(sp, slo, shi, 1, np.full(s.size, C.STEP_HEIGHT), scmin, scmax)
        raised = sp[:, 1] - base_y
        sp[:, 0], s_hit_x = _sweep_axis(sp, slo, shi, 0, dx[s], scmin, scmax)
        sp[:, 2], s_hit_z = _sweep_axis(sp, slo, shi, 2, dz[s], scmin, scmax)
        sp[:, 1], supported = _sweep_axis(sp, slo, shi, 1, -raised, scmin, scmax)
        plain = (pos[s, 0] - start[s, 0]) ** 2 + (pos[s, 2] - start[s, 2]) ** 2
        climbed = (sp[:, 0] - start[s, 0]) ** 2 + (sp[:, 2] - start[s, 2]) ** 2
        accept = supported & (raised > 1.0e-6) & (climbed > plain + 1.0e-10)
        if accept.any():
            a = s[accept]
            pos[a] = sp[accept]
            hit_x[a] = s_hit_x[accept]
            hit_z[a] = s_hit_z[accept]
            stepped[a] = True
    vx = np.where(hit_x, 0.0, vx)
    vz = np.where(hit_z, 0.0, vz)

    # ---- vertical move ---------------------------------------------------------------
    impact = np.maximum(-vy, 0.0)
    dy = vy * dt
    pos[:, 1], hit_y = _sweep_axis(pos, lo, hi, 1, dy, cmin, cmax)
    on_ground = hit_y & (dy < 0.0)
    vy = np.where(hit_y, 0.0, vy)
    snap = grounded & ~on_ground & (vy <= 0.0)
    if snap.any():
        s = np.flatnonzero(snap)
        sp = pos[s].copy()
        sp[:, 1], found = _sweep_axis(sp, lo[s], hi[s], 1, np.full(s.size, -C.STEP_HEIGHT), cmin[s], cmax[s])
        if found.any():
            f = s[found]
            pos[f, 1] = sp[found, 1]
            on_ground[f] = True
            vy[f] = 0.0
    landed = on_ground & ~was_ground

    # ---- write back ----------------------------------------------------------------------
    new_flags = (np.where(on_ground, C.MF_ON_GROUND, 0) | np.where(crouched, C.MF_CROUCHED, 0)
                 | np.where(sliding, C.MF_SLIDING, 0) | np.where(sprinting, C.MF_SPRINTING, 0)
                 | np.where(ads, C.MF_ADS, 0) | np.where(do_jump, C.MF_JUMPED, 0)
                 | np.where(landed, C.MF_LANDED, 0) | np.where(stepped, C.MF_STEPPED, 0))
    ks.pos[rows] = pos
    ks.vel[rows, 0] = vx
    ks.vel[rows, 1] = vy
    ks.vel[rows, 2] = vz
    ks.mflags[rows] = new_flags.astype(np.uint16)
    ks.prev_buttons[rows] = buttons
    ks.slide_t[rows] = slide_t
    ks.slide_cd[rows] = slide_cd
    ks.crouch_t[rows] = crouch_t
    ks.air_t[rows] = air_t
    ks.jump_buf[rows] = jump_buf
    ks.slide_dir[rows, 0] = sdx
    ks.slide_dir[rows, 1] = sdz
    ks.fall_speed[rows] = np.where(landed, impact, ks.fall_speed[rows])


# =========================================================================== #
#  4. ANATOMICAL HITBOXES
# =========================================================================== #
HITBOX_NAMES = ("head", "chest", "abdomen", "arm_l", "arm_r", "leg_l", "leg_r")
HITBOX_COUNT = len(HITBOX_NAMES)
HITBOX_ZONE = np.array([C.ZONE_HEAD, C.ZONE_TORSO, C.ZONE_TORSO, C.ZONE_LIMB, C.ZONE_LIMB,
                        C.ZONE_LIMB, C.ZONE_LIMB], dtype=np.int64)
HITBOX_MULTIPLIER = C.ZONE_MULTIPLIER[HITBOX_ZONE]

# Local frame: origin at the feet, +X right, +Y up, +Z facing direction.
HITBOX_STAND_CENTER = np.array([
    [0.00, 1.62, 0.02],     # head
    [0.00, 1.22, 0.00],     # chest
    [0.00, 0.84, 0.00],     # abdomen
    [-0.33, 1.14, 0.00],    # left arm
    [0.33, 1.14, 0.00],     # right arm
    [-0.12, 0.35, 0.00],    # left leg
    [0.12, 0.35, 0.00],     # right leg
], dtype=np.float64)
HITBOX_STAND_HALF = np.array([
    [0.13, 0.14, 0.14],
    [0.24, 0.24, 0.15],
    [0.21, 0.14, 0.14],
    [0.08, 0.32, 0.09],
    [0.08, 0.32, 0.09],
    [0.10, 0.35, 0.11],
    [0.10, 0.35, 0.11],
], dtype=np.float64)
HITBOX_CROUCH_CENTER = np.array([
    [0.00, 0.97, 0.06],
    [0.00, 0.66, 0.03],
    [0.00, 0.38, 0.00],
    [-0.33, 0.62, 0.04],
    [0.33, 0.62, 0.04],
    [-0.12, 0.16, 0.12],
    [0.12, 0.16, 0.12],
], dtype=np.float64)
HITBOX_CROUCH_HALF = np.array([
    [0.13, 0.14, 0.14],
    [0.24, 0.20, 0.17],
    [0.21, 0.10, 0.16],
    [0.08, 0.24, 0.10],
    [0.08, 0.24, 0.10],
    [0.10, 0.16, 0.26],
    [0.10, 0.16, 0.26],
], dtype=np.float64)
HITBOX_BOUND_RADIUS = 1.25          # sphere around the body centre used as a ray broad phase


class HitboxSet:
    """
    Critical-zone box groups for ``capacity`` entities.

    Each entity owns ``HITBOX_COUNT`` oriented boxes expressed in its local
    frame (``local_center`` / ``local_half``).  ``update`` re-derives them from
    the root movement state every tick -- translation, heading and the crouch
    blend -- so the group always moves rigidly with the entity.  The arrays
    are plain attributes on purpose: the lag compensator swaps historical
    copies in and out of them around a rewound shot.
    """

    def __init__(self, capacity: int):
        n = int(capacity)
        self.capacity = n
        self.root = np.tile(np.eye(4), (n, 1, 1))
        self.root_inv = np.tile(np.eye(4), (n, 1, 1))
        self.local_center = np.tile(HITBOX_STAND_CENTER, (n, 1, 1))
        self.local_half = np.tile(HITBOX_STAND_HALF, (n, 1, 1))
        self.world_center = np.zeros((n, HITBOX_COUNT, 3), dtype=np.float64)
        self.bound_center = np.zeros((n, 3), dtype=np.float64)
        self.valid = np.zeros(n, dtype=bool)

    def update(self, rows, pos, yaw, crouch_t, alive) -> None:
        """Deterministically rebuild the boxes of ``rows`` from their root state."""
        rows = np.atleast_1d(np.asarray(rows, dtype=np.intp))
        if rows.size == 0:
            return
        root = M.rigid_from_pos_yaw(pos, yaw)
        blend = M.ease_in_out_cos(np.asarray(crouch_t, dtype=np.float64))[:, None, None]
        center = HITBOX_STAND_CENTER[None] + (HITBOX_CROUCH_CENTER - HITBOX_STAND_CENTER)[None] * blend
        half = HITBOX_STAND_HALF[None] + (HITBOX_CROUCH_HALF - HITBOX_STAND_HALF)[None] * blend
        self.root[rows] = root
        self.root_inv[rows] = M.rigid_inverse(root)
        self.local_center[rows] = center
        self.local_half[rows] = half
        self.world_center[rows] = np.einsum("nij,nhj->nhi", root[:, :3, :3], center) + root[:, None, :3, 3]
        body = np.asarray(pos, dtype=np.float64).copy()
        body[:, 1] += 0.5 * (C.PLAYER_HEIGHT_STAND + (C.PLAYER_HEIGHT_CROUCH - C.PLAYER_HEIGHT_STAND) * blend[:, 0, 0])
        self.bound_center[rows] = body
        self.valid[rows] = alive

    def set_from_matrices(self, root, crouch_t, valid) -> None:
        """Overwrite every entity from externally supplied root matrices (rollback path)."""
        blend = M.ease_in_out_cos(np.asarray(crouch_t, dtype=np.float64))[:, None, None]
        self.root = np.asarray(root, dtype=np.float64).copy()
        self.root_inv = M.rigid_inverse(self.root)
        self.local_center = HITBOX_STAND_CENTER[None] + (HITBOX_CROUCH_CENTER - HITBOX_STAND_CENTER)[None] * blend
        self.local_half = HITBOX_STAND_HALF[None] + (HITBOX_CROUCH_HALF - HITBOX_STAND_HALF)[None] * blend
        self.world_center = (np.einsum("nij,nhj->nhi", self.root[:, :3, :3], self.local_center)
                             + self.root[:, None, :3, 3])
        body = self.root[:, :3, 3].copy()
        body[:, 1] += 0.5 * (C.PLAYER_HEIGHT_STAND + (C.PLAYER_HEIGHT_CROUCH - C.PLAYER_HEIGHT_STAND) * blend[:, 0, 0])
        self.bound_center = body
        self.valid = np.asarray(valid, dtype=bool).copy()

    def snapshot(self) -> tuple:
        """Reference copy of the live arrays (used to restore after a rewind)."""
        return (self.root, self.root_inv, self.local_center, self.local_half, self.world_center,
                self.bound_center, self.valid)

    def restore(self, saved: tuple) -> None:
        (self.root, self.root_inv, self.local_center, self.local_half, self.world_center,
         self.bound_center, self.valid) = saved

    def raycast(self, origins, directions, max_t, target_mask):
        """
        Batched ray test of ``R`` rays against every entity selected by
        ``target_mask``.  Returns ``(t, entity, box)`` with -1 for misses;
        ``t`` is never larger than the supplied ``max_t`` (pass the world hit
        distance to get occlusion for free).
        """
        origins = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        directions = np.atleast_2d(np.asarray(directions, dtype=np.float64))
        count = origins.shape[0]
        best_t = np.array(np.broadcast_to(np.asarray(max_t, dtype=np.float64), (count,)), dtype=np.float64)
        entity = np.full(count, -1, dtype=np.int64)
        box = np.full(count, -1, dtype=np.int64)
        cand = np.flatnonzero(np.asarray(target_mask, dtype=bool) & self.valid)
        if cand.size == 0 or count == 0:
            return best_t, entity, box
        # broad phase: closest approach of each ray to each body sphere
        oc = self.bound_center[cand][None, :, :] - origins[:, None, :]
        along = np.sum(oc * directions[:, None, :], axis=2)
        perp2 = np.sum(oc * oc, axis=2) - along * along
        near = ((perp2 <= HITBOX_BOUND_RADIUS ** 2) & (along > -HITBOX_BOUND_RADIUS)
                & (along - HITBOX_BOUND_RADIUS <= best_t[:, None]))
        column = near.any(axis=0)
        if not column.any():
            return best_t, entity, box
        cand = cand[column]
        rot = self.root_inv[cand, :3, :3]
        trans = self.root_inv[cand, :3, 3]
        o_local = np.einsum("cij,rj->rci", rot, origins) + trans[None]
        d_local = np.einsum("cij,rj->rci", rot, directions)
        inv_d = M.safe_inverse(d_local)
        bmin = (self.local_center[cand] - self.local_half[cand])[None]
        bmax = (self.local_center[cand] + self.local_half[cand])[None]
        t1 = (bmin - o_local[:, :, None, :]) * inv_d[:, :, None, :]
        t2 = (bmax - o_local[:, :, None, :]) * inv_d[:, :, None, :]
        t_near = np.minimum(t1, t2).max(axis=3)
        t_far = np.maximum(t1, t2).min(axis=3)
        hit = (t_far >= np.maximum(t_near, 0.0)) & (t_near <= best_t[:, None, None])
        t = np.where(hit, np.maximum(t_near, 0.0), np.inf).reshape(count, -1)
        k = t.argmin(axis=1)
        tk = t[np.arange(count), k]
        ok = np.isfinite(tk)
        entity[ok] = cand[k[ok] // HITBOX_COUNT]
        box[ok] = k[ok] % HITBOX_COUNT
        best_t[ok] = tk[ok]
        return best_t, entity, box


# =========================================================================== #
#  5. CAMERA RIG (client side view feel)
# =========================================================================== #
def spring_step(position, velocity, stiffness: float, damping: float, dt: float):
    """
    One step of the damped spring ``x'' = -k x - c x'`` with implicit (backward)
    Euler.  Unlike the explicit update it is stable for every ``dt``: a frame
    of 100 ms (a level load, a slow software renderer) can no longer make a
    stiff spring oscillate with growing amplitude.  Returns the new
    ``(position, velocity)``; works on scalars and NumPy arrays alike.
    """
    velocity = (velocity - dt * stiffness * position) / (1.0 + dt * damping + dt * dt * stiffness)
    return position + velocity * dt, velocity


class CameraRig:
    """
    Converts the predicted body state into a first-person camera pose and a
    view-model pose.  Purely visual: nothing in here feeds back into the
    simulation except the recoil aim offsets returned by :meth:`consume_aim_kick`.

    Roll is positive when the horizon tilts clockwise on screen.
    """

    def __init__(self, base_fov: float = C.BASE_FOV):
        self.base_fov = float(base_fov)
        self.fov = float(base_fov)
        self.time = 0.0
        self.bob_phase = 0.0
        self.bob_amount = 0.0
        self.roll = 0.0
        self.slide_side = -1.0
        self.land_offset = 0.0
        self.land_velocity = 0.0
        self.step_offset = 0.0
        self.sprint_blend = 0.0
        self.slide_blend = 0.0
        self.ads_blend = 0.0
        self.reload_blend = 0.0
        # recoil: aim offsets really move the view; punch is a cosmetic spring
        self.recoil_target = np.zeros(2)       # (pitch, yaw) radians still to be delivered / recovered
        self.recoil_current = np.zeros(2)
        self._aim_delta = np.zeros(2)
        self.punch = np.zeros(3)               # (pitch, yaw, roll) radians
        self.punch_velocity = np.zeros(3)
        self.kickback = 0.0                    # view-model recoil travel (world units)
        self.trauma = 0.0
        self.look_lag = np.zeros(2)
        # outputs
        self.cam_pos = np.zeros(3)
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_roll = 0.0
        self.vm_offset = np.zeros(3)
        self.vm_rotation = np.zeros(3)         # (pitch, yaw, roll) radians
        self.shake_uv = np.zeros(2)

    # ---- impulses -------------------------------------------------------------
    def add_recoil(self, pitch_rad: float, yaw_rad: float, punch: float = 1.0) -> None:
        """Queue a recoil kick (aim climbs by ``pitch_rad``) and a cosmetic punch."""
        self.recoil_target[0] += pitch_rad
        self.recoil_target[1] += yaw_rad
        self.punch_velocity[0] += pitch_rad * 9.0 * punch
        self.punch_velocity[2] += yaw_rad * 14.0 * punch
        self.kickback = min(self.kickback + 0.035 * punch + pitch_rad * 1.6, 0.16)

    def add_damage_punch(self, direction_sign: float, strength: float) -> None:
        self.punch_velocity[0] += 0.9 * strength
        self.punch_velocity[2] += 1.6 * strength * direction_sign
        self.trauma = min(1.0, self.trauma + 0.22 * strength)

    def add_landing(self, impact_speed: float) -> None:
        self.land_velocity -= min(impact_speed * 0.22, 3.2)

    def add_step(self, height: float) -> None:
        """Hide a stair step: the camera lags behind and catches up smoothly."""
        self.step_offset = float(np.clip(self.step_offset - height, -C.STEP_HEIGHT * 1.5, C.STEP_HEIGHT * 1.5))

    def add_explosion_shake(self, cam_pos, explosion_positions, power: float = 14.0) -> None:
        """
        Camera trauma from explosions, vectorised over every blast this frame:
        ``trauma += power / max(d^2, 1)`` (inverse-square distance falloff).
        """
        points = np.atleast_2d(np.asarray(explosion_positions, dtype=np.float64))
        if points.size == 0:
            return
        d2 = np.sum((points - np.asarray(cam_pos, dtype=np.float64)) ** 2, axis=1)
        self.trauma = float(min(1.0, self.trauma + np.sum(power / np.maximum(d2, 1.0))))

    def consume_aim_kick(self) -> np.ndarray:
        """Aim offset (pitch, yaw) produced since the last call; add it to the view angles."""
        out = self._aim_delta.copy()
        self._aim_delta[:] = 0.0
        return out

    # ---- per frame ------------------------------------------------------------
    def update(self, dt: float, feet, velocity, mflags: int, crouch_t: float, yaw: float, pitch: float,
               look_delta=(0.0, 0.0), ads_zoom: float = 1.0, reloading: bool = False,
               recoil_recover: float = 6.0) -> None:
        dt = float(min(max(dt, 0.0), 0.1))
        self.time += dt
        on_ground = bool(mflags & C.MF_ON_GROUND)
        sliding = bool(mflags & C.MF_SLIDING)
        sprinting = bool(mflags & C.MF_SPRINTING)
        ads = bool(mflags & C.MF_ADS)
        vx, vy, vz = float(velocity[0]), float(velocity[1]), float(velocity[2])
        speed_h = float(np.hypot(vx, vz))
        sin_y, cos_y = float(np.sin(yaw)), float(np.cos(yaw))
        lateral = vx * cos_y - vz * sin_y
        move_frac = min(speed_h / C.SPRINT_SPEED, 1.25)

        # blends -------------------------------------------------------------
        self.sprint_blend = float(M.damp(self.sprint_blend, 1.0 if sprinting else 0.0, 9.0, dt))
        self.slide_blend = float(M.damp(self.slide_blend, 1.0 if sliding else 0.0, 12.0, dt))
        self.ads_blend = float(M.damp(self.ads_blend, 1.0 if ads else 0.0, 16.0, dt))
        self.reload_blend = float(M.damp(self.reload_blend, 1.0 if reloading else 0.0, 10.0, dt))

        # stride --------------------------------------------------------------
        striding = on_ground and not sliding
        self.bob_amount = float(M.damp(self.bob_amount, move_frac if striding else 0.0, 10.0, dt))
        if striding:
            self.bob_phase += speed_h * C.BOB_FREQUENCY * dt
        bob_scale = self.bob_amount * (1.0 - 0.75 * self.ads_blend)
        cam_bob_y = float(np.sin(2.0 * self.bob_phase)) * C.BOB_AMPLITUDE * bob_scale
        cam_bob_x = float(np.cos(self.bob_phase)) * C.BOB_AMPLITUDE * 0.5 * bob_scale

        # banking -------------------------------------------------------------
        if sliding and self.slide_blend < 0.2 and abs(lateral) > 0.6:
            self.slide_side = 1.0 if lateral > 0.0 else -1.0
        bank = -float(np.clip(lateral / C.WALK_SPEED, -1.3, 1.3)) * C.BANK_STRAFE_DEG * (1.0 - self.slide_blend)
        bank += C.BANK_SLIDE_DEG * self.slide_side * self.slide_blend
        self.roll = float(M.damp(self.roll, np.radians(bank), 9.0, dt))

        # landing spring (critically damped) -----------------------------------
        stiffness = 190.0
        self.land_offset, self.land_velocity = spring_step(self.land_offset, self.land_velocity, stiffness,
                                                           2.0 * np.sqrt(stiffness), dt)
        self.land_offset = float(np.clip(self.land_offset, -C.LAND_DIP * 2.0, 0.05))
        self.step_offset = float(M.damp(self.step_offset, 0.0, 14.0, dt))

        # recoil ---------------------------------------------------------------
        self.recoil_target *= float(np.exp(-recoil_recover * dt))
        step = (self.recoil_target - self.recoil_current) * (1.0 - float(np.exp(-28.0 * dt)))
        self.recoil_current += step
        self._aim_delta += step
        punch_k = 260.0
        self.punch, self.punch_velocity = spring_step(self.punch, self.punch_velocity, punch_k,
                                                      2.0 * np.sqrt(punch_k) * 0.85, dt)
        self.kickback = float(M.damp(self.kickback, 0.0, 16.0, dt))

        # explosion / damage shake ----------------------------------------------
        self.trauma = max(0.0, self.trauma - 1.25 * dt)
        shake = self.trauma * self.trauma
        t = self.time * 26.0
        shake_pitch = float(M.smooth_noise(t, 11)) * shake * np.radians(3.2)
        shake_yaw = float(M.smooth_noise(t, 23)) * shake * np.radians(3.2)
        shake_roll = float(M.smooth_noise(t, 37)) * shake * np.radians(5.0)
        self.shake_uv[0] = float(M.smooth_noise(t, 53)) * shake * 0.012
        self.shake_uv[1] = float(M.smooth_noise(t, 71)) * shake * 0.012

        # field of view ----------------------------------------------------------
        target_fov = (self.base_fov + C.SPRINT_FOV_BONUS * self.sprint_blend + C.SLIDE_FOV_BONUS * self.slide_blend)
        target_fov = target_fov * (1.0 - self.ads_blend) + (self.base_fov / max(ads_zoom, 1.0)) * self.ads_blend
        self.fov = float(M.damp(self.fov, target_fov, 12.0, dt))

        # camera pose --------------------------------------------------------------
        height = float(eye_height(crouch_t)) + self.land_offset + self.step_offset + cam_bob_y
        self.cam_pos[0] = float(feet[0]) + cos_y * cam_bob_x
        self.cam_pos[1] = float(feet[1]) + height
        self.cam_pos[2] = float(feet[2]) - sin_y * cam_bob_x
        self.cam_yaw = float(yaw) + float(self.punch[1]) + shake_yaw
        self.cam_pitch = float(np.clip(pitch + self.punch[0] + shake_pitch, -C.PITCH_LIMIT, C.PITCH_LIMIT))
        self.cam_roll = self.roll + float(self.punch[2]) + shake_roll

        # view-model sway: low frequency double sine driven by travelled distance -------
        hip = 1.0 - self.ads_blend
        idle = self.time
        sway_gain = C.SWAY_AMPLITUDE * (0.25 + move_frac) * (0.25 + 0.75 * hip)
        sway_x = (np.sin(self.bob_phase) + 0.5 * np.sin(2.0 * self.bob_phase + 0.6)) * sway_gain * self.bob_amount
        sway_y = (np.sin(2.0 * self.bob_phase + 1.1) + 0.5 * np.sin(0.5 * self.bob_phase)) * sway_gain * 0.8 * self.bob_amount
        breathe_x = (np.sin(idle * 0.9) + 0.5 * np.sin(idle * 2.3 + 1.3)) * 0.0022 * hip
        breathe_y = (np.sin(idle * 1.3 + 0.4) + 0.5 * np.sin(idle * 2.9)) * 0.0020 * hip
        lag_target = np.array([-float(look_delta[0]), -float(look_delta[1])]) * 0.045
        self.look_lag = M.damp(self.look_lag, np.clip(lag_target, -0.03, 0.03), 14.0, dt)
        air_drift = float(np.clip(vy * 0.004, -0.03, 0.03)) if not on_ground else 0.0
        self.vm_offset[0] = float(sway_x + breathe_x + self.look_lag[0]) - lateral * 0.0016 * hip
        self.vm_offset[1] = (float(sway_y + breathe_y + self.look_lag[1]) - air_drift + self.land_offset * 0.35
                             - 0.055 * self.sprint_blend - 0.11 * self.reload_blend - 0.02 * self.slide_blend)
        self.vm_offset[2] = -self.kickback - 0.03 * self.sprint_blend
        self.vm_rotation[0] = (float(self.punch[0]) * 0.6 - self.kickback * 0.9 + np.radians(14.0) * self.sprint_blend
                               - np.radians(24.0) * self.reload_blend)
        self.vm_rotation[1] = float(self.look_lag[0]) * 1.8 + np.radians(-22.0) * self.sprint_blend
        self.vm_rotation[2] = (self.roll * 1.6 + float(self.punch[2]) * 0.5 + np.radians(12.0) * self.slide_blend * self.slide_side
                               + float(self.look_lag[0]) * 2.4 + np.radians(8.0) * self.reload_blend)
